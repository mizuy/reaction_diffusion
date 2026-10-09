"""Fixed-threshold, periodic morphology measurements; never used by evolution.

Branch graph removes redundant diagonal edges when an orthogonal route exists.
Short terminal spurs are removed before contracting junction clusters.
"""
from collections import deque
import cv2
import numpy as np
from evaluate_pit_steps import zhang_suen_thin


def periodic_labels(binary,connectivity=8):
    n,labels=cv2.connectedComponents(binary.astype(np.uint8),connectivity=connectivity)
    parent=np.arange(n)
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        if a and b:parent[root(a)]=root(b)
    height,width=binary.shape
    shifts=[-1,0,1] if connectivity==8 else [0]
    for j in range(width):
        for shift in shifts:union(labels[0,j],labels[-1,(j+shift)%width])
    for i in range(height):
        for shift in shifts:union(labels[i,0],labels[(i+shift)%height,-1])
    roots=np.array([root(i) for i in range(n)])
    out=roots[labels]
    ids=np.unique(out);ids=ids[ids>0]
    mapping=np.zeros(n,dtype=int);mapping[ids]=np.arange(1,len(ids)+1)
    return len(ids),mapping[out]


def skeleton_graph(sk):
    shape=sk.shape
    coords=list(zip(*np.where(sk)))
    index={p:i for i,p in enumerate(coords)}
    adjacency=[set() for _ in coords]
    for i,(y,x) in enumerate(coords):
        for dy,dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            q=((y+dy)%shape[0],(x+dx)%shape[1])
            if q not in index:continue
            if dy and dx and (sk[(y+dy)%shape[0],x] or sk[y,(x+dx)%shape[1]]):continue
            adjacency[i].add(index[q])
    return coords,adjacency


def branches(sk,dx=1):
    coords,adj=skeleton_graph(sk)
    active=set(range(len(coords)))
    # Remove terminal branches shorter than 2 physical units, not whole rods.
    for _ in range(3):
        remove=set()
        for start in active:
            if len(adj[start]&active)!=1:continue
            path=[start];previous=None;current=start
            while len(path)<=max(2,int(2/dx)):
                nexts=(adj[current]&active)-({previous} if previous is not None else set())
                if len(nexts)!=1:break
                nxt=next(iter(nexts));degree=len(adj[nxt]&active)
                if degree>=3:remove.update(path);break
                if degree==1:break
                previous,current=current,nxt;path.append(current)
        if not remove:break
        active-=remove
    junctions={i for i in active if len(adj[i]&active)>=3}
    clusters=[];unseen=set(junctions)
    while unseen:
        seed=unseen.pop();cluster={seed};todo=[seed]
        while todo:
            node=todo.pop()
            for j in adj[node]&unseen:
                unseen.remove(j);cluster.add(j);todo.append(j)
        clusters.append(cluster)
    count=0;arm_lengths=[]
    for cluster in clusters:
        corridors=[];used=set()
        for start in sorted(set().union(*(adj[i] for i in cluster))-cluster):
            if start not in active or start in used:continue
            previous=next(iter(adj[start]&cluster));current=start;path=[];length=0.0
            for _ in range(len(active)+1):
                y,x=coords[current];yp,xp=coords[previous]
                dy=min(abs(y-yp),sk.shape[0]-abs(y-yp));ddx=min(abs(x-xp),sk.shape[1]-abs(x-xp))
                length+=np.hypot(dy,ddx)*dx
                path.append(current)
                if current in junctions or len(adj[current]&active)!=2:break
                nxts=(adj[current]&active)-{previous}
                if not nxts:break
                previous,current=current,next(iter(nxts))
                if current in cluster:break
            used.update(path);corridors.append(length)
        good=[l for l in corridors if l>=3.0]
        if len(good)>=3:count+=1;arm_lengths.extend(good)
    return count,arm_lengths


def measure(u,dx=1,threshold=0):
    binary=u>threshold;n=binary.shape[0]
    # Tile for correct boundary geometry, crop once skeleton/EDT are computed.
    tiled=np.tile(binary,(3,3))
    sk=zhang_suen_thin(tiled)[n:2*n,n:2*n]
    dist=cv2.distanceTransform(tiled.astype(np.uint8),cv2.DIST_L2,5)[n:2*n,n:2*n]
    widths=2*dist[sk]*dx
    count,labels=periodic_labels(binary,8)
    bg_count,_=periodic_labels(~binary,4)
    areas=np.bincount(labels.ravel())[1:]*dx*dx
    lengths=np.bincount(labels[sk],minlength=count+1)[1:]*dx
    width=float(np.median(widths)) if len(widths) else 0
    ratios=[];aspect=[]
    for lab in range(1,count+1):
        yy,xx=np.where(labels==lab)
        if len(xx)<5:continue
        # Periodic circular centering for component covariance; unreliable once
        # components wrap globally, so elongation is for finite early patterns.
        xx=xx*dx;yy=yy*dx;L=n*dx
        cx=np.angle(np.mean(np.exp(2j*np.pi*xx/L)))*L/(2*np.pi)
        cy=np.angle(np.mean(np.exp(2j*np.pi*yy/L)))*L/(2*np.pi)
        x=(xx-cx+L/2)%L-L/2;y=(yy-cy+L/2)%L-L/2
        vals=np.linalg.eigvalsh(np.cov(np.array([x,y])))
        aspect.append(np.sqrt(max(vals[1],1e-10)/max(vals[0],1e-10)))
        local_width=2*dist[sk&(labels==lab)]*dx
        w=float(np.median(local_width)) if len(local_width) else width
        ratios.append(lengths[lab-1]/max(w,dx))
    branch_count,arm_lengths=branches(sk,dx)
    gy,gx=np.gradient(u,dx)
    band=np.exp(-(u/.3)**2)
    a=np.sum(band*gx*gx);b=np.sum(band*gy*gy);c=np.sum(band*gx*gy)
    orientation=np.sqrt((a-b)**2+4*c*c)/max(a+b,1e-10)
    return dict(coverage=float(binary.mean()),components=count,background_components=bg_count,
                largest_fraction=float(areas.max()/areas.sum()) if len(areas) else 0,
                width_median=width,width_p90=float(np.percentile(widths,90)) if len(widths) else 0,
                component_length_width_p90=float(np.percentile(ratios,90)) if ratios else 0,
                elongation_p90=float(np.percentile(aspect,90)) if aspect else 0,
                branch_count=branch_count,branches_per_component=branch_count/max(count,1),
                arm_length_p90=float(np.percentile(arm_lengths,90)) if arm_lengths else 0,
                orientation=float(orientation))


def classify(metrics,reference_width,area=128*128):
    r=metrics
    if not .6<=r['width_median']/max(reference_width,1e-10)<=1.5 or r['orientation']>=.5:
        return 'unresolved'
    branched=r['branch_count']*128*128/area>=3
    mesh=r['largest_fraction']>=.4 or r['background_components']*128*128/area>=6
    if branched and mesh:return 'IV_V'
    if branched and r['component_length_width_p90']<=12:return 'IV_B'
    if not branched and r['branch_count']==0 and r['elongation_p90']<1.8 and r['component_length_width_p90']<2:return 'I'
    if not branched and r['elongation_p90']<=3.5 and r['component_length_width_p90']<=6:return 'IIIL'
    return 'unresolved'
