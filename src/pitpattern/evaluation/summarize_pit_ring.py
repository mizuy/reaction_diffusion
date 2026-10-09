#!/usr/bin/env python3
"""Inspect saved simulations; geometry measurements do not feed back into forces."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitpattern.metrics.evaluate_pit_steps import zhang_suen_thin, prepare_skeleton_topology
from pitpattern.metrics.pit_goal_metrics import measure_feature_width
from pitpattern.models.simulate_pit_ring import contact_sheet


def geometry(xy, pixels_per_unit=64):
    shifted=(xy-xy.min(axis=0))*pixels_per_unit+12
    size=np.ceil(shifted.max(axis=0)).astype(int)+13
    binary=np.zeros((size[1],size[0]),dtype=np.uint8)
    cv2.fillPoly(binary,[np.rint(shifted).astype(np.int32)],1)
    skeleton=zhang_suen_thin(binary.astype(bool))
    sk,branches=prepare_skeleton_topology(skeleton,prune_spurs=4,min_arm_length=12)
    widths=measure_feature_width(binary.astype(bool),sk)
    branch_count=cv2.connectedComponents(branches.astype(np.uint8),8)[0]-1
    return dict(width_median=widths['feature_width_median']/pixels_per_unit,
                width_p90=widths['feature_width_p90']/pixels_per_unit,
                three_way_junctions=int(branch_count),pixels_per_unit=pixels_per_unit)


def plot_panel(draw, box, series, xlim, ylim, title, xlabel, ylabel):
    l,t,r,b=box
    def xy(x,y):
        return (l+(x-xlim[0])/(xlim[1]-xlim[0])*(r-l),b-(y-ylim[0])/(ylim[1]-ylim[0])*(b-t))
    draw.text((l,t-34),title,fill='#152c39')
    for val in np.linspace(ylim[0],ylim[1],6):
        _,yy=xy(xlim[0],val)
        draw.line((l,yy,r,yy),fill='#dddddd')
        draw.text((l-44,yy-5),f'{val:.2g}',fill='#152c39')
    for val in np.linspace(xlim[0],xlim[1],6):
        xx,_=xy(val,ylim[0])
        draw.text((xx-8,b+8),f'{val:.2g}',fill='#152c39')
    draw.rectangle(box,outline='#152c39')
    for idx,(label,points,color) in enumerate(series):
        points=[xy(x,y) for x,y in points if ylim[0]<=y<=ylim[1]]
        if len(points)>1: draw.line(points,fill=color,width=3)
        draw.text((l+8,t+8+idx*18),label,fill=color)
    draw.text(((l+r)/2-40,b+30),xlabel,fill='#152c39')
    draw.text((l,t-18),ylabel,fill='#152c39')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/pit_ring/validation'))
    args=parser.parse_args()
    output=args.output
    data=json.loads((output/'validation.json').read_text())
    contours=np.load(output/'validation_contours.npz')
    cases=data['cases']
    for i,c in enumerate(cases): c['xy']=contours[f'case_{i:03}']
    def select(route,seed=7):
        return sorted([c for c in cases if c['metrics']['path']==route and c['metrics']['seed']==seed],key=lambda c:c['metrics']['growth'])
    main_cases=select('independent')
    forward,reverse=select('forward'),select('reverse')
    contact_sheet(main_cases+forward+reverse,output/'history.png','Rows: independent initialization / increasing g / decreasing g. Same model, different histories.',9)
    levels=[0.1,0.2,0.5,0.7,0.9]
    refinement=[c for c in main_cases if c['metrics']['growth'] in levels]
    for modes in [36,48]:
        refinement.extend(c for c in select('refinement') if c['metrics']['modes']==modes)
    contact_sheet(refinement,output/'refinement.png','Rows: 24, 36, 48 Fourier modes. Higher resolution relaxes the embedded 24-mode solution.',5)
    rows=[]
    for c in main_cases+forward+reverse:
        m=c['metrics']
        for resolution in [64,128]:
            rows.append(dict(path=m['path'],growth=m['growth'],**geometry(c['xy'],resolution)))
    with (output/'geometry.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    circles=json.loads((output/'circular_stability.json').read_text())
    fig=Image.new('RGB',(1200,500),'white');draw=ImageDraw.Draw(fig)
    colors=['#247ba0','#e06c32','#9a419c']
    series=[]
    critical=[]
    for ks,color in zip([0.03,0.3,3.0],colors):
        arr=[c for c in circles if c['stroma']==ks]
        series.append((f'Ks={ks}',[(c['growth'],c['shape_eigenvalue']) for c in arr],color))
        i=next(i for i,c in enumerate(arr) if c['shape_eigenvalue']<0)
        critical.append(dict(stroma=ks,g_lower=arr[i-1]['growth'],g_upper=arr[i]['growth'],mode=arr[i]['unstable_mode']))
    plot_panel(draw,(65,75,565,420),series,(0,0.05),(-4,6),'Circular state loses shape stability','growth demand g','smallest shape Hessian eigenvalue')
    series=[]
    for route,color in zip(['independent','forward','reverse'],colors):
        series.append((route,[(c['metrics']['growth'],c['metrics']['elongation']) for c in select(route)],color))
    plot_panel(draw,(670,75,1160,420),series,(0,0.9),(0,10),'Elongation depends on history (Ks=0.3)','growth demand g','filled-lumen aspect ratio')
    fig.save(output/'stability_and_history.png')
    details={'circle_instability_brackets':critical,
             'max_perturbed_relative_diameter_change':max(abs(c['metrics']['relative_diameter_change']) for c in cases if c['metrics']['path']=='perturbation'),
             'max_perturbed_relative_energy_change':max(abs(c['metrics']['relative_energy_change']) for c in cases if c['metrics']['path']=='perturbation'),
             'all_cases_converged':all(c['metrics']['converged'] for c in cases),
             'max_gradient':max(c['metrics']['gradient_max'] for c in cases),
             'minimum_hessian':min(c['metrics']['hessian_min'] for c in cases)}
    (output/'diagnostics.json').write_text(json.dumps(details,indent=2))
    print(json.dumps(details,indent=2))
    for r in rows:
        if r['path']=='independent' and r['pixels_per_unit']==128: print(r)


if __name__=='__main__':
    main()
