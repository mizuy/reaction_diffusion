#!/usr/bin/env python3
"""Export all successful and unsuccessful series with fixed, provisional labels."""
import argparse,base64,csv,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pit_microphase_metrics import classify,measure


def png(u,pixels=256):
    # Display the zero contour of the interpolated field. Metrics always use
    # the original simulation grid; no image smoothing feeds back into dynamics.
    view=np.array(Image.fromarray(u.astype(np.float32)).resize((pixels,pixels),Image.Resampling.BICUBIC))
    rgb=np.empty((pixels,pixels,3),dtype=np.uint8);rgb[:]=[231,222,204];rgb[view>0]=[32,62,74]
    return Image.fromarray(rgb)


def ordered_four(labels):
    required=['I','IIIL','IV_B','IV_V'];i=0
    for label in labels:
        if label==required[i]:
            i+=1
            if i==4:return True
    return False


def figure(series,output,select_means=None):
    series={k:v for k,v in series.items()}
    if select_means is not None:
        series={k:[c for c in v if any(abs(c['metrics']['m']-m)<1e-9 for m in select_means)] for k,v in series.items()}
    cols=max(map(len,series.values()));cell=256;ch=cell+65
    image=Image.new('RGB',(cols*cell,len(series)*(ch+25)),'#17262b');d=ImageDraw.Draw(image)
    for row,(name,cases) in enumerate(series.items()):
        top=row*(ch+25);d.text((8,top+5),name,fill='white')
        for col,c in enumerate(cases):
            x=col*cell;y=top+25;r=c['metrics'];image.paste(png(c['u']),(x,y))
            d.text((x+8,y+cell+6),f"m={r['m']:.3f}  {r['label']}",fill='white')
            d.text((x+8,y+cell+23),f"width x{r['width_ratio']:.2f} | forks {r['branch_count']} | largest {r['largest_fraction']:.2f}",fill='white')
            d.text((x+8,y+cell+40),f"length/width p90={r['component_length_width_p90']:.1f}",fill='white')
    image.save(output)


def html(series,out):
    data=[]
    for name,cases in series.items():
        for c in cases:
            buf=io.BytesIO();png(c['u'],384).save(buf,format='PNG')
            data.append(dict(series=name,metrics=c['metrics'],image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))
    page='''<!doctype html><meta charset="utf-8"><title>Pit pattern — one model</title>
+<style>body{font:16px system-ui;background:#17262b;color:#eee;max-width:1100px;margin:25px auto}p{line-height:1.7}select,input{font-size:16px;margin:10px}img{width:512px;max-width:90vw;image-rendering:pixelated}pre{white-space:pre-wrap}a{color:#9de}</style>
+<h1>I → IIIL → IV_B → IV_V：一つの非局所相分離モデル</h1>
+<p>平均秩序量mのみを操作。主系列は係数と間質不均一場を全mで固定し、前の場を引き継いで計算しています。青はpit/溝候補、背景はその間の領域。ラベルは形態指標からの暫定分類です。unresolvedは四分類に無理に当てはめていません。</p>
+<label>系列<select id="series"></select></label><br><label>m<input type="range" id="slider" min="0" step="1"><span id="label"></span></label><br>
+<img id="pattern"><pre id="metrics"></pre>
+<p>異なるmの間を画像補間していません。各フレームは計算した場です。I〜IV_Vは臨床診断の確定結果ではなく、特にIV_Bの範囲は履歴・seedで変化します。これは現象論的な表面模様のモデルであり、3D組織学的villousの再現ではありません。</p>
+<script>const data=DATA;const sel=document.getElementById('series'),s=document.getElementById('slider');let cases=[];
+for(const name of new Set(data.map(c=>c.series))){let o=document.createElement('option');o.textContent=name;sel.add(o)}
+function draw(){let c=cases[+s.value];document.getElementById('pattern').src=c.image;document.getElementById('label').textContent=c.metrics.m.toFixed(3)+' / '+c.metrics.label;document.getElementById('metrics').textContent=JSON.stringify(c.metrics,null,2)}
+function change(){cases=data.filter(c=>c.series===sel.value);s.max=cases.length-1;s.value=0;draw()}sel.onchange=change;s.oninput=draw;change();</script>'''.replace('\n+','\n')
    (out/'interactive.html').write_text(page.replace('DATA',json.dumps(data)))


def plots(series,out):
    from summarize_pit_ring import plot_panel
    image=Image.new('RGB',(1200,960),'white');draw=ImageDraw.Draw(image)
    routes=['main_seed7','seed1','seed42','hold']
    colors=['#156f92','#ce682a','#914793','#299354']
    specs=[('branch_count',(0,50),'Branch junction count'),
           ('largest_fraction',(0,1),'Largest connected pit fraction'),
           ('width_ratio',(.8,1.6),'Width relative to initial I'),
           ('component_length_width_p90',(0,120),'Component skeleton length / width, p90')]
    for i,(key,ylim,title) in enumerate(specs):
        row,col=divmod(i,2);left=65+col*590;top=70+row*465
        values=[(route,[(c['metrics']['m'],c['metrics'][key]) for c in series[route]],color)
                for route,color in zip(routes,colors) if route in series]
        if key=='component_length_width_p90':
            ylim=(0,max(y for _,points,_ in values for _,y in points)*1.05)
        plot_panel(draw,(left,top,left+490,top+340),values,(-.55,.25),ylim,title,'mean m',key)
    image.save(out/'order_parameters.png')
    frames=[]
    for c in series['main_seed7']:
        frame=Image.new('RGB',(512,565),'#17262b');frame.paste(png(c['u'],512),(0,0))
        d=ImageDraw.Draw(frame);r=c['metrics']
        d.text((10,523),f"m={r['m']:.3f} | {r['label']} | same field, stepwise parameter continuation",fill='white')
        frames.append(frame)
    frames[0].save(out/'progression.gif',save_all=True,append_images=frames[1:],duration=900,loop=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/pit_microphase'));args=parser.parse_args()
    root=args.root;out=root/'summary';out.mkdir(parents=True,exist_ok=True)
    paths=[root/'fine',*sorted((root/'validation').glob('*')),
           *sorted((root/'candidate2').glob('*')),*sorted((root/'search').glob('large*'))]
    series={};summary=[];flat=[]
    for path in paths:
        if not (path/'report.json').exists():continue
        report=json.loads((path/'report.json').read_text());saved=np.load(path/'fields.npz');cfg=report['config'];records=report['records']
        order=np.argsort([r['m'] for r in records]);reference=next((r['width_median'] for r in records if abs(r['m']+.55)<1e-8),7.19378662109375)
        if reference<=1e-10:reference=7.19378662109375  # empty ablation has no I width
        name='main_seed7' if path.name=='fine' else path.name
        if path.parent.name=='candidate2':name='eta012_'+name
        cases=[]
        for i in order:
            r={k:v for k,v in records[i].items() if k!='history'}
            r.update(label=classify(r,reference,(cfg['size']*cfg['dx'])**2),width_ratio=r['width_median']/reference)
            c=dict(metrics=r,u=saved['u'][i]);cases.append(c);flat.append(dict(series=name,**r))
        series[name]=cases
        labels=[c['metrics']['label'] for c in cases]
        energies=[max(np.diff([h['energy_density'] for h in r['history']])) for r in records]
        summary.append(dict(series=name,labels=labels,contains_ordered_four=ordered_four(labels),unresolved_count=labels.count('unresolved'),
                            eta=cfg['heterogeneity'],domain_length=cfg['size']*cfg['dx'],
                            max_mass_error=max(abs(r['mean']-r['m']) for r in records),
                            max_energy_increase_within_hold=float(max(energies)),
                            max_late_binary_change=max(r['late_binary_change'] for r in records)))
    if 'seed42_transition' in series and 'seed42' in series:
        # Only join at the saved m=.05 state; the refined run recomputes the
        # entire suffix, rather than splicing back to a different trajectory.
        combined=[c for c in series['seed42'] if c['metrics']['m']<=.05]+series['seed42_transition']
        series['seed42_refined_path']=combined
        labs=[c['metrics']['label'] for c in combined]
        summary.append(dict(series='seed42_refined_path',labels=labs,contains_ordered_four=ordered_four(labs),unresolved_count=labs.count('unresolved')))
    keys=sorted(set().union(*(r.keys() for r in flat)))
    with (out/'all_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    (out/'series_verdicts.json').write_text(json.dumps(summary,indent=2))
    figure({'main seed7; only mean m changes':series['main_seed7']},out/'four_patterns.png',[-.55,-.15,0,.2])
    figure({k:series[k] for k in ['main_seed7','seed1','seed42'] if k in series},out/'seeds.png')
    figure({k:series[k] for k in ['main_seed7','hold','independent','reverse'] if k in series},out/'history_and_hold.png')
    figure({k:series[k] for k in ['dt04','dt02','dx05'] if k in series},out/'numerical_checks.png')
    html(series,out)
    plots(series,out)
    # Threshold sensitivity on the same four states, not retuned trajectories.
    rows=[]
    for c in series['main_seed7']:
        if not any(abs(c['metrics']['m']-m)<1e-9 for m in [-.55,-.15,0,.2]):continue
        for threshold in [-.1,0,.1]:
            r=measure(c['u'],threshold=threshold)
            rows.append(dict(m=c['metrics']['m'],threshold=threshold,label=classify(r,7.19378662109375),**r))
    (out/'threshold_sensitivity.json').write_text(json.dumps(rows,indent=2))
    numerical=[]
    if all(k in series for k in ['dt04','dt02','dx05']):
        for a,b,c in zip(series['dt04'],series['dt02'],series['dx05']):
            ma,mb,mc=a['metrics'],b['metrics'],c['metrics']
            numerical.append(dict(m=ma['m'],dt_rms=float(np.sqrt(np.mean((a['u']-b['u'])**2))),
                                  dt_binary_change=float(np.mean((a['u']>0)!=(b['u']>0))),
                                  dx_width_ratio=mc['width_median']/ma['width_median'],
                                  dx_largest_difference=mc['largest_fraction']-ma['largest_fraction'],
                                  labels=[ma['label'],mb['label'],mc['label']]))
    (out/'numerical_comparison.json').write_text(json.dumps(numerical,indent=2))
    lines=['# 非局所相分離モデル：検証一覧','', '| 系列 | 四形態を順に含む | ラベル列（m昇順） |','|---|---|---|']
    lines.extend(f"| {r['series']} | {r['contains_ordered_four']} | {' → '.join(r['labels'])} |" for r in summary)
    (out/'validation_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
