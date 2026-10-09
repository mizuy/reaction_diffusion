#!/usr/bin/env python3
"""Extend the existing one-parameter trajectory into inverse-spot morphology."""
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from simulate_pit_microphase import Config,run
from pit_microphase_metrics import measure
from summarize_pit_microphase import png


def panel(cases,output,title,columns=4):
    size=256;height=310
    rows=(len(cases)+columns-1)//columns
    image=Image.new('RGB',(columns*size,40+rows*height),'#17262b');draw=ImageDraw.Draw(image)
    draw.text((10,12),title,fill='white')
    for i,(m,u,description) in enumerate(cases):
        row,col=divmod(i,columns);x=col*size;y=40+row*height
        image.paste(png(u,size),(x,y))
        draw.text((x+10,y+size+6),f'm={m:.2f} | blue coverage {np.mean(u>0):.1%}',fill='white')
        draw.text((x+10,y+size+23),description,fill='white')
    image.save(output)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('artifacts/pit_microphase/fine'))
    p.add_argument('--output',type=Path,default=Path('artifacts/pit_microphase/high_m'))
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    source=np.load(args.source/'fields.npz')
    old=json.loads((args.source/'report.json').read_text())
    cfg=replace(Config(**old['config']),steps=3000)
    index=int(np.argmax(source['means']))
    means=[.30,.35,.40,.45,.50,.55,.60,.65,.70]
    if not np.isclose(source['means'][index],.25):raise ValueError('Expected saved m=.25 trajectory endpoint')
    fields,records=run(cfg,means,out/'extension','forward_from_saved_m0.25',heterogeneity_field=source['h'],initial_state=source['u'][index])
    inverse=[]
    for m,u,record in zip(means,fields,records):
        r=measure(-u,dx=cfg.dx)
        inverse.append(dict(m=m,blue_coverage=record['coverage'],blue_width=record['width_median'],
                            background_coverage=r['coverage'],background_components=r['components'],
                            background_largest_fraction=r['largest_fraction'],background_elongation_p90=r['elongation_p90'],
                            background_width=r['width_median'],background_branches=r['branch_count']))
    with (out/'inverse_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=inverse[0].keys());writer.writeheader();writer.writerows(inverse)
    indices=[1,5,8]
    held,hold_records=run(replace(cfg,steps=8000),[means[i] for i in indices],out/'hold','hold',
                           initial_fields=[fields[i] for i in indices],heterogeneity_field=source['h'])
    checks=[]
    for i,u,r in zip(indices,held,hold_records):
        im=measure(-u,dx=cfg.dx)
        checks.append(dict(m=means[i],binary_change_from_extension=float(np.mean((u>0)!=(fields[i]>0))),
                           late_binary_change=r['late_binary_change'],rhs_rms=r['rhs_rms'],
                           background_components=im['components'],background_elongation_p90=im['elongation_p90'],
                           background_largest_fraction=im['largest_fraction']))
    (out/'hold_checks.json').write_text(json.dumps(checks,indent=2))
    display=[]
    for m in [.15,.20,.25]:
        i=int(np.argmin(abs(source['means']-m)));display.append((m,source['u'][i],'saved original trajectory'))
    for m,u,r in zip(means,fields,inverse):
        display.append((m,u,f"background: {r['background_components']} units, AR90 {r['background_elongation_p90']:.2f}"))
    panel(display,out/'extended_sequence.png','Same model and material field; only m increases from 0.15 to 0.70')
    overview=[c for c in display if any(np.isclose(c[0],m) for m in [.2,.35,.55,.65])]
    panel(overview,out/'overview.png','Same colors throughout: connected network -> fragmented background -> inverse spots')
    i=int(np.argmin(abs(source['means']+.55)));j=means.index(.55)
    panel([(-.55,source['u'][i],'original I-like state'),(.35,fields[1],'denser connected pattern'),
           (.55,fields[j],'background becomes isolated spots'),(.55,-fields[j],'same field, colors reversed for comparison')],
          out/'inversion_comparison.png','Original spots, dense network, inverse spots; last panel is visualization only')
    frames=[]
    for m,u,description in display:
        im=Image.new('RGB',(512,555),'#17262b');im.paste(png(u,512),(0,0));d=ImageDraw.Draw(im)
        d.text((10,524),f'm={m:.2f} | same trajectory continued; no shape editing',fill='white');frames.append(im)
    frames[0].save(out/'extension.gif',save_all=True,append_images=frames[1:],duration=900,loop=0)
    validation=dict(source=str(args.source),config=cfg.__dict__,
                    maximum_mass_error=max(abs(r['mean']-r['m']) for r in records+hold_records),
                    maximum_recorded_energy_increase=max(max(np.diff([h['energy_density'] for h in r['history']])) for r in records+hold_records))
    (out/'provenance.json').write_text(json.dumps(validation,indent=2))
    print('INVERSE',json.dumps(inverse),flush=True);print('HOLD',json.dumps(checks),flush=True)

if __name__=='__main__':main()
