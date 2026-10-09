#!/usr/bin/env python3
"""Reproducible robustness, ablation and circular-instability experiments."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from simulate_pit_ring import (Parameters, Ring, minimize, measure, hessian,
                               contact_sheet, write_cases)


def case(ring, growth, initial, seed, path):
    x, info = minimize(ring, growth, initial, maxiter=2500)
    metrics = measure(ring, x, growth, info, curvature_check=True)
    metrics.update(seed=seed, path=path, modes=ring.p.modes, points=ring.p.points,
                   area_stiffness=ring.p.area)
    return dict(metrics=metrics, coefficients=x.tolist(), xy=ring.coordinates(x), trace=info['trace'])


def circular_state(ring, growth):
    x=np.zeros(ring.b.shape[1])
    for _ in range(50):
        grad=ring.energy(x,growth)[1][0]
        dx=np.zeros_like(x)
        dx[0]=1e-5
        curvature=(ring.energy(x+dx,growth)[1][0]-ring.energy(x-dx,growth)[1][0])/2e-5
        x[0]-=np.clip(grad/curvature,-0.1,0.1)
        if abs(grad)<1e-10:
            break
    matrix=hessian(ring,x,growth)
    eig=np.linalg.eigvalsh(matrix[1:,1:])
    index=int(np.argmin(np.diag(matrix)[1:]))+1
    return dict(growth=float(growth),stroma=ring.p.stroma,
                radius=float(np.exp(x[0])),shape_eigenvalue=float(eig[0]),
                unstable_mode=ring.frequencies[index])


def html_view(cases, path):
    payload=json.dumps([{'xy':c['xy'].tolist(),'metrics':c['metrics']} for c in cases])
    page='''<!doctype html><meta charset="utf-8"><title>Pit ring simulation</title>
<style>body{font:16px system-ui;background:#182126;color:#eee;max-width:1000px;margin:30px auto}canvas{background:#e6dcc8;width:640px;height:500px}select,input{font-size:16px;margin:10px}pre{white-space:pre-wrap}p{line-height:1.6}</style>
<h1>管腔を囲む上皮の力学モデル</h1>
<p>計算済みの平衡形状です。青：管腔、赤：上皮壁、背景：局所的な間質支持。画像を切って長さを制限する処理はありません。単一腺口のモデルなので合流・villousは未実装です。</p>
<label>計算系列 <select id="series"></select></label><br>
<label>余剰周長要求 g <input id="slider" type="range" min="0" step="1"><span id="value"></span></label><br>
<canvas id="canvas" width="800" height="625"></canvas><pre id="metrics"></pre>
<p>全系列で同じ座標スケール。独立初期化の隣り合う結果は物理時間の動画ではありません。ARは管腔面積分布から求めた伸長比です。数値は無次元であり、実測の物性値に較正していません。</p>
<script>const cases=DATA;const s=document.getElementById('series'), slider=document.getElementById('slider');
const key=c=>`${c.metrics.path} / Ks=${c.metrics.stroma} / seed=${c.metrics.seed} / modes=${c.metrics.modes} / KA=${c.metrics.area_stiffness}`;
const keys=[...new Set(cases.map(key))];for(const k of keys){const o=document.createElement('option');o.textContent=k;s.add(o)}
const extent=Math.max(1.7,...cases.flatMap(c=>c.xy.flat().map(Math.abs)))*1.12;let subset=[];
function draw(){const c=subset[+slider.value], m=c.metrics;document.getElementById('value').textContent=m.growth.toFixed(3);
const ctx=document.getElementById('canvas').getContext('2d');ctx.clearRect(0,0,800,625);ctx.beginPath();c.xy.forEach((p,i)=>{const x=400+p[0]*270/extent,y=310-p[1]*270/extent;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.fillStyle='#263f4b';ctx.fill();ctx.strokeStyle='#b24948';ctx.lineWidth=4;ctx.stroke();
document.getElementById('metrics').textContent=JSON.stringify(m,null,2)}
function change(){subset=cases.filter(c=>key(c)===s.value).sort((a,b)=>a.metrics.growth-b.metrics.growth);slider.max=subset.length-1;slider.value=0;draw()}s.onchange=change;slider.oninput=draw;change();</script>'''
    path.write_text(page.replace('DATA',payload))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/pit_ring/validation'))
    args=parser.parse_args()
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    growths=[0,0.025,0.05,0.1,0.2,0.35,0.5,0.7,0.9]
    base=Parameters(stroma=0.3,modes=24,points=384)
    cases=[]
    for seed in [1,7,42]:
        ring=Ring(base)
        for g in growths:
            c=case(ring,g,ring.initial(seed),seed,'independent')
            cases.append(c)
        print(f'independent seed={seed} done',flush=True)
    for route in ['forward','reverse']:
        ring=Ring(base)
        previous=ring.initial(7)
        for g in sorted(growths,reverse=route=='reverse'):
            c=case(ring,g,previous+0.1*ring.initial(8),7,route)
            previous=np.array(c['coefficients'])
            cases.append(c)
        print(f'{route} done',flush=True)
    # Embed the same low-resolution equilibrium before refinement: distinguish
    # discretization change from independent attraction-basin selection.
    for modes,points in [(36,576),(48,768)]:
        ring=Ring(replace(base,modes=modes,points=points))
        for g in [0.1,0.2,0.5,0.7,0.9]:
            old=next(c for c in cases if c['metrics']['path']=='independent' and c['metrics']['seed']==7 and c['metrics']['growth']==g)
            initial=np.zeros(ring.b.shape[1]);initial[:len(old['coefficients'])]=old['coefficients']
            c=case(ring,g,initial,7,'refinement')
            c['metrics']['energy_before_refinement']=ring.energy(initial,g)[0]
            cases.append(c)
        print(f'refinement modes={modes} done',flush=True)
    ring=Ring(replace(base,area=0))
    for g in growths:
        cases.append(case(ring,g,ring.initial(7),7,'no_area_constraint'))
    print('area ablation done',flush=True)
    for original in [c for c in cases if c['metrics']['path']=='independent' and c['metrics']['seed']==7]:
        ring=Ring(base)
        g=original['metrics']['growth']
        perturbed=np.array(original['coefficients'])+ring.initial(101)*2
        c=case(ring,g,perturbed,101,'perturbation')
        c['metrics']['relative_energy_change']=(c['metrics']['energy']-original['metrics']['energy'])/max(1,original['metrics']['energy'])
        c['metrics']['relative_diameter_change']=c['metrics']['diameter']/original['metrics']['diameter']-1
        cases.append(c)
    print('perturbations done',flush=True)
    circles=[]
    for ks in [0.03,0.3,3.0]:
        ring=Ring(replace(base,stroma=ks,modes=12,points=192))
        circles.extend(circular_state(ring,float(g)) for g in np.linspace(0,0.08,161))
    (output/'circular_stability.json').write_text(json.dumps(circles,indent=2))
    write_cases(cases,output,asdict(base),'validation')
    main_cases=[c for c in cases if c['metrics']['path']=='independent' and c['metrics']['seed']==7]
    contact_sheet(main_cases,output/'main_sequence.png','Same model, same stromal support Ks=0.3; only growth demand g changes',len(growths))
    compare=[c for c in cases if c['metrics']['path']=='independent']
    contact_sheet(compare,output/'seeds.png','Independent initial perturbations: seeds 1, 7, 42 (rows)',len(growths))
    comparison=[c for c in cases if c['metrics']['path']=='no_area_constraint']
    contact_sheet(main_cases+comparison,output/'area_ablation.png','Top: finite lumen area; bottom: area constraint removed',len(growths))
    html_view(cases,output/'interactive.html')
    summary={'cases':len(cases),'converged':sum(c['metrics']['converged'] for c in cases),
             'negative_hessian_below_minus_1e_minus_4':sum(c['metrics']['hessian_min'] < -1e-4 for c in cases)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
