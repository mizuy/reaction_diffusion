#!/usr/bin/env python3
"""Nonlocal phase separation: one mean-composition parameter, fixed disorder.

u_t = Laplacian(u^3-u-epsilon^2 Laplacian(u)+eta*h)-alpha*(u-m).
Periodic domain. The zero Fourier mode is constrained to mean m.
No pattern classifier, cutting, prescribed pit shapes or type-specific switches
participate in the evolution. This is a phenomenological morphological model.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

@dataclass
class Config:
    size:int=128
    dx:float=1.0
    dt:float=0.4
    steps:int=3000
    epsilon:float=1.0
    alpha:float=0.025
    heterogeneity:float=0.05
    correlation:float=3.0
    seed:int=7
    stabilizer:float=2.0

class Microphase:
    def __init__(self,cfg):
        self.c=cfg
        k=2*np.pi*np.fft.fftfreq(cfg.size,d=cfg.dx)
        self.k2=k[:,None]**2+k[None,:]**2
        self.inv=np.zeros_like(self.k2)
        np.divide(1,self.k2,out=self.inv,where=self.k2>0)
        # Independent random streams for material heterogeneity and initial state.
        noise=np.random.default_rng(cfg.seed).normal(size=(cfg.size,cfg.size))
        self.h=np.fft.ifft2(np.fft.fft2(noise)*np.exp(-self.k2*cfg.correlation**2/2)).real
        self.h-=self.h.mean();self.h/=self.h.std()
        self.noise=np.random.default_rng(cfg.seed+10000).normal(size=self.h.shape)
        self.denominator=1+cfg.dt*(cfg.epsilon**2*self.k2**2+cfg.stabilizer*self.k2+cfg.alpha)
    def initial(self,m):
        u=m+.05*self.noise
        return u+m-u.mean()
    def energy_residual(self,u,m):
        c=self.c;hat=np.fft.fft2(u)
        lap=np.fft.ifft2(-self.k2*hat).real
        psi=np.fft.ifft2(self.inv*hat).real
        mu=u**3-u-c.epsilon**2*lap+c.heterogeneity*self.h+c.alpha*psi
        rhs=np.fft.ifft2(-self.k2*np.fft.fft2(mu)).real
        energy=float(np.mean(.25*(u*u-1)**2-.5*c.epsilon**2*u*lap+c.heterogeneity*self.h*u+.5*c.alpha*(u-m)*psi))
        return dict(energy_density=energy,rhs_rms=float(np.sqrt(np.mean(rhs*rhs))),mean=float(u.mean()),u_min=float(u.min()),u_max=float(u.max()))
    def evolve(self,u,m,steps=None):
        c=self.c;u=u+m-u.mean();records=[]
        steps=c.steps if steps is None else steps
        start=u.copy();late=None
        for i in range(steps):
            if i%max(1,steps//5)==0:
                records.append(dict(step=i,**self.energy_residual(u,m)))
            if i==int(.8*steps):late=u.copy()
            nonlinear=u**3-(1+c.stabilizer)*u+c.heterogeneity*self.h
            hat=(np.fft.fft2(u)-c.dt*self.k2*np.fft.fft2(nonlinear))/self.denominator
            hat[0,0]=m*c.size*c.size
            u=np.fft.ifft2(hat).real
            if not np.isfinite(u).all():raise FloatingPointError('nonfinite evolution')
        records.append(dict(step=steps,**self.energy_residual(u,m)))
        return u,dict(history=records,late_rms=float(np.sqrt(np.mean((u-late)**2))),
                      late_binary_change=float(np.mean((u>0)!=(late>0))),
                      change_from_start=float(np.sqrt(np.mean((u-start)**2))))

def sheet(fields,means,output,title,scale=2):
    rows=len(fields);n=fields[0][0].shape[0]
    w=n*scale;header=40;cellh=w+35
    image=Image.new('RGB',(len(means)*w,header+rows*cellh),'#17262b');draw=ImageDraw.Draw(image)
    draw.text((10,12),title,fill='white')
    for i,series in enumerate(fields):
        for j,u in enumerate(series):
            rgb=np.empty((*u.shape,3),dtype=np.uint8);rgb[:]=[231,222,204];rgb[u>0]=[32,62,74]
            image.paste(Image.fromarray(rgb).resize((w,w),Image.Resampling.NEAREST),(j*w,header+i*cellh))
            draw.text((j*w+8,header+i*cellh+w+8),f'm = {means[j]:.3f}',fill='white')
    image.save(output)

def run(cfg,means,output,route='forward',initial_fields=None,heterogeneity_field=None,initial_state=None):
    from pitpattern.metrics.pit_microphase_metrics import measure
    output.mkdir(parents=True,exist_ok=True)
    model=Microphase(cfg)
    if heterogeneity_field is not None:model.h=heterogeneity_field.copy()
    u=model.initial(means[0]) if initial_state is None else initial_state.copy()
    fields=[];records=[]
    for i,m in enumerate(means):
        if initial_fields is not None:u=initial_fields[i].copy()
        elif route=='independent':u=model.initial(m)
        u,info=model.evolve(u,m)
        metrics=measure(u,dx=cfg.dx)
        record=dict(m=float(m),**metrics,**model.energy_residual(u,m),**info)
        fields.append(u.copy());records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='history'}),flush=True)
    np.savez_compressed(output/'fields.npz',u=np.array(fields),h=model.h,means=np.array(means))
    (output/'report.json').write_text(json.dumps(dict(config=asdict(cfg),route=route,records=records),indent=2))
    keys=[k for k in records[0] if k!='history']
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in records)
    sheet([fields],means,output/'sequence.png',f'Nonlocal phase separation | {route} | fixed eta={cfg.heterogeneity}; alpha={cfg.alpha}')
    return fields,records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/pit_microphase/main'))
    p.add_argument('--means',type=float,nargs='+',default=list(np.linspace(-.55,.25,9)))
    for name in ['size','steps','seed']:p.add_argument('--'+name,type=int,default=getattr(Config(),name))
    for name in ['dx','dt','epsilon','alpha','heterogeneity','correlation']:p.add_argument('--'+name,type=float,default=getattr(Config(),name))
    p.add_argument('--route',choices=['forward','reverse','independent'],default='forward')
    args=p.parse_args();cfg=Config(**{k:getattr(args,k) for k in asdict(Config()) if hasattr(args,k)})
    means=sorted(args.means,reverse=args.route=='reverse')
    run(cfg,means,args.output,args.route)
if __name__=='__main__':main()
