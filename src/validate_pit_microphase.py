#!/usr/bin/env python3
"""Frozen-parameter validation of a single nonlocal model and composition path."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from simulate_pit_microphase import Config,Microphase,run

MEANS=[-.55,-.35,-.15,-.05,0.,.05,.1,.15,.2,.25]
BASE=Config(heterogeneity=.08,steps=2500)


def upsample(field,factor=2):
    n=field.shape[0];N=n*factor
    spectrum=np.fft.fftshift(np.fft.fft2(field))
    padded=np.zeros((N,N),dtype=complex)
    start=(N-n)//2;padded[start:start+n,start:start+n]=spectrum
    return np.fft.ifft2(np.fft.ifftshift(padded)).real*factor**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/pit_microphase/validation'));args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    for seed in [1,42]:
        print(f'START seed {seed}',flush=True)
        run(replace(BASE,seed=seed),MEANS,out/f'seed{seed}')
    print('START independent',flush=True)
    run(BASE,MEANS,out/'independent','independent')
    print('START reverse',flush=True)
    run(BASE,list(reversed(MEANS)),out/'reverse','reverse')
    saved=np.load('artifacts/pit_microphase/fine/fields.npz')
    print('START long hold',flush=True)
    run(replace(BASE,steps=10000),MEANS,out/'hold','hold',list(saved['u']),saved['h'])
    selected=[0,2,4,8];means=[MEANS[i] for i in selected];initial=[saved['u'][i] for i in selected]
    print('START matched dt',flush=True)
    run(replace(BASE,steps=5000),means,out/'dt04','hold',initial,saved['h'])
    run(replace(BASE,steps=10000,dt=.2),means,out/'dt02','hold',initial,saved['h'])
    print('START doubled resolution',flush=True)
    run(replace(BASE,size=256,dx=.5,steps=5000),means,out/'dx05','hold',[upsample(u) for u in initial],upsample(saved['h']))
    print('START larger domain',flush=True)
    run(replace(BASE,size=192),MEANS,out/'domain192')
    print('START no disorder ablation',flush=True)
    run(replace(BASE,heterogeneity=0),MEANS,out/'no_disorder')
    print('START seed42 refined transition',flush=True)
    saved42=np.load(out/'seed42'/'fields.npz')
    index=int(np.argmin(abs(saved42['means']-.05)))
    run(replace(BASE,seed=42),[.06,.07,.08,.09,.1,.15,.2,.25],out/'seed42_transition',
        'forward_refined_from_m0.05',heterogeneity_field=saved42['h'],initial_state=saved42['u'][index])
    print('START domain192 refined transition',flush=True)
    saved_large=np.load(out/'domain192'/'fields.npz')
    index=int(np.argmin(abs(saved_large['means']-.05)))
    run(replace(BASE,size=192),[.06,.07,.08,.09,.1,.15,.2,.25],out/'domain192_transition',
        'forward_refined_from_m0.05',heterogeneity_field=saved_large['h'],initial_state=saved_large['u'][index])
    (out/'complete.json').write_text(json.dumps({'finished':True,'frozen_parameters':BASE.__dict__}))

if __name__=='__main__':main()
