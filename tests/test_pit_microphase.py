import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pitpattern.models.simulate_pit_microphase import Config,Microphase
from pitpattern.metrics.pit_microphase_metrics import periodic_labels,measure

def test_periodic_component_merges_seam():
    b=np.zeros((24,24),np.uint8);b[8:15,:3]=1;b[8:15,-3:]=1
    count,labels=periodic_labels(b)
    assert count==1
    assert labels[10,0]==labels[10,-1]

def test_measured_y_branches_and_disk():
    y=np.zeros((96,96),np.uint8)
    for end in [(48,15),(18,75),(78,75)]:cv2.line(y,(48,48),end,9)
    a=measure(y.astype(float)*2-1)
    assert a['branch_count']==1
    disk=np.zeros_like(y);cv2.circle(disk,(48,48),15,1,-1)
    assert measure(disk.astype(float)*2-1)['branch_count']==0

def test_conservation_and_energy_descent():
    model=Microphase(Config(size=32,steps=100,dt=.1))
    u,info=model.evolve(model.initial(-.35),-.35)
    assert abs(u.mean()+.35)<1e-12
    e=[r['energy_density'] for r in info['history']]
    assert np.max(np.diff(e))<1e-10

def test_uniform_linear_growth_rate():
    c=Config(size=32,heterogeneity=0,dt=.01,steps=1)
    model=Microphase(c);m=-.3
    x=np.arange(32)*2*np.pi/32
    u=m+1e-5*np.broadcast_to(np.cos(2*x),(32,32))
    v,_=model.evolve(u,m)
    q2=(4*np.pi/32)**2
    expected=(1-c.dt*q2*(3*m*m-1-c.stabilizer))/(1+c.dt*(c.epsilon**2*q2*q2+c.stabilizer*q2+c.alpha))
    ratio=np.fft.fft2(v)[0,2].real/np.fft.fft2(u)[0,2].real
    np.testing.assert_allclose(ratio,expected,rtol=1e-8)
