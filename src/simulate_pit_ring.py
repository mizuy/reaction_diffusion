#!/usr/bin/env python3
"""Equilibrium search for a growing epithelial ring around a finite lumen.

Dimensionless, star-shaped Galerkin approximation with a Winkler foundation.
No RD, shape templates, component cutting, or prescribed maximum pit length.
Uses only NumPy and Pillow; optimizer iterations are NOT physical time.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Parameters:
    area: float = 40.0
    perimeter: float = 20.0
    bending: float = 0.03
    stroma: float = 0.1
    contact: float = 20.0
    wall_distance: float = 0.15
    modes: int = 12
    points: int = 192


class Ring:
    def __init__(self, parameters: Parameters):
        self.p = parameters
        self.theta = np.arange(parameters.points) * (2 * np.pi / parameters.points)
        self.h = 2 * np.pi / parameters.points
        t = self.theta
        columns = [np.ones_like(t)]
        d1 = [np.zeros_like(t)]
        d2 = [np.zeros_like(t)]
        self.frequencies = [0]
        for m in range(2, parameters.modes + 1):
            columns.extend([np.cos(m*t), np.sin(m*t)])
            d1.extend([-m*np.sin(m*t), m*np.cos(m*t)])
            d2.extend([-m*m*np.cos(m*t), -m*m*np.sin(m*t)])
            self.frequencies.extend([m, m])
        self.b = np.array(columns).T
        self.d1 = np.array(d1).T
        self.d2 = np.array(d2).T
        self.unit = np.column_stack([np.cos(t), np.sin(t)])
        # Exclude material neighbors within four wall thicknesses on reference ring.
        i, j = np.triu_indices(parameters.points, 1)
        arc = np.minimum(j-i, parameters.points-(j-i)) * self.h
        keep = arc > 4 * parameters.wall_distance
        self.pair_i, self.pair_j = i[keep], j[keep]

    def energy(self, x: np.ndarray, growth: float):
        p = self.p
        rho, q, z = self.b @ x, self.d1 @ x, self.d2 @ x
        # Numerical overflow guard, not a pit-length cutoff.
        if np.max(np.abs(rho)) > 20:
            return np.inf, np.full_like(x, np.nan), {}
        r = np.exp(rho)
        b = 1 + q*q
        c = b-z
        area = 0.5 * self.h * np.sum(r*r)
        perimeter = self.h * np.sum(r*np.sqrt(b))
        area_force = p.area / np.pi * (area-np.pi)
        perimeter_force = p.perimeter / (2*np.pi) * (perimeter-2*np.pi*(1+growth))
        bend = c*c / (r*b**2.5)
        terms = {
            'area_energy': p.area/(2*np.pi)*(area-np.pi)**2,
            'perimeter_energy': p.perimeter/(4*np.pi)*(perimeter-2*np.pi*(1+growth))**2,
            'bending_energy': 0.5*p.bending*self.h*np.sum(bend),
            'stroma_energy': 0.5*p.stroma*self.h*np.sum((r-1)**2),
        }
        gr = area_force*r*r + perimeter_force*r*np.sqrt(b)
        gr += -0.5*p.bending*bend + p.stroma*(r-1)*r
        gq = perimeter_force*r*q/np.sqrt(b)
        gq += 0.5*p.bending*(4*c*q/(r*b**2.5)-5*q*c*c/(r*b**3.5))
        gz = -p.bending*c/(r*b**2.5)
        gradient = self.h*(self.b.T@gr + self.d1.T@gq + self.d2.T@gz)
        xy = r[:, None]*self.unit
        delta = xy[self.pair_i]-xy[self.pair_j]
        distance = np.sqrt(np.sum(delta*delta, axis=1)+1e-24)
        active = distance < p.wall_distance
        gap = p.wall_distance-distance[active]
        terms['contact_energy'] = 0.5*p.contact*self.h**2*np.sum(gap*gap)
        if np.any(active) and p.contact:
            force = -p.contact*self.h**2*gap[:, None]*delta[active]/distance[active, None]
            point_grad = np.zeros_like(xy)
            np.add.at(point_grad, self.pair_i[active], force)
            np.add.at(point_grad, self.pair_j[active], -force)
            gradient += self.b.T@np.sum(point_grad*xy, axis=1)
        terms.update(area=float(area), perimeter=float(perimeter))
        total = sum(v for k, v in terms.items() if k.endswith('_energy'))
        return float(total), gradient, terms

    def coordinates(self, x):
        return np.exp(self.b@x)[:, None]*self.unit

    def initial(self, seed):
        rng = np.random.default_rng(seed)
        x = 0.003*rng.normal(size=self.b.shape[1]) / np.maximum(1, np.array(self.frequencies))
        x[0] = 0
        return x


def minimize(ring, growth, initial, maxiter=1500, tolerance=1e-6):
    """Analytic-gradient BFGS with Armijo descent and spectral preconditioner."""
    x = initial.copy()
    diagonal = 1 + ring.p.bending*np.maximum(1, np.array(ring.frequencies))**4
    diagonal[0] = 4*np.pi*ring.p.area + 2*np.pi*ring.p.perimeter + 1
    base = np.diag(1/diagonal)
    inverse = base.copy()
    f, grad, _ = ring.energy(x, growth)
    trace = [float(f)]
    converged = False
    reason = 'maxiter'
    for iteration in range(maxiter):
        if np.max(np.abs(grad)) < tolerance:
            converged, reason = True, 'gradient'
            break
        direction = -inverse@grad
        if grad@direction >= 0:
            inverse = base.copy()
            direction = -inverse@grad
        scale = 1.0
        for _ in range(45):
            new_x = x+scale*direction
            new_f, new_grad, _ = ring.energy(new_x, growth)
            if np.isfinite(new_f) and new_f <= f+1e-4*scale*(grad@direction):
                break
            scale *= 0.5
        else:
            reason = 'line_search'
            break
        step, change = new_x-x, new_grad-grad
        sy = step@change
        if sy > 1e-12*np.linalg.norm(step)*np.linalg.norm(change):
            hy = inverse@change
            inverse += ((sy+change@hy)/(sy*sy))*np.outer(step, step)
            inverse -= (np.outer(hy, step)+np.outer(step, hy))/sy
        else:
            inverse = base.copy()
        x, f, grad = new_x, new_f, new_grad
        trace.append(float(f))
    return x, {'converged': converged, 'reason': reason, 'iterations': iteration+1,
               'gradient_max': float(np.max(np.abs(grad))), 'trace': trace}


def hessian(ring, x, growth, eps=1e-5):
    columns = []
    for j in range(len(x)):
        dx = np.zeros_like(x)
        dx[j] = eps
        columns.append((ring.energy(x+dx, growth)[1]-ring.energy(x-dx, growth)[1])/(2*eps))
    matrix = np.array(columns).T
    return 0.5*(matrix+matrix.T)


def measure(ring, x, growth, info, curvature_check=False):
    f, grad, terms = ring.energy(x, growth)
    xy = ring.coordinates(x)
    # Filled lumen covariance and area from polar quadrature (not contour PCA).
    r = np.exp(ring.b@x)
    a = terms['area']
    centroid = ring.h*np.sum(r[:, None]**3*ring.unit, axis=0)/(3*a)
    second = ring.h*np.einsum('i,ij,ik->jk', r**4, ring.unit, ring.unit)/(4*a)
    eigen = np.linalg.eigvalsh(second-np.outer(centroid, centroid))
    amplitudes = np.sqrt(x[1::2]**2+x[2::2]**2)
    dominant = int(2+np.argmax(amplitudes)) if np.max(amplitudes)>0.005 else 0
    diameter = float(np.max(np.linalg.norm(xy[:, None]-xy[None, :], axis=2)))
    result = dict(growth=float(growth), stroma=ring.p.stroma, energy=f,
                  area_ratio=a/np.pi, perimeter_ratio=terms['perimeter']/(2*np.pi),
                  elongation=float(np.sqrt(eigen[1]/eigen[0])), diameter=diameter,
                  circularity=4*np.pi*a/terms['perimeter']**2,
                  dominant_mode=dominant, mode_amplitude=float(np.max(amplitudes)),
                  radius_min=float(r.min()), radius_max=float(r.max()),
                  gradient_max=float(np.max(np.abs(grad))),
                  converged=info['converged'], reason=info['reason'], iterations=info['iterations'])
    result.update({k:float(v) for k,v in terms.items()})
    if curvature_check:
        result['hessian_min'] = float(np.linalg.eigvalsh(hessian(ring,x,growth))[0])
    return result


def contact_sheet(cases, output, title, columns=9):
    """Same spatial scale for every panel; colors mark lumen/wall/stroma."""
    cell_w, cell_h = 190, 220
    rows = (len(cases)+columns-1)//columns
    image = Image.new('RGB', (columns*cell_w, 55+rows*cell_h), '#182126')
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), title, fill='white')
    extent = max(1.7, max(float(np.abs(case['xy']).max()) for case in cases)*1.1)
    scale = 73/extent
    for index, case in enumerate(cases):
        row, col = divmod(index, columns)
        left, top = col*cell_w, 55+row*cell_h
        draw.rectangle((left+4,top+3,left+cell_w-4,top+cell_h-4), fill='#e6dcc8')
        xy = case['xy']*scale+np.array([left+cell_w/2,top+100])
        coords = [tuple(point) for point in xy]
        draw.polygon(coords, fill='#263f4b')
        draw.line(coords+[coords[0]], fill='#b24948', width=3)
        m=case['metrics']
        draw.text((left+10,top+9),f"g={m['growth']:.2f}  Ks={m['stroma']:.2g}", fill='#182126')
        draw.text((left+10,top+176),f"AR {m['elongation']:.2f}  mode {m['dominant_mode']}",fill='#182126')
        draw.text((left+10,top+191),f"A/A0 {m['area_ratio']:.2f}  grad {m['gradient_max']:.0e}",fill='#182126')
    image.save(output)


def write_cases(cases, output, parameters, name='sweep'):
    output.mkdir(parents=True, exist_ok=True)
    fields=sorted(set().union(*(case['metrics'].keys() for case in cases)))
    with (output/f'{name}.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(case['metrics'] for case in cases)
    (output/f'{name}.json').write_text(json.dumps({'parameters': parameters,
        'model': 'star-shaped log-radius ring; local elastic foundation; equilibrium search',
        'cases':[{k:v for k,v in c.items() if k!='xy'} for c in cases]},indent=2))
    np.savez_compressed(output/f'{name}_contours.npz',**{f'case_{i:03}':c['xy'] for i,c in enumerate(cases)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/pit_ring'))
    parser.add_argument('--stroma',type=float,nargs='+',default=[0.03,0.3,3.0])
    parser.add_argument('--growth',type=float,nargs='+',default=list(np.linspace(0,1.2,9)))
    parser.add_argument('--modes',type=int,default=12)
    parser.add_argument('--points',type=int,default=192)
    parser.add_argument('--maxiter',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--area',type=float,default=40)
    parser.add_argument('--bending',type=float,default=0.03)
    parser.add_argument('--path',choices=['independent','forward','reverse'],default='independent')
    parser.add_argument('--hessian',action='store_true')
    args=parser.parse_args()
    if args.points < 8*args.modes or args.modes<2 or args.maxiter<1:
        parser.error('Require modes >= 2, points >= 8*modes and maxiter >= 1')
    cases=[]
    parameters=[]
    for ks in args.stroma:
        p=Parameters(stroma=ks,modes=args.modes,points=args.points,area=args.area,bending=args.bending)
        parameters.append(asdict(p))
        ring=Ring(p)
        previous=ring.initial(args.seed)
        sequence=sorted(args.growth,reverse=args.path=='reverse')
        for growth in sequence:
            initial=ring.initial(args.seed) if args.path=='independent' else previous+0.1*ring.initial(args.seed+1)
            x,info=minimize(ring,growth,initial,args.maxiter)
            previous=x.copy()
            metrics=measure(ring,x,growth,info,args.hessian)
            metrics.update(seed=args.seed,path=args.path)
            cases.append(dict(metrics=metrics,coefficients=x.tolist(),xy=ring.coordinates(x),trace=info['trace']))
            print(json.dumps(metrics),flush=True)
    write_cases(cases,args.output,parameters)
    contact_sheet(cases,args.output/'contact_sheet.png','Growing epithelial ring | blue: lumen; red: wall; beige: local stromal support',len(args.growth))


if __name__=='__main__':
    main()
