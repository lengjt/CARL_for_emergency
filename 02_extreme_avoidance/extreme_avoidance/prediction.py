"""Independent, approximate batched nonlinear bicycle predictor.

NumPy RK4, friction ellipse, longitudinal load transfer, steering/force states.
Use for candidate screening only. No autograd, wheel-speed dynamics, ABS, or
validated match to OCD. mu_x/mu_y are effective coefficients, NOT lambda_mue.
State: [x, y, yaw, vx, vy, yaw_rate, steering, longitudinal_force].
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BicycleParameters:
    mass: float=800.
    iz: float=1000.
    lf: float=1.724
    lr: float=1.476
    height: float=0.3
    cornering_front: float=85000.
    cornering_rear: float=100000.
    steer_tau: float=0.06
    force_tau: float=0.08
    steer_max: float=0.3
    steer_rate: float=0.5
    force_rate: float=80000.

def derivative(state, action, mu_x, mu_y, p):
    x=np.asarray(state,dtype=float)
    u=np.asarray(action,dtype=float)
    vx,vy,r,delta,force=(x[...,i] for i in range(3,8))
    v=np.maximum(vx,1.0)
    fzf=np.clip((p.mass*9.81*p.lr-p.height*force)/(p.lf+p.lr),.05*p.mass*9.81,.95*p.mass*9.81)
    fzr=p.mass*9.81-fzf
    fxf=np.where(force<0,.6*force,0.)
    fxr=np.where(force<0,.4*force,force)
    mx,my=np.asarray(mu_x),np.asarray(mu_y)
    fxf=np.clip(fxf,-mx*fzf,mx*fzf);fxr=np.clip(fxr,-mx*fzr,mx*fzr)
    capf=my*fzf*np.sqrt(np.maximum(0,1-(fxf/(mx*fzf))**2))
    capr=my*fzr*np.sqrt(np.maximum(0,1-(fxr/(mx*fzr))**2))
    af=np.arctan2(vy+p.lf*r,v)-delta
    ar=np.arctan2(vy-p.lr*r,v)
    fyf=-capf*np.tanh(p.cornering_front*af/np.maximum(capf,1e-6))
    fyr=-capr*np.tanh(p.cornering_rear*ar/np.maximum(capr,1e-6))
    fx=fxf*np.cos(delta)-fyf*np.sin(delta)+fxr
    fy=fyf*np.cos(delta)+fxf*np.sin(delta)+fyr
    steering_rate=np.clip((np.clip(u[...,0],-p.steer_max,p.steer_max)-delta)/p.steer_tau,-p.steer_rate,p.steer_rate)
    force_rate=np.clip((np.clip(u[...,1],-10000,4000)-force)/p.force_tau,-p.force_rate,p.force_rate)
    return np.stack((vx*np.cos(x[...,2])-vy*np.sin(x[...,2]),
                     vx*np.sin(x[...,2])+vy*np.cos(x[...,2]),r,
                     (fx-.025*p.mass*9.81*np.tanh(vx)-.6125*vx*abs(vx))/p.mass+r*vy,
                     fy/p.mass-r*vx,(p.lf*(fyf*np.cos(delta)+fxf*np.sin(delta))-p.lr*fyr)/p.iz,
                     steering_rate,force_rate),axis=-1)

def rollout(initial, actions, dt=0.01, mu_x=1.45, mu_y=1.16, params=None):
    """actions shape (batch, steps, 2); output (batch, steps+1, 8)."""
    a=np.asarray(actions,dtype=float)
    if a.ndim!=3 or a.shape[-1]!=2 or a.shape[1]<1 or not np.all(np.isfinite(a)):
        raise ValueError("Expected finite actions with shape (batch, steps, 2)")
    if not 0<dt<=.01 or not np.all(np.asarray(mu_x)>0) or not np.all(np.asarray(mu_y)>0):
        raise ValueError("Use positive effective friction and dt <= 0.01")
    p=params or BicycleParameters()
    x=np.broadcast_to(np.asarray(initial,dtype=float),(a.shape[0],8)).copy()
    if not np.all(np.isfinite(x)): raise ValueError("Nonfinite initial state")
    out=[x.copy()]
    for i in range(a.shape[1]):
        f=lambda y:derivative(y,a[:,i],mu_x,mu_y,p)
        k1=f(x);k2=f(x+.5*dt*k1);k3=f(x+.5*dt*k2);k4=f(x+dt*k3)
        x=x+dt*(k1+2*k2+2*k3+k4)/6
        out.append(x.copy())
    return np.stack(out,axis=1)
