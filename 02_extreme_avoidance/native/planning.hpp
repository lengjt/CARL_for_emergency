#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

// Independent 3-DOF prediction model. This does not substitute for OCD execution.
namespace planning {
using State=std::array<double,8>;
using Action=std::array<double,2>;
using Maneuver=std::array<double,6>; // signed offset, ramp, brake, brake end s, return length, stop flag
struct Parameters {
  double mass,iz,lf,lr,height,cf,cr,mux,muy;
  double steer_tau,force_tau,steer_max,steer_rate,force_rate,brake_max,drive_max,brake_bias;
};
inline double clip(double x,double lo,double hi) { return std::clamp(x,lo,hi); }
inline double wrap(double a) { return std::atan2(std::sin(a),std::cos(a)); }
inline std::array<double,3> smooth(double u) {
  u=clip(u,0,1);
  return {10*u*u*u-15*std::pow(u,4)+6*std::pow(u,5),
          30*u*u-60*u*u*u+30*std::pow(u,4),60*u-180*u*u+120*u*u*u};
}
inline Action command(const State& x,const Maneuver& z,double trigger,double obstacle_s,
                      double obstacle_length,double length,double speed,const Parameters& p) {
  double d=0,dp=0,ddp=0;
  if (z[5]<.5) {
    auto a=smooth((x[0]-trigger)/z[1]);
    auto b=smooth((x[0]-(obstacle_s+obstacle_length/2+length/2+5))/z[4]);
    d=z[0]*(a[0]-b[0]);dp=z[0]*(a[1]/z[1]-b[1]/z[4]);
    ddp=z[0]*(a[2]/(z[1]*z[1])-b[2]/(z[4]*z[4]));
  }
  double v=std::max(x[3],0.0),beta=std::atan2(x[4],std::max(v,1.0));
  double curvature=ddp/std::pow(1+dp*dp,1.5);
  double delta=std::atan((p.lf+p.lr)*curvature)+.9*wrap(std::atan(dp)-x[2])
    +std::atan2(1.2*(d-x[1]),v+3)-.25*beta+.10*(v*curvature-x[5]);
  double force=0;
  if (z[5]>.5) force=-p.brake_max;
  else if (x[0]<z[3] && v>6) force=-z[2];
  else if (x[0]>obstacle_s+obstacle_length/2+length/2+5 && std::abs(x[1])<.35
           && std::abs(x[2])<.05 && std::abs(beta)<.04 && std::abs(x[5])<.08)
    force=clip(p.mass*1.5*(speed-v)+.025*p.mass*9.81+.6125*v*v,-1500,1000);
  return {clip(delta,-p.steer_max,p.steer_max),clip(force,-p.brake_max,p.drive_max)};
}
inline State derivative(const State& x,const Action& u,const Parameters& p) {
  double v=std::max(x[3],1.0),vy=x[4],r=x[5],delta=x[6],force=x[7];
  double weight=p.mass*9.81;
  // Account for aerodynamic downforce and simple longitudinal load transfer.
  double total=weight+.91875*x[3]*x[3];
  double fzf=clip((total*p.lr-p.height*force)/(p.lf+p.lr),.05*total,.95*total);
  double fzr=total-fzf;
  double fxf=force<0?p.brake_bias*force:0;
  double fxr=force<0?(1-p.brake_bias)*force:force;
  // Approximate the shared brake regulator; no wheel rotational states or ABS cycle.
  fxf=clip(fxf,-.9*p.mux*fzf,.9*p.mux*fzf);
  fxr=clip(fxr,-.9*p.mux*fzr,.9*p.mux*fzr);
  double capf=p.muy*fzf*std::sqrt(std::max(0.0,1-std::pow(fxf/(p.mux*fzf),2)));
  double capr=p.muy*fzr*std::sqrt(std::max(0.0,1-std::pow(fxr/(p.mux*fzr),2)));
  double af=std::atan2(vy+p.lf*r,v)-delta, ar=std::atan2(vy-p.lr*r,v);
  double fyf=-capf*std::tanh(p.cf*af/std::max(capf,1e-6));
  double fyr=-capr*std::tanh(p.cr*ar/std::max(capr,1e-6));
  double fx=fxf*std::cos(delta)-fyf*std::sin(delta)+fxr;
  double fy=fyf*std::cos(delta)+fxf*std::sin(delta)+fyr;
  double ax=(fx-.025*weight*std::tanh(x[3])-.6125*x[3]*std::abs(x[3]))/p.mass+r*vy;
  // Smooth near-zero braking: keep forward-only planning well posed at a safe stop.
  if (x[3]<.5 && ax<0) ax=std::max(ax,-2*x[3]);
  return {x[3]*std::cos(x[2])-vy*std::sin(x[2]),x[3]*std::sin(x[2])+vy*std::cos(x[2]),r,
          ax,fy/p.mass-r*x[3],(p.lf*(fyf*std::cos(delta)+fxf*std::sin(delta))-p.lr*fyr)/p.iz,
          clip((clip(u[0],-p.steer_max,p.steer_max)-delta)/p.steer_tau,-p.steer_rate,p.steer_rate),
          clip((clip(u[1],-p.brake_max,p.drive_max)-force)/p.force_tau,-p.force_rate,p.force_rate)};
}
inline State advance(const State& x,const Action& u,double h,const Parameters& p) {
  auto a=derivative(x,u,p); State t;
  for(int j=0;j<8;++j)t[j]=x[j]+h*.5*a[j];
  auto b=derivative(t,u,p);
  for(int j=0;j<8;++j)t[j]=x[j]+h*.5*b[j];
  auto c=derivative(t,u,p);
  for(int j=0;j<8;++j)t[j]=x[j]+h*c[j];
  auto d=derivative(t,u,p);
  for(int j=0;j<8;++j)t[j]=x[j]+h*(a[j]+2*b[j]+2*c[j]+d[j])/6;
  return t;
}
// SAT gap of oriented ego rectangle vs axis-aligned obstacle rectangle.
inline double gap(const State& x,const std::array<double,4>& ob,double length,double width,double inflate) {
  double c=std::cos(x[2]),s=std::sin(x[2]),dx=x[0]-ob[0],dy=x[1]-ob[1];
  double l=length/2+inflate,w=width/2+inflate;
  return std::max({std::abs(dx)-(std::abs(c)*l+std::abs(s)*w+ob[2]/2),
    std::abs(dy)-(std::abs(s)*l+std::abs(c)*w+ob[3]/2),
    std::abs(dx*c+dy*s)-(l+std::abs(c)*ob[2]/2+std::abs(s)*ob[3]/2),
    std::abs(-dx*s+dy*c)-(w+std::abs(s)*ob[2]/2+std::abs(c)*ob[3]/2)});
}
inline Parameters parameters(const std::vector<double>& v) {
  if(v.size()!=17)throw std::invalid_argument("Expected 17 predictor parameters");
  for(auto n:v) if(!std::isfinite(n)||n<=0)throw std::invalid_argument("Invalid predictor parameters");
  return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16]};
}
} // namespace planning

inline void bind_planning(pybind11::module_& m) {
  namespace py=pybind11;
  using namespace planning;
  m.def("predict_actions",[](State x,py::array_t<double,py::array::c_style|py::array::forcecast> actions,
                             double dt,const std::vector<double>& values) {
    auto a=actions.unchecked<2>();auto p=parameters(values);
    if(a.shape(1)!=2||dt<=0||dt>.02)throw std::invalid_argument("Invalid rollout input");
    py::array_t<double> result({a.shape(0)+1,py::ssize_t(8)});auto out=result.mutable_unchecked<2>();
    for(int j=0;j<8;++j)out(0,j)=x[j];
    for(py::ssize_t i=0;i<a.shape(0);++i) {
      x=advance(x,{a(i,0),a(i,1)},dt,p);
      for(int j=0;j<8;++j)out(i+1,j)=x[j];
    }
    return result;
  });
  m.def("maneuver_action",[](const State& x,const Maneuver& z,double trigger,double obstacle_s,
                            double obstacle_length,double length,double speed,const std::vector<double>& values) {
    return command(x,z,trigger,obstacle_s,obstacle_length,length,speed,parameters(values));
  });
  m.def("screen_maneuvers",[](State initial,const std::vector<Maneuver>& candidates,
       const std::vector<std::array<double,4>>& obstacles,const std::vector<Action>& delayed,
       double trigger,double speed,double road_width,double length,double width,
       double horizon,double dt,double control_dt,double margin,const std::vector<double>& values) {
    if(candidates.empty()||obstacles.empty()||horizon<=0||horizon>12||dt<=0||dt>.02||margin<0)
      throw std::invalid_argument("Invalid screening input");
    auto p=parameters(values);int n=int(std::round(horizon/dt));
    int hold=int(std::round(control_dt/dt));
    if(hold<1||std::abs(hold*dt-control_dt)>1e-9)throw std::invalid_argument("Control period must align with prediction step");
    // columns: feasible, score, minimum gap, road margin, max beta, terminal d, terminal yaw, final vx, final x
    py::array_t<double> result({py::ssize_t(candidates.size()),py::ssize_t(9)});
    auto out=result.mutable_unchecked<2>();
    py::gil_scoped_release release;
    for(size_t k=0;k<candidates.size();++k) {
      auto z=candidates[k];State x=initial;std::vector<Action> queue=delayed;
      double mingap=1e6,minroad=1e6,maxbeta=0,score=0;bool valid=true;
      Action proposed{};
      for(int i=0;i<n;++i) {
        if(i%hold==0)proposed=command(x,z,trigger,obstacles[0][0],obstacles[0][2],length,speed,p);
        queue.push_back(proposed);auto prev=x;
        x=advance(x,queue[i],dt,p);
        if(!std::all_of(x.begin(),x.end(),[](double v){return std::isfinite(v);})) {valid=false;break;}
        double beta=std::atan2(x[4],std::max(x[3],1.0));maxbeta=std::max(maxbeta,std::abs(beta));
        State mid=x;mid[0]=(x[0]+prev[0])/2;mid[1]=(x[1]+prev[1])/2;mid[2]=(x[2]+prev[2])/2;
        double inflate=margin+.5*std::hypot(x[0]-prev[0],x[1]-prev[1])
          +.25*std::hypot(length,width)*std::abs(x[2]-prev[2]);
        double road=road_width/2-std::abs(mid[1])-std::abs(std::sin(mid[2]))*(length/2+inflate)
          -std::abs(std::cos(mid[2]))*(width/2+inflate);
        minroad=std::min(minroad,road);
        for(auto ob:obstacles)mingap=std::min(mingap,gap(mid,ob,length,width,inflate));
        if(mingap<=0||road<=0||std::abs(beta)>.35||std::abs(x[2])>.65||x[3]<-.1) {valid=false;break;}
        score+=dt*(.15*x[1]*x[1]+15*beta*beta+3*x[5]*x[5]+1e-7*z[2]*z[2]);
      }
      bool stopped=x[3]<.2 && x[0]+length/2<obstacles[0][0]-obstacles[0][2]/2;
      score+=10*x[1]*x[1]+80*x[2]*x[2]+10*x[4]*x[4]+20*x[5]*x[5];
      score+=stopped?25.0:std::max(0.0,obstacles[0][0]+length+5-x[0]);
      score-=.05*(x[0]-initial[0]);
      if(!valid)score=1e9;
      out(k,0)=valid;out(k,1)=score;out(k,2)=mingap;out(k,3)=minroad;out(k,4)=maxbeta;
      out(k,5)=std::isfinite(x[1])?x[1]:0;out(k,6)=std::isfinite(x[2])?x[2]:0;
      out(k,7)=std::isfinite(x[3])?x[3]:0;out(k,8)=std::isfinite(x[0])?x[0]:0;
    }
    return result;
  });
}
