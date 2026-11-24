#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include "planning.hpp"
#include "tum_types_cpp/common.hpp"
#include "tsl_logger_cpp/value_logger.hpp"
#include "ocd_vehicle_model_cpp/vehicle_model.hpp"
#include "ocd_drivetrain_fx_cpp/drivetrain_fx_model.hpp"
#include "ocd_steering_actuator_pt1_cpp/steering_actuator_pt1_model.hpp"
#include "ocd_vehicle_dynamics_double_track_cpp/vehicle_dynamics_model.hpp"
#include "ocd_tire_models_cpp/mf_52.hpp"
#include "ocd_aerodynamics_models_cpp/default.hpp"
namespace py = pybind11;
using Tire = tam::ocd::tire_models::MF52;
// Expose the integrated steering state. Upstream's measured-angle log is a cached
// value from the final RK stage and can create spurious finite-difference rates.
class ObservedSteering : public tam::ocd::steering_actuator::PT1SteeringActuatorModel {
  double integrated_ = 0;
public:
  ObservedSteering() {
    auto logger=std::dynamic_pointer_cast<tam::tsl::ReferenceLogger>(get_logger());
    logger->log("integrated_steering_angle_rad",&integrated_);
  }
  void set_x_vec(const StateVectorType& x) override {
    tam::ocd::steering_actuator::PT1SteeringActuatorModel::set_x_vec(x);
    integrated_=x[0];
  }
};
using Car = tam::ocd::VehicleModel<tam::ocd::drivetrain::DrivetrainFxModel,
  ObservedSteering,
  tam::ocd::vehicle_dynamics::VehicleDynamicsDoubleTrackModel<Tire,
    tam::ocd::aerodynamics::DefaultAerodynamicsModel>>;
using Wheels = tam::types::common::DataPerWheel<double>;
std::array<double, 4> array(const Wheels& v) {
  return {v.front_left, v.front_right, v.rear_left, v.rear_right};
}
void set_params(tam::pmg::MgmtInterface& pm, const py::dict& values) {
  for (auto kv : values) {
    auto name = py::cast<std::string>(kv.first);
    using T = tam::pmg::ParameterType;
    switch (pm.get_type(name)) {
      case T::DOUBLE: pm.set_value(name, py::cast<double>(kv.second)); break;
      case T::DOUBLE_ARRAY: pm.set_value(name, py::cast<std::vector<double>>(kv.second)); break;
      case T::BOOL: pm.set_value(name, py::cast<bool>(kv.second)); break;
      case T::INTEGER: pm.set_value(name, py::cast<std::int64_t>(kv.second)); break;
      default: throw std::invalid_argument("Unsupported parameter type: " + name);
    }
  }
}
class Model {
  std::unique_ptr<Car> car_;
  std::array<double, 8> baseline_mu_{};
  std::array<double, 4> last_internal_scale_{1,1,1,1};
  const std::array<std::string,4> wheel_names_{"front_left","front_right","rear_left","rear_right"};
public:
  explicit Model(const py::dict& params) { reset(params); }
  void reset(const py::dict& params) {
    car_ = std::make_unique<Car>();
    set_params(*car_->get_param_manager(), params);
    for (size_t i=0;i<4;++i) {
      auto prefix="vehicle_dynamics_double_track.z.tire_model."+wheel_names_[i]+".";
      baseline_mu_[2*i]=car_->get_param_manager()->get_value(prefix+"LMUX").as_double();
      baseline_mu_[2*i+1]=car_->get_param_manager()->get_value(prefix+"LMUY").as_double();
    }
    last_internal_scale_={1,1,1,1};
    car_->reset();
  }
  py::dict step(double steering_rad, double force_N, const std::array<double, 4>& scale,
                bool internal_mu = false) {
    if (!std::isfinite(steering_rad) || !std::isfinite(force_N))
      throw std::invalid_argument("Nonfinite command");
    for (double v : scale) if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("Friction scale must be positive and finite");
    Car::SteeringActuatorDriverInputType steer;
    steer.steering_angle_rad = steering_rad;
    car_->set_steering_input(steer);
    car_->set_drivetrain_input(force_N);
    tam::ocd::types::ExternalInfluences ext;
    for (size_t i=0;i<4;++i) {
      const double factor=internal_mu ? scale[i] : 1.0;
      if (factor!=last_internal_scale_[i]) {
        auto prefix="vehicle_dynamics_double_track.z.tire_model."+wheel_names_[i]+".";
        car_->get_param_manager()->set_value(prefix+"LMUX",baseline_mu_[2*i]*factor);
        car_->get_param_manager()->set_value(prefix+"LMUY",baseline_mu_[2*i+1]*factor);
        last_internal_scale_[i]=factor;
      }
    }
    ext.lambda_mue = internal_mu ? Wheels{1,1,1,1} : Wheels{scale[0], scale[1], scale[2], scale[3]};
    car_->set_external_influences(ext);
    car_->step();
    return output();
  }
  py::dict output() const {
    const auto out = car_->get_vehicle_model_output();
    const auto& v = out.vehicle_dynamics_output;
    const auto& logs = car_->get_logger()->get_data();
    auto value = [&](const std::string& k) {
      return std::visit([](auto x) { return static_cast<double>(x); }, logs.at(k));
    };
    // Integrated state is read from the logger after set_state_vector_of_models.
    // Upstream output()/force diagnostics are from the final RK evaluation stage.
    py::dict d;
    for (const auto& name : {"x_m", "y_m", "psi_rad", "v_x_mps", "v_y_mps",
                             "psi_dot_radps", "phi_rad", "theta_rad"})
      d[py::str(name)] = value(std::string("vehicle_dynamics/x_vec/") + name);
    d["steering_rad"] = value("steering_actuator/integrated_steering_angle_rad");
    d["ax_mps2"] = v.acceleration_mps2.x;
    d["ay_mps2"] = v.acceleration_mps2.y;
    d["fx_N"] = array(v.longitudinal_tire_force_tire_frame_per_wheel_N);
    d["fy_N"] = array(v.lateral_tire_force_tire_frame_per_wheel_N);
    d["fz_N"] = array(v.vertical_tire_force_per_wheel_N);
    d["kappa"] = array(v.tire_longitudinal_slip_per_wheel);
    d["alpha_rad"] = array(v.tire_slip_angle_per_wheel_rad);
    d["omega_radps"] = array(out.wheel_speeds_radps);
    d["load_torque_Nm"] = array(out.drivetrain_load_torque_per_wheel_Nm);
    return d;
  }
  py::dict logs() const {
    py::dict d;
    for (const auto& [k,v] : car_->get_logger()->get_data())
      d[py::str(k)] = std::visit([](auto x) -> double { return x; }, v);
    return d;
  }
};
class TireBench {
  Tire tire_;
  tam::pmg::ParamReferenceManager pm_;
public:
  explicit TireBench(const py::dict& params) {
    tire_.declare_parameters(&pm_, "");
    set_params(pm_, params);
  }
  std::array<double,2> forces(double kappa, double alpha, double fz, double scale) {
    if (fz <= 0 || scale <= 0 || !std::isfinite(kappa+alpha+fz+scale))
      throw std::invalid_argument("Invalid tire bench input");
    tam::ocd::types::TireModelInput input;
    input.longitudinal_slip=kappa; input.slip_angle_rad=alpha; input.F_z_N=fz;
    const auto out=tire_.evaluate(input);
    return {scale*out.longitudinal_force_N,scale*out.lateral_force_N};
  }
};
PYBIND11_MODULE(_ocd_native, m) {
  m.attr("UPSTREAM_COMMIT") = "94f8fb187fb0ed22bba1d809bd74f66d1ff75af4";
  py::class_<Model>(m, "Model").def(py::init<const py::dict&>())
    .def("reset", &Model::reset).def("step", &Model::step, py::arg("steering_rad"),
         py::arg("force_N"),py::arg("scale"),py::arg("internal_mu")=false).def("logs", &Model::logs);
  py::class_<TireBench>(m,"TireBench").def(py::init<const py::dict&>()).def("forces",&TireBench::forces);
  bind_planning(m);
}
