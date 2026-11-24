cmake_minimum_required(VERSION 3.18)
project(extreme_ocd LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
set(OCD ${CMAKE_CURRENT_SOURCE_DIR}/../upstream/Open-Car-Dynamics)
set(COMMON ${OCD}/submodules/tam__common)
set(PMG ${COMMON}/submodules/parameter_management/param_management_cpp)
set(TSL ${COMMON}/submodules/time_series_logging/tsl_logger_cpp)
# A single research configuration, using upstream equations without modifications.
# This is our binding, not the upstream 48-model Python factory or ROS build.
pybind11_add_module(_ocd_native NO_EXTRAS bridge.cpp
  ${OCD}/lib/ocd_aerodynamics_models_cpp/src/default.cpp
  ${OCD}/lib/ocd_drivetrain_fx_cpp/src/drivetrain_fx_model.cpp
  ${OCD}/lib/ocd_drivetrain_rwd_lsd_cpp/src/drivetrain_rwd_lsd_model.cpp
  ${OCD}/lib/ocd_drivetrain_rwd_lsd_cpp/src/drivetrain_rwd_lsd_eqns.cpp
  ${OCD}/lib/ocd_steering_actuator_pt1_cpp/src/steering_actuator_pt1_model.cpp
  ${OCD}/lib/ocd_tire_models_cpp/src/mf_52.cpp
  ${PMG}/src/param_manager_composer.cpp
  ${PMG}/src/initialization.cpp
  ${PMG}/src/init_backend.cpp
  ${PMG}/src/param_reference_manager.cpp
  ${PMG}/src/param_value_manager.cpp)
file(GLOB OCD_PACKAGES ${OCD}/lib/*)
foreach(PKG ${OCD_PACKAGES})
  if(IS_DIRECTORY ${PKG}/include)
    target_include_directories(_ocd_native PRIVATE ${PKG}/include)
  endif()
endforeach()
target_include_directories(_ocd_native PRIVATE ${PMG}/include ${TSL}/include
  ${COMMON}/tum_helpers_cpp/include ${COMMON}/tum_types_cpp/include)
target_link_libraries(_ocd_native PRIVATE Eigen3::Eigen)
target_compile_features(_ocd_native PRIVATE cxx_std_20)
target_compile_options(_ocd_native PRIVATE -Wall -Wextra)
set_target_properties(_ocd_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../extreme_avoidance)
