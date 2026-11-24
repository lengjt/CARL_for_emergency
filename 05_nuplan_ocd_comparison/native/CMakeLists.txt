cmake_minimum_required(VERSION 3.18)
project(nuplan_ocd_python39 LANGUAGES CXX)

# Rebuild the project's existing OCD binding for Flow Planner's Python ABI.
# The upstream OCD sources and the existing bridge remain unchanged.
add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../02_extreme_avoidance/native"
  "${CMAKE_CURRENT_BINARY_DIR}/ocd_binding"
)
set_target_properties(_ocd_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../02_extreme_avoidance/extreme_avoidance"
)
