cmake_minimum_required(VERSION 3.20)
project(sparse_branch_machine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SBM_ENABLE_NATIVE "Enable native CPU tuning" ON)
option(SBM_BUILD_TESTS "Build tests" ON)
option(SBM_BUILD_BENCHMARK "Build benchmark executable" ON)

function(sbm_configure_target target)
  target_include_directories(${target} PUBLIC include)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(SBM_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(${target} PRIVATE -O3 -march=native)
  endif()
endfunction()

add_library(sbm_core STATIC
  src/modules/vector_ops.cpp
  src/modules/signature.cpp
)
sbm_configure_target(sbm_core)

add_library(sbm_machine STATIC
  src/modules/machine_storage.cpp
  src/modules/machine_routing.cpp
  src/modules/machine_learning.cpp
  src/modules/machine_maintenance.cpp
)
sbm_configure_target(sbm_machine)
target_link_libraries(sbm_machine PUBLIC sbm_core)

add_library(sbm_dataset STATIC
  src/modules/dataset.cpp
)
sbm_configure_target(sbm_dataset)
target_link_libraries(sbm_dataset PUBLIC sbm_core)

add_library(sbm_experiment STATIC
  src/modules/experiment.cpp
)
sbm_configure_target(sbm_experiment)
target_link_libraries(sbm_experiment PUBLIC sbm_machine sbm_dataset sbm_core)

# Compatibility umbrella target: downstream code can continue linking `sbm`.
add_library(sbm INTERFACE)
target_include_directories(sbm INTERFACE include)
target_link_libraries(sbm INTERFACE sbm_experiment sbm_machine sbm_dataset sbm_core)

if(SBM_BUILD_BENCHMARK)
  add_executable(sbm_benchmark src/main.cpp)
  target_link_libraries(sbm_benchmark PRIVATE sbm)
  if(SBM_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(sbm_benchmark PRIVATE -O3 -march=native)
  endif()
endif()

if(SBM_BUILD_TESTS)
  enable_testing()
  add_executable(sbm_tests tests/test_sbm.cpp)
  target_link_libraries(sbm_tests PRIVATE sbm)
  add_test(NAME sbm_tests COMMAND sbm_tests)
endif()
