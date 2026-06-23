cmake_minimum_required(VERSION 3.20)
project(sparse_branch_machine VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

option(SBM_ENABLE_NATIVE "Enable native CPU tuning" ON)
option(SBM_BUILD_TESTS "Build tests" ON)
option(SBM_BUILD_CLI "Build the thin CLI executable" ON)
option(SBM_BUILD_PYTHON_SMOKE_TEST "Run the Python ctypes smoke test with CTest" ON)

function(sbm_configure_target target)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  if(SBM_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(${target} PRIVATE -O3 -march=native)
  endif()
  set_target_properties(${target} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
endfunction()

add_library(sbm_core SHARED
  src/modules/vector_ops.cpp
  src/modules/signature.cpp
)
sbm_configure_target(sbm_core)

add_library(sbm_machine SHARED
  src/modules/machine_storage.cpp
  src/modules/machine_routing.cpp
  src/modules/machine_learning.cpp
  src/modules/token_objective.cpp
  src/modules/machine_maintenance.cpp
)
sbm_configure_target(sbm_machine)
target_link_libraries(sbm_machine PUBLIC sbm_core)

add_library(sbm_dataset SHARED
  src/modules/dataset.cpp
)
sbm_configure_target(sbm_dataset)
target_link_libraries(sbm_dataset PUBLIC sbm_core)

add_library(sbm_experiment SHARED
  src/modules/experiment.cpp
  src/modules/token_experiment.cpp
)
sbm_configure_target(sbm_experiment)
target_link_libraries(sbm_experiment PUBLIC sbm_machine sbm_dataset sbm_core)

# Stable C ABI loaded by the CLI and Python ctypes wrapper.
add_library(sbm_api SHARED
  src/api/c_api.cpp
)
sbm_configure_target(sbm_api)
target_compile_definitions(sbm_api PRIVATE SBM_API_BUILD)
set_target_properties(sbm_api PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
target_link_libraries(sbm_api PRIVATE sbm_experiment sbm_machine sbm_dataset sbm_core)

# Compatibility umbrella target for direct C++ embedding.
add_library(sbm INTERFACE)
target_include_directories(sbm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(sbm INTERFACE sbm_experiment sbm_machine sbm_dataset sbm_core)

if(SBM_BUILD_CLI)
  add_executable(sbm_cli src/cli/main.cpp)
  set_target_properties(sbm_cli PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  target_include_directories(sbm_cli PRIVATE include)
  target_link_libraries(sbm_cli PRIVATE sbm_api)
  if(SBM_ENABLE_NATIVE AND NOT MSVC)
    target_compile_options(sbm_cli PRIVATE -O3 -march=native)
  endif()
  # Preserve the historical build target name without producing another exe.
  add_executable(sbm_benchmark ALIAS sbm_cli)
endif()

if(SBM_BUILD_TESTS)
  enable_testing()

  add_executable(sbm_tests tests/test_sbm.cpp)
  target_link_libraries(sbm_tests PRIVATE sbm)
  add_test(NAME sbm_cpp_api COMMAND sbm_tests)

  add_executable(sbm_c_api_tests tests/test_c_api.cpp)
  target_include_directories(sbm_c_api_tests PRIVATE include)
  target_link_libraries(sbm_c_api_tests PRIVATE sbm_api)
  add_test(NAME sbm_c_api COMMAND sbm_c_api_tests)

  if(SBM_BUILD_PYTHON_SMOKE_TEST)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      add_test(
        NAME sbm_python_api
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_api.py
                --library $<TARGET_FILE:sbm_api>)
    endif()
  endif()
endif()

install(TARGETS sbm_core sbm_machine sbm_dataset sbm_experiment sbm_api
  EXPORT sbmTargets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
if(SBM_BUILD_CLI)
  install(TARGETS sbm_cli RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)
install(FILES python/sbm_runtime.py DESTINATION python)
install(PROGRAMS scripts/tune.py scripts/run_python_api.py DESTINATION bin)
