# DirectLUSolvers -- header-only sparse direct LU solvers for Eigen.
#
# The solvers themselves need no build: they are headers under src/. This file
# exists to build and register the test suites with CTest, so that adding a test
# costs one add_dlu_test() line instead of a hand-maintained compiler invocation
# in a comment.
#
# Configure + build + test (from this directory):
#   cmake -S . -B build -G Ninja
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Fast subset only (skips the large testdata matrices):
#   ctest --test-dir build -L quick
#
# Optional dependencies are each an independent switch; both default OFF:
#   cmake -S . -B build -G Ninja -DDLU_WITH_METIS=ON -DDLU_WITH_PARDISO=ON
#
# Paths default to this project's layout (Eigen, METIS and MKL as siblings of
# the DirectLUSolvers directory); override the DLU_*_DIR cache variables when
# they live elsewhere.

cmake_minimum_required(VERSION 3.16)

# Default to Release -- and note this MUST come BEFORE project().
#
# On a toolchain targeting MSVC (including clang++ with a *-windows-msvc triple),
# Platform/Windows-MSVC.cmake sets CMAKE_BUILD_TYPE_INIT to Debug, so project()
# populates CMAKE_BUILD_TYPE itself. The usual `if(NOT CMAKE_BUILD_TYPE)` guard
# placed after project() therefore never fires there, and everyone silently gets
# an -O0 build -- which for this project means benchmark and regression timings
# roughly 50-100x off. Before project(), the variable reflects only what the user
# actually passed on the command line (or an existing cache), so an explicit
# -DCMAKE_BUILD_TYPE=Debug is still honoured.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Build type (Debug Release RelWithDebInfo MinSizeRel)" FORCE)
endif()

project(DirectLUSolvers VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd / IDE include resolution

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---------------------------------------------------------------------------
#  The solver library (header-only)
# ---------------------------------------------------------------------------

add_library(DirectLUSolvers INTERFACE)
add_library(DirectLUSolvers::DirectLUSolvers ALIAS DirectLUSolvers)
target_include_directories(DirectLUSolvers
                           INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(DirectLUSolvers INTERFACE cxx_std_17)

# ---------------------------------------------------------------------------
#  Eigen
# ---------------------------------------------------------------------------

set(DLU_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen"
    CACHE PATH "Path to the Eigen source tree (the directory containing Eigen/Core)")

# DLU_EIGEN_DIR is checked FIRST, deliberately. find_package(Eigen3) resolves
# against whatever Eigen happens to be registered in the user's CMake package
# registry, which on a developer machine is frequently an unrelated copy of a
# different version -- and these solvers are benchmarked and baselined against
# the Eigen that ships beside them. An in-tree Eigen therefore wins; the
# system-installed one is only a fallback.
if(EXISTS "${DLU_EIGEN_DIR}/Eigen/Core")
  target_include_directories(DirectLUSolvers INTERFACE "${DLU_EIGEN_DIR}")
  get_filename_component(DLU_EIGEN_ABS "${DLU_EIGEN_DIR}" ABSOLUTE)
  message(STATUS "Eigen: ${DLU_EIGEN_ABS}")
else()
  find_package(Eigen3 3.3 QUIET NO_MODULE)
  if(TARGET Eigen3::Eigen)
    target_link_libraries(DirectLUSolvers INTERFACE Eigen3::Eigen)
    get_target_property(DLU_EIGEN_FOUND_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Eigen: via find_package -- ${DLU_EIGEN_FOUND_INC}")
    message(WARNING "Using a system-installed Eigen rather than an in-tree one. "
                    "Fill baselines were recorded against this project's own Eigen; "
                    "set -DDLU_EIGEN_DIR=<path containing Eigen/Core> if results differ.")
  else()
    message(FATAL_ERROR
            "Eigen not found. Set -DDLU_EIGEN_DIR=<path containing Eigen/Core> "
            "(tried '${DLU_EIGEN_DIR}'), or install it so find_package(Eigen3) succeeds.")
  endif()
endif()

# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(DLU_BUILD_TESTS_DEFAULT ON)
else()
  set(DLU_BUILD_TESTS_DEFAULT OFF)  # do not impose tests on a parent project
endif()
option(DLU_BUILD_TESTS "Build the DirectLUSolvers test suites" ${DLU_BUILD_TESTS_DEFAULT})

if(NOT DLU_BUILD_TESTS)
  return()
endif()

enable_testing()

option(DLU_WITH_METIS   "Build the METIS / AutoOrdering comparisons" OFF)
option(DLU_WITH_PARDISO "Build the MKL PARDISO comparison"           OFF)

# No DLU_WITH_OPENMP / DLU_WITH_TBB switches yet: there is no suite that
# exercises OpenMPExecutor or TBBExecutor, so an option here would only promise
# a test that does not exist. test_parallel_lu covers StdThreadExecutor only.
# Adding executor-parameterised variants of it is tracked as follow-up work.

set(DLU_TESTDATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../testdata"
    CACHE PATH "Directory holding the MatrixMarket benchmark matrices")
if(NOT EXISTS "${DLU_TESTDATA_DIR}")
  message(WARNING "testdata directory '${DLU_TESTDATA_DIR}' not found -- "
                  "matrix-driven tests will be registered but will fail to load. "
                  "Set -DDLU_TESTDATA_DIR=<path> to fix.")
endif()

find_package(Threads REQUIRED)

# Common settings for every test binary. The testdata path is baked in as an
# absolute path so tests do not depend on the working directory CTest picks.
add_library(dlu_test_common INTERFACE)
target_link_libraries(dlu_test_common INTERFACE DirectLUSolvers Threads::Threads)
target_include_directories(dlu_test_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/test")
target_compile_definitions(dlu_test_common INTERFACE
                           "DLU_TESTDATA_DIR=\"${DLU_TESTDATA_DIR}\""
                           "DLU_MATRIX_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/test/matrices\""
                           "DLU_BASELINE_FILE=\"${CMAKE_CURRENT_SOURCE_DIR}/test/baselines/testdata.baseline\"")

if(MSVC)
  # /bigobj: the heavily templated solver headers overflow the default object
  # section limit in debug configurations.
  target_compile_options(dlu_test_common INTERFACE /bigobj /EHsc)
endif()

if(WIN32)
  # The tests read DLU_TESTDATA_DIR / DLU_BASELINE_FILE with std::getenv, which
  # the Windows CRT headers mark deprecated in favour of the non-portable
  # _dupenv_s. Portable code wins here; silence the warning rather than fork the
  # code per platform.
  target_compile_definitions(dlu_test_common INTERFACE _CRT_SECURE_NO_WARNINGS)
endif()

# --- optional: METIS -------------------------------------------------------
if(DLU_WITH_METIS)
  set(DLU_METIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../install"
      CACHE PATH "Install prefix containing METIS + GKlib (include/ and lib/)")
  find_path(DLU_METIS_INCLUDE_DIR metis.h HINTS "${DLU_METIS_DIR}/include")
  find_library(DLU_METIS_LIBRARY NAMES metis HINTS "${DLU_METIS_DIR}/lib")
  find_library(DLU_GKLIB_LIBRARY NAMES GKlib gklib HINTS "${DLU_METIS_DIR}/lib")
  if(NOT DLU_METIS_INCLUDE_DIR OR NOT DLU_METIS_LIBRARY)
    message(FATAL_ERROR "DLU_WITH_METIS=ON but METIS was not found under '${DLU_METIS_DIR}'. "
                        "Set -DDLU_METIS_DIR=<install prefix>.")
  endif()
  add_library(dlu_metis INTERFACE)
  target_include_directories(dlu_metis INTERFACE "${DLU_METIS_INCLUDE_DIR}")
  target_link_libraries(dlu_metis INTERFACE "${DLU_METIS_LIBRARY}")
  if(DLU_GKLIB_LIBRARY)
    target_link_libraries(dlu_metis INTERFACE "${DLU_GKLIB_LIBRARY}")
  endif()
  target_compile_definitions(dlu_metis INTERFACE HAVE_METIS)
  message(STATUS "METIS: ${DLU_METIS_LIBRARY}")
endif()

# --- optional: MKL PARDISO -------------------------------------------------
if(DLU_WITH_PARDISO)
  set(DLU_MKL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mkl"
      CACHE PATH "Intel MKL root (containing include/ and lib/)")
  find_path(DLU_MKL_INCLUDE_DIR mkl.h HINTS "${DLU_MKL_DIR}/include")
  find_library(DLU_MKL_RT_LIBRARY NAMES mkl_rt HINTS "${DLU_MKL_DIR}/lib")
  if(NOT DLU_MKL_INCLUDE_DIR OR NOT DLU_MKL_RT_LIBRARY)
    message(FATAL_ERROR "DLU_WITH_PARDISO=ON but MKL was not found under '${DLU_MKL_DIR}'. "
                        "Set -DDLU_MKL_DIR=<mkl root>.")
  endif()
  add_library(dlu_pardiso INTERFACE)
  target_include_directories(dlu_pardiso INTERFACE "${DLU_MKL_INCLUDE_DIR}")
  target_link_libraries(dlu_pardiso INTERFACE "${DLU_MKL_RT_LIBRARY}")
  target_compile_definitions(dlu_pardiso INTERFACE HAVE_PARDISO)
  message(STATUS "MKL PARDISO: ${DLU_MKL_RT_LIBRARY}")
endif()

# add_dlu_test(<name> SOURCE <file> [ARGS ...] [LABELS ...] [LINK ...])
#
# Builds one test binary and registers it with CTest. Labels drive subsetting:
# "quick" tests need no testdata and run in seconds; "testdata" tests read the
# benchmark matrices and take substantially longer.
function(add_dlu_test name)
  cmake_parse_arguments(T "" "SOURCE" "ARGS;LABELS;LINK" ${ARGN})
  add_executable(${name} "${T_SOURCE}")
  target_link_libraries(${name} PRIVATE dlu_test_common ${T_LINK})
  add_test(NAME ${name} COMMAND ${name} ${T_ARGS})
  if(T_LABELS)
    set_tests_properties(${name} PROPERTIES LABELS "${T_LABELS}")
  endif()
  # The matrix-driven suites are far slower than CTest's 1500s default.
  set_tests_properties(${name} PROPERTIES TIMEOUT 3600)
endfunction()

# add_dlu_test_variant(<name> TARGET <existing target> [ARGS ...] [LABELS ...])
#
# Registers another CTest entry running an already-built binary with different
# arguments -- no second compile.
function(add_dlu_test_variant name)
  cmake_parse_arguments(V "" "TARGET" "ARGS;LABELS" ${ARGN})
  add_test(NAME ${name} COMMAND ${V_TARGET} ${V_ARGS})
  set_tests_properties(${name} PROPERTIES LABELS "${V_LABELS}" TIMEOUT 3600)
endfunction()

add_dlu_test(test_supernodal_lu SOURCE test/test_supernodal_lu.cpp LABELS "quick;correctness")
add_dlu_test(test_leftright_lu  SOURCE test/test_leftright_lu.cpp  LABELS "quick;correctness")
add_dlu_test(test_parallel_lu   SOURCE test/test_parallel_lu.cpp   LABELS "testdata;parallel")

# The reader's own unit tests are instant; the registry sweep that follows them
# parses every benchmark matrix (including the 659k-row pre2) and dominates the
# runtime, so the two are registered separately and only the first is "quick".
add_dlu_test(test_matrixmarket SOURCE test/test_matrixmarket.cpp
             ARGS --no-testdata LABELS "quick;correctness")
add_dlu_test_variant(test_matrixmarket_registry TARGET test_matrixmarket
                     LABELS "testdata;correctness")

set(DLU_REGRESSION_LINK "")
set(DLU_COMPARE_LINK "")
if(DLU_WITH_METIS)
  list(APPEND DLU_REGRESSION_LINK dlu_metis)
  list(APPEND DLU_COMPARE_LINK dlu_metis)
endif()
if(DLU_WITH_PARDISO)
  list(APPEND DLU_COMPARE_LINK dlu_pardiso)
endif()

add_dlu_test(test_regression SOURCE test/test_regression.cpp
             LABELS "testdata;regression" LINK ${DLU_REGRESSION_LINK})

# SuiteSparse corpus sweep. The matrices are fetched on demand into a git-ignored
# cache (test/matrices/fetch_suitesparse.py); with nothing downloaded the suite
# reports how to get them and passes, so this is safe to register unconditionally.
add_dlu_test(test_scalar_types SOURCE test/test_scalar_types.cpp LABELS "quick;correctness")
add_dlu_test(test_edge_cases   SOURCE test/test_edge_cases.cpp   LABELS "quick;correctness")

add_dlu_test(test_suitesparse SOURCE test/test_suitesparse.cpp
             LABELS "suitesparse;correctness")
# `--tier large` deliberately omits pre2: Eigen::SparseLU alone runs for minutes
# at 3+ GB there, which no default `ctest` invocation should trigger. Run the
# binary by hand (no arguments) for the full corpus.
add_dlu_test(compare_testdata SOURCE test/compare_testdata.cpp
             ARGS --tier large LABELS "testdata;benchmark" LINK ${DLU_COMPARE_LINK})

# Thread-count scaling sweep with per-phase timing. Registered with --quick
# (synthetic matrices only) so CTest exercises the code path without spending
# minutes on the large 3D FEM systems; run the binary bare for the full sweep.
add_dlu_test(bench_parallel SOURCE test/bench_parallel.cpp
             ARGS --quick --reps 1 --threads 1,4 LABELS "benchmark;parallel")

# Machine ceiling: what the hardware can deliver before any of our scheduling.
# The --quick form uses only the synthetic matrix so CTest stays fast.
add_dlu_test(bench_ceiling SOURCE test/bench_ceiling.cpp
             ARGS --quick LABELS "benchmark;parallel")

# The regression suite also has a synthetic-only mode that needs no testdata and
# runs in well under a second -- worth having in the "quick" set so the fill
# guard is exercised even on a checkout without the benchmark matrices.
add_dlu_test_variant(test_regression_synthetic TARGET test_regression
                     ARGS --synthetic-only LABELS "quick;regression")

message(STATUS "DirectLUSolvers tests: METIS=${DLU_WITH_METIS} PARDISO=${DLU_WITH_PARDISO}")
message(STATUS "  testdata: ${DLU_TESTDATA_DIR}")
