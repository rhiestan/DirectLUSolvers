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
option(DLU_WITH_OPENMP  "Test the OpenMPExecutor backend"            OFF)
option(DLU_WITH_TBB     "Test the TBBExecutor backend"               OFF)

set(DLU_TESTDATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../testdata"
    CACHE PATH "Directory holding the MatrixMarket benchmark matrices")
if(NOT EXISTS "${DLU_TESTDATA_DIR}")
  message(WARNING "testdata directory '${DLU_TESTDATA_DIR}' not found -- "
                  "matrix-driven tests will be registered but will fail to load. "
                  "Set -DDLU_TESTDATA_DIR=<path> to fix.")
endif()

find_package(Threads REQUIRED)

# dlu_test_runtime_path(<test> <native dir>)
#
# Prepend a directory to a test's PATH so a DLL it needs at run time is found.
#
# The escaping is the whole point. A CTest ENVIRONMENT property is a CMake LIST,
# and on Windows $ENV{PATH} is full of semicolons -- pasted in raw, every PATH
# entry becomes a separate environment assignment, leaving the test with a PATH
# of just the one directory. That fails in the least helpful way available: the
# named DLL loads (its directory is the surviving entry) and then its own
# dependencies do not, so CTest reports exit code 0xc06d007e with no output and
# it reads as a solver crash. Escaping the separators keeps PATH one element.
function(dlu_test_runtime_path test_name dir)
  if(NOT WIN32 OR NOT dir)
    return()
  endif()
  string(REPLACE ";" "\\;" _host_path "$ENV{PATH}")
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT "PATH=${dir}\\;${_host_path}")
endfunction()

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

  # mkl_rt.dll must be locatable at run time -- linking against mkl_rt.lib is not
  # enough, and a missing DLL surfaces as CTest exit code 0xc0000135 with no
  # output, which reads as a solver crash rather than a PATH problem. Hand CTest
  # the directory instead of making the developer remember it (same treatment as
  # tbb12.dll below).
  if(WIN32 AND EXISTS "${DLU_MKL_DIR}/bin")
    file(TO_NATIVE_PATH "${DLU_MKL_DIR}/bin" DLU_MKL_BIN)
  endif()
endif()

# --- optional: OpenMP executor ---------------------------------------------
if(DLU_WITH_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  add_library(dlu_openmp INTERFACE)
  target_link_libraries(dlu_openmp INTERFACE OpenMP::OpenMP_CXX)
  target_compile_definitions(dlu_openmp INTERFACE DLU_HAVE_OPENMP)
  message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")
endif()

# --- optional: oneTBB executor ---------------------------------------------
if(DLU_WITH_TBB)
  # oneAPI ships a TBBConfig.cmake, so find_package is the right entry point --
  # but the layout has a trap: the `latest` symlink can point at a version that
  # installed libraries WITHOUT headers (2022.3 here), while an older versioned
  # directory beside it is complete (2022.0). Prefer a config whose include tree
  # actually exists rather than whatever `latest` resolves to.
  if(NOT DLU_TBB_DIR)
    file(GLOB _dlu_tbb_candidates
         "C:/Program Files (x86)/Intel/oneAPI/tbb/*/lib/cmake/tbb"
         "C:/Program Files/Intel/oneAPI/tbb/*/lib/cmake/tbb"
         "/opt/intel/oneapi/tbb/*/lib/cmake/tbb")
    list(SORT _dlu_tbb_candidates)
    list(REVERSE _dlu_tbb_candidates)  # newest first
    foreach(_cand IN LISTS _dlu_tbb_candidates)
      get_filename_component(_root "${_cand}/../../.." ABSOLUTE)
      if(EXISTS "${_root}/include/oneapi/tbb.h")
        set(DLU_TBB_DIR "${_cand}" CACHE PATH "Directory containing TBBConfig.cmake")
        break()
      endif()
    endforeach()
  endif()

  if(DLU_TBB_DIR)
    set(TBB_DIR "${DLU_TBB_DIR}" CACHE PATH "" FORCE)
  endif()
  find_package(TBB REQUIRED)
  add_library(dlu_tbb INTERFACE)
  target_link_libraries(dlu_tbb INTERFACE TBB::tbb)
  target_compile_definitions(dlu_tbb INTERFACE DLU_HAVE_TBB)
  message(STATUS "TBB: ${DLU_TBB_DIR}")

  # tbb12.dll must be locatable at run time. Rather than make the developer
  # remember that, hand CTest the directory in the test's PATH.
  get_filename_component(_dlu_tbb_root "${DLU_TBB_DIR}/../../.." ABSOLUTE)
  if(WIN32 AND EXISTS "${_dlu_tbb_root}/bin")
    file(TO_NATIVE_PATH "${_dlu_tbb_root}/bin" DLU_TBB_BIN)
  endif()
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
add_dlu_test(test_pointblock_lu SOURCE test/test_pointblock_lu.cpp LABELS "quick;correctness")
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

# Oracle harness for the header-only METIS nested-dissection port (see
# DirectLUSolvers/src/HeaderOnlyMetis* once it exists). Registered
# unconditionally like test_mc64/test_scalar_types below: without
# DLU_WITH_METIS it just reports "nothing to check" and passes.
set(DLU_HEADER_ONLY_METIS_LINK "")
if(DLU_WITH_METIS)
  list(APPEND DLU_HEADER_ONLY_METIS_LINK dlu_metis)
endif()
add_dlu_test(test_header_only_metis SOURCE test/test_header_only_metis.cpp
             LABELS "testdata;correctness;metis" LINK ${DLU_HEADER_ONLY_METIS_LINK})
add_dlu_test(test_header_only_metis_internal SOURCE test/test_header_only_metis_internal.cpp
             LABELS "quick;correctness;metis" LINK ${DLU_HEADER_ONLY_METIS_LINK})
# Phase 4 wiring check for Eigen::HeaderOnlyMetisOrdering (src/HeaderOnlyMetis.h).
# Registered unconditionally and WITHOUT dlu_metis in the no-METIS build: that
# it still links and passes there is precisely the "no link-time METIS
# dependency" claim under test.
add_dlu_test(test_header_only_metis_ordering SOURCE test/test_header_only_metis_ordering.cpp
             LABELS "quick;correctness;metis" LINK ${DLU_HEADER_ONLY_METIS_LINK})
# Bit-identical parity with the reference depends on float arithmetic (the
# ubfactor/balance comparisons in coarsening and FM refinement) producing the
# EXACT same rounding as the reference build, not just an equally-valid one.
# Without this, clang is free to fuse a*b+c into a single-rounding FMA within
# one statement (the C/C++ standard's default FP_CONTRACT=DEFAULT) wherever
# the port's source happens to keep an expression in one statement and the
# reference's doesn't (or vice versa) -- a source-shape accident, not an
# algorithmic difference, that can flip a borderline float comparison and
# cascade into a completely different (but still "valid") separator choice.
foreach(dlu_metis_parity_target test_header_only_metis test_header_only_metis_internal
                               test_header_only_metis_ordering)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${dlu_metis_parity_target} PRIVATE -ffp-contract=off)
  elseif(MSVC)
    target_compile_options(${dlu_metis_parity_target} PRIVATE /fp:strict)
  endif()
endforeach()
if(DLU_WITH_METIS)
  # metis_internal_bridge.cpp needs METIS's own (non-installed) internal
  # headers -- metislib.h, struct.h, rename.h, gklib_rename.h, ... -- to reach
  # ctrl_t/graph_t/CompressGraph/etc. See that file for why it is a separate
  # translation unit from test_header_only_metis_internal.cpp.
  target_sources(test_header_only_metis_internal PRIVATE test/metis_internal_bridge.cpp)
  # Only libmetis/ (for metislib.h/struct.h/rename.h/...), NOT METIS/include:
  # the raw source metis.h needs -DIDXTYPEWIDTH/-DREALTYPEWIDTH injected by
  # METIS's own CMake build to define idx_t/real_t: pulling it in directly
  # here (instead of resolving metis.h via dlu_metis's already-installed,
  # already-configured copy) fails with "unknown type name 'idx_t'".
  target_include_directories(test_header_only_metis_internal PRIVATE
                             "${CMAKE_CURRENT_SOURCE_DIR}/../METIS/libmetis")
endif()

# SuiteSparse corpus sweep. The matrices are fetched on demand into a git-ignored
# cache (test/matrices/fetch_suitesparse.py); with nothing downloaded the suite
# reports how to get them and passes, so this is safe to register unconditionally.
add_dlu_test(test_mc64         SOURCE test/test_mc64.cpp         LABELS "quick;correctness")
add_dlu_test(test_scalar_types SOURCE test/test_scalar_types.cpp LABELS "quick;correctness")

# Executor backends. StdThreadExecutor is always covered; OpenMP and TBB join in
# when their switches are on, in the same binary, so the three are held to one
# shared contract instead of drifting apart in separate tests.
set(DLU_EXECUTOR_LINK "")
if(DLU_WITH_OPENMP)
  list(APPEND DLU_EXECUTOR_LINK dlu_openmp)
endif()
if(DLU_WITH_TBB)
  list(APPEND DLU_EXECUTOR_LINK dlu_tbb)
endif()
add_dlu_test(test_executors SOURCE test/test_executors.cpp
             LABELS "quick;correctness;parallel" LINK ${DLU_EXECUTOR_LINK})
dlu_test_runtime_path(test_executors "${DLU_TBB_BIN}")
add_dlu_test(test_edge_cases   SOURCE test/test_edge_cases.cpp   LABELS "quick;correctness")

# Serial-vs-parallel consistency for both solvers' intra-supernode parallelism.
# Registered twice: --quick (synthetic matrices only, seconds) joins the "quick"
# set so every build exercises the chunked panel paths, while the bare form adds
# the SuiteSparse corpus and is left to the on-demand runs.
add_dlu_test(test_parallel_consistency SOURCE test/test_parallel_consistency.cpp
             ARGS --quick LABELS "quick;correctness;parallel")
add_dlu_test_variant(test_parallel_consistency_corpus TARGET test_parallel_consistency
                     LABELS "suitesparse;correctness;parallel")

add_dlu_test(test_suitesparse SOURCE test/test_suitesparse.cpp
             LABELS "suitesparse;correctness")
# `--tier large` deliberately omits pre2: Eigen::SparseLU alone runs for minutes
# at 3+ GB there, which no default `ctest` invocation should trigger. Run the
# binary by hand (no arguments) for the full corpus.
add_dlu_test(compare_testdata SOURCE test/compare_testdata.cpp
             ARGS --tier large LABELS "testdata;benchmark" LINK ${DLU_COMPARE_LINK})
dlu_test_runtime_path(compare_testdata "${DLU_MKL_BIN}")

# Thread-count scaling sweep with per-phase timing. Registered with --quick
# (synthetic matrices only) so CTest exercises the code path without spending
# minutes on the large 3D FEM systems; run the binary bare for the full sweep.
add_dlu_test(bench_parallel SOURCE test/bench_parallel.cpp
             ARGS --quick --reps 1 --threads 1,4 LABELS "benchmark;parallel")

# Machine ceiling: what the hardware can deliver before any of our scheduling.
# The --quick form uses only the synthetic matrix so CTest stays fast.
add_dlu_test(bench_ceiling SOURCE test/bench_ceiling.cpp
             ARGS --quick LABELS "benchmark;parallel")

# Per-matrix solver/ordering shootout with warm-up and per-phase timing. Picks up
# METIS and PARDISO rows when those switches are on; --quick keeps CTest to two
# synthetic matrices, so run the binary bare (or with matrix paths) for the real
# comparison.
add_dlu_test(bench_solvers SOURCE test/bench_solvers.cpp
             ARGS --quick --reps 1 --threads 1,4 LABELS "benchmark"
             LINK ${DLU_COMPARE_LINK})
dlu_test_runtime_path(bench_solvers "${DLU_MKL_BIN}")

# The regression suite also has a synthetic-only mode that needs no testdata and
# runs in well under a second -- worth having in the "quick" set so the fill
# guard is exercised even on a checkout without the benchmark matrices.
add_dlu_test_variant(test_regression_synthetic TARGET test_regression
                     ARGS --synthetic-only LABELS "quick;regression")

# ---------------------------------------------------------------------------
#  Profiling driver (not a test)
# ---------------------------------------------------------------------------
#
# A profiler needs the three phases separately attributable, which compute()
# does not give it. profile_driver runs analyzePattern/factorize/solve as
# distinct, individually selectable phases over the SuiteSparse corpus. It is
# NOT registered with CTest -- it asserts nothing.
#
#   cmake --build build --target profile_driver
#   build/profile_driver --solver snlu --phase factorize --reps 5
#
# With DLU_WITH_ITT=ON the phases also emit VTune ITT task markers, so one run
# can be split by "Task Type" in the GUI instead of profiling each phase twice.
option(DLU_BUILD_PROFILE_DRIVER "Build the VTune profiling driver" OFF)
option(DLU_WITH_ITT "Emit VTune ITT task markers from the profiling driver" OFF)

if(DLU_BUILD_PROFILE_DRIVER)
  add_executable(profile_driver test/profile_driver.cpp)
  target_link_libraries(profile_driver PRIVATE dlu_test_common)
  if(DLU_WITH_ITT)
    set(DLU_ITT_DIR "" CACHE PATH "VTune SDK root (the directory containing include/ittnotify.h)")
    if(NOT DLU_ITT_DIR AND DEFINED ENV{VTUNE_PROFILER_DIR})
      set(DLU_ITT_DIR "$ENV{VTUNE_PROFILER_DIR}/sdk")
    endif()
    if(NOT EXISTS "${DLU_ITT_DIR}/include/ittnotify.h")
      message(FATAL_ERROR "DLU_WITH_ITT=ON but ittnotify.h not found under "
                          "'${DLU_ITT_DIR}'. Set -DDLU_ITT_DIR=<VTune>/sdk.")
    endif()
    target_include_directories(profile_driver PRIVATE "${DLU_ITT_DIR}/include")
    target_link_libraries(profile_driver PRIVATE "${DLU_ITT_DIR}/lib64/libittnotify.lib")
    target_compile_definitions(profile_driver PRIVATE DLU_WITH_ITT)
  endif()

  # Ordering-only driver for the header-only METIS port: profile_driver can only
  # attribute the ordering as part of analyzePattern. Links dlu_metis when
  # available so it can time the port against the C library it replaces.
  add_executable(profile_header_only_metis test/profile_header_only_metis.cpp)
  target_link_libraries(profile_header_only_metis PRIVATE dlu_test_common)
  if(DLU_WITH_METIS)
    target_link_libraries(profile_header_only_metis PRIVATE dlu_metis)
  endif()
  # Same FP pinning as the parity suites -- a profiling build that contracts
  # float ops differently would not be measuring the shipped ordering.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(profile_header_only_metis PRIVATE -ffp-contract=off)
  elseif(MSVC)
    target_compile_options(profile_header_only_metis PRIVATE /fp:strict)
  endif()
  if(DLU_WITH_ITT)
    target_include_directories(profile_header_only_metis PRIVATE "${DLU_ITT_DIR}/include")
    target_link_libraries(profile_header_only_metis PRIVATE "${DLU_ITT_DIR}/lib64/libittnotify.lib")
    target_compile_definitions(profile_header_only_metis PRIVATE DLU_WITH_ITT)
  endif()
endif()

message(STATUS "DirectLUSolvers tests: METIS=${DLU_WITH_METIS} PARDISO=${DLU_WITH_PARDISO}")
message(STATUS "  testdata: ${DLU_TESTDATA_DIR}")
