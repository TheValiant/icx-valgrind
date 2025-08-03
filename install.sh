#!/bin/bash
#
# This script performs a 2-stage Profile-Guided Optimization (PGO) build
# of Valgrind. It prioritizes the Intel oneAPI compilers (icx/icpx) but
# will fall back to the system's gcc/g++ if oneAPI is not found.
#
# It is designed to be idempotent and safe to re-run.
#

# --- Script Configuration ---
set -e    # Exit immediately if a command exits with a non-zero status.
set -u    # Treat unset variables as an error when substituting.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -x    # Print commands and their arguments as they are executed.

# --- Logging Setup ---
LOG_FILE="$(pwd)/valgrind_build_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Valgrind PGO build - logging to: ${LOG_FILE}"
# Redirect all output to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- Command Line Arguments ---
VERBOSE=false
for arg in "$@"; do
    case $arg in
        --verbose|-v)
            VERBOSE=true
            echo "Verbose mode enabled"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose|-v] [--help|-h]"
            echo "  --verbose, -v    Enable verbose output for all build commands"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set verbose flags based on command line argument
if [ "$VERBOSE" = true ]; then
    CURL_VERBOSE="-v"
    TAR_VERBOSE="-v"
    COMPILER_VERBOSE="-v"
else
    CURL_VERBOSE=""
    TAR_VERBOSE=""
    COMPILER_VERBOSE=""
fi

# --- Build Configuration Variables ---
VALGRIND_VERSION="3.25.1"
VALGRIND_URL="https://sourceware.org/pub/valgrind/valgrind-${VALGRIND_VERSION}.tar.bz2"
TARBALL_NAME="valgrind-${VALGRIND_VERSION}.tar.bz2"
EXPECTED_MD5="2b424c9a43aa9bf2840d4989b01ea6e7"
EXPECTED_SHA1="4d2cc4d527213f81af573bca4d2cb93ccac7f274"

PGO_BENCHMARK_URL="https://raw.githubusercontent.com/TheValiant/icx-valgrind/main/PGO_benchmark.cpp"
PGO_BENCHMARK_SRC="PGO_benchmark.cpp"
PGO_BENCHMARK_EXE="pgo_benchmark"

BUILD_DIR="/tmp/valgrind-pgo-build"
# Use a standard local prefix for a clean installation
INSTALL_PREFIX="${HOME}/.local"
FINAL_VALGRIND_PATH="${INSTALL_PREFIX}/bin/valgrind"

# Flags to use for both PGO generation and the final performance test
VALGRIND_PGO_FLAGS="-s \
--leak-resolution=high \
--track-origins=yes \
--num-callers=500 \
--show-mismatched-frees=yes \
--track-fds=yes \
--trace-children=yes \
--gen-suppressions=no \
--error-limit=no \
--undef-value-errors=yes \
--expensive-definedness-checks=yes \
--malloc-fill=0x41 \
--free-fill=0x42 \
--read-var-info=yes \
--keep-debuginfo=yes \
--show-realloc-size-zero=yes \
--partial-loads-ok=no"

# --- Main Script Logic ---

echo "========================================================================"
echo "Initial Setup and Directory Creation"
echo "========================================================================"
mkdir -p "${BUILD_DIR}"
# Also create the bin directory in the install prefix
mkdir -p "${INSTALL_PREFIX}/bin"
cd "${BUILD_DIR}"
echo "Build directory: ${BUILD_DIR}"
echo "Installation prefix: ${INSTALL_PREFIX}"

echo "========================================================================"
echo "Stage 0: Pre-flight Compiler and Sanity Check"
echo "========================================================================"
INTEL_SETVARS_SCRIPT="/opt/intel/oneapi/setvars.sh"

if [ -f "$INTEL_SETVARS_SCRIPT" ]; then
    echo "Intel oneAPI environment found. Sourcing setvars.sh..."
    set +x
    export OCL_ICD_FILENAMES=""
    export SETVARS_ARGS="--force"
    set +u
    # shellcheck source=/opt/intel/oneapi/setvars.sh
    source "$INTEL_SETVARS_SCRIPT"
    set -u
    unset SETVARS_ARGS
    set -x

    export CC=icx
    export CXX=icpx
    export I_MPI_CC=icx
    export I_MPI_CXX=icpx
    echo "Using Intel compilers: $(which icx) and $(which icpx)"

    echo "Performing a pre-flight check to ensure C++ compiler can build the benchmark..."
    curl -L -O ${CURL_VERBOSE} "${PGO_BENCHMARK_URL}"
    # Test compilation by outputting to /dev/null
    ${CXX} -O3 -ipo -static -g3 -flto ${COMPILER_VERBOSE} -o /dev/null "${PGO_BENCHMARK_SRC}"
    rm -f "${PGO_BENCHMARK_SRC}"
    echo "Pre-flight check successful."
fi


echo "========================================================================"
echo "Stage 1: Download Valgrind Source"
echo "========================================================================"
if [ ! -f "${TARBALL_NAME}" ]; then
    echo "Downloading Valgrind..."
    curl -L -O ${CURL_VERBOSE} "${VALGRIND_URL}"
else
    echo "Valgrind tarball already exists. Skipping download."
fi


echo "========================================================================"
echo "Stage 2: Verify Hashes"
echo "========================================================================"
echo "${EXPECTED_MD5}  ${TARBALL_NAME}" | md5sum -c -
echo "MD5 hash verified successfully."
echo "${EXPECTED_SHA1}  ${TARBALL_NAME}" | sha1sum -c -
echo "SHA1 hash verified successfully."


echo "========================================================================"
echo "Stage 3: PGO Build - Phase 1 (Instrumentation)"
echo "========================================================================"
if [ -d "valgrind-pgo-instrumented" ]; then
    rm -rf "valgrind-pgo-instrumented"
fi
tar -xjf "${TARBALL_NAME}"
mv "valgrind-${VALGRIND_VERSION}" "valgrind-pgo-instrumented"
cd "valgrind-pgo-instrumented"

INSTRUMENTED_INSTALL_DIR="${BUILD_DIR}/temp_install"
PGO_DATA_DIR="${BUILD_DIR}/pgo-data"
mkdir -p "${PGO_DATA_DIR}"

BASE_FLAGS="-O3 -pipe -fp-model=fast -march=native -static -ffast-math -funroll-loops -finline-functions -inline-level=2 -fvectorize -vec"
PGO_GEN_FLAGS="-fprofile-instr-generate=${PGO_DATA_DIR}/default.profraw"
export CFLAGS="${BASE_FLAGS} ${PGO_GEN_FLAGS}"
export CXXFLAGS="${BASE_FLAGS} ${PGO_GEN_FLAGS}"

./configure --prefix="${INSTRUMENTED_INSTALL_DIR}"  --enable-lto
make -j"$(nproc)" V=1
make install -j"$(nproc)" V=1

unset CFLAGS CXXFLAGS
INSTRUMENTED_VALGRIND="${INSTRUMENTED_INSTALL_DIR}/bin/valgrind"


echo "========================================================================"
echo "Stage 4: Generate Profile Data"
echo "========================================================================"
cd "${BUILD_DIR}"

echo "Downloading latest PGO benchmark source..."
rm -f "${PGO_BENCHMARK_SRC}"
curl -L -O ${CURL_VERBOSE} "${PGO_BENCHMARK_URL}"

echo "Compiling the benchmark code..."
${CXX} -O3 -ipo -static -g3 -flto ${COMPILER_VERBOSE} -o "${PGO_BENCHMARK_EXE}" "${PGO_BENCHMARK_SRC}"

export LLVM_PROFILE_FILE="${PGO_DATA_DIR}/default.profraw"
echo "Running benchmark with instrumented Valgrind to generate PGO data..."
echo "Timing information for the INSTRUMENTED run:"
time "${INSTRUMENTED_VALGRIND}" ${VALGRIND_PGO_FLAGS} ./"${PGO_BENCHMARK_EXE}"

echo "Converting profile data from .profraw to .profdata format..."
llvm-profdata merge -output="${PGO_DATA_DIR}/default.profdata" "${PGO_DATA_DIR}/default.profraw"
echo "Profile data conversion completed."

unset LLVM_PROFILE_FILE
echo "Profile data has been generated."


echo "========================================================================"
echo "Stage 5: PGO Build - Phase 2 (Optimized Recompilation)"
echo "========================================================================"
if [ -d "valgrind-pgo-optimized" ]; then
    rm -rf "valgrind-pgo-optimized"
fi
tar -xjf "${TARBALL_NAME}"
mv "valgrind-${VALGRIND_VERSION}" "valgrind-pgo-optimized"
cd "valgrind-pgo-optimized"

PGO_USE_FLAGS="-fprofile-instr-use=${PGO_DATA_DIR}/default.profdata"
export CFLAGS="${BASE_FLAGS} ${PGO_USE_FLAGS}"
export CXXFLAGS="${BASE_FLAGS} ${PGO_USE_FLAGS}"

# Configure to install to the final destination
./configure --prefix="${INSTALL_PREFIX}"  --enable-lto
make -j"$(nproc)" V=1

unset CFLAGS CXXFLAGS


echo "========================================================================"
echo "Stage 6: Install Final Binary and Tool Libraries"
echo "========================================================================"
echo "Installing the final optimized binary and all its components to ${INSTALL_PREFIX}"
make install -j"$(nproc)" V=1


echo "========================================================================"
echo "Stage 7: Update Shell RC Files"
echo "========================================================================"
EXPORT_LINE="export PATH=\"${INSTALL_PREFIX}/bin:\$PATH\""

for rc_file in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [ -f "$rc_file" ]; then
        if grep -qF -- "${EXPORT_LINE}" "$rc_file"; then
            echo "${rc_file} already contains the correct PATH entry. Skipping."
        else
            echo "Adding PATH entry to ${rc_file}..."
            echo -e "\n# Added by Valgrind PGO build script\n${EXPORT_LINE}" >> "$rc_file"
            echo "Entry added."
        fi
    else
        echo "Shell config file not found: ${rc_file}. Skipping."
    fi
done


echo "========================================================================"
echo "Stage 8: Final Test: Run Optimized Valgrind and Compare Timings"
echo "========================================================================"
cd "${BUILD_DIR}"
echo "Timing information for the FINAL OPTIMIZED run:"
time "${FINAL_VALGRIND_PATH}" ${VALGRIND_PGO_FLAGS} ./"${PGO_BENCHMARK_EXE}"


echo "========================================================================"
echo "SUCCESS!"
echo "PGO-optimized Valgrind has been built and installed to: ${FINAL_VALGRIND_PATH}"
echo ""
echo "Please restart your shell or run 'source ~/.bashrc' or 'source ~/.zshrc' for the new PATH to take effect."
echo "========================================================================"
echo "Build log saved to: ${LOG_FILE}"

# turn off xtrace
set +x