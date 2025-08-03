#!/bin/bash
#
# This script performs a 2-stage Profile-Guided Optimization (PGO) build
# of Valgrind with full Link Time Optimization (LTO). It prioritizes the Intel
# oneAPI compilers (icx/icpx) and includes extensive pre-flight checks to
# fail early if the environment is not correctly configured.
#
# Key optimizations:
# - Profile-Guided Optimization (PGO) for runtime optimization
# - Full Link Time Optimization (LTO) applied in the final stage
# - Intel-specific optimization flags and vectorization
# - Static linking for better performance
#
# It is designed to be idempotent and safe to re-run.
#

# --- Script Configuration ---
set -e
set -u
set -o pipefail
set -x

# --- Logging Setup ---
SCRIPT_CWD=$(pwd)
LOG_FILE="${SCRIPT_CWD}/valgrind_build_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Valgrind PGO build - logging to: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- Automatic Cleanup ---
BUILD_DIR=$(mktemp -d -t valgrind-pgo-build-XXXXXXXXXX)
trap 'echo "Cleaning up temporary directory: ${BUILD_DIR}"; rm -rf "${BUILD_DIR}"' EXIT

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

INSTALL_PREFIX="${HOME}/.local"
FINAL_VALGRIND_PATH="${INSTALL_PREFIX}/bin/valgrind"

VALGRIND_PGO_FLAGS="-s \
--leak-resolution=high --track-origins=yes --num-callers=500 \
--show-mismatched-frees=yes --track-fds=yes --trace-children=yes \
--gen-suppressions=no --error-limit=no --undef-value-errors=yes \
--expensive-definedness-checks=yes --malloc-fill=0x41 --free-fill=0x42 \
--read-var-info=yes --keep-debuginfo=yes --show-realloc-size-zero=yes \
--partial-loads-ok=no"

# --- Helper Functions ---
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: Required command '$1' not found. Please install it." >&2
        exit 1
    fi
}

# --- Main Script Logic ---

echo "========================================================================"
echo "Stage 0: Pre-flight Checks & Environment Setup"
echo "========================================================================"

echo "Checking for essential system commands..."
check_command curl
check_command tar
check_command bzip2
check_command md5sum
check_command sha1sum
check_command make
check_command readelf
check_command file
echo "✓ All essential system commands are present."

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
else
    echo "ERROR: Intel oneAPI setvars.sh not found at ${INTEL_SETVARS_SCRIPT}" >&2
    exit 1
fi

echo "Verifying compiler and toolchain..."
check_command icx
check_command icpx
check_command llvm-profdata

INTEL_COMPILER_DIR="$(dirname "$(which icx)")/compiler"
if [ ! -d "${INTEL_COMPILER_DIR}" ]; then
    echo "ERROR: Could not dynamically locate Intel compiler toolchain directory at '${INTEL_COMPILER_DIR}'." >&2
    exit 1
fi
echo "Dynamically located Intel compiler toolchain at: ${INTEL_COMPILER_DIR}"

if [ ! -f "${INTEL_COMPILER_DIR}/llvm-ar" ] || [ ! -f "${INTEL_COMPILER_DIR}/llvm-ranlib" ] || [ ! -f "${INTEL_COMPILER_DIR}/ld.lld" ]; then
    echo "ERROR: Essential LLVM LTO tools are missing from the Intel oneAPI installation." >&2
    echo "Please ensure 'llvm-ar', 'llvm-ranlib', and 'ld.lld' exist in ${INTEL_COMPILER_DIR}" >&2
    exit 1
fi
echo "✓ All required Intel and LLVM toolchain commands are present."

export CC=icx
export CXX=icpx
export I_MPI_CC=icx
export I_MPI_CXX=icpx

echo "Checking directory permissions..."
mkdir -p "${INSTALL_PREFIX}/bin"
if ! touch "${BUILD_DIR}/.permission_test" || ! touch "${INSTALL_PREFIX}/.permission_test"; then
    echo "ERROR: Do not have write permissions for build dir (${BUILD_DIR}) or install prefix (${INSTALL_PREFIX})." >&2
    exit 1
fi
rm "${BUILD_DIR}/.permission_test" "${INSTALL_PREFIX}/.permission_test"
echo "✓ Write permissions are OK."

echo "Performing a quick compilation sanity check..."
cd "${BUILD_DIR}"
curl -s -L -O "${PGO_BENCHMARK_URL}"
${CXX} -O3 -static -flto ${COMPILER_VERBOSE} -o /dev/null "${PGO_BENCHMARK_SRC}"
rm -f "${PGO_BENCHMARK_SRC}"
echo "✓ Pre-flight check successful. Proceeding with build."


echo "========================================================================"
echo "Stage 1: Download or Locate Valgrind Source"
echo "========================================================================"
LOCAL_TARBALL_SRC="${SCRIPT_CWD}/${TARBALL_NAME}"
USE_LOCAL_TARBALL=false

if [ -f "${LOCAL_TARBALL_SRC}" ]; then
    echo "Found local tarball at ${LOCAL_TARBALL_SRC}. Verifying hashes..."
    if (cd "${SCRIPT_CWD}" && echo "${EXPECTED_MD5}  ${TARBALL_NAME}" | md5sum -c - && echo "${EXPECTED_SHA1}  ${TARBALL_NAME}" | sha1sum -c -); then
        USE_LOCAL_TARBALL=true
    else
        echo "⚠ WARNING: Local tarball hash mismatch. Will re-download."
    fi
fi

cd "${BUILD_DIR}"
if [ "$USE_LOCAL_TARBALL" = true ]; then
    echo "✓ Hashes verified. Copying local tarball to build directory."
    cp "${LOCAL_TARBALL_SRC}" .
else
    echo "Downloading Valgrind to build directory..."
    curl -L -O ${CURL_VERBOSE} "${VALGRIND_URL}"
fi


echo "========================================================================"
echo "Stage 2: Verify Hashes of Build Directory Tarball"
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

BASE_FLAGS_INSTRUMENT="-O3 -pipe -fp-model=fast -march=native -static -ffast-math -funroll-loops -fvectorize -vec"
PGO_GEN_FLAGS="-fprofile-instr-generate"

./configure --prefix="${INSTRUMENTED_INSTALL_DIR}" \
    CFLAGS="${BASE_FLAGS_INSTRUMENT} ${PGO_GEN_FLAGS}" \
    CXXFLAGS="${BASE_FLAGS_INSTRUMENT} ${PGO_GEN_FLAGS}"

make -j"$(nproc)" V=1
make install -j"$(nproc)" V=1

INSTRUMENTED_VALGRIND="${INSTRUMENTED_INSTALL_DIR}/bin/valgrind"


echo "========================================================================"
echo "Stage 4: Generate Profile Data"
echo "========================================================================"
cd "${BUILD_DIR}"

echo "Compiling the benchmark code..."
curl -s -L -O "${PGO_BENCHMARK_URL}"
${CXX} -O3 -ipo -static -g3 -flto ${COMPILER_VERBOSE} -o "${PGO_BENCHMARK_EXE}" "${PGO_BENCHMARK_SRC}"

export LLVM_PROFILE_FILE="${PGO_DATA_DIR}/default-%p.profraw"
echo "Running benchmark with instrumented Valgrind to generate PGO data..."
echo "Timing information for the INSTRUMENTED run:"
time "${INSTRUMENTED_VALGRIND}" ${VALGRIND_PGO_FLAGS} ./"${PGO_BENCHMARK_EXE}"

echo "Converting profile data from .profraw to .profdata format..."
llvm-profdata merge -output="${PGO_DATA_DIR}/default.profdata" "${PGO_DATA_DIR}/default-"*.profraw
echo "Profile data conversion completed."

unset LLVM_PROFILE_FILE


echo "========================================================================"
echo "Stage 5: Verify Profile Data"
echo "========================================================================"
echo "Verifying PGO data was generated successfully..."
if [ ! -s "${PGO_DATA_DIR}/default.profdata" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "!! ERROR: PGO data file is missing or empty." >&2
    echo "!! The instrumentation run in Stage 4 likely failed to generate profile data." >&2
    echo "!! Check the log for errors during the benchmark run." >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    exit 1
fi
echo "✓ PGO profile data is present and valid."


echo "========================================================================"
echo "Stage 6: PGO Build - Phase 2 (Optimized Recompilation with LTO)"
echo "========================================================================"
if [ -d "valgrind-pgo-optimized" ]; then
    rm -rf "valgrind-pgo-optimized"
fi
tar -xjf "${TARBALL_NAME}"
mv "valgrind-${VALGRIND_VERSION}" "valgrind-pgo-optimized"
cd "valgrind-pgo-optimized"

### FIX: Removed redundant/problematic flags and separated PGO-use for the 'make' stage ###
BASE_FLAGS_OPTIMIZED="-O3 -pipe -fp-model=fast -march=native -static -ffast-math -funroll-loops -fvectorize -vec -flto=full"
PGO_USE_FLAGS="-fprofile-instr-use=${PGO_DATA_DIR}/default.profdata"

# Run configure WITHOUT PGO flags to ensure its LTO check passes cleanly.
./configure --prefix="${INSTALL_PREFIX}" --enable-lto \
    CFLAGS="${BASE_FLAGS_OPTIMIZED}" \
    CXXFLAGS="${BASE_FLAGS_OPTIMIZED}" \
    LDFLAGS="-fuse-ld=lld -flto=full" \
    AR="${INTEL_COMPILER_DIR}/llvm-ar" \
    RANLIB="${INTEL_COMPILER_DIR}/llvm-ranlib"

# Apply the PGO flags during the 'make' stage using AM_CFLAGS.
# This appends the flags to the existing CFLAGS set by configure.
make -j"$(nproc)" V=1 AM_CFLAGS="${PGO_USE_FLAGS}" AM_CXXFLAGS="${PGO_USE_FLAGS}"


echo "========================================================================"
echo "Stage 7: Install Final Binary and Tool Libraries"
echo "========================================================================"
echo "Installing the final optimized binary and all its components to ${INSTALL_PREFIX}"
make install -j"$(nproc)" V=1


echo "========================================================================"
echo "Stage 8: LTO Verification"
echo "========================================================================"
echo "Verifying that LTO was properly applied to the final binary..."

if [ -f "${FINAL_VALGRIND_PATH}" ]; then
    echo "Binary size analysis:"
    ls -lh "${FINAL_VALGRIND_PATH}"

    echo ""
    echo "Checking compiler information in binary:"
    readelf -p .comment "${FINAL_VALGRIND_PATH}" 2>/dev/null | head -5 || echo "Could not read compiler info"

    echo ""
    echo "LTO verification: Checking if build used LLVM tools..."
    if grep -q "checking if toolchain accepts lto... yes" "${BUILD_DIR}/valgrind-pgo-optimized/config.log" 2>/dev/null; then
        echo "✓ SUCCESS: Build configuration shows LTO was accepted."
    else
        echo "⚠ WARNING: LTO check failed in configure. LTO may not be properly enabled."
    fi

    echo ""
    echo "Final binary information:"
    file "${FINAL_VALGRIND_PATH}"
else
    echo "ERROR: Final Valgrind binary not found at ${FINAL_VALGRIND_PATH}"
    exit 1
fi


echo "========================================================================"
echo "Stage 9: Update Shell RC Files"
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
echo "Stage 10: Final Test: Run Optimized Valgrind and Compare Timings"
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