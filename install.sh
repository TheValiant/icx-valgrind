#!/bin/bash
#
# This script performs a 2-stage Profile-Guided Optimization (PGO) build
# of Valgrind. It prioritizes the Intel oneAPI compilers (icx/icpc) but
# will fall back to the system's gcc/g++ if oneAPI is not found.
#
# It is designed to be idempotent and safe to re-run.
#

# --- Script Configuration ---
set -e    # Exit immediately if a command exits with a non-zero status.
set -u    # Treat unset variables as an error when substituting.
set -o pipefail # The return value of a pipeline is the status of the last command to exit with a non-zero status.
set -x    # Print commands and their arguments as they are executed.

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
INSTALL_PREFIX="${HOME}/bin"
FINAL_VALGRIND_PATH="${INSTALL_PREFIX}/valgrind"

# --- Main Script Logic ---

echo "========================================================================"
echo "Stage 0: Initial Setup and Directory Creation"
echo "========================================================================"
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_PREFIX}"
cd "${BUILD_DIR}"
echo "Build directory: ${BUILD_DIR}"
echo "Installation prefix: ${INSTALL_PREFIX}"


echo "========================================================================"
echo "Stage 1: Download Valgrind Source"
echo "========================================================================"
if [ ! -f "${TARBALL_NAME}" ]; then
    echo "Downloading Valgrind..."
    curl -L -O "${VALGRIND_URL}"
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

if [ -d "valgrind-${VALGRIND_VERSION}" ]; then
    echo "Valgrind source directory already exists. Removing for a clean state."
    rm -rf "valgrind-${VALGRIND_VERSION}"
fi
echo "Extracting Valgrind source..."
tar -xjf "${TARBALL_NAME}"


echo "========================================================================"
echo "Stage 3: Configure Compiler and Flags"
echo "========================================================================"
INTEL_SETVARS_SCRIPT="/opt/intel/oneapi/setvars.sh"
PGO_DATA_DIR="${BUILD_DIR}/valgrind-${VALGRIND_VERSION}/pgo-data"

# These variables will be assigned based on the compiler choice below
BUILD_CFLAGS_GEN=""
BUILD_CFLAGS_USE=""
BUILD_LDFLAGS=""

if [ -f "$INTEL_SETVARS_SCRIPT" ]; then
    echo "Intel oneAPI environment found. Sourcing setvars.sh..."
    set +x
    source "$INTEL_SETVARS_SCRIPT"
    set -x
    export CC=icx
    export CXX=icpc
    echo "Using Intel compilers: $(which icx)"

    INTEL_BASE_FLAGS="-O3 -ipo -flto -static \
-fno-strict-aliasing -fno-omit-frame-pointer \
-fvisibility=hidden -fvisibility-inlines-hidden \
-pipe \
-axCORE-AVX2,AVX512_VNNI -qopt-zmm-usage=high \
-fp-model=fast \
-ffunction-sections -fdata-sections -march=native -fwhole-program-vtables"

    BUILD_CFLAGS_GEN="${INTEL_BASE_FLAGS} -prof-gen=threadsafe"
    BUILD_CFLAGS_USE="${INTEL_BASE_FLAGS} -prof-use"
    BUILD_LDFLAGS="-Wl,--gc-sections,-ipo,-static,--as-needed"

else
    echo "------------------------------------------------------------------------"
    echo "WARNING: Intel oneAPI setvars.sh not found at ${INTEL_SETVARS_SCRIPT}."
    echo "Falling back to default system compilers (gcc/g++) with PGO flags."
    echo "------------------------------------------------------------------------"
    export CC=gcc
    export CXX=g++
    echo "Using GCC compilers: $(which gcc)"

    # Create a dedicated directory for GCC's profile data
    mkdir -p "${PGO_DATA_DIR}"

    GCC_BASE_FLAGS="-O3 -march=native -flto -static"
    BUILD_CFLAGS_GEN="${GCC_BASE_FLAGS} -fprofile-generate=${PGO_DATA_DIR}"
    # -fprofile-correction helps with inconsistent profiling data
    BUILD_CFLAGS_USE="${GCC_BASE_FLAGS} -fprofile-use=${PGO_DATA_DIR} -fprofile-correction"
    BUILD_LDFLAGS="-Wl,--gc-sections -flto -static"
fi


echo "========================================================================"
echo "Stage 4: PGO Build - Phase 1 (Instrumentation)"
echo "========================================================================"
cd "valgrind-${VALGRIND_VERSION}"

# Clean any previous build artifacts to ensure a fresh PGO build
if [ -f "Makefile" ]; then
    make distclean || echo "distclean failed, continuing anyway..."
fi

echo "Configuring Valgrind with instrumentation flags..."
TEMP_INSTALL_DIR="${BUILD_DIR}/temp_install"
mkdir -p "$TEMP_INSTALL_DIR"

# Pass the chosen PGO-generate flags to configure
./configure --prefix="${TEMP_INSTALL_DIR}" \
    CFLAGS="${BUILD_CFLAGS_GEN}" \
    CXXFLAGS="${BUILD_CFLAGS_GEN}" \
    LDFLAGS="${BUILD_LDFLAGS}"

echo "Building instrumented Valgrind..."
make -j$(nproc)
make install -j$(nproc)
INSTRUMENTED_VALGRIND="${TEMP_INSTALL_DIR}/bin/valgrind"


echo "========================================================================"
echo "Stage 5: Generate Profile Data"
echo "========================================================================"
cd "${BUILD_DIR}"

if [ ! -f "${PGO_BENCHMARK_SRC}" ]; then
    echo "Downloading PGO benchmark source..."
    curl -L -O "${PGO_BENCHMARK_URL}"
else
    echo "PGO benchmark source already exists."
fi

echo "Compiling the benchmark code with the selected compiler..."
${CXX} -O2 -o "${PGO_BENCHMARK_EXE}" "${PGO_BENCHMARK_SRC}"

echo "Running benchmark with instrumented Valgrind to generate PGO data..."
echo "Timing information for the INSTRUMENTED run:"
# The 'time' command provides stderr output with timing stats
time "${INSTRUMENTED_VALGRIND}" ./"${PGO_BENCHMARK_EXE}"

echo "Profile data has been generated."


echo "========================================================================"
echo "Stage 6: PGO Build - Phase 2 (Optimized Recompilation)"
echo "========================================================================"
cd "valgrind-${VALGRIND_VERSION}"
echo "Re-compiling Valgrind using the generated profile data..."

# We don't need to re-run configure, just 'make' with the prof-use flags.
make -j$(nproc) \
    CFLAGS="${BUILD_CFLAGS_USE}" \
    CXXFLAGS="${BUILD_CFLAGS_USE}" \
    LDFLAGS="${BUILD_LDFLAGS}"


echo "========================================================================"
echo "Stage 7: Strip and Install Final Binary"
echo "========================================================================"
# The final, optimized binary is in ./coregrind/valgrind
echo "Stripping and installing the final optimized binary to ${FINAL_VALGRIND_PATH}"
install -s -m 755 ./coregrind/valgrind "${FINAL_VALGRIND_PATH}"


echo "========================================================================"
echo "Stage 8: Update Shell RC Files"
echo "========================================================================"
EXPORT_LINE='export PATH="'${INSTALL_PREFIX}':$PATH"'

for rc_file in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
    if [ -f "$rc_file" ]; then
        if grep -qF -- "${EXPORT_LINE}" "$rc_file"; then
            echo "${rc_file} already contains the correct PATH entry. Skipping."
        else
            echo "Adding PATH entry to ${rc_file}..."
            # Add a newline just in case the file doesn't end with one
            echo -e "\n# Added by Valgrind PGO build script\n${EXPORT_LINE}" >> "$rc_file"
            echo "Entry added."
        fi
    else
        echo "Shell config file not found: ${rc_file}. Skipping."
    fi
done


echo "========================================================================"
echo "Final Test: Run Optimized Valgrind and Compare Timings"
echo "========================================================================"
cd "${BUILD_DIR}"
echo "Timing information for the FINAL OPTIMIZED run:"
time "${FINAL_VALGRIND_PATH}" ./"${PGO_BENCHMARK_EXE}"


echo "========================================================================"
echo "SUCCESS!"
echo "PGO-optimized Valgrind has been built and installed to: ${FINAL_VALGRIND_PATH}"
echo ""
echo "Please restart your shell or run 'source ~/.bashrc' or 'source ~/.zshrc' for the new PATH to take effect."
echo "========================================================================"

# turn off xtrace
set +x
