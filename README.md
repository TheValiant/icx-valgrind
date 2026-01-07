# icx-valgrind

Automated Valgrind build script with advanced compiler optimizations using Intel oneAPI compilers. Designed for users who want a more performant Valgrind installation.

## Overview

This project provides an optimized build script for Valgrind that leverages Intel's oneAPI compilers (ICX/ICPX) to produce a more performant binary through:

- **Profile-Guided Optimization (PGO)** - Uses runtime profiling to optimize hot paths
- **Link-Time Optimization (LTO)** - Enables whole-program optimization across translation units
- **Intel-specific flags** - Utilizes `-march=native`, `-ffast-math`, vectorization, and other Intel compiler optimizations

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/TheValiant/icx-valgrind/main/install.sh | bash
```

Then restart your terminal or source your shell configuration:

```bash
source ~/.bashrc   # or ~/.zshrc
```

## Requirements

- Intel oneAPI Base Toolkit (includes ICX compiler)
- Standard build tools (make, autoconf)
- LLVM tools (llvm-ar, llvm-ranlib, lld)
- Internet connection (downloads Valgrind source)

### Installing Intel oneAPI

On Linux:
```bash
# Add Intel repository
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

# Install base toolkit
sudo apt install intel-basekit
```

On macOS (Intel):
```bash
# Install via installer from Intel's website
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html
```

## What the Script Does

1. **Pre-flight checks** - Verifies Intel compilers and required tools are available
2. **Downloads Valgrind** - Fetches the latest stable Valgrind release (3.25.1)
3. **Stage 1 Build** - Compiles with basic optimizations
4. **PGO Training** - Runs benchmark suite to generate optimization profile
5. **Stage 2 Build** - Recompiles with PGO data and full LTO enabled
6. **Installation** - Installs to `~/.local/bin/valgrind`
7. **PATH Setup** - Automatically updates shell configuration files

## Build Flags

### Stage 1 (Initial Build)
```bash
-O3 -pipe -fp-model=fast -march=native
```

### Stage 2 (Optimized Build)
```bash
-O3 -pipe -fp-model=fast -march=native -static -ffast-math 
-funroll-loops -fvectorize -vec -flto=full 
-fprofile-instr-use=<profile_data>
```

## Manual Build

If you prefer to run the build manually or need to customize:

```bash
git clone https://github.com/TheValiant/icx-valgrind.git
cd icx-valgrind
chmod +x install.sh
./install.sh --verbose
```

### Options

- `--verbose`, `-v` - Enable verbose build output
- `--help`, `-h` - Show usage information

## Verification

After installation, verify the optimized build:

```bash
which valgrind
# Should show: ~/.local/bin/valgrind

valgrind --version
# Shows installed version

# Check for LTO in binary
readelf -p .comment ~/.local/bin/valgrind
```

## Alternative: GCC Build

For systems without Intel compilers, a GCC fallback script is provided:

```bash
./gcc.sh
```

This uses GCC with similar optimization flags but without Intel-specific optimizations.

## Notes

- Build takes approximately 15-30 minutes depending on system
- Requires ~500MB of temporary disk space during build
- PGO training phase runs Valgrind against a comprehensive test suite
- The script is idempotent and safe to re-run

## Troubleshooting

### Intel compiler not found
Ensure you've sourced the oneAPI environment:
```bash
source /opt/intel/oneapi/setvars.sh
```

### LTO verification fails
LTO requires matching versions of compiler tools. Ensure `llvm-ar`, `llvm-ranlib`, and `lld` are from the Intel oneAPI installation.

### Build fails with OOM
Reduce parallel jobs by editing the script to use `make -j4` instead of `make -j$(nproc)`.

## License

This project is provided as-is. Valgrind itself is licensed under GPLv2.

