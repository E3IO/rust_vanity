# Rust Vanity Address Generator

A high-performance vanity address generator for Solana or other ed25519-based blockchains, written in Rust. This tool can generate addresses matching specified prefixes or suffixes, with GPU acceleration via CUDA.

## Features

- Generate ed25519 addresses with customizable prefixes and suffixes
- High-performance multi-threaded CPU generation using Rayon
- CUDA GPU acceleration for even faster generation
- Case-insensitive matching option
- Wildcard support in patterns ('?' matches any character)
- Multiple GPU support
- Resilient error handling
- Comprehensive logging and progress tracking

## Requirements

- Rust (stable)
- CUDA Toolkit (for GPU acceleration)
- NVIDIA GPU with Compute Capability 5.0 or newer (for GPU acceleration)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rust_vanity.git
   cd rust_vanity
   ```

2. If using CUDA, compile the PTX files:
   ```
   make cuda-ptx
   ```

3. Build the project:
   ```
   make build
   ```

## Usage

Run the vanity address generator with the following options:

```
./target/release/rust_vanity [OPTIONS]
```

### Options

- `--prefix <PREFIX>`: Generate addresses starting with this prefix (can be specified multiple times)
- `--suffix <SUFFIX>`: Generate addresses ending with this suffix (can be specified multiple times)
- `--threads <THREADS>`: Number of CPU threads to use (default: number of logical cores)
- `--batch-size <SIZE>`: Number of attempts per GPU batch (default: 100000)
- `--gpu`: Enable GPU acceleration (requires CUDA support)
- `--ignore-case`: Case-insensitive matching
- `--output <FILE>`: Output file to save results (default: "keys.txt")

### Examples

Generate addresses starting with "RUST":
```
./target/release/rust_vanity --prefix RUST
```

Generate addresses ending with "777":
```
./target/release/rust_vanity --suffix 777
```

Generate addresses either starting with "ABC" or ending with "XYZ", case-insensitive:
```
./target/release/rust_vanity --prefix ABC --suffix XYZ --ignore-case
```

Use GPU acceleration:
```
./target/release/rust_vanity --prefix COOL --gpu
```

## Performance

Performance varies depending on hardware:

- Single CPU thread: ~10,000 attempts/second
- Multi-threaded CPU (8 cores): ~80,000 attempts/second
- Mid-range GPU (e.g., GTX 1060): ~500,000 attempts/second
- High-end GPU (e.g., RTX 3080): ~2,000,000 attempts/second
- Multiple GPUs: Linear scaling with number of devices

## Architecture

The project is structured as follows:

- `src/main.rs`: Command-line interface and application entry point
- `src/config.rs`: Configuration handling
- `src/vanity.rs`: Core address generation algorithms
- `src/cuda/`: CUDA integration for GPU acceleration
  - `src/cuda/mod.rs`: CUDA context management
  - `src/cuda/vanity_kernel.rs`: GPU kernel implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on [ed25519-dalek](https://github.com/dalek-cryptography/ed25519-dalek) for Rust Ed25519 implementation
- Uses [rustacuda](https://github.com/bheisler/RustaCUDA) for CUDA integration 