# Contributing to data-core

Bug reports, fixes and improvements are welcome. Please be respectful and
constructive in issues and pull requests.

## Development setup

You need:

- Rust stable (CI uses the version pinned in `.github/workflows/ci.yml`)
- Python 3.10 to 3.13
- FFmpeg development libraries, `pkg-config` and `clang`/`libclang` (bindgen
  generates the FFmpeg bindings at build time)

On Ubuntu or Debian:

```bash
sudo apt-get install -y \
  libavcodec-dev libavformat-dev libavutil-dev \
  libavfilter-dev libavdevice-dev libswscale-dev libswresample-dev \
  pkg-config clang libclang-dev
```

Build the Python module into a virtual environment, with the test
dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "maturin[patchelf]"
maturin develop --extras dev
```

Add `--release` when you want to measure performance; debug builds are much
slower.

The Python package lives in `python/data_core`. `maturin develop` installs it
in editable mode and places the compiled extension `data_core/_native*.so`
next to it, so Python edits take effect immediately and Rust edits need
another `maturin develop`. When you change a binding in
`crates/python/src/lib.rs`, update its stub in `python/data_core/_native.pyi`.

If bindgen fails because it cannot find `limits.h` or `stddef.h`, point it at
your GCC headers. Adjust the version (`13`) to match
`ls /usr/lib/gcc/x86_64-linux-gnu/`:

```bash
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/gcc/x86_64-linux-gnu/13/include"
```

## Running the tests

Rust:

```bash
cargo test --workspace
```

Python, after `maturin develop` (rebuild after every Rust change):

```bash
pytest
```

The PyTorch frontend tests are skipped when `torch` is not installed. To run
them without a GPU, install the CPU build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Formatting and linting

CI runs these, and pull requests must pass them:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Run `cargo fmt --all` to fix formatting.

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests for behavior you change.
- Make sure `cargo test --workspace`, `pytest`, rustfmt, clippy and
  `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` pass.

## Reporting issues

Open an issue at
<https://github.com/PAIR-data-platform/data-core-rs/issues>. For bugs,
include:

- what you ran and what happened, with the full error message
- OS, Python version, `rustc --version` and the FFmpeg version
  (`pkg-config --modversion libavcodec`)
- the dataset format (LeRobot v2 or v3) and, if possible, a public dataset
  or a small script that reproduces the problem

By contributing, you agree that your contributions are licensed under the MIT
License.
