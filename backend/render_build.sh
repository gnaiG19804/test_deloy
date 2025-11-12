#!/bin/sh
set -e

# tạo thư mục cache cho cargo/rustup (vùng có thể ghi)
mkdir -p /tmp/cargo /tmp/rustup
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
export PATH=$CARGO_HOME/bin:$PATH

# cài rustup non-interactive và set default stable
curl https://sh.rustup.rs -sSf | sh -s -- -y
# source environment (rustup installer puts cargo in ~/.cargo by default)
export PATH=$HOME/.cargo/bin:$PATH
rustup default stable

# upgrade pip
python -m pip install --upgrade pip setuptools wheel

# (tùy chọn) dùng PyTorch CPU wheel index
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt
