#!/bin/sh
set -e
mkdir -p /tmp/cargo /tmp/rustup
export CARGO_HOME=/tmp/cargo
export RUSTUP_HOME=/tmp/rustup
python -m pip install --upgrade pip
pip install -r requirements.txt
