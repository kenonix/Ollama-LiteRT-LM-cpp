#!/bin/bash
MODEL="${1:-./gemma-4-E2B-it.litertlm}"
./bazel-bin/multimodal_cli --server --gpu --port 11434 "$MODEL"
