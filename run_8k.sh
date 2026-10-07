#!/bin/bash
MODEL="${1:-./gemma-4-E4B-it.litertlm}"
MAX_TOKENS="${MAX_TOKENS:-6192}"
./bazel-bin/multimodal_cli --server --gpu --max-tokens "$MAX_TOKENS" --port 11434 "$MODEL"
