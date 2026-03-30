#!/bin/bash
# setup.sh — Copy GNS files into a G-Retriever installation.
# Usage: bash setup.sh /path/to/G-Retriever

set -e

if [ -z "$1" ]; then
    echo "Usage: bash setup.sh /path/to/G-Retriever"
    exit 1
fi

TARGET="$1"

# Validate target
if [ ! -f "$TARGET/train.py" ] || [ ! -d "$TARGET/src/model" ]; then
    echo "Error: $TARGET does not look like a G-Retriever directory."
    echo "Expected to find train.py and src/model/ inside it."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Copying GNS files into $TARGET ..."

# Model files
cp "$SCRIPT_DIR/src/model/gnn.py"       "$TARGET/src/model/gnn.py"
cp "$SCRIPT_DIR/src/model/gns_llm.py"   "$TARGET/src/model/gns_llm.py"
cp "$SCRIPT_DIR/src/model/__init__.py"   "$TARGET/src/model/__init__.py"

# Config
cp "$SCRIPT_DIR/src/config.py"           "$TARGET/src/config.py"

# Training script
cp "$SCRIPT_DIR/train_gns.py"            "$TARGET/train_gns.py"

echo "Done. You can now run:"
echo "  cd $TARGET"
echo "  python train_gns.py --dataset expla_graphs --model_name gns_llm --num_graph_token 8"
