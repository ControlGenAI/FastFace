#!/bin/bash
mkdir -p data

if ! command -v gdown &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
fi

FILE_ID="1S-hYdGMYgMkkXc7Rc7p6pK_Z0TNnKKZY"
ZIP_FILE="data/data.zip"
OUT_DIR="data/"

gdown --id "$FILE_ID" -O "$ZIP_FILE"
unzip -o "$ZIP_FILE" -d "$OUT_DIR"
rm "$ZIP_FILE"

mv "${OUT_DIR}evaluation_ds_v2/"* "$OUT_DIR"
rmdir "${OUT_DIR}evaluation_ds_v2"
