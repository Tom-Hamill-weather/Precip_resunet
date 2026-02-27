#!/bin/bash
# Batch convert all pickle files to NetCDF4 with compression
# This will save ~73% disk space (519 GB → ~140 GB)

# Usage: ./batch_convert_to_netcdf.sh [directory]
# Default directory: /data2/resnet_data/trainings

DIR="${1:-/data2/resnet_data/trainings}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Converting pickle files in: $DIR"
echo "This will save ~73% disk space"
echo ""

# Count files
TOTAL=$(find "$DIR" -name "*.cPick" | wc -l)
echo "Found $TOTAL pickle files to convert"
echo ""

# Convert each file
CONVERTED=0
FAILED=0

for pickle_file in "$DIR"/*.cPick; do
    if [ ! -f "$pickle_file" ]; then
        continue
    fi

    # Generate NetCDF filename
    base=$(basename "$pickle_file" .cPick)
    netcdf_file="$DIR/${base}.nc"

    # Skip if already converted
    if [ -f "$netcdf_file" ]; then
        echo "SKIP: $base (already exists)"
        continue
    fi

    echo "Converting: $base"

    # Convert
    if python "$SCRIPT_DIR/convert_pickle_to_netcdf.py" "$pickle_file" "$netcdf_file" > /tmp/convert_${base}.log 2>&1; then
        # Show compression stats
        tail -4 /tmp/convert_${base}.log | head -3
        CONVERTED=$((CONVERTED + 1))

        # Delete pickle immediately after successful conversion (disk full!)
        rm "$pickle_file"
        echo "  ✓ Deleted original pickle file"
    else
        echo "  ERROR: Conversion failed (see /tmp/convert_${base}.log)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "============================================"
echo "Conversion complete!"
echo "  Converted: $CONVERTED files"
echo "  Failed: $FAILED files"
echo ""
echo "Next steps:"
echo "  1. Verify a few NetCDF files load correctly in training"
echo "  2. Uncomment 'rm' line in this script"
echo "  3. Re-run to delete old pickle files"
echo "  4. Expected space savings: ~379 GB"
