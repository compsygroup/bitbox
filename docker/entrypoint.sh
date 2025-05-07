#!/bin/bash

# Check if the model file exists
if [[ ! -f /app/3DI/models/raw/01_MorphableModel.mat || ! -f /app/3DI_lite/data/raw/01_MorphableModel.mat ]]; then
    echo ""
    echo "!!! ERROR !!!"
    echo ""
    echo "Model file '01_MorphableModel.mat' not found. Please download it from"
    echo "    https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads"
    echo "and follow instructions at"
    echo "    https://github.com/compsygroup/bitbox"
    echo "to install it."
    echo ""
fi

# Proceed with the application
exec "$@"
