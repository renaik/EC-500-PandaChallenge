#!/bin/bash

# --------- TRAIN DATASET ----------
# Tile WSIs to generate train patches.
python 'src/graph_construction/tile_wsi.py' \
    -s 512 \
    -e 0 \
    -j 32 \
    -B 50 \
    -M 1 \
    -o 'data/train_patches' \
    'data/train_images/*.tiff' \

# Build graphs from generated train patches.
python 'src/graph_construction/build_graphs.py' \
    --weights 'src/graph_construction/feature_extractor.pth' \
    --dataset 'data/train_patches/*/*/' \
    --output 'data/train_graphs'\


# --------- VAL DATASET ----------
# Tile WSIs to generate val patches.
python 'src/graph_construction/tile_wsi.py' \
    -s 512 \
    -e 0 \
    -j 32 \
    -B 50 \
    -M 1 \
    -o 'data/val_patches' \
    'data/val_images/*.tiff' \

# Build graphs from generated val patches.
python 'src/graph_construction/build_graphs.py' \
    --weights 'src/graph_construction/feature_extractor.pth' \
    --dataset 'data/val_patches/*/*/' \
    --output 'data/val_graphs'\


# --------- TEST DATASET ----------
# Tile WSIs to generate test patches.
python 'src/graph_construction/tile_wsi.py' \
    -s 512 \
    -e 0 \
    -j 32 \
    -B 50 \
    -M 1 \
    -o 'data/test_patches' \
    'data/test_images/*.tiff' \

# Build graphs from generated test patches.
python 'src/graph_construction/build_graphs.py' \
    --weights 'src/graph_construction/feature_extractor.pth' \
    --dataset 'data/test_patches/*/*/' \
    --output 'data/test_graphs'\