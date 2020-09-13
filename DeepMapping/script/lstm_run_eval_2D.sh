#!/bin/bash

SCENE_NAME=$1
CHECKPOINT_DIR="../results/2D/lstm_${SCENE_NAME}/"
python eval_2D.py -c $CHECKPOINT_DIR
