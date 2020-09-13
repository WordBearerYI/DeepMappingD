#!/bin/bash

CHECKPOINT_DIR="../results/2D/$1/"
python eval_2D.py -c $CHECKPOINT_DIR
