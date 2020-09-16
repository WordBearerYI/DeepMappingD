#!/bin/bash

CHECKPOINT_DIR='../results/AVD/AVD_Home_008_1_traj3/'
python eval_vis_AVD.py -c $CHECKPOINT_DIR
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR
