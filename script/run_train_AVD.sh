#!/bin/bash

TRAJ=traj3
SCENE=Home_010_1
NAME=AVD_${SCENE}_${TRAJ}
DATA_DIR=/home/mmvc/mmvc-ny-nas/Yi_Shi/data/ActiveVisionDataset/${SCENE}/
# training epochs

EPOCH=15000
# batch size
BS=16
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=35
# logging interval
GPUID=2
LOG=8
LAT=16
CONV=1
MODE='z_single'
### training from scratch
python train_AVD.py -o $MODE -g $GPUID  -y $LAT -c $CONV --name $NAME -d $DATA_DIR -t ${TRAJ}.txt -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
