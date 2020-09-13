#!/bin/bash

NAME=$1
# path to dataset
DATA_DIR=../data/2D/${NAME}
echo ${DATA_DIR}

# training epochs
EPOCH=10000
# batch size
BS=128
# loss function
LOSS=bce_ch
# number of points sampled from line-of-sight
N=19
# logging interval
LOG=10
LAT_SIZE=12

### training from scratch
python lstm_train_2D.py --lat_size ${LAT_SIZE} --name $NAME -d $DATA_DIR -e $EPOCH -b $BS -l $LOSS -n $N --log_interval $LOG
