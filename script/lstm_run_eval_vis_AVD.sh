INDEX=$1
CHECKPOINT_DIR="../results/AVD/AVD_Home_010_1_traj${INDEX}/"
python eval_vis_AVD.py -c $CHECKPOINT_DIR
#vglrun-wrapper python eval_vis_AVD.py -c $CHECKPOINT_DIR
