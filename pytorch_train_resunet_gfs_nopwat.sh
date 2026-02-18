#!/bin/zsh
# ./pytorch_train_resunet_gfs_nopwat_gfs_nopwat.sh
# simple script to train over multiple lead times.
# the pytorch_train_resunet_gfs_nopwat is smart enough to 
# begin its training with the weights of the 
# training 3 h previous rather than starting from
# random.
# Tom Hamill, 14 Dec 2025

python pytorch_train_resunet_gfs_nopwat.py 2025120100 3
python pytorch_train_resunet_gfs_nopwat.py 2025120100 6
python pytorch_train_resunet_gfs_nopwat.py 2025120100 9
python pytorch_train_resunet_gfs_nopwat.py 2025120100 12
python pytorch_train_resunet_gfs_nopwat.py 2025120100 15
python pytorch_train_resunet_gfs_nopwat.py 2025120100 18
python pytorch_train_resunet_gfs_nopwat.py 2025120100 21
python pytorch_train_resunet_gfs_nopwat.py 2025120100 24

python pytorch_train_resunet_gfs_nopwat.py 2025120100 27
python pytorch_train_resunet_gfs_nopwat.py 2025120100 30
python pytorch_train_resunet_gfs_nopwat.py 2025120100 33
python pytorch_train_resunet_gfs_nopwat.py 2025120100 36
python pytorch_train_resunet_gfs_nopwat.py 2025120100 39
python pytorch_train_resunet_gfs_nopwat.py 2025120100 42
python pytorch_train_resunet_gfs_nopwat.py 2025120100 45
python pytorch_train_resunet_gfs_nopwat.py 2025120100 48

