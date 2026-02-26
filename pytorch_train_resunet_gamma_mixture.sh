#!/bin/bash
# ./pytorch_train_resunet_gamma_mixture.sh
# Simple script to train 2-component Gamma mixture model over multiple lead times.
# The pytorch_train_resunet_gamma_mixture.py is smart enough to
# begin its training with the weights of the training 3 h previous
# rather than starting from random.
# Tom Hamill with Claude Code assistance, Feb 2026

#sleep 4h
#cd ../resnet_data
#gunzip g.tar.gz
#tar xvf g.tar
#cd ../resnet

python pytorch_train_resunet_gamma_mixture.py 2025060100 3
python pytorch_train_resunet_gamma_mixture.py 2025060100 6
python pytorch_train_resunet_gamma_mixture.py 2025060100 9
python pytorch_train_resunet_gamma_mixture.py 2025060100 12
python pytorch_train_resunet_gamma_mixture.py 2025060100 15
python pytorch_train_resunet_gamma_mixture.py 2025060100 18
python pytorch_train_resunet_gamma_mixture.py 2025060100 21
python pytorch_train_resunet_gamma_mixture.py 2025060100 24

python pytorch_train_resunet_gamma_mixture.py 2025060100 27
python pytorch_train_resunet_gamma_mixture.py 2025060100 30
python pytorch_train_resunet_gamma_mixture.py 2025060100 33
python pytorch_train_resunet_gamma_mixture.py 2025060100 36
python pytorch_train_resunet_gamma_mixture.py 2025060100 39
python pytorch_train_resunet_gamma_mixture.py 2025060100 42
python pytorch_train_resunet_gamma_mixture.py 2025060100 45
python pytorch_train_resunet_gamma_mixture.py 2025060100 48

