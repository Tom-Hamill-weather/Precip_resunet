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

python pytorch_train_resunet_gamma_mixture.py 2025090100 3
python pytorch_train_resunet_gamma_mixture.py 2025090100 6
python pytorch_train_resunet_gamma_mixture.py 2025090100 9
python pytorch_train_resunet_gamma_mixture.py 2025090100 12
python pytorch_train_resunet_gamma_mixture.py 2025090100 15
python pytorch_train_resunet_gamma_mixture.py 2025090100 18
python pytorch_train_resunet_gamma_mixture.py 2025090100 21
python pytorch_train_resunet_gamma_mixture.py 2025090100 24

python pytorch_train_resunet_gamma_mixture.py 2025090100 27
python pytorch_train_resunet_gamma_mixture.py 2025090100 30
python pytorch_train_resunet_gamma_mixture.py 2025090100 33
python pytorch_train_resunet_gamma_mixture.py 2025090100 36
python pytorch_train_resunet_gamma_mixture.py 2025090100 39
python pytorch_train_resunet_gamma_mixture.py 2025090100 42
python pytorch_train_resunet_gamma_mixture.py 2025090100 45
python pytorch_train_resunet_gamma_mixture.py 2025090100 48

