#!/bin/bash
# sbatch resnet_train.slurm cyyyymmddhh clead

#SBATCH --job-name=resnet_train    # create a short name for your job
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --cpus-per-task=2          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G           # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00            # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin          # send email when job begins
#SBATCH --mail-type=end            # send email when job ends
#SBATCH --mail-user=tom.hamill@weather.com

echo cyyyymmddhh=$1
echo clead=$2

source /storage/home/thamill/.bashrc
source /storage/home/thamill/anaconda3/bin/activate deep2
export HDF5_USE_FILE_LOCKING=FALSE
which python3
python3 pytorch_train_resunet.py $1 $2
