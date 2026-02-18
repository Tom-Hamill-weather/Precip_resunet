#!/bin/bash
# this sets off slurm jobs that will download MRMS and 
# interpolate to the GRAF grid.  Bounding dates defined.
# submit with $ ./control_MRMS_download.sh
# monitor with $ squeue | grep thamill

sbatch control_MRMS_download.slurm 2025121200 2025121623
sbatch control_MRMS_download.slurm 2025121700 2025122123
sbatch control_MRMS_download.slurm 2025122200 2025122623
sbatch control_MRMS_download.slurm 2025122700 2025123123



