#!/bin/bash
# this sets off slurm jobs that will download MRMS and 
# interpolate to the GRAF grid.  Bounding dates defined.
# submit with $ ./control_MRMS_download.sh
# monitor with $ squeue | grep thamill


sbatch control_MRMS_download.slurm 2024100100 2024100523
sbatch control_MRMS_download.slurm 2024100600 2024101023
sbatch control_MRMS_download.slurm 2024101100 2024101523
sbatch control_MRMS_download.slurm 2024101600 2024102023
sbatch control_MRMS_download.slurm 2024102100 2024102523
sbatch control_MRMS_download.slurm 2024102600 2024103123

sbatch control_MRMS_download.slurm 2024110100 2024110523
sbatch control_MRMS_download.slurm 2024110600 2024111023
sbatch control_MRMS_download.slurm 2024111100 2025011523
sbatch control_MRMS_download.slurm 2024111600 2024112023
sbatch control_MRMS_download.slurm 2024112100 2024112523
sbatch control_MRMS_download.slurm 2024112600 2024113023

sbatch control_MRMS_download.slurm 2024120100 2024120523
sbatch control_MRMS_download.slurm 2024120600 2024121023
sbatch control_MRMS_download.slurm 2024121100 2024121523
sbatch control_MRMS_download.slurm 2024121600 2024122023
sbatch control_MRMS_download.slurm 2024122100 2024122523
sbatch control_MRMS_download.slurm 2024122600 2024123123


