#!/bin/bash
# this sets off slurm jobs that will download MRMS and 
# interpolate to the GRAF grid.  Bounding dates defined.
# submit with $ ./control_MRMS_download.sh
# monitor with $ squeue | grep thamill

sbatch control_MRMS_download.slurm 2024110100 2024110523
sbatch control_MRMS_download.slurm 2024110600 2024111023
sbatch control_MRMS_download.slurm 2024111100 2024111523
sbatch control_MRMS_download.slurm 2024111600 2024112023
sbatch control_MRMS_download.slurm 2024112100 2024112523
sbatch control_MRMS_download.slurm 2024112600 2024113023

sbatch control_MRMS_download.slurm 2024120100 2024120523
sbatch control_MRMS_download.slurm 2024120600 2024121023
sbatch control_MRMS_download.slurm 2024121100 2024121523
sbatch control_MRMS_download.slurm 2024121600 2024122023
sbatch control_MRMS_download.slurm 2024122100 2024122523
sbatch control_MRMS_download.slurm 2024122600 2024123123

sbatch control_MRMS_download.slurm 2025010100 2025010523
sbatch control_MRMS_download.slurm 2025010600 2025011023
sbatch control_MRMS_download.slurm 2025011100 2025011523
sbatch control_MRMS_download.slurm 2025011600 2025012023
sbatch control_MRMS_download.slurm 2025012100 2025012523
sbatch control_MRMS_download.slurm 2025012600 2025013123

sbatch control_MRMS_download.slurm 2025020100 2025020523
sbatch control_MRMS_download.slurm 2025020600 2025021023
sbatch control_MRMS_download.slurm 2025021100 2025011523
sbatch control_MRMS_download.slurm 2025021600 2025022023
sbatch control_MRMS_download.slurm 2025022100 2025022523
sbatch control_MRMS_download.slurm 2025022600 2025022823

sbatch control_MRMS_download.slurm 2025030100 2025030523
sbatch control_MRMS_download.slurm 2025030600 2025031023
sbatch control_MRMS_download.slurm 2025031100 2025031523
sbatch control_MRMS_download.slurm 2025031600 2025032023
sbatch control_MRMS_download.slurm 2025032100 2025032523
sbatch control_MRMS_download.slurm 2025032600 2025033123


