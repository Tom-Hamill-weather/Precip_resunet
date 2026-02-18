#!/bin/bash
# submit with $./control_save_patched_GRAF_MRMS_GFS.sh
# run from resnet/resnet directory
# uncomment to submit batch jobs on the Cray to extract
# patches of training data appropriate to the chosen date and lead time

sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 3
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 6
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 9
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 12
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 15
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 18
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 21
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 24
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 27
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 30
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 33
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 36
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 39
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 42
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 45
sleep 25m
sbatch control_save_patched_GRAF_MRMS_GFS.slurm 2025120100 48





