#!/bin/bash

#SBATCH --nodes=1
# #SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --job-name=diff_train
#SBATCH --output=logs/log.diff
#SBATCH -p gpu_p100
#SBATCH -A ccsd

# load modules and conda
source ~/.bashrc
conda activate /lustre/or-scratch/cades-ccsd/z6f/conda_envs/kloop

module load gcc
module load openmpi

cd /lustre/or-scratch/cades-ccsd/z6f/KineticsLoop/Training_Code
mkdir results/$SLURM_JOB_ID
DATAPATH=/lustre/or-scratch/cades-ccsd/z6f/lombardo-ft/data/lombardo_clean_v3.csv

# 'MW,human CLsys (mL/min/kg),fraction unbound in plasma (fu),Pfizer logD,pKa_acidic,pKa_basic,fub_transformed,fub_spec,fub_spec_transformed,log_clearance'

mpirun -n 1 python fine_tuning_regression.py $DATAPATH --epochs 20 --normalize --output_directory results/$SLURM_JOB_ID --score_columns 'MW,Pfizer logD,pKa_acidic,pKa_basic,fub_spec_transformed,log_clearance'

# mpirun -n 1 python fine_tuning_regression.py ./data/lombardo_all_v1.csv --epochs 50 --batch_size 8 --normalize --output_directory results/$SLURM_JOB_ID --score_columns 'log_clearance'

# mpirun -n 1 python fine_tuning_regression.py ./data/activity_data.csv --epochs 80 --batch_size 16 --log_transform --output_directory results/$SLURM_JOB_ID --score_columns 'f_avg_IC50'

# mpirun -n 1 python fine_tuning_regression.py ./data/lombardo_clean_v4.csv --epochs 10 --normalize --output_directory results/$SLURM_JOB_ID --score_columns 'MW,log_clint,Pfizer logD,pKa_acidic,pKa_basic,fub_spec_transformed'

#srun -n 2 --mpi=pmi2 python fine_tuning_regression.py data/Lombardo_PK_dataset_forZach.csv
#srun -n 2 python fine_tuning_regression.py data/Lombardo_PK_dataset_forZach.csv
