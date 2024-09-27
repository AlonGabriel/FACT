#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --time {time}
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/slurm-%j.log
{directives}

# Modules
module load cuda-12.1
module load python/3.10

# Environment
source ~/environments/FACT/bin/activate

python train.py with {config}