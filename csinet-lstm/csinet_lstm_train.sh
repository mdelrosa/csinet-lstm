#!/bin/bash -l

#SBATCH
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdelrosa@ucdavis.edu
#SBATCH -p GPU-shared 
#SBATCH --job-name=out80ms
#SBATCH --time=2-0
#SBATCH --gres=gpu:4
#SBATCH --mem=64000 # memory required by job

module load cuda
conda activate conda activate /ocean/projects/ecs190004p/mdelrosa/.conda/tf114

python csinet_lstm_train.py -d 0 -g -1 -e outdoor -r 512 -l csinet80ms -ep 500 -sr 2

#wait
