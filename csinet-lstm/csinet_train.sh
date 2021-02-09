#!/bin/bash -l

#SBATCH
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdelrosa@ucdavis.edu
#SBATCH -p GPU-AI 
#SBATCH --job-name=cr512_out
#SBATCH --time=0-12
#SBATCH --gres=gpu:volta16:1
#SBATCH --mem=64000 # memory required by job

source $HOME/.bashrc 

python csinet_train.py -d 0 -g 0 -e indoor -r 512 -l csinet -ep 1000

#wait
