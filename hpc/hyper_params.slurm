#!/usr/bin/bash
#SBATCH --job-name="tfspeed"
#SBATCH --nodelist=node012
#SBATCH --gres=gpu:1 
#SBATCH --time=00:04:00
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./run.sh $HOME/DLVENV ../scripts/hyper_params.py 
