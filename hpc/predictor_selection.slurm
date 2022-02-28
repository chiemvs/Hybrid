#!/usr/bin/bash
#SBATCH --job-name="31q5_extra"
#SBATCH --nodelist=node242
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20gb
#SBATCH --time=100:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./run.sh $HOME/DLVENV ../scripts/predictor_selection.py False 31 0.5 # do_climdev, [timeagg, quantile] or [ndaythreshold] 
