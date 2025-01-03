#!/bin/bash
#PBS -N stylespeech           
#PBS -P ey69           
#PBS -q gpuvolta               
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=70GB
#PBS -l walltime=48:00:00      
#PBS -j oe                     
#PBS -M haowei.lou@unsw.edu.au
#PBS -m abe                    
#PBS -l wd                    

# Change to working directory
cd $PBS_O_WORKDIR

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "Error: Virtual environment 'venv' not found. Exiting."
  exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the Python script
echo "Starting Python script..."
python3 train_ss.py

# Deactivate virtual environment
deactivate