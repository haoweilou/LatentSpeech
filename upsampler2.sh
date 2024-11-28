#!/bin/bash
#PBS -N upsampler2           
#PBS -P ey69           
#PBS -q gpuvolta               
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=40GB
#PBS -l walltime=24:00:00      
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
python3 train_upsampler2.py

# Deactivate virtual environment
deactivate