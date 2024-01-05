#!/bin/bash
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=1000G
#SBATCH --time=24:00:00
#SBATCH --job-name=iq_julia_job
#SBATCH --output=logs/iq_julia_job-%J.log

# Load required modules
module load SciPy-bundle
module load mosek
module load Julia

# Change to your working directory
cd /mnt/personal/cignalud/QD_LDS_readout/Julia

# Run the Julia script
julia Julia_1D_trajectorywise.jl