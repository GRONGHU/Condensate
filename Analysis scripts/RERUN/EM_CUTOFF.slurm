#!/bin/bash
#SBATCH --job-name=EM_CUTOFF  
#SBATCH --output=EM_CUTOFF_output_%j.log      
#SBATCH --error=EM_CUTOFF_error_%j.log       
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=10        
#SBATCH --partition=fatcpu96         


python -u EDR_convert.py --EDR rerun_CUTOFF_3.edr --group groups.dat --enmat EM_CUTOFF_3.npy






