#!/bin/bash
#SBATCH --job-name=rerun_cutoff_3 
#SBATCH --output=rerun_cutoff_3_output_%j.log        
#SBATCH --error=rerun_cutoff_3_error_%j.log          
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=128         
#SBATCH --cpus-per-task=1            
#SBATCH --partition=cpu128          


source /share/home/huguorong/gromacs/bin/GMXRC
gmx_mpi grompp -f rerun_CUTOFF.mdp -c ../../md3.gro -p ../../condensate.top -n index_cutoff.ndx -o rerun_CUTOFF_3.tpr

/usr/mpi/gcc/openmpi-4.1.7rc1/bin/mpirun --bind-to core  -np 128 gmx_mpi mdrun -ntomp 1 -s rerun_CUTOFF_3.tpr -rerun ../../md3.xtc -deffnm rerun_CUTOFF_3





