import MDAnalysis as mda
import numpy as np
import time
import pickle
import argparse
import multiprocessing
from functools import partial

def residue_MSD_DT(residue_nomove, residue_move, dt,tmin,tmax):
    """
    Calculate Mean Squared Displacement (MSD) for residues over a given time interval.
    
    Args:
        residue_nomove: Displacement vectors of residues without condensate motion correction
        residue_move: Displacement vectors of residues with condensate motion correction  
        dt: Time interval for MSD calculation
        tmin: Starting time frame index
        tmax: Ending time frame index
    
    Returns:
        Array containing MSD values for both corrected and uncorrected residue motions
    """
    MSD_DT_residues_move = []
    MSD_DT_residues_nomove = []
    
    # Calculate MSD for each time window within the specified range
    for i in range(tmin,tmax):
        # Calculate center of mass displacement for the time interval
        residue_com_move = np.sum(residue_move[i:i+dt], axis=0)
        residue_com_nomove = np.sum(residue_nomove[i:i+dt], axis=0)
                    
        # Square the displacements to get squared distances
        ave_move= residue_com_move**2
        ave_nomove = residue_com_nomove**2
        
        # Sum over all dimensions and append to results
        MSD_DT_residues_move.append(np.sum(ave_move,axis=1))
        MSD_DT_residues_nomove.append(np.sum(ave_nomove,axis=1))
    
    # Calculate mean MSD across all time windows
    MSD_DT_residues_move=np.mean(np.array(MSD_DT_residues_move),axis=0)
    MSD_DT_residues_nomove=np.mean(np.array(MSD_DT_residues_nomove),axis=0)
    
    return np.array([MSD_DT_residues_move,MSD_DT_residues_nomove])

def calculate_MSD_for_dt(dt, residue_nomove, residue_move,tmin, tmax):
    """
    Wrapper function for MSD calculation with error handling and timing.
    
    Args:
        dt: Time interval for MSD calculation
        residue_nomove: Displacement vectors without condensate correction
        residue_move: Displacement vectors with condensate correction
        tmin: Starting time frame index
        tmax: Ending time frame index
    
    Returns:
        Tuple containing dt value and corresponding MSD results
    """
    try:
        start_time = time.time()
        result = residue_MSD_DT(residue_nomove, residue_move, dt,tmin, tmax)
        elapsed = time.time() - start_time
        
        print(f"dt={dt} calculated in {elapsed:.1f} seconds")
        return (dt, result)
    except Exception as e:
        print(f"Error calculating dt={dt}: {str(e)}")
        return (dt, None)

def main():
    """
    Main function to coordinate MSD calculation workflow.
    Handles argument parsing, data loading, parallel processing, and result saving.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate residue MSD relative to condensate COM')
    parser.add_argument('--xtc', type=str, help='Comma-separated list of trajectory file indices')
    parser.add_argument('--points', type=int, default=1000, help='Number of time points to sample')
    parser.add_argument('--processes', type=int, default=None, help='Number of parallel processes to use')
    parser.add_argument('--tmin', type=int, default=1, help='Minimum time frame index')
    parser.add_argument('--tmax', type=int, default=10001, help='Maximum time frame index')
    
    args = parser.parse_args()
    XTC_list=args.xtc.split(",")
    
    # Load displacement data from pickle files
    residue_data_temp=[]
    condensate_data_temp=[]
    for name in XTC_list:
        # Load residue displacement data
        with open(f"residue_displacement_vector_{str(name)}.pkl", 'rb') as f:
            residue_data_temp.append(pickle.load(f))
   
        # Load condensate displacement data  
        with open(f"condensate_displacement_vector_{str(name)}.pkl", 'rb') as f:
            condensate_data_temp.append(pickle.load(f))
    
    # Combine data from multiple trajectories
    residue_nomove=np.concatenate(residue_data_temp)
    condensate_data=np.concatenate(condensate_data_temp)
    
    # Calculate corrected residue displacements by subtracting condensate motion
    residue_move=residue_nomove-condensate_data
    
    # Set calculation parameters
    t_min = args.tmin
    t_max = args.tmax
    num_points = min(args.points, t_max - t_min)  # Ensure we don't exceed available points
    
    # Generate logarithmically spaced time points for MSD calculation
    log_points = np.logspace(np.log10(t_min), np.log10(t_max), num_points, dtype=int)
    linear_points = np.unique(log_points)  # Remove duplicates and sort
    
    # Initialize parallel processing
    start_total = time.time()
    num_processes = args.processes or multiprocessing.cpu_count()
    
    print(f"Starting calculation with {num_processes} processes for {len(linear_points)} dt values...")
    
    # Create partial function with fixed parameters for parallel processing
    worker = partial(calculate_MSD_for_dt, 
                    residue_nomove=residue_nomove, 
                    residue_move=residue_move, 
                    tmin=t_min,
                    tmax=t_max)
    
    # Execute calculations in parallel using process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker, linear_points)
    
    # Sort results by dt value
    results.sort(key=lambda x: x[0])
    TMSD_DT_residues = [result for _, result in results]
    
    # Save results to pickle file
    with open(f"residue_MSD_{args.xtc}.pkl", 'wb') as f:
        pickle.dump({
            'dt_values': linear_points,
            'msd_data': TMSD_DT_residues
        }, f)
    
    # Print performance statistics
    total_time = time.time() - start_total
    print(f"Total calculation time: {total_time:.2f} seconds")
    print(f"Processed {len(linear_points)} dt values with {num_processes} processes")

if __name__ == "__main__":
    # Support for frozen applications (Windows compatibility)
    multiprocessing.freeze_support()
    main()