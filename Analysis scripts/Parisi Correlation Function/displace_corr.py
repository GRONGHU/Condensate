import numpy as np
import MDAnalysis as mda
from scipy.spatial.distance import pdist
import argparse
import time
import pickle
import multiprocessing as mp
from tqdm import tqdm

def interval_cal(args):
    """Calculate correlation function for a single time interval"""
    start, dt, res_com, res_velocities = args
    cos_res = {}
    count_res = {}
    
    for i in range(start, start + dt - 1):
        # Calculate cosine similarity and Euclidean distance
        Y = 1 - pdist(res_velocities[i], 'cosine')
        X = np.round(pdist(res_com[i], 'euclidean')).astype(int)
        
        # Aggregate data for identical distances
        unique_dists, inverse_indices = np.unique(X, return_inverse=True)
        sums = np.bincount(inverse_indices, weights=Y)
        counts = np.bincount(inverse_indices)
        
        for d, s, c in zip(unique_dists, sums, counts):
            cos_res[d] = cos_res.get(d, 0) + s
            count_res[d] = count_res.get(d, 0) + c

    return cos_res, count_res

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Calculate Parisi Correlation Function')
    parser.add_argument('--xtc', type=int, default=1, help='Trajectory index')
    parser.add_argument('--dt', type=int, default=100, help='Time interval length')
    parser.add_argument('--procs', type=int, default=10, help='Number of processes')
    args = parser.parse_args()
    
    # Display key parameters
    print("="*60)
    print(f"Starting analysis with {args.procs} processes")
    print(f"• XTC index: {args.xtc}")
    print(f"• Time interval (dt): {args.dt}")
    print("="*60)
    
    # Load trajectory data
    address = "../../"
    XTC = address + f"md{args.xtc}_p_nojump.xtc"
    GRO = address + f"md{args.xtc}_p_whole.gro"
    u = mda.Universe(GRO, XTC)
    
    # Precompute center of mass coordinates
    print("Precomputing center of mass...")
    res_com = []
    conden_com = []
    for ts in u.trajectory:
        protein = u.select_atoms("protein")
        res_com.append(protein.residues.center_of_mass(compound="residues"))
        conden_com.append(protein.residues.center_of_mass())
        print(ts.frame)
    
    res_com = np.array(res_com)
    conden_com = np.array(conden_com)
    
    # Calculate relative velocities
    print("Calculating velocities...")
    res_velocities = res_com[1:] - res_com[:-1]
    com_velocities = conden_com[1:] - conden_com[:-1]
    res_velocities = res_velocities - com_velocities[:, np.newaxis, :]
    
    # Generate task list for parallel processing
    total_frames = len(u.trajectory)
    start_list = list(range(0, total_frames - args.dt, args.dt))
    print(f"Total intervals to process: {len(start_list)}")
    
    # Prepare multiprocessing parameters
    tasks = [(start, args.dt, res_com, res_velocities) for start in start_list]
    
    # Use multiprocessing pool for parallel computation
    print("Processing intervals...")
    cos_res_list = []
    count_res_list = []
    
    with mp.Pool(processes=args.procs) as pool:
        results = list(tqdm(pool.imap(interval_cal, tasks), total=len(tasks)))
    
    # Unpack results from all processes
    for cos_res, count_res in results:
        cos_res_list.append(cos_res)
        count_res_list.append(count_res)
    
    # Save results to pickle files
    print("Saving results...")
    with open(f"cos_res_list_xtc{args.xtc}.pkl", 'wb') as f:
        pickle.dump(cos_res_list, f)
    with open(f"count_res_list_xtc{args.xtc}.pkl", 'wb') as f:
        pickle.dump(count_res_list, f)
    
    print(f"Total execution time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    main()