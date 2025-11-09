import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity
import numpy as np
import pickle
import os
import argparse
from multiprocessing import Pool
import time
import gzip

def trans2bool(Bulk, Cond, water_resid, water_num):
    """Convert water residue IDs to boolean classification flags.
    
    Args:
        Bulk: List of bulk water residue IDs
        Cond: List of condensate water residue IDs  
        water_resid: Array of all water residue IDs
        water_num: Total number of water molecules
    
    Returns:
        water_tf: Array with classification flags (0=unclassified, 1=condensate, 2=bulk, 3=both)
    """
    water_tf = np.zeros(water_num, dtype=np.uint8)
    condense_set = set(Cond)
    bulk_set = set(Bulk)
    
    for idx, resid in enumerate(water_resid):
        if resid in condense_set:
            water_tf[idx] += 1
        if resid in bulk_set:
            water_tf[idx] += 2

    return water_tf

def main():
    """Main function to analyze water molecule localization in molecular dynamics trajectories."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Calculate z-direction density profile per frame')
    parser.add_argument('--xtc', type=int, default=1)
    parser.add_argument('--cutoff', type=float, default=0.300)
    parser.add_argument('--buffer', type=float, default=20)

    args = parser.parse_args()
    
    address = "..//..//"    
    tpr = address + f"md{args.xtc}.tpr"
    xtc = address + f"md{args.xtc}.xtc"
    u = mda.Universe(tpr, xtc)   
    data_file = address + f"Density profile//XTC{args.xtc}//Density_xtc{args.xtc}.pkl.gz"
    with gzip.open(data_file, 'rb') as f:
        data = pickle.load(f) 
            
    cutoff = args.cutoff
    buffer = args.buffer
    selections = [
        'resname SOL',
        'protein', 
        'resname NA',
        'resname CL',
    ]
    
    # Display key parameters
    print("=" * 60)
    print("Starting water id analysis with parameters:")
    print(f"• Interface water buffering distance: {args.buffer}")
    print(f"• XTC index: {args.xtc}") 
    print(f"• CUTOFF: {args.cutoff} A")
    print("=" * 60)    
    
    # Initialize lists to store boundary positions
    conden_leftbound = []
    conden_rightbound = []
    water_leftbound = [] 
    water_rightbound = []
    
    # Calculate phase boundaries from density profiles
    for i in range(len(data["densities"])):
        for j in range(len(data["densities"][i])):
            # Find condensate boundaries using density cutoff
            Pindex = np.where((data["densities"][i][j][2] > cutoff) == True)
            lower, upper = np.min(Pindex), np.max(Pindex)     
            conden_leftbound.append(data["hist_edges"][i][j][lower])
            conden_rightbound.append(data["hist_edges"][i][j][upper])
    
            # Find water boundaries (non-zero density regions)
            Windex = np.where((data["densities"][i][j][2] == 0) == False)
            lower, upper = np.min(Windex), np.max(Windex)     
            water_leftbound.append(data["hist_edges"][i][j][lower])
            water_rightbound.append(data["hist_edges"][i][j][upper])
       
    # Calculate average phase boundaries
    Conden_phase = [np.mean(conden_leftbound), np.mean(conden_rightbound)]
    Bulk_phase = [np.mean(water_leftbound) - buffer, np.mean(water_rightbound) + buffer]
    
    print(f"condensate range from {Conden_phase[0]:.2f} A to {Conden_phase[1]:.2f} A")
    print(f"Bulk water range: 0 to {Bulk_phase[0]:.2f} A and {Bulk_phase[1]:.2f} to {u.dimensions[2]:.2f} A")
    
    # Get water molecule information
    water_num = len(u.select_atoms("resname SOL").residues.resids)
    water_resid = u.select_atoms("resname SOL").residues.resids
    residenceid = []
    
    start = time.time()
    print("=" * 60)    
    
    # Process each frame in trajectory
    for ts in u.trajectory:
        # Select water molecules in bulk and condensate regions based on z-coordinates
        water_in_bulk = u.select_atoms(f"(name OW and prop z > {Bulk_phase[1]}) or (name OW and prop z < {Bulk_phase[0]})").resids
        water_in_condense = u.select_atoms(f"(name OW and prop z < {Conden_phase[1]}) and (name OW and prop z > {Conden_phase[0]})").resids
        
        # Classify water molecules
        effect_id = trans2bool(water_in_bulk, water_in_condense, water_resid, water_num)
        
        # Check for classification conflicts (should not exceed 3)
        if np.max(effect_id) > 2:
            print(f"overlaping at frame {ts.frame}")
            
        residenceid.append(effect_id)
        end = time.time()
        print(f"frame: {ts.frame} and calculate time: {end - start:.1f} seconds")
        
    residenceid = np.array(residenceid)
    
    # Save results to compressed pickle file
    with gzip.open(f"waterid_{args.xtc}.pkl.gz", 'wb') as f:
        pickle.dump(residenceid, f) 
    
if __name__ == "__main__":
    main()