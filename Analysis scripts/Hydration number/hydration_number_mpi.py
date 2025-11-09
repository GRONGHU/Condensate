import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import numpy as np
import os
import time
import argparse
import pickle
from multiprocessing import Pool
from collections import defaultdict

def compute_hydration_chunk(args):
    """Parallel computation of hydration numbers for trajectory chunk"""
    # Unpack arguments
    (tpr_path, xtc_path, chunk_start, chunk_end, 
     cutoff, tmp_dir, chunk_id, water_sel) = args

    u = mda.Universe(tpr_path, xtc_path)
    chunk_results = defaultdict(dict)
    
    # Predefine selection strings
    protein_sel = "protein and not type H*"
    water_sel = water_sel  # Get water selection from parameters
    
    # Process each frame in chunk
    for frame_idx in range(chunk_start, chunk_end):
        chunk_results[frame_idx] = {}
        try:
            u.trajectory[frame_idx]
        except:
            print(f"Error loading frame {frame_idx}")
            continue
        
        # Select protein atoms and get residue IDs
        protein = u.select_atoms(protein_sel)
        resids = protein.resids
        
        # Select water oxygen atoms
        water_oxygens = u.select_atoms(water_sel)
        if len(water_oxygens) == 0:
            print(f"Warning: No water found in frame {frame_idx}")
            continue
        
        # Compute contact pairs between protein and water atoms
        pairs = capped_distance(
            protein.positions, water_oxygens.positions,
            max_cutoff=cutoff,
            box=u.dimensions,
            return_distances=False
        )
        
        # Calculate hydration numbers for each residue
        hydration = {}
        if len(pairs) > 0:
            # Get contacting residues and water indices
            contact_resids = resids[pairs[:, 0]]
            water_indices = pairs[:, 1]
            
            # Deduplicate: count each residue-water pair once
            unique_pairs = np.unique(
                np.column_stack((contact_resids, water_indices)), 
                axis=0
            )
            
            # Count residue contacts
            unique_resids, counts = np.unique(
                unique_pairs[:, 0], 
                return_counts=True
            )
            hydration = dict(zip(unique_resids, counts))
        
        # Store results for current frame
        chunk_results[frame_idx] = hydration
        
        # Progress reporting
        if (frame_idx - chunk_start) % 50 == 0:
            print(f"Chunk {chunk_id} | Frame {frame_idx} | "
                  f"Residues: {len(hydration)}")

    # Save chunk results to temporary file
    temp_file = os.path.join(tmp_dir, f"hydration_{chunk_id:03d}.pkl")
    with open(temp_file, 'wb') as f:
        pickle.dump(dict(chunk_results), f)
    return temp_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parallel computation of protein residue hydration numbers')
    parser.add_argument('--xtc', type=int, required=True)
    parser.add_argument('--water', default='resname SOL and name OW', type=str)
    parser.add_argument('--cutoff', default=5.0,type=float)
    parser.add_argument('--processes', default=10, type=int)
    parser.add_argument('--chunk', default=200, type=int)
 
    args = parser.parse_args()

    # Set file paths
    address = "..//..//"    
    xtc = address + f"md{args.xtc}.xtc"  
    tpr = address + f"md{args.xtc}.tpr"

    # Display analysis parameters
    print("="*60)
    print("Initializing analysis parameters:")    
    print(f"• Water selection: {args.water}")
    print(f"• Chunk size: {args.chunk}")
    print(f"• XTC index: {args.xtc}")
    print(f"• Contact cutoff: {args.cutoff} A")
    print("="*60)
       
    # Load trajectory and get total frame count
    print("\nLoading trajectory...")
    u = mda.Universe(tpr, xtc)
    total_frames = len(u.trajectory)
    print(f"Loaded {total_frames} frames from:\n  {tpr}\n  {xtc}")

    # Split trajectory into chunks for parallel processing
    chunk_size = args.chunk
    chunks = [(i, min(i+chunk_size, total_frames)) 
             for i in range(0, total_frames, chunk_size)]
    print(f"\nSplitting into {len(chunks)} chunks (each ≤{chunk_size} frames):")
    for i, (s, e) in enumerate(chunks):
        print(f"  Chunk {i}: frames {s}-{e}")

    # Create temporary directory for intermediate results
    tmp_dir = "HYDRATION_TMP"
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Temporary files stored in: {tmp_dir}")

    # Prepare arguments for parallel processing
    task_args = [
        (tpr, xtc, chunk_start, chunk_end, args.cutoff, tmp_dir, chunk_id, args.water)
        for chunk_id, (chunk_start, chunk_end) in enumerate(chunks)
    ]

    # Execute parallel computation using process pool
    print("\nStarting parallel computation with {} processes...".format(args.processes))
    with Pool(processes=args.processes) as pool:
        temp_files = pool.map(compute_hydration_chunk, task_args)

    # Merge results from all temporary files
    final_data = {}
    for temp_file in temp_files:
        with open(temp_file, 'rb') as f:
            chunk_data = pickle.load(f)
            final_data.update(chunk_data)
        # Remove temporary file after loading data
        os.remove(temp_file)
    print("Merged all chunk results.")

    # Save final results to output file
    with open(f"hydration_num_cut_{args.cutoff}_XTC{args.xtc}.pkl", 'wb') as f:
        for ts in range(total_frames):
            pickle.dump(final_data[ts], f)
    print(f"Saved final results")

    # Cleanup temporary directory
    os.rmdir(tmp_dir)

    # Display final statistics
    print("\n" + "="*50)
    print(f"Total processed frames: {len(final_data)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time()-start_time:.2f} seconds")