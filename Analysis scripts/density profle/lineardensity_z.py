import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity
import numpy as np
import pickle
import os
import argparse
from multiprocessing import Pool
import time
import gzip

def compute_density_chunk(args):
    """Calculate density distribution for a single chunk of frames"""
    tpr_file, xtc_file, chunk_start, chunk_end, selections, bins, output_dir, chunk_id = args
    start_time = time.time()
    
    u = mda.Universe(tpr_file, xtc_file)
    chunk_mass_data = []
    hist_data=[]
    print(f"Processing chunk {chunk_id} [frames {chunk_start}-{chunk_end}]...")
    
    for frame_idx in range(chunk_start, chunk_end):
        # Calculate mass density for all atoms
        frame_mass_density=[]
        u.trajectory[frame_idx]
        atoms = u.atoms 
        ld = lineardensity.LinearDensity(atoms, binsize=bins)
        ld.run(start=frame_idx, stop=frame_idx + 1)
        hist_data.append(ld.results.z.hist_bin_edges)
        frame_mass_density.append(ld.results.z.mass_density)
              
        # Calculate mass density for each selection
        for sel in selections:
            u.trajectory[frame_idx]
            atoms = u.select_atoms(sel) 
            ld = lineardensity.LinearDensity(atoms, binsize=bins)
            ld.run(start=frame_idx, stop=frame_idx + 1)
            frame_mass_density.append(ld.results.z.mass_density)
        chunk_mass_data.append(np.array(frame_mass_density))
        
        

    # Save temporary results for this chunk
    temp_file = os.path.join(output_dir, f"temp_{chunk_id:04d}.pkl")
    with open(temp_file, 'wb') as f:
        pickle.dump({
            'mass_data': chunk_mass_data,
            'hist': hist_data
        }, f)
    
    # Calculate and report processing time
    processing_time = time.time() - start_time
    print(f"Chunk {chunk_id} completed (frames {chunk_start}-{chunk_end}) "
          f"in {processing_time:.1f}s. Saved to {temp_file}")
    
    return temp_file

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Calculate z-direction density profile per frame')
    parser.add_argument('--csize', type=int, default=400)
    parser.add_argument('--processes', type=int, default=10)
    parser.add_argument('--xtc', type=int, default=1)
    parser.add_argument('--bins', type=float, default=1)

    args = parser.parse_args()

    # File path configuration
    address ="..//..//"    
    tpr = address + f"md{args.xtc}.tpr"
    xtc = address + f"md{args.xtc}.xtc"

    # Display key parameters
    print("="*60)
    print("Starting density analysis with parameters:")
    print(f"• Chunk size: {args.csize}")
    print(f"• XTC index: {args.xtc}")
    print(f"• Binsize: {args.bins} A")
    print("="*60)

    # Atom selections for different components
    selections = [
        'resname SOL',
        'protein',
        'resname NA',
        'resname CL',
    ]
    
    print("\nSelection criteria:")
    for i, sel in enumerate(selections):
        print(f"  {i+1}. {sel}")

    # Get total number of frames
    print("\nLoading trajectory...")
    u = mda.Universe(tpr, xtc)
    total_frames = len(u.trajectory)
    print(f"Loaded {total_frames} frames from:\n  {tpr}\n  {xtc}")

    # Split trajectory into chunks for parallel processing
    chunk_size = args.csize
    chunks = [(i, min(i+chunk_size, total_frames)) 
             for i in range(0, total_frames, chunk_size)]
    print(f"\nSplitting into {len(chunks)} chunks (each ≤{chunk_size} frames):")
    for i, (s, e) in enumerate(chunks):
        print(f"  Chunk {i}: frames {s}-{e}")

    # Create output directory for temporary files
    tmp_dir =  f"density_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Temporary files stored in: {tmp_dir}")

    # Parallel computation using multiprocessing
    print(f"\nStarting parallel processing with {args.processes} workers...")
    with Pool(args.processes) as pool:
        results = pool.map(compute_density_chunk, [
            (tpr, xtc, start, end, selections, args.bins, tmp_dir, i)
            for i, (start, end) in enumerate(chunks)
        ])

    # Merge results from all chunks
    print(f"\nMerging {len(results)} temporary files...")
    all_hist_edges = []
    all_densities = []
    
    for i, temp_file in enumerate(results):
        print(f"  Processing chunk {i+1}/{len(results)}: {temp_file}")
        with open(temp_file, 'rb') as f:
            data = pickle.load(f)
            all_hist_edges.append(data["hist"])
            all_densities.append(data["mass_data"])
        os.remove(temp_file)
    # Save final results using compressed pickle format
    output_file = f"Density_xtc{args.xtc}.pkl.gz"
    print(f"\nSaving final results to {output_file}")
    with gzip.open(output_file, 'wb') as f:
        pickle.dump({
            'hist_edges': all_hist_edges,
            'densities': all_densities,
        }, f) 
    os.rmdir(tmp_dir)

    print(f"Successfully saved {len(all_hist_edges)} frames data")

    # Display summary statistics
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Analysis completed in {total_time:.1f} seconds")
    print(f"Total frames processed: {total_frames}")
    print(f"Output file: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()