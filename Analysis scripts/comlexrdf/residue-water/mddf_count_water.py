import argparse
import os
import time
import ComplexMixtures as cm
import MDAnalysis as mda

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate MDDF for protein chains')
    
    # Required arguments
    parser.add_argument('--pdb', required=True, help='Input PDB file path')
    parser.add_argument('--xtc', required=True, help='Input trajectory file path')
    
    # Output arguments
    parser.add_argument('-o', '--output', default='mddf_results', 
                      help='Output directory (default: mddf_results)')
    
    # Calculation parameters
    parser.add_argument('--bulk_range', type=lambda s: tuple(map(float, s.split(','))),
                      default=(40.0, 50.0), help='Bulk range as "min,max" (default: 40.0,50.0)')
    parser.add_argument('--n_samples', type=int, default=80,
                      help='Number of random samples (default: 80)')
    parser.add_argument('--binstep', type=float, default=0.02,
                      help='Bin step size (default: 0.02)')
   
    
    # Trajectory processing
    parser.add_argument('--start', type=int, default=0,
                      help='Starting frame index (default: 0)')
    parser.add_argument('--stop', type=int, 
                      help='End frame index (default: full trajectory)')
    parser.add_argument('--step', type=int, default=1,
                      help='Frame step size (default: 1)')
    
    return parser.parse_args()

def compute_mddf(args):
    """Main function to execute MDDF calculation"""
    # Create main output directory
    os.makedirs(args.output, exist_ok=True)

    # Directory for storing trajectory frames (keep original path)
    frame_dir = os.path.join(args.output, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Initialize Universe
    u = mda.Universe(args.pdb, args.xtc)
    
    # Preprocess structure file by removing hydrogen atoms
    heavy_pdb = os.path.join(args.output, "heavy.pdb")
    u.select_atoms("not type H* DUMMY").write(heavy_pdb)
    
    # Read processed structure
    atoms = cm.readPDB(heavy_pdb)
    
    # Get protein chain information
    segids = u.select_atoms("not type H* and protein").segments.segids
    
    # Set up solvent selection
    water = cm.select(atoms, "name OW")
    solvent = cm.AtomSelection(water, natomspermol=1)
    
    # Configure calculation options
    options = cm.Options(
        bulk_range=args.bulk_range,
        n_random_samples=args.n_samples,
        binstep=args.binstep
    )
    
    # Get trajectory range
    start = args.start
    stop = args.stop if args.stop else len(u.trajectory)
    step = args.step
    
    total_time = time.time()
    
    # Iterate through specified frame range
    for ts in range(start, stop, step):
        frame_time = time.time()
        u.trajectory[ts]
        
        # Create results directory for current frame
        frame_subdir = f"frame_{ts:06d}"
        frame_output_dir = os.path.join(args.output, frame_subdir)
        os.makedirs(frame_output_dir, exist_ok=True)
        
        # Write current frame trajectory file (keep original path)
        frame_path = os.path.join(frame_dir, f"frame_{ts:06d}.xtc")
        with mda.Writer(frame_path) as w:
            w.write(u.select_atoms("not type H* DUMMY"))
        
        # Iterate through each protein chain
        for chain_idx, segid in enumerate(segids):
            # Select current chain
            protein = cm.select(atoms, f"chain {segid} and protein")
            print(len(protein))
            solute = cm.AtomSelection(protein, nmols=1)
            
            # Execute calculation
            results = cm.mddf(frame_path, solute, solvent, options)
            
            # Save results to frame directory
            output_file = os.path.join(
                frame_output_dir, 
                f"chain{chain_idx}.json"  # Filename no longer includes frame number
            )
            
            cm.save(results, output_file)
            
            # Output progress
            elapsed = time.time() - frame_time
            print(f"[Frame {ts:06d}] Chain {chain_idx} | Saved to {frame_subdir} | Time: {elapsed:.2f}s", flush=True)
    
    print(f"Total execution time: {time.time() - total_time:.2f} seconds", flush=True)

def main():
    args = parse_arguments()
    compute_mddf(args)

if __name__ == "__main__":
    main()