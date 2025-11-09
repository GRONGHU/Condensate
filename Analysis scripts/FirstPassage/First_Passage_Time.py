import MDAnalysis as mda
from MDAnalysis.analysis import lineardensity
import numpy as np
import pickle
import os
import argparse
from multiprocessing import Pool, RawArray
import time
import gzip

# Define global variables (shared across child processes)
global_shared_array = None
global_shape = None

def init_worker(shared_array, shape):
    """Initialize worker processes and set up shared memory data"""
    global global_shared_array, global_shape
    global_shared_array = shared_array
    global_shape = shape

def process_one_water(i):
    """Process a single water molecule sequence to calculate residence times"""
    # Reconstruct array from shared memory
    residence_data = np.frombuffer(global_shared_array, dtype=np.int32).reshape(global_shape)
    seq = residence_data[:, i]
    
    # Calculate two types of residence times
    liftime_out = find_dwell_times_out(seq)
    liftime_in = find_dwell_times_in(seq)
    
    return liftime_out, liftime_in

def find_dwell_times_out(seq):
    """Calculate residence time outside the channel"""
    events = []
    in_event = False
    start_index = -1

    for i in range(len(seq)):
        if seq[i] == 1:  # Enter event
            if not in_event:
                in_event = True
                start_index = i
        elif seq[i] == 2:  # Exit event
            if in_event:
                end_index = i
                length = end_index - start_index + 1
                events.append(length)
                in_event = False
        else:  # seq[i] == 0
            pass
    return events

def find_dwell_times_in(seq):
    """Calculate residence time inside the channel"""
    events = []
    in_event = False
    start_index = -1

    for i in range(len(seq)):
        if seq[i] == 2:  # Enter event
            if not in_event:
                in_event = True
                start_index = i
        elif seq[i] == 1:  # Exit event
            if in_event:
                end_index = i
                length = end_index - start_index + 1
                events.append(length)
                in_event = False
        else:  # seq[i] == 0
            pass
    return events

def main():
    parser = argparse.ArgumentParser(description='Calculate first passage time!')
    parser.add_argument('--xtc', type=int, default=1)
    parser.add_argument('--processes', type=int, default=20,
                        help='Number of parallel processes to use')
    args = parser.parse_args()
    
    start_total = time.time()
    address = f"..//..//Waterid//XTC{args.xtc}//"    
    print(f"Loading data for XTC{args.xtc}...")
    
    # Load data
    with gzip.open(address + f"waterid_{args.xtc}.pkl.gz", 'rb') as f:
        Residenceid = pickle.load(f) 
    
    # Convert data type to int32 to save memory
    Residenceid = Residenceid.astype(np.int32)
    total_frames, total_waters = Residenceid.shape
    print(f"Data loaded: {total_waters} waters, {total_frames} frames")
    
    # Create shared memory
    print("Creating shared memory...")
    flat_data = Residenceid.ravel()
    shared_array = RawArray('i', flat_data)
    del Residenceid  # Delete original data to free memory
    
    # Initialize process pool
    print(f"Starting pool with {args.processes} processes...")
    with Pool(
        processes=args.processes,
        initializer=init_worker,
        initargs=(shared_array, (total_frames, total_waters))
    ) as pool:
        
        residence_lifetime_out = []
        residence_lifetime_in = []
        count = 0
        
        # Process all water molecules in parallel
        print("Processing waters...")
        start_processing = time.time()
        
        # Use imap_unordered to get results
        results = pool.imap_unordered(process_one_water, range(total_waters), chunksize=100)
        
        for out_events, in_events in results:
            residence_lifetime_out.extend(out_events)
            residence_lifetime_in.extend(in_events)
            count += 1
            
            # Print progress every 100 water molecules
            if count % 100 == 0:
                elapsed = time.time() - start_processing
                print(f"Processed {count}/{total_waters} waters "
                      f"({count/total_waters*100:.1f}%), "
                      f"elapsed: {elapsed:.1f}s")

    # Save results
    print("Saving results...")
    with open(f"residence_lifetime_out_xtc{args.xtc}.pkl", 'wb') as f:
        pickle.dump(residence_lifetime_out, f)
        
    with open(f"residence_lifetime_in_xtc{args.xtc}.pkl", 'wb') as f:
        pickle.dump(residence_lifetime_in, f)   
        
    total_time = time.time() - start_total
    print(f"Total calculation time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()