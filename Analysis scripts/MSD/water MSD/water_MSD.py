import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rms import rmsd
import matplotlib.pyplot as plt
import time
import pickle
import gzip
import argparse
import re


def extract_ranges(log_file_path):
    """
    Extract condensate and bulk water ranges from log file.
    
    Args:
        log_file_path (str): Path to the log file containing range information
        
    Returns:
        tuple: (condensate_ranges, bulk_ranges) where:
            - condensate_ranges: List of tuples (start, end) for condensate zones
            - bulk_ranges: List of tuples representing bulk water zones
    """
    condensate_ranges = []
    bulk_ranges = []
    condensate_pattern = r"condensate range from (\d+\.\d+) A to (\d+\.\d+) A"
    bulk_pattern = r"Bulk water range: (\d+\.\d+|\d+) to (\d+\.\d+|\d+) A and\s+(\d+\.\d+|\d+) to (\d+\.\d+|\d+) A"
    
    # Try different encodings to handle file encoding variations
    encodings = ['utf-8', 'latin1', 'cp1252', 'gb18030']
    
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding) as file:
                for line in file:
                    # Extract condensate range
                    condensate_match = re.search(condensate_pattern, line)
                    if condensate_match:
                        start = float(condensate_match.group(1))
                        end = float(condensate_match.group(2))
                        condensate_ranges.append((start, end))
                    
                    # Extract bulk water range
                    bulk_match = re.search(bulk_pattern, line)
                    if bulk_match:
                        bulk_start1 = float(bulk_match.group(1))
                        bulk_end1 = float(bulk_match.group(2))
                        bulk_start2 = float(bulk_match.group(3))
                        bulk_end2 = float(bulk_match.group(4))
                        bulk_ranges.append([(bulk_start1, bulk_end1), (bulk_start2, bulk_end2)])
            # Break loop if successful read
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    return condensate_ranges, bulk_ranges


def water_MSD_DT(residue, dt, tmax):
    """
    Calculate Mean Squared Displacement (MSD) for water residues over time intervals.
    
    Args:
        residue (numpy.ndarray): Displacement vectors for water residues
        dt (int): Time interval for MSD calculation
        tmax (int): Maximum time point to calculate
        
    Returns:
        numpy.ndarray: MSD values for each starting time point
    """
    MSD_DT_residues = []
    for i in range(tmax):
        # Calculate relative movement over time interval dt
        relative_move = np.sum(residue[i:i+dt], axis=0)
        ave = relative_move**2
        msd_ave = np.mean(ave)*3  # Multiply by 3 for 3D MSD
        MSD_DT_residues.append(msd_ave)
    MSD_DT_residues=np.array(MSD_DT_residues)
    return MSD_DT_residues

def main():
    """
    Main function to calculate water molecule Mean Squared Displacement (MSD).
    Separates calculation for bulk water (BW) and interfacial water (IW).
    """
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Calculate water MSD')
    parser.add_argument('--xtc', type=int, default=1)
    args = parser.parse_args()
    
    # Define file paths and load trajectory
    address ="..//..//..//"    
    XTC = address + f"md{args.xtc}_water_nojump.xtc"
    GRO = address + f"md{args.xtc}_water_nojump.gro"
    u = mda.Universe(GRO,XTC) 
    
    # Load condensate phase range information
    Cond_Address=address+f"Waterid//XTC{args.xtc}//"
    log_path=Cond_Address+f"water_loc_{args.xtc}.log"
    Condensed, Bulk=extract_ranges(log_path)    
    
    # Load water identification data (bulk vs interfacial waters)
    file=open(address+f"waterid_{args.xtc}.npy","rb")
    iwater=pickle.load(file)  # Interfacial water indices
    bwater=pickle.load(file)  # Bulk water indices
    file.close()
 
    # Display key parameters
    print("="*60)
    print("Starting MSD analysis with parameters:")
    print(f"• XTC index: {args.xtc}")
    print("="*60)    
    
    # Calculate displacement vectors for all water molecules
    residue_displacement_vector=[]
    start_frame=0
    u.trajectory[start_frame]
    residue_com_temp=u.select_atoms("resname SOL").center_of_mass(compound='residues')
    
    start = time.time()
    for ts in u.trajectory:
        # Calculate center of mass for water residues
        residue_com=u.select_atoms("resname SOL").center_of_mass(compound='residues')
    
        # Store displacement from initial position
        residue_displacement_vector.append(residue_com-residue_com_temp)
    
        residue_com_temp=residue_com
        end = time.time()
        print(f"Frame: {ts.frame} and calculate time: {end-start:.1f} seconds")
    
    residue_displacement_vector=np.array(residue_displacement_vector)

    # Separate displacement vectors for bulk and interfacial waters
    residue_displacement_vector_BW=residue_displacement_vector[:,bwater,:]
    residue_displacement_vector_IW=residue_displacement_vector[:,iwater,:]

    # Set up time parameters for MSD calculation
    t_min = 1
    t_max = len(u.trajectory)//2
    dt=0.05
    linear_points=np.array(range(1,t_max))
    Time=linear_points*dt

    # Calculate MSD for bulk water
    MSD_BW=[]
    for dt in linear_points:
        MSD_BW.append(water_MSD_DT(residue_displacement_vector_BW, dt, t_max))
        print(dt)
    MSD_BW_nm2=np.array(MSD_BW)/100  # Convert from Å² to nm²

    # Calculate MSD for interfacial water
    MSD_IW=[]
    for dt in linear_points:
        MSD_IW.append(water_MSD_DT(residue_displacement_vector_IW, dt, t_max))
        print(dt)
    MSD_IW_nm2=np.array(MSD_IW)/100  # Convert from Å² to nm²

    # Save results to pickle file
    with open(f"water_MSD_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(MSD_BW_nm2, f) 
        pickle.dump(MSD_IW_nm2, f) 
        pickle.dump(Time, f) 
    

if __name__ == "__main__":
    main()