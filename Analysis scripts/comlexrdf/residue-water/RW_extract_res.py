import MDAnalysis as mda
import numpy as np
import time
import pickle
import ComplexMixtures as cm
import re
import argparse

def extract_condensate_ranges(log_file_path):
    """
    Extract condensate range values from log file.
    
    Args:
        log_file_path (str): Path to the log file containing condensate ranges
        
    Returns:
        list: List of tuples containing (start, end) ranges in Angstroms
    """
    ranges = []
    pattern = r"condensate range from (\d+\.\d+) A to (\d+\.\d+) A"
    
    # Try different encoding formats to handle various file encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'gb18030']
    
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding) as file:
                for line in file:
                    match = re.search(pattern, line)
                    if match:
                        start = float(match.group(1))
                        end = float(match.group(2))
                        ranges.append((start, end))
            # Break loop if successful read
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    return ranges

def extrarestype(Incondense_num, pRDF_Address, Selection):
    """
    Extract residue-wise pRDF contributions for chains in condensate phase.
    
    Args:
        Incondense_num (list): List of chain indices in condensate phase
        pRDF_Address (str): Directory path containing pRDF JSON files
        Selection (list): List of residue selections for each chain
        
    Returns:
        list: List of pRDF contributions for each residue type in each chain
    """
    prdf = []
    for num in Incondense_num:
        prdf_chain = []
        # Load pRDF results for current chain
        results = cm.load(pRDF_Address + f"chain{num}.json")
        for my_selection in Selection[num]:
            # Calculate residue contributions and scale by bulk solvent density
            residue_contributions = cm.contributions(results, cm.SoluteGroup(my_selection))
            prdf_chain.append(np.array(residue_contributions) * results.density.solvent_bulk)
        prdf.append(np.array(prdf_chain))
        print(num)
    return prdf

def main():
    """
    Main function to calculate residue-wise water distributions in protein condensates.
    Processes MD trajectories and extracts pRDF data for residues in condensate phase.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate MDDF for protein chains')
    parser.add_argument('--xtc', required=True, help='Trajectory identifier number')
    parser.add_argument('--addition', type=int, default=10, help='Buffer distance from condensate boundaries')
    args = parser.parse_args()
    
    # File paths for trajectory and structure
    PDB = f"md{args.xtc}_whole_PRDF.pdb"
    XTC = f"md{args.xtc}_whole_PRDF.xtc"
    
    # Create MDAnalysis universe
    u = mda.Universe(PDB, XTC)
    
    # Count number of protein chains
    protein_num = len(u.select_atoms("protein").segments)
    
    # Import condensate phase boundaries from log file
    log_path = f"..//water_loc_{args.xtc}.log"
    downbound, upbound = extract_condensate_ranges(log_path)[0]
    
    # Constants and parameters
    N_Avogadro = 6.02214129e+23  # Avogadro's number
    add = args.addition  # Buffer distance from condensate boundaries
    
    # Get unique residue types and chain identifiers
    restype = np.sort(list(set(u.select_atoms("protein").resnames)))
    chainID = u.select_atoms("protein").segments.segids

    # Load heavy atom structure for ComplexMixtures
    heavy_pdb = "mddf_results//heavy.pdb"
    atoms = cm.readPDB(heavy_pdb)
    
    # Create residue selections for each chain and residue type
    Selection = []
    for i in chainID:
        my_selection = []
        for j in restype:
            # Select atoms for current chain and residue type
            my_selection.append(cm.select(atoms, f"chain {i} and protein and resname {j} "))
        print(i)
        Selection.append(my_selection)

    # Main processing loop over trajectory frames
    pRDF_ALL = []
    start = time.time()
    
    for ts in u.trajectory:
        # Calculate Z-coordinate of center of mass for each protein chain
        protein_zcom = u.select_atoms("protein").center_of_mass(compound="segments")[:, 2]
        
        # Identify chains within condensate phase (with buffer distance)
        Incondense = (protein_zcom < upbound - add) * (protein_zcom > downbound + add)
        Incondense_num = np.where(Incondense == True)[0]
        print(np.sum(Incondense))
        
        # Process current frame
        frame_number = ts.frame
        pRDF_Address = f"mddf_results//frame_{frame_number:06d}//"
        pRDF_frame = extrarestype(Incondense_num, pRDF_Address, Selection)
        pRDF_ALL += pRDF_frame
        
        # Print timing information
        end = time.time()
        print(f"frame: {ts.frame} and calculate time: {end - start:.1f} seconds")
        
    # Convert results to numpy array
    pRDF_ALL = np.array(pRDF_ALL)

    # Save results to pickle file
    file = open(f"residue_water_res_xtc{args.xtc}.pkl", "wb")
    pickle.dump(pRDF_ALL, file)
    file.close()

if __name__ == "__main__":
    main()