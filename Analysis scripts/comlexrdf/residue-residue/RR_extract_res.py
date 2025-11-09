import MDAnalysis as mda
import numpy as np
import time
import pickle
import ComplexMixtures as cm
import re
import argparse

def extract_condensate_ranges(log_file_path):
    """
    Extract condensate ranges from log file
    
    Args:
        log_file_path (str): Path to the log file containing condensate range information
    
    Returns:
        list: List of tuples containing (start, end) ranges in Angstroms
    """
    ranges = []
    pattern = r"condensate range from (\d+\.\d+) A to (\d+\.\d+) A"
    
    # Try different encoding formats to handle file reading issues
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
            # Break loop if successful reading
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    
    return ranges

def extrarestype(Incondense_num, pRDF_Address, Selection):
    """
    Extract residue-specific pRDF contributions for chains within condensate
    
    Args:
        Incondense_num (list): List of chain indices within condensate
        pRDF_Address (str): Directory path containing pRDF JSON files
        Selection (list): List of residue selections for each chain
    
    Returns:
        list: List containing residue-specific MDDF contributions for each chain
    """
    prdf = []
    for num in Incondense_num:
        prdf_chain = []
        results = cm.load(pRDF_Address + f"chain{num}.json")
        for my_selection in Selection[num]:
            # Calculate residue contributions to MDDF
            residue_contributions = cm.contributions(results, cm.SoluteGroup(my_selection))
            prdf_chain.append(np.array(residue_contributions) * results.density.solvent_bulk)
        prdf.append(np.array(prdf_chain))
        print(num)
    return prdf

def main():
    """
    Main function to calculate residue-specific MDDF contributions in condensate
    """
    parser = argparse.ArgumentParser(description='Calculate MDDF for protein chains')
    parser.add_argument('--xtc', required=True, help='XTC trajectory number')
    parser.add_argument('--addition', type=int, default=10, help='Buffer zone addition around condensate')
    args = parser.parse_args()
    
    # File paths for PDB and XTC files
    PDB = f"md{args.xtc}_whole_PRDF.pdb"
    XTC = f"md{args.xtc}_whole_PRDF.xtc"
    u = mda.Universe(PDB, XTC)
    protein_num = len(u.select_atoms("protein").segments)
    
    # Import condensate range information
    log_path = f"..//water_loc_{args.xtc}.log"
    downbound, upbound = extract_condensate_ranges(log_path)[0]
    N_Avogadro = 6.02214129e+23  # Avogadro's number
    add = args.addition
    
    # Get unique residue types and chain identifiers
    restype = np.sort(list(set(u.select_atoms("protein").resnames)))
    chainID = u.select_atoms("protein").segments.segids

    # Load heavy atom structure for residue selection
    heavy_pdb = "mddf_results//heavy.pdb"
    atoms = cm.readPDB(heavy_pdb)
    
    # Create residue selections for each chain and residue type
    Selection = []
    for i in chainID:
        my_selection = []
        for j in restype:
            # Select atoms for specific chain and residue type
            my_selection.append(cm.select(atoms, f"chain {i} and protein and resname {j} "))
        print(i)
        Selection.append(my_selection)

    # Process residue-specific pRDF data
    pRDF_ALL = []
    start = time.time()
    for ts in u.trajectory:
        # Identify chains within condensate region
        protein_zcom = u.select_atoms("protein").center_of_mass(compound="segments")[:, 2]
        Incondense = (protein_zcom < upbound - add) * (protein_zcom > downbound + add)
        Incondense_num = np.where(Incondense == True)[0]
        print(np.sum(Incondense))
        
        # Load and process residue-specific pRDF data for current frame
        frame_number = ts.frame
        pRDF_Address = f"mddf_results//frame_{frame_number:06d}//"
        pRDF_frame = extrarestype(Incondense_num, pRDF_Address, Selection)
        pRDF_ALL += pRDF_frame
        
        end = time.time()
        print(f"frame: {ts.frame} and calculate time: {end-start:.1f} seconds")
        
    pRDF_ALL = np.array(pRDF_ALL)

    # Save results to pickle file
    file = open(f"residue_residue_res_xtc{args.xtc}.pkl", "wb")
    pickle.dump(pRDF_ALL, file)
    file.close()
        
if __name__ == "__main__":
    main()