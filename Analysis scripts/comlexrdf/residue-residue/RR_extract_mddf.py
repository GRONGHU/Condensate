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

def extractchain(Incondense_num, pRDF_Address):
    """
    Extract pRDF data for chains within condensate
    
    Args:
        Incondense_num (list): List of chain indices within condensate
        pRDF_Address (str): Directory path containing pRDF JSON files
    
    Returns:
        list: List containing distance and MDDF data for each chain
    """
    prdf = []
    for num in Incondense_num:
        prdf_chain = []
        results = cm.load(pRDF_Address + f"chain{num}.json")
        prdf_chain.append(np.array(results.d))  # Distance array
        prdf_chain.append(np.array(results.mddf) * results.density.solvent_bulk)  # MDDF values
        prdf.append(prdf_chain)
    return prdf

def main():
    """
    Main function to calculate residue-residue MDDF for protein chains in condensate
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
    
    # Lists to store density calculations
    massdensity_list = []
    numdensity_list = []
    
    massdensity_noH_list = []
    numdensity_noH_list = []
    
    # Calculate density for each frame in trajectory
    for ts in u.trajectory:
        # Calculate mass and number density for protein atoms in condensate region
        protein_atommass = np.sum(u.select_atoms(f"protein and prop z > {downbound-add} and prop z < {upbound+add}").masses)
        protein_atomnum = len(u.select_atoms(f"protein and prop z > {downbound-add} and prop z < {upbound+add}"))
        volume = (u.dimensions[0] * u.dimensions[1] * (upbound - downbound + 2 * add))
        massdensity = protein_atommass / volume / N_Avogadro * 1000 * (10 ** 24)
        numdensity = protein_atomnum / volume
        
        # Calculate density excluding hydrogen atoms
        protein_atommass_noH = np.sum(u.select_atoms(f"protein and not type H* and prop z > {downbound-add} and prop z < {upbound+add}").masses)
        protein_atomnum_noH = len(u.select_atoms(f"protein and not type H* and prop z > {downbound-add} and prop z < {upbound+add}"))
        volume_noH = (u.dimensions[0] * u.dimensions[1] * (upbound - downbound + 2 * add))
        massdensity_noH = protein_atommass_noH / volume_noH / N_Avogadro * 1000 * (10 ** 24)
        numdensity_noH = protein_atomnum_noH / volume_noH
        
        # Store calculated densities
        massdensity_list.append(massdensity)
        numdensity_list.append(numdensity)
        
        massdensity_noH_list.append(massdensity_noH)
        numdensity_noH_list.append(numdensity_noH)
        
        print(ts.frame)

    # Process pRDF data for chains within condensate
    pRDF_ALL = []
    start = time.time()
    for ts in u.trajectory:
        # Identify chains within condensate region
        protein_zcom = u.select_atoms("protein").center_of_mass(compound="segments")[:, 2]
        Incondense = (protein_zcom < upbound - add) * (protein_zcom > downbound + add)
        Incondense_num = np.where(Incondense == True)[0]
        print(np.sum(Incondense))
        
        # Load and process pRDF data for current frame
        frame_number = ts.frame
        pRDF_Address = f"mddf_results//frame_{frame_number:06d}//"
        pRDF_frame = extractchain(Incondense_num, pRDF_Address)
        pRDF_ALL += pRDF_frame
        
        end = time.time()
        print(f"frame: {ts.frame} and calculate time: {end-start:.1f} seconds")
        
    pRDF_ALL = np.array(pRDF_ALL)

    # Calculate average pRDF and radius values
    pRDF = np.mean(pRDF_ALL[:, 1, :], axis=0)
    RADIUS = np.mean(pRDF_ALL[:, 0, :], axis=0)
    
    # Save results to pickle file
    file = open(f"residue_residue_xtc{args.xtc}.pkl", "wb")
    pickle.dump(pRDF_ALL, file)
    pickle.dump(np.mean(massdensity_noH_list) / np.mean(numdensity_noH_list), file)  # Average mass per atom
    file.close()
        
if __name__ == "__main__":
    main()