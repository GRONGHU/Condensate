import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rms import rmsd
import matplotlib.pyplot as plt
import time
import pickle
import gzip
import argparse


def NW_ASA_MSD(temp_hydration,temp_MSD):
    """
    Calculate mean and standard deviation of MSD for each hydration level
    
    Args:
        temp_hydration: Array containing hydration numbers for each residue
        temp_MSD: Array containing MSD values for each residue
    
    Returns:
        range_len: Range of hydration levels
        temp_mean: Mean MSD for each hydration level
        temp_std: Standard deviation of MSD for each hydration level
    """
    temp_mean=[]
    temp_std=[]
    range_len=range(0,np.max(temp_hydration)+1)
    for le in range_len:
        temp_mean.append(np.mean(temp_MSD[np.where(temp_hydration==le)]))
        temp_std.append(np.std(temp_MSD[np.where(temp_hydration==le)]))
        print(le)
    return range_len,temp_mean,temp_std

    
def main():
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate local diffusivity of residue with given hydration degree')
    parser.add_argument('--xtc', type=str)
    args = parser.parse_args()

    args = parser.parse_args()
    XTC_list=args.xtc.split(",")

    # Display key parameters
    print("="*60)
    print("Starting analysis with parameters:")
    print(f"• XTC index: {args.xtc}")
    print("="*60)    
    
    # Maximum accessible surface area for each amino acid (in nm²)
    MaxASA={
        "ARG":2.74,
        "ALA":1.29,
        "ASN":1.95,
        "ASP":1.93,
        "CYS":1.67,
        "GLN":2.25,
        "GLU":2.23,
        "GLY":1.04,    
        "HIS":2.24,
        "ILE":1.97,
        "LEU":2.01,
        "LYS":2.36,
        "MET":2.24,
        "PHE":2.40,
        "PRO":1.59,
        "SER":1.55,
        "THR":1.72,
        "TRP":2.85,
        "TYR":2.63,
        "VAL":1.74, 
    }
    

    # Load trajectory data
    address="..//"
    GRO = address + "npt.gro"
    TPR = address + "npt.tpr"
    u = mda.Universe(TPR, GRO)  
    
    frame_length=20001
    # Select protein atoms excluding hydrogens
    protein = u.select_atoms("protein and not type H*")
    resids_initial = protein.resids  
    unique_resids = np.unique(resids_initial)
    # Create dictionary mapping residue IDs to atom indices
    residue_groups = {resid: np.where(resids_initial == resid)[0] for resid in unique_resids}
    resname=u.select_atoms("protein").residues.resnames
    MAXASA_res=[]
    for i in resname:
        MAXASA_res.append(MaxASA[i])


    # Load displacement data
    residue_data_temp=[]
    condensate_data_temp=[]
    for name in XTC_list:
        # Load residue displacement vectors
        with open(address+f"MSD//XTC{str(name)}//protein-residue-COM//residue_displacement_vector_{str(name)}.pkl", 'rb') as f:
            residue_data_temp.append(pickle.load(f))
            print(f"residue_displacement_vector_{str(name)}.pkl")
        # Load condensate displacement vectors
        with open(address+f"MSD//XTC{str(name)}//protein-residue-COM//condensate_displacement_vector_{str(name)}.pkl", 'rb') as f:
            condensate_data_temp.append(pickle.load(f))
            print(f"condensate_displacement_vector_{str(name)}.pkl")

    # Concatenate data from multiple trajectories
    residue_data=np.concatenate(residue_data_temp)
    condensate_data=np.concatenate(condensate_data_temp)
    
    # Calculate MSD without subtracting condensate motion
    MSD_nomove=np.sum(residue_data*residue_data,axis=2)
    
    # Calculate MSD with condensate motion subtracted
    MSD_move=np.sum((residue_data-condensate_data)*(residue_data-condensate_data),axis=2)

    # Load hydration data
    hydration_data_temp=[]
    for name in XTC_list:
        output=address+f"HydrationN//XTC{str(name)}//hydration_num_cut_5.0_XTC{str(name)}.pkl"
        hydration_data_ = np.zeros((frame_length, len(unique_resids)))
        file=open(output, 'rb') 
        for i in range(frame_length):
            data=pickle.load(file)        
            hydration_data_[i,np.array(list(data.keys()))-1]=np.array(list(data.values()))
            print(i)
        file.close()
        # Calculate hydration at midpoints between frames
        hydration_data_temp.append((hydration_data_[0:-1]+hydration_data_[1:])/2)   
        
    hydration_data_mid=np.concatenate(hydration_data_temp)
    
    # Normalize hydration number by maximum accessible surface area
    nw_data_mid=hydration_data_mid/MAXASA_res
    nw_data_mid_int = np.round(nw_data_mid).astype(int)
    
    # Calculate MSD statistics for each hydration level
    all_range_len_msa,all_mean_msa,all_std_msa=NW_ASA_MSD(nw_data_mid_int,MSD_nomove)
    all_range_len_msa_move,all_mean_msa_move,all_std_msa_move=NW_ASA_MSD(nw_data_mid_int,MSD_move)

    # Save results without condensate motion correction
    with open(f"hydration_diffusivity_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(all_range_len_msa, f) 
        pickle.dump(all_mean_msa, f) 
        pickle.dump(all_std_msa, f) 
        
    # Save results with condensate motion correction
    with open(f"hydration_diffusivity_move_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(all_range_len_msa_move, f) 
        pickle.dump(all_mean_msa_move, f) 
        pickle.dump(all_std_msa_move, f) 
    
if __name__ == "__main__":
    main()