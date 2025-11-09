import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rms import rmsd
import matplotlib.pyplot as plt
import time
import pickle
import gzip
import argparse

    
def main():
    start_time = time.time()
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate COM displacement of protein and residue')
    parser.add_argument('--xtc', type=int, default=1)
    args = parser.parse_args()
    
    # Define file paths
    address ="..//..//..//"    
    XTC = address + f"md{args.xtc}.xtc"
    TPR = address + f"md{args.xtc}.tpr"
    u = mda.Universe(TPR,XTC)   

    # Display key parameters
    print("="*60)
    print("Starting MSD analysis with parameters:")
    print(f"• XTC index: {args.xtc}")
    print("="*60)    

    # Extract system dimensions and protein information
    x_length, y_length, z_length = u.dimensions[0:3]
    protein_indices = u.select_atoms("protein").indices
    protein_length = 2254
    box = np.array([x_length, y_length, z_length])
    protein_mass = u.select_atoms("protein").masses
    
    # Create chain atom index mapping
    n_chains = int(len(protein_indices) / protein_length)
    chain_atom_indices = np.array([u.select_atoms("protein").atoms[i*protein_length:(i+1)*protein_length].indices for i in range(n_chains)])
    chain_masses = []
    for id in chain_atom_indices:
        chain_masses.append(u.select_atoms("protein").atoms[id].masses)
    
    # Create residue atom index mapping
    residues = u.select_atoms("protein").residues
    n_residues = len(residues)
    residue_atom_indices = []
    residue_masses = []
    for res in residues:
        atom_indices = res.atoms.indices
        # Find positions of these atoms in protein_indices
        pos_in_protein = np.where(np.isin(protein_indices, atom_indices))[0]
        residue_atom_indices.append(pos_in_protein)
        residue_masses.append(res.atoms.masses)

    # Initialize lists to store displacement vectors
    displacement_residue_list = []
    displacement_chain_list = []
    displacement_condensate_list = []

    # Store initial positions
    temp_pos = u.atoms[protein_indices].positions
    start = time.time()

    # Process each frame in trajectory (skip first frame)
    for ts in u.trajectory[1:]:
        current_pos = u.atoms[protein_indices].positions
        # Calculate raw displacement vectors
        displacement = current_pos - temp_pos
        # Apply periodic boundary condition correction
        displacement -= np.round(displacement / box) * box
        
        # Calculate center of mass displacement for each residue
        residue_displacements = np.zeros((n_residues, 3))
        for i in range(n_residues):
            # Get atom indices for current residue
            atom_indices = residue_atom_indices[i]
            # Calculate mass-weighted average displacement
            masses = residue_masses[i]
            total_mass = np.sum(masses)
            weighted_disp = np.sum(displacement[atom_indices] * masses[:, np.newaxis], axis=0)
            residue_displacements[i] = weighted_disp / total_mass
    
        # Calculate center of mass displacement for each chain
        chain_displacements = np.zeros((n_chains, 3))
        for i in range(n_chains):
            # Get atom indices for current chain
            atom_indices = chain_atom_indices[i]
            # Calculate mass-weighted average displacement
            masses = chain_masses[i]
            total_mass = np.sum(masses)
            weighted_disp = np.sum(displacement[atom_indices] * masses[:, np.newaxis], axis=0)
            chain_displacements[i] = weighted_disp / total_mass
    
        # Calculate center of mass displacement for entire condensate
        condensate_displacements = np.zeros((1, 3))
        total_mass = np.sum(protein_mass)
        weighted_disp = np.sum(displacement * protein_mass[:, np.newaxis], axis=0)
        condensate_displacements[0] = weighted_disp / total_mass

        # Store displacement vectors for current frame
        displacement_residue_list.append(residue_displacements)
        displacement_chain_list.append(chain_displacements)
        displacement_condensate_list.append(condensate_displacements)
    
        # Update previous frame coordinates
        temp_pos = current_pos.copy()
        end = time.time()

        # Print progress information
        print(f"Frame: {ts.frame} and calculate time: {end-start:.1f} seconds")

    # Convert lists to numpy arrays
    displacement_residue_list = np.array(displacement_residue_list)
    displacement_chain_list = np.array(displacement_chain_list)
    displacement_condensate_list = np.array(displacement_condensate_list)

    # Save displacement vectors to pickle files
    with open(f"residue_displacement_vector_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(displacement_residue_list, f) 
    
    with open(f"chain_displacement_vector_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(displacement_chain_list, f) 

    with open(f"condensate_displacement_vector_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(displacement_condensate_list, f) 
  
if __name__ == "__main__":
    main()