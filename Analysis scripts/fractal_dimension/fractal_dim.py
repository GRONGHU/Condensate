import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib import distances
from MDAnalysis.analysis.rms import rmsd
import matplotlib.pyplot as plt
import time
import pickle
import argparse

    
def main():
    start_time = time.time()
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Calculate fractal dimensions ')
    parser.add_argument('--xtc', type=int, default=1)
    parser.add_argument('--max_box', type=int, default=128)
    args = parser.parse_args()
    
    # Define file paths
    address ="..//..//"    
    XTC = address + f"md{args.xtc}.xtc"
    TPR = address + f"md{args.xtc}.tpr"
    u = mda.Universe(TPR,XTC)   

    # Display key parameters
    print("="*60)
    print("Starting  fractal dimensions analysis with parameters:")
    print(f"• XTC index: {args.xtc}")
    print("="*60)    

    # Define grid sizes for fractal dimension calculation
    max_box = args.max_box
    edge_num = [2,4,6,8,10,12,16,18,20,24,28,30,32,36,40,42,48,52,54,56,60,64]
    box_length_log = np.log10(max_box/np.array(edge_num))
    
    dimension_count_list=[]
    # Iterate through each frame in the trajectory
    for ts in u.trajectory:
        start_time = time.time()
        
        # Calculate center of mass of protein condensate
        condensate_com = u.select_atoms("protein").center_of_mass()
    
        # Define bounding box around center of mass
        upx,upy,upz=condensate_com+max_box/2
        downx,downy,downz=condensate_com-max_box/2
    
        # Select atoms within the bounding box
        box = u.select_atoms(f'protein and prop x > {downx} and prop x < {upx} and prop y > {downy} and prop y < {upy} and prop z > {downz} and prop z < {upz}')
        atom_pos = box.positions
    
        # Extract coordinates
        x_pos = atom_pos[:, 0]
        y_pos = atom_pos[:, 1]
        z_pos = atom_pos[:, 2]
        dimension_count=[]
        
        # Calculate fractal dimension for different grid sizes
        for grid_size in edge_num:
            grid_length = max_box / grid_size
    
            # Calculate grid indices for each atom
            x_indices = np.floor((x_pos - downx) / grid_length).astype(int)
            y_indices = np.floor((y_pos - downy) / grid_length).astype(int)
            z_indices = np.floor((z_pos - downz) / grid_length).astype(int)
    
            # Ensure indices are within valid range
            x_indices = np.clip(x_indices, 0, grid_size - 1)
            y_indices = np.clip(y_indices, 0, grid_size - 1)
            z_indices = np.clip(z_indices, 0, grid_size - 1)
    
            # Create 3D grid and mark occupied cells
            grid_filled = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
            grid_filled[x_indices, y_indices, z_indices] = True
    
            # Count number of occupied grid cells
            filled_count = np.sum(grid_filled)
            print(f"分割{grid_size} 的box被填充的 grid 数量: {filled_count}")
    
            dimension_count.append(np.log10(filled_count))
        dimension_count_list.append(dimension_count)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"帧{ts.frame} 循环执行时间: {elapsed_time:.2f} 秒")
    dimension_count_list=np.array(dimension_count_list)

    # Save results to pickle file
    with open(f"_fractal_dimensions_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(dimension_count_list, f) 
        pickle.dump(box_length_log,f) 
        pickle.dump(edge_num,f) 
        pickle.dump(max_box,f) 

if __name__ == "__main__":
    main()