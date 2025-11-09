import numpy as np
import MDAnalysis as mda
from numpy.linalg import norm
import time
from MDAnalysis.analysis import distances
import pickle
from MDAnalysis.lib.pkdtree import PeriodicKDTree
from scipy.spatial.distance import pdist
from MDAnalysis.lib.distances import capped_distance
from collections import defaultdict
from scipy.fft import fftn, fftfreq
import argparse
import multiprocessing as mp
from scipy.signal import savgol_filter
import os

def compute_energy_flux_corrected(u, v, w, lx, ly, lz, n_bins=50):
    """
    Corrected energy flux calculation following spherical shell integration definition
    
    Parameters:
        u, v, w: 3D arrays (nx, ny, nz)
            Velocity components in x, y, z directions
        lx, ly, lz: float
            Domain size in x, y, z directions
        n_bins: int
            Number of wave number bins
            
    Returns:
        k_bins: 1D array
            Center values of wave number intervals
        Pi_k: 1D array
            Energy flux Π(k) for each wave number interval
        T_k: 1D array  
            Nonlinear transfer term T(k) for each wave number interval
    """
    # Get grid dimensions
    nx, ny, nz = u.shape
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    
    # ===================================================================
    # 1. Prepare wave number grid
    # ===================================================================
    kx = 2 * np.pi * fftfreq(nx, d=dx)
    ky = 2 * np.pi * fftfreq(ny, d=dy) 
    kz = 2 * np.pi * fftfreq(nz, d=dz)
    
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
    
    # ===================================================================
    # 2. Compute Fourier transform of velocity field
    # ===================================================================
    u_hat = fftn(u)
    v_hat = fftn(v)
    w_hat = fftn(w)
    
    # ===================================================================
    # 3. Calculate nonlinear term (u·∇)u directly in Fourier space
    # ===================================================================
    # Use convolution theorem to compute nonlinear term
    # Note: Assumes incompressible flow, so pressure term is zero
    uu_hat = fftn(u * u)
    uv_hat = fftn(u * v)
    uw_hat = fftn(u * w)
    vv_hat = fftn(v * v)
    vw_hat = fftn(v * w)
    ww_hat = fftn(w * w)
    
    # Calculate Fourier transform of (u·∇)u
    conv_x_hat = 1j * (Kx * uu_hat + Ky * uv_hat + Kz * uw_hat)
    conv_y_hat = 1j * (Kx * uv_hat + Ky * vv_hat + Kz * vw_hat)
    conv_z_hat = 1j * (Kx * uw_hat + Ky * vw_hat + Kz * ww_hat)
    
    # ===================================================================
    # 4. Compute dot product: û*(k)·ℱ[(u·∇)u](k)
    # ===================================================================
    dot_product = (
        np.conj(u_hat) * conv_x_hat +
        np.conj(v_hat) * conv_y_hat +
        np.conj(w_hat) * conv_z_hat
    )
    dot_product_real = np.real(dot_product)
    
    # ===================================================================
    # 5. Set wave number intervals - spherical shell integration
    # ===================================================================
    # Use logarithmically spaced wave number intervals to better capture turbulence scales
    k_min = 2 * np.pi / max(lx, ly, lz)  # Minimum wave number
    k_max = np.pi / min(dx, dy, dz)      # Maximum wave number (Nyquist wave number)
    
    k_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_bins = 0.5 * (k_edges[1:] + k_edges[:-1])
    
    # Wave number space volume element - correction factor
    dk_vol = (2 * np.pi)**3 / (lx * ly * lz)
    
    # ===================================================================
    # 6. Calculate nonlinear transfer term T(k) with spherical shell integration - follows Lin equation
    # ===================================================================
    T_k = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Select all points where k_edges[i] ≤ |k| < k_edges[i+1] (spherical shell integration)
        mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i+1])
        num_modes = np.sum(mask)  # Count number of modes in this spherical shell
        
        if num_modes > 0:
            # Sum and average all modes in the spherical shell
            integral_sum = np.sum(dot_product_real[mask])
            T_k[i] = integral_sum * dk_vol
        else:
            T_k[i] = 0.0
    
    # ===================================================================
    # 7. Calculate energy flux Π(k) - obtained by accumulating T(k)
    # ===================================================================
    # According to turbulence theory: Π(k) = -∫_0^k T(k') dk' = ∫_k^∞ T(k') dk'
    Pi_k = -np.cumsum(T_k)
    
    # Optional: Smooth results to reduce noise
    if len(Pi_k) > 5:
        Pi_k = savgol_filter(Pi_k, 5, 2)
    
    return k_bins, Pi_k, T_k

def gaussian_kernel(distances, sigma):
    """Calculate Gaussian weight kernel function"""
    return np.exp(-distances**2 / (2 * sigma**2))

def group_neighbors_dict(pairs):
    """Group neighbors with same center atom using dictionary"""
    from collections import defaultdict
    grouped = defaultdict(list)
    for center, neighbor in pairs:
        grouped[center].append(neighbor)
    return dict(grouped)

def process_frame_batch(args):
    """
    Parallel function to process a batch of frames
    
    Parameters:
        args: tuple containing (xtc_idx, frame_batch, grid_size, grid_count, sigma, cutoff_sigma)
        
    Returns:
        point_velocities_batch: list of velocity arrays for each frame
        frame_batch: list of processed frame indices
    """
    xtc_idx, frame_batch, grid_size, grid_count, sigma, cutoff_sigma = args
    print(f"Process {os.getpid()} starting to process {len(frame_batch)} frames")
    
    address = "../..//"
    XTC = address + f"md{xtc_idx}.xtc"
    TPR = address + f"md{xtc_idx}.tpr"
    
    # Each process creates its own Universe
    u_xtc = mda.Universe(TPR, XTC)
    time_interval = u_xtc.trajectory.dt
    
    point_velocities_batch = []
    start=time.time()

    for t in frame_batch:
        print(f"Process {os.getpid()}: processing frame {t}")
        
        u_xtc.trajectory[t]
        box = u_xtc.dimensions
        protein = u_xtc.select_atoms("protein")
        P_Z = protein.center_of_mass()[2]
        P_COM = np.array([box[0]/2, box[1]/2, P_Z])
        all_atoms = u_xtc.atoms
        positions = all_atoms.positions
        
        u_xtc.trajectory[t+1]
        all_atoms = u_xtc.atoms
        positions_move = all_atoms.positions    
        displacements = positions_move - positions
        displacements -= np.round(displacements / box[0:3]) * box[0:3]
        velocities = displacements / time_interval
        
        masses = all_atoms.masses
        
        # Calculate grid coverage range
        half_box = grid_size * grid_count / 2
        grid_min = P_COM - half_box
        grid_max = P_COM + half_box
        
        # Create grid point coordinates
        x = np.linspace(grid_min[0], grid_max[0], grid_count, endpoint=False)
        y = np.linspace(grid_min[1], grid_max[1], grid_count, endpoint=False)
        z = np.linspace(grid_min[2], grid_max[2], grid_count, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        grid_centers = grid_points + grid_size/2.0
        
        # Calculate cutoff radius
        cutoff = cutoff_sigma * sigma
        radius = (cutoff_sigma + 1) * sigma
        
        # Create PeriodicKDTree object
        pkdtree = PeriodicKDTree(box=box)
        pkdtree.set_coords(positions, radius)
        neighbors = pkdtree.search_tree(grid_centers, cutoff)
        group_neibour = group_neighbors_dict(neighbors)
        
        point_velocities = []
        for i in range(len(group_neibour)):
            mass_re = np.array(masses[group_neibour[i]]).reshape(-1, 1)  
            r_d = np.linalg.norm(positions[group_neibour[i]] - grid_centers[i], axis=1)
            Weight = gaussian_kernel(r_d, sigma) * mass_re.T
            Velocity_weight = np.sum(velocities[group_neibour[i]] * Weight.T, axis=0) / np.sum(Weight)
            point_velocities.append(Velocity_weight)
            
        point_velocities = np.array(point_velocities)
        point_velocities_batch.append(point_velocities)
        end=time.time()
        print('Running time: %s Seconds' % (end - start))    
    # Unit conversion, nm/ps
    point_velocities_batch = [pv / 10 for pv in point_velocities_batch]
    
    print(f"Process {os.getpid()} completed processing")
    return point_velocities_batch, frame_batch

def main():
    """Main function to calculate energy flux from molecular dynamics trajectories"""
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Calculate z-direction density profile per frame')
    parser.add_argument('--xtc', type=int, default=1)
    parser.add_argument('--dt', type=int, default=10)
    parser.add_argument('--grid_size', type=int, default=2)
    parser.add_argument('--grid_count', type=int, default=50)
    parser.add_argument('--sigma', type=int, default=3)
    parser.add_argument('--cutoff_sigma', type=int, default=3)
    parser.add_argument('--n_processes', type=int, default=10, help='Number of processes to use')
    parser.add_argument('--frames_per_process', type=int, default=200, help='Frames per process')
    
    args = parser.parse_args()
    
    # Display key parameters
    print("="*60)
    print("Starting MSD analysis with parameters:")
    print(f"• XTC index: {args.xtc}")
    print(f"• Frame step: {args.dt}")
    print(f"• Grid size: {args.grid_size} A")
    print(f"• Grid count: {args.grid_count}")
    print(f"• Gaussian kernel width sigma: {args.sigma}")    
    print(f"• Cut-off distance: {args.cutoff_sigma} sigma")    
    print(f"• Number of processes: {args.n_processes}")
    print(f"• Frames per process: {args.frames_per_process}")
    print("="*60)     
    
    # Load trajectory to get total number of frames
    address = "../..//"
    XTC = address + f"md{args.xtc}.xtc"
    TPR = address + f"md{args.xtc}.tpr"
    u_xtc = mda.Universe(TPR, XTC)
    
    total_frames = len(u_xtc.trajectory) - 1
    Frames = np.array(range(0, total_frames, args.dt))
    
    print(f"Total frames: {total_frames}, Sampled frames: {len(Frames)}")
    
    # Split frames into batches for processing
    frame_batches = []
    for i in range(0, len(Frames), args.frames_per_process):
        batch = Frames[i:i + args.frames_per_process]
        if len(batch) > 0:
            frame_batches.append(batch)
    
    # Limit number of processes
    n_processes = min(args.n_processes, len(frame_batches))
    print(f"Actual number of processes used: {n_processes}, Number of batches: {len(frame_batches)}")
    
    # Prepare multiprocessing arguments
    mp_args = [(args.xtc, batch, args.grid_size, args.grid_count, args.sigma, args.cutoff_sigma) 
               for batch in frame_batches[:n_processes]]
    
    # Use multiprocessing for computation
    print("Starting multiprocessing calculation...")
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_frame_batch, mp_args)
    
    # Collect all results
    point_velocities_all = []
    frame_indices_all = []
    
    for point_velocities_batch, frame_batch in results:
        point_velocities_all.extend(point_velocities_batch)
        frame_indices_all.extend(frame_batch)
    
    # Sort by frame index
    sorted_indices = np.argsort(frame_indices_all)
    point_velocities_all = [point_velocities_all[i] for i in sorted_indices]
    
    # Save velocity field data
    with open(f"point_velocities_all_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(point_velocities_all, f)
    
    print(f"Velocity field data saved, total frames: {len(point_velocities_all)}")
    
    # Calculate energy flux
    print("Starting energy flux calculation...")
    k_bins_list = []
    Pi_E_list = []
    T_k_list = []
    
    half_box = args.grid_size * args.grid_count / 2
    lx = half_box * 2 / 10
    ly = half_box * 2 / 10
    lz = half_box * 2 / 10
    
    for pv in point_velocities_all:
        u = pv[:, 0].reshape((args.grid_count, args.grid_count, args.grid_count))
        v = pv[:, 1].reshape((args.grid_count, args.grid_count, args.grid_count))
        w = pv[:, 2].reshape((args.grid_count, args.grid_count, args.grid_count))
        
        k_bins, Pi_E, T_k = compute_energy_flux_corrected(u, v, w, lx, ly, lz)
        k_bins_list.append(k_bins)
        Pi_E_list.append(Pi_E)
        T_k_list.append(T_k)
    
    k_bins_list = np.array(k_bins_list)
    Pi_E_list = np.array(Pi_E_list)
    T_k_list = np.array(T_k_list)
    
    # Save energy flux results
    with open(f"energy_flux_{args.xtc}.pkl", 'wb') as f:
        pickle.dump(k_bins_list, f) 
        pickle.dump(Pi_E_list, f) 
        pickle.dump(T_k_list, f)
    
    total_time = time.time() - start_time
    print(f"All calculations completed! Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()