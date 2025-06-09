import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve, savgol_filter
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from pinkrigs_tools.dataset.query import load_data, queryCSV
import os
from datetime import datetime
import seaborn as sns
import pandas as pd
from scipy import stats, signal
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from glob import glob
from scipy.ndimage import gaussian_filter1d
from itertools import combinations 

from scipy.ndimage import uniform_filter1d 
from tqdm import tqdm
import pickle
import json



def prepare_correlation_data_freq(freq_results):
    """
    Extract sleep periods as start/end times instead of creating full mask
    """
    import numpy as np
    import pandas as pd
    
    all_counts = []
    all_cluster_info = []
    min_time_bins = float('inf')
    total_clusters = 0
    filtered_clusters = 0
    
    # 1. Find minimum length across probes
    for probe_name, probe_data in freq_results.items():
        if probe_name.startswith('probe'):
            counts = probe_data['counts']
            min_time_bins = min(min_time_bins, counts.shape[1])
            total_clusters += counts.shape[0]
            print(f"{probe_name}: {counts.shape}")
    
    print(f"Will truncate to {min_time_bins} time bins")
    
    # 2. Combine truncated data with quality filtering
    for probe_name, probe_data in freq_results.items():
        if probe_name.startswith('probe'):
            counts = probe_data['counts'][:, :min_time_bins]
            cluster_ids = probe_data['cluster_ids']
            
            # Get quality filter if available
            if 'good_mua_cluster_mask' in probe_data:
                quality_mask = probe_data['good_mua_cluster_mask']
                filtered_counts = counts[quality_mask]
                filtered_cluster_ids = [cid for i, cid in enumerate(cluster_ids) if quality_mask[i]]
                filtered_clusters += filtered_counts.shape[0]
                
                # Get cluster qualities for filtered clusters
                if 'cluster_quality' in probe_data:
                    cluster_quality = probe_data['cluster_quality']
                    filtered_quality = [cluster_quality[i] for i in range(len(cluster_ids)) if quality_mask[i]]
                else:
                    filtered_quality = ['unknown'] * len(filtered_cluster_ids)
                    
                print(f"{probe_name}: Kept {filtered_counts.shape[0]}/{counts.shape[0]} clusters (good+mua only)")
            else:
                # If no quality filter available, use all
                filtered_counts = counts
                filtered_cluster_ids = cluster_ids
                filtered_clusters += filtered_counts.shape[0]
                filtered_quality = probe_data.get('cluster_quality', ['unknown'] * len(cluster_ids))
                print(f"{probe_name}: Using all {counts.shape[0]} clusters (no quality filter found)")
            
            # Add the filtered data
            all_counts.append(filtered_counts)
            
            # Create cluster info for filtered clusters
            for i in range(len(filtered_cluster_ids)):
                all_cluster_info.append({
                    'probe': probe_name,
                    'cluster_id': filtered_cluster_ids[i],
                    'quality': filtered_quality[i]
                })
    
    # 3. Combine data
    combined_counts = np.vstack(all_counts) if all_counts else np.array([])
    time_bins = freq_results['probe0']['time_bins'][:min_time_bins]
    
    # Convert cluster info to DataFrame
    neuron_info = pd.DataFrame(all_cluster_info)
    
    # 4. EFFICIENT: Extract sleep periods as start/end time pairs
    sleep_periods = []
    
    # Try to get sleep data from the first probe's sleep_bout_mapping
    sleep_bout_mapping = None
    for probe_name, probe_data in freq_results.items():
        if probe_name.startswith('probe') and 'sleep_bout_mapping' in probe_data:
            sleep_bout_mapping = probe_data['sleep_bout_mapping']
            print(f"Found sleep_bout_mapping in {probe_name}")
            break
    
    if sleep_bout_mapping is not None and len(sleep_bout_mapping) > 0:
        print(f"Extracting {len(sleep_bout_mapping)} sleep periods...")
        for _, bout in sleep_bout_mapping.iterrows():
            if 'start_timestamp_s' in bout and 'end_timestamp_s' in bout:
                # Use the actual timestamps directly
                start_time = bout['start_timestamp_s']
                end_time = bout['end_timestamp_s']
                
                # Make sure times are within the recording range
                recording_start = time_bins[0]
                recording_end = time_bins[-1]
                
                if start_time <= recording_end and end_time >= recording_start:
                    # Clip to recording bounds
                    clipped_start = max(start_time, recording_start)
                    clipped_end = min(end_time, recording_end)
                    sleep_periods.append((clipped_start, clipped_end))
                    
        print(f"Extracted {len(sleep_periods)} sleep periods")

    else:
        print("Warning: No sleep_bout_mapping found in freq_results")
        print("Available keys in probe data:", list(freq_results['probe0'].keys()) if 'probe0' in freq_results else "No probe0")
    
    # 5. Create basic sleep mask for backward compatibility (optional)
    sleep_mask = np.zeros(len(time_bins), dtype=bool)
    for start_time, end_time in sleep_periods:
        start_idx = np.searchsorted(time_bins, start_time)
        end_idx = np.searchsorted(time_bins, end_time)
        start_idx = max(0, min(start_idx, len(sleep_mask) - 1))
        end_idx = max(0, min(end_idx, len(sleep_mask)))
        sleep_mask[start_idx:end_idx] = True
    
    wake_mask = ~sleep_mask
    
    print(f"Final combined data: {combined_counts.shape[0]} neurons ({filtered_clusters}/{total_clusters} after filtering), {combined_counts.shape[1]} time bins")
    print(f"Time resolution: {time_bins[1] - time_bins[0]:.3f} seconds")
    
    if len(sleep_periods) > 0:
        total_sleep_duration = sum(end - start for start, end in sleep_periods)
        total_duration = time_bins[-1] - time_bins[0]
        print(f"Sleep periods: {len(sleep_periods)} periods, {total_sleep_duration:.1f}s total ({total_sleep_duration/total_duration*100:.1f}%)")
    else:
        print("WARNING: No sleep periods found! Check sleep_bout_mapping data.")
    
    return combined_counts, neuron_info, time_bins, sleep_mask, wake_mask, sleep_periods



def analyze_time_resolved_synchrony_matrix(combined_counts, time_bins, sleep_mask, wake_mask, sleep_periods,
                                          window_size_s=10, step_size_s=2, 
                                          bin_size_ms=100, output_dir=None):
    """
    Fixed version with proper sleep period overlay
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
    print(f"Input data: {combined_counts.shape[0]} neurons, {combined_counts.shape[1]} time bins")
    print(f"Sleep mask: {np.sum(sleep_mask)} sleep bins out of {len(sleep_mask)} total")
    
    # Convert window parameters to bin indices
    window_size_bins = int(window_size_s * 1000 / bin_size_ms)  # 10s → 100 bins
    step_size_bins = int(step_size_s * 1000 / bin_size_ms)      # 2s → 20 bins
    
    print(f"Window: {window_size_bins} bins ({window_size_s}s)")
    print(f"Step: {step_size_bins} bins ({step_size_s}s)")
    
    # Calculate number of windows
    max_start_bin = len(time_bins) - window_size_bins
    n_windows = (max_start_bin // step_size_bins) + 1
    
    print(f"Total windows: {n_windows}")
    
    # Pre-allocate results
    window_times = []
    mean_correlations = []
    
    # Process each window with matrix correlation
    print("Computing sliding window correlations...")
    
    for w in tqdm(range(n_windows), desc="Matrix correlations"):
        # Define window boundaries
        start_bin = w * step_size_bins
        end_bin = start_bin + window_size_bins
        
        # Extract data for this window (neurons × time)
        window_data = combined_counts[:, start_bin:end_bin]
        
        # Skip windows with insufficient activity
        if np.sum(window_data) < 10:
            window_center_time = time_bins[start_bin + window_size_bins // 2]
            window_times.append(window_center_time)
            mean_correlations.append(0.0)
            continue
            
        # Calculate FULL correlation matrix in one operation!
        corr_matrix = np.corrcoef(window_data)  # 639×639 matrix
        
        # Extract upper triangular (unique pairs only)
        upper_tri_mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        pair_correlations = corr_matrix[upper_tri_mask]
        
        # Remove NaN values (from zero-variance neurons)
        valid_correlations = pair_correlations[~np.isnan(pair_correlations)]
        
        if len(valid_correlations) > 0:
            mean_corr = np.mean(valid_correlations)
        else:
            mean_corr = 0.0
            
        # Store results
        window_center_time = time_bins[start_bin + window_size_bins // 2]
        window_times.append(window_center_time)
        mean_correlations.append(mean_corr)
    
    # Convert to arrays
    window_times = np.array(window_times)
    mean_correlations = np.array(mean_correlations)
    
    print(f"✅ Completed: {len(window_times)} windows processed")
    print(f"   Mean correlation range: {np.min(mean_correlations):.3f} to {np.max(mean_correlations):.3f}")
    
    # Plot results
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot correlation over time
    ax.plot(window_times, mean_correlations, linewidth=2, color='black', alpha=0.8)
    
    # EFFICIENT: Add sleep periods directly (one axvspan per period)
    if len(sleep_periods) > 0:
        print(f"Adding {len(sleep_periods)} sleep periods to plot...")
        
        for sleep_start, sleep_end in sleep_periods:
            ax.axvspan(sleep_start, sleep_end, alpha=0.3, color='blue', linewidth=0)
        
        print(f"Sleep periods plotted: {len(sleep_periods)} spans")
    else:
        print("Warning: No sleep periods provided")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Population Correlation')
    ax.set_title(f'Population Synchrony Over Time\n'
                f'{window_size_s}s windows, {step_size_s}s steps, {combined_counts.shape[0]} neurons')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_bins[0], time_bins[-1])
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='Population Synchrony')
    ]
    
    if np.sum(sleep_mask) > 0:
        legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='Sleep Periods'))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/population_synchrony_over_time.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'window_times': window_times,
        'mean_correlations': mean_correlations,
        'n_windows': len(window_times),
        'window_size_s': window_size_s,
        'step_size_s': step_size_s,
        'n_neurons': combined_counts.shape[0]
    }


def cross_correlation_corrcoef_memory_efficient(spike_data, lag_range_ms=200,
                                        lag_resolution_ms=5, min_firing_rate=0.001):
    """
    Use corrcoef but with memory-efficient slicing like direct_fast
    """

    print(f"Input data shape: {spike_data.shape}")
    n_neurons, n_time_bins = spike_data.shape

    # 1. Filter by firing rate but DON'T create copies yet
    firing_rates = np.mean(spike_data, axis=1)
    active_mask = firing_rates > min_firing_rate
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)

    print(f"Using {n_active}/{n_neurons} neurons (no memory copy yet)")

    # 2. Setup lags
    max_lag_bins = int(lag_range_ms / lag_resolution_ms)
    lag_bins = np.arange(-max_lag_bins, max_lag_bins + 1) * int(lag_resolution_ms)
    n_lags = len(lag_bins)

    # 3. Process using memory-efficient slicing
    n_pairs = n_active * (n_active - 1) // 2
    all_lag_correlations = np.zeros((n_pairs, n_lags))

    with tqdm(total=n_lags, desc="Memory-efficient corrcoef") as pbar:
        
        for lag_idx, lag_offset in enumerate(lag_bins):
            
            if lag_offset == 0:
                # Zero lag: extract active neurons directly (small copy)
                data_subset = spike_data[active_indices]  # Only copy what we need
                corr_matrix = np.corrcoef(data_subset)
                
            else:
                # Create slices (views, not copies)
                if lag_offset > 0:
                    slice1 = spike_data[active_indices, :-lag_offset]
                    slice2 = spike_data[active_indices, lag_offset:]
                else:
                    slice1 = spike_data[active_indices, -lag_offset:]
                    slice2 = spike_data[active_indices, :lag_offset]
                
                # Stack only the small slices
                combined_small = np.vstack([slice1, slice2])
                
                # corrcoef on much smaller array
                full_corr = np.corrcoef(combined_small)
                corr_matrix = full_corr[:n_active, n_active:]
            
            # Extract pairs
            pair_idx = 0
            for i in range(n_active):
                for j in range(i + 1, n_active):
                    if lag_offset == 0:
                        all_lag_correlations[pair_idx, lag_idx] = corr_matrix[i, j]
                    else:
                        all_lag_correlations[pair_idx, lag_idx] = corr_matrix[i, j]
                    pair_idx += 1
            
            pbar.update(1)

    # Find peaks
    peak_correlations = np.zeros(n_pairs)
    peak_lags = np.zeros(n_pairs)

    for pair_idx in range(n_pairs):
        lag_correlations = all_lag_correlations[pair_idx]
        peak_idx = np.argmax(np.abs(lag_correlations))
        peak_correlations[pair_idx] = lag_correlations[peak_idx]
        peak_lags[pair_idx] = lag_bins[peak_idx]

    return {
        'peak_correlations': peak_correlations,
        'peak_lags': peak_lags,
        'n_pairs': n_pairs
    }

def save_analysis_results(variables_to_save, output_folder, prefix=''):
    """
    Fixed version of save_analysis_results that doesn't rely on global variables
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    # Create a subfolder for saved results
    saved_folder = os.path.join(output_folder, "saved_analysis")
    os.makedirs(saved_folder, exist_ok=True)
    
    # Add timestamp to filenames if requested
    if prefix:
        prefix = f"{prefix}_"
    
    saved_paths = {}
    
    for var_name, var_object in variables_to_save.items():
        # Skip if variable doesn't exist
        if var_object is None:
            print(f"Skipping {var_name}: variable not found")
            continue
            
        print(f"Saving {var_name}...", end="")
        
        # Choose appropriate format based on data type
        if isinstance(var_object, str) and os.path.exists(var_object) and var_object.endswith('.csv'):
            # For CSV paths, just record the path
            saved_paths[var_name] = var_object
            print(f" recorded path: {var_object}")
            
        elif isinstance(var_object, np.ndarray):
            # NumPy arrays -> .npy
            save_path = os.path.join(saved_folder, f"{prefix}{var_name}.npy")
            np.save(save_path, var_object)
            saved_paths[var_name] = save_path
            print(f" saved as .npy ({var_object.shape})")
            
        elif isinstance(var_object, pd.DataFrame):
            # Pandas DataFrames -> .csv
            save_path = os.path.join(saved_folder, f"{prefix}{var_name}.csv")
            var_object.to_csv(save_path, index=False)
            saved_paths[var_name] = save_path
            print(f" saved as .csv ({var_object.shape})")
            
        elif isinstance(var_object, dict):
            # Check if dictionary contains mostly numpy arrays
            numpy_keys = [k for k, v in var_object.items() if isinstance(v, np.ndarray)]
            if len(numpy_keys) > len(var_object) * 0.5:
                # Dictionary of arrays -> .npz
                save_path = os.path.join(saved_folder, f"{prefix}{var_name}.npz")
                np.savez_compressed(save_path, **var_object)
                saved_paths[var_name] = save_path
                print(f" saved as .npz (dict with {len(var_object)} keys)")
            else:
                # General dictionary -> .pkl
                save_path = os.path.join(saved_folder, f"{prefix}{var_name}.pkl")
                with open(save_path, 'wb') as f:
                    pickle.dump(var_object, f)
                saved_paths[var_name] = save_path
                print(f" saved as .pkl (dict with {len(var_object)} keys)")
        else:
            # Other objects -> pickle
            save_path = os.path.join(saved_folder, f"{prefix}{var_name}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(var_object, f)
            saved_paths[var_name] = save_path
            print(f" saved as .pkl")
    
    # Create a simplified metadata file without subject info
    metadata = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'saved_files': saved_paths
    }
    
    metadata_path = os.path.join(saved_folder, f"{prefix}files_index.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAll results saved to {saved_folder}")
    print(f"File index saved to {os.path.basename(metadata_path)}")
    
    return saved_paths, metadata_path

# Define a proper wrapper with all the subject parameters
def save_analysis_results_with_metadata(variables_to_save, output_folder, prefix='', 
                                             subject=None, exp_date=None, exp_num=None):
    """
    Fixed wrapper for save_analysis_results that also adds experiment metadata
    """

    
    # First call our fixed function with the parameters it expects
    saved_paths, files_path = save_analysis_results(
        variables_to_save=variables_to_save, 
        output_folder=output_folder,
        prefix=prefix
    )
    
    # Add optional timestamp prefix
    prefix_str = f"{prefix}_" if prefix else ""
    
    # Create a separate metadata file with experiment info
    saved_folder = os.path.join(output_folder, "saved_analysis")
    os.makedirs(saved_folder, exist_ok=True)
    
    # Generate metadata with experiment info
    metadata = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'saved_files': saved_paths,
    }
    
    # Add the subject and experiment info if provided
    if subject:
        metadata['subject'] = subject
    if exp_date:
        metadata['exp_date'] = exp_date
    if exp_num:
        metadata['exp_num'] = exp_num
    
    # Save the metadata file
    metadata_path = os.path.join(saved_folder, f"{prefix_str}experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment metadata saved to {os.path.basename(metadata_path)}")
    
    return saved_paths, metadata_path


# Function to load saved analysis results
def load_analysis_results(metadata_path=None, output_folder=None):
    """
    Load saved analysis results from disk
    
    Parameters:
        metadata_path: path to metadata file (optional)
        output_folder: folder containing saved_analysis directory (optional)
        
    Returns:
        Dictionary of loaded variables
    """

    # Find metadata file if not provided
    if metadata_path is None:
        if output_folder is None:
            raise ValueError("Either metadata_path or output_folder must be provided")
        
        saved_folder = os.path.join(output_folder, "saved_analysis")
        if not os.path.exists(saved_folder):
            raise FileNotFoundError(f"Saved analysis folder not found: {saved_folder}")
        
        # Find most recent metadata file
        metadata_files = [f for f in os.listdir(saved_folder) if f.endswith('metadata.json')]
        if not metadata_files:
            raise FileNotFoundError(f"No metadata files found in {saved_folder}")
        
        # Sort by modification time (newest first)
        metadata_files.sort(key=lambda f: os.path.getmtime(os.path.join(saved_folder, f)), reverse=True)
        metadata_path = os.path.join(saved_folder, metadata_files[0])
        print(f"Using most recent metadata file: {metadata_files[0]}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading results from {metadata['date']} for subject {metadata['subject']}")
    
    # Load each variable
    loaded_variables = {}
    for var_name, file_path in metadata['saved_files'].items():
        print(f"Loading {var_name}...", end="")
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f" SKIPPED (file not found: {file_path})")
            continue
            
        # Load based on file extension
        if file_path.endswith('.npy'):
            loaded_variables[var_name] = np.load(file_path)
            print(f" loaded from .npy ({loaded_variables[var_name].shape})")
            
        elif file_path.endswith('.npz'):
            npz_data = np.load(file_path)
            loaded_variables[var_name] = {key: npz_data[key] for key in npz_data.files}
            print(f" loaded from .npz (dict with {len(loaded_variables[var_name])} keys)")
            
        elif file_path.endswith('.csv'):
            if var_name.endswith('_csv'):  # Just a path to a CSV
                loaded_variables[var_name] = file_path
                print(f" recorded path: {file_path}")
            else:
                loaded_variables[var_name] = pd.read_csv(file_path)
                print(f" loaded from .csv ({loaded_variables[var_name].shape})")
            
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                loaded_variables[var_name] = pickle.load(f)
            
            if isinstance(loaded_variables[var_name], dict):
                print(f" loaded from .pkl (dict with {len(loaded_variables[var_name])} keys)")
            else:
                print(f" loaded from .pkl")
    
    print(f"\nSuccessfully loaded {len(loaded_variables)} variables")
    return loaded_variables




# ------------------------VISUALIZATION FUNCTIONS------------------------

def plot_selected_neuron_pairs(freq_counts, peak_correlations, peak_lags, lag_range_ms=200, lag_resolution_ms=5, n_pairs=20, output_folder=None):
    """
    Plot cross-correlation for selected neuron pairs spanning different correlation strengths
    """

    # Sort pairs by correlation strength
    sorted_indices = np.argsort(peak_correlations)
    
    # Select pairs spanning the range of correlations
    n_samples = min(n_pairs, len(peak_correlations))
    step = len(sorted_indices) // n_samples
    sample_indices = sorted_indices[::step][:n_samples]
    
    # Setup lag range
    max_lag_bins = int(lag_range_ms / lag_resolution_ms)
    lag_bins = np.arange(-max_lag_bins, max_lag_bins + 1) * int(lag_resolution_ms)
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    # Get neuron pairs (this is an approximation since we don't have the original indices)
    n_neurons = freq_counts.shape[0]
    pair_indices = []
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            pair_indices.append((i, j))
            if len(pair_indices) > max(sample_indices):
                break
        if len(pair_indices) > max(sample_indices):
            break
    
    # Plot each selected pair
    for i, idx in enumerate(sample_indices):
        if i >= len(axes):
            break
            
        corr = peak_correlations[idx]
        lag = peak_lags[idx]
        
        # Calculate full cross-correlation
        if idx < len(pair_indices):
            neuron_i, neuron_j = pair_indices[idx]
            
            # Compute cross-correlation for different lags
            cross_corrs = []
            for lag_offset in lag_bins:
                lag_offset = int(lag_offset)
                if lag_offset == 0:
                    # Zero lag
                    corr_val = np.corrcoef(freq_counts[neuron_i], freq_counts[neuron_j])[0, 1]
                else:
                    # Non-zero lag
                    if lag_offset > 0:
                        slice1 = freq_counts[neuron_i, :-lag_offset]
                        slice2 = freq_counts[neuron_j, lag_offset:]
                    else:
                        slice1 = freq_counts[neuron_i, -lag_offset:]
                        slice2 = freq_counts[neuron_j, :lag_offset]
                    
                    corr_val = np.corrcoef(slice1, slice2)[0, 1]
                
                cross_corrs.append(corr_val)
            
            # Plot cross-correlation
            axes[i].plot(lag_bins, cross_corrs, 'k-', linewidth=2)
            axes[i].axvline(x=lag, color='red', linestyle='--')
            axes[i].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            axes[i].set_title(f"Pair {idx} (Neurons {neuron_i}, {neuron_j})\nPeak r={corr:.3f} at lag={lag}ms")
            axes[i].set_xlabel('Lag (ms)')
            axes[i].set_ylabel('Correlation')
            axes[i].grid(True, alpha=0.3)
        else:
            # Just plot the peak information
            axes[i].text(0.5, 0.5, f"Pair {idx}\nPeak r={corr:.3f}\nLag={lag}ms", 
                       ha='center', va='center', fontsize=12)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle("Cross-correlation for Selected Neuron Pairs", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "selected_pair_correlations.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_strength_heatmap(peak_correlations, n_neurons, top_n=300, output_folder=None):
    """
    Plot heatmap of maximum correlation strengths for the most correlated neurons
    """

    
    # Get the top N most correlated pairs
    sorted_indices = np.argsort(-np.abs(peak_correlations))[:top_n]
    top_corrs = peak_correlations[sorted_indices]
    
    # Reconstruct the correlation matrix
    corr_matrix = np.zeros((n_neurons, n_neurons))
    
    # Fill in the matrix from the pairs
    pair_idx = 0
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if pair_idx < len(peak_correlations):
                corr_matrix[i, j] = peak_correlations[pair_idx]
                corr_matrix[j, i] = peak_correlations[pair_idx]  # Symmetric
            pair_idx += 1
            if pair_idx >= len(peak_correlations):
                break
        if pair_idx >= len(peak_correlations):
            break
    
    # Calculate the number of neurons to include to see top_n pairs
    pairs_per_neuron = n_neurons - 1
    neurons_to_show = min(n_neurons, int(np.sqrt(2 * top_n)) + 5)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(corr_matrix[:neurons_to_show, :neurons_to_show], dtype=bool)
    np.fill_diagonal(mask, True)  # Mask the diagonal
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix[:neurons_to_show, :neurons_to_show], 
                mask=mask, 
                cmap=cmap, 
                vmax=1.0, 
                vmin=-1.0,
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8})
    
    plt.title(f'Peak Correlation Strength Heatmap (Top {neurons_to_show} Neurons)', fontsize=14)
    if output_folder:
        plt.savefig(os.path.join(output_folder, "correlation_strength_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix[:neurons_to_show, :neurons_to_show]

def plot_lag_heatmap(peak_lags, n_neurons, top_n=300, output_folder=None):
    """
    Plot heatmap showing at what lag the maximum correlation occurred
    """

    
    # Get the pairs with highest absolute correlation
    sorted_indices = np.argsort(-np.abs(peak_lags))[:top_n]
    top_lags = peak_lags[sorted_indices]
    
    # Reconstruct the lag matrix
    lag_matrix = np.zeros((n_neurons, n_neurons))
    lag_matrix.fill(np.nan)  # Use NaN for undefined lags
    
    # Fill in the matrix from the pairs
    pair_idx = 0
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if pair_idx < len(peak_lags):
                lag_matrix[i, j] = peak_lags[pair_idx]
                lag_matrix[j, i] = -peak_lags[pair_idx]  # Antisymmetric
            pair_idx += 1
            if pair_idx >= len(peak_lags):
                break
        if pair_idx >= len(peak_lags):
            break
    
    # Calculate the number of neurons to include to see top_n pairs
    neurons_to_show = min(n_neurons, int(np.sqrt(2 * top_n)) + 5)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    mask = np.zeros_like(lag_matrix[:neurons_to_show, :neurons_to_show], dtype=bool)
    np.fill_diagonal(mask, True)  # Mask the diagonal
    
    max_lag = np.nanmax(np.abs(lag_matrix[:neurons_to_show, :neurons_to_show]))
    cmap = 'coolwarm'
    sns.heatmap(lag_matrix[:neurons_to_show, :neurons_to_show], 
                mask=mask, 
                cmap=cmap, 
                vmax=max_lag, 
                vmin=-max_lag,
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8, "label": "Lag (ms)"})
    
    plt.title(f'Peak Correlation Lag Heatmap (Top {neurons_to_show} Neurons)', fontsize=14)
    if output_folder:
        plt.savefig(os.path.join(output_folder, "correlation_lag_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return lag_matrix[:neurons_to_show, :neurons_to_show]

def plot_correlation_summary(super_fast_results, output_folder=None):
    """
    Create a summary plot of correlation statistics
    """

    
    peak_correlations = super_fast_results['peak_correlations']
    peak_lags = super_fast_results['peak_lags']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram of correlation strength
    sns.histplot(peak_correlations, bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].axvline(x=0, color='r', linestyle='--')
    axes[0, 0].set_title('Distribution of Peak Correlation Strengths')
    axes[0, 0].set_xlabel('Correlation Coefficient')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Histogram of lag times
    sns.histplot(peak_lags, bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_title('Distribution of Peak Correlation Lags')
    axes[0, 1].set_xlabel('Lag (ms)')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Scatter plot of lag vs correlation strength
    # Sample if too many points
    if len(peak_correlations) > 5000:
        sample_idx = np.random.choice(len(peak_correlations), 5000, replace=False)
        sample_corrs = peak_correlations[sample_idx]
        sample_lags = peak_lags[sample_idx]
    else:
        sample_corrs = peak_correlations
        sample_lags = peak_lags
    
    axes[1, 0].scatter(sample_lags, sample_corrs, alpha=0.3, s=5)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_title('Correlation Strength vs Lag')
    axes[1, 0].set_xlabel('Lag (ms)')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    
    # 4. Summary statistics table
    axes[1, 1].axis('off')
    stats_text = (
        f"SUMMARY STATISTICS\n\n"
        f"Total pairs analyzed: {len(peak_correlations)}\n\n"
        f"Correlation strength:\n"
        f"  Mean: {np.mean(peak_correlations):.3f}\n"
        f"  Median: {np.median(peak_correlations):.3f}\n"
        f"  Std Dev: {np.std(peak_correlations):.3f}\n"
        f"  Min: {np.min(peak_correlations):.3f}\n"
        f"  Max: {np.max(peak_correlations):.3f}\n\n"
        f"Pairs with |r| > 0.3: {np.sum(np.abs(peak_correlations) > 0.3)} ({np.sum(np.abs(peak_correlations) > 0.3)/len(peak_correlations)*100:.1f}%)\n"
        f"Positive correlations: {np.sum(peak_correlations > 0)} ({np.sum(peak_correlations > 0)/len(peak_correlations)*100:.1f}%)\n"
        f"Negative correlations: {np.sum(peak_correlations < 0)} ({np.sum(peak_correlations < 0)/len(peak_correlations)*100:.1f}%)\n\n"
        f"Lag times:\n"
        f"  Mean: {np.mean(peak_lags):.1f} ms\n"
        f"  Median: {np.median(peak_lags):.1f} ms\n"
        f"  Std Dev: {np.std(peak_lags):.1f} ms\n"
        f"  Min: {np.min(peak_lags):.1f} ms\n"
        f"  Max: {np.max(peak_lags):.1f} ms\n\n"
        f"Zero lag pairs: {np.sum(peak_lags == 0)} ({np.sum(peak_lags == 0)/len(peak_lags)*100:.1f}%)\n"
        f"Positive lag pairs: {np.sum(peak_lags > 0)} ({np.sum(peak_lags > 0)/len(peak_lags)*100:.1f}%)\n"
        f"Negative lag pairs: {np.sum(peak_lags < 0)} ({np.sum(peak_lags < 0)/len(peak_lags)*100:.1f}%)"
    )
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.suptitle("Neural Correlation Analysis Summary", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "correlation_analysis_summary.png"), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix_matshow(peak_correlations, n_neurons=None, max_neurons=100, 
                                           value_range=(-0.4, 0.4), output_folder=None):
    """
    Plot correlation matrix using matshow with customized color range
    
    Parameters:
        peak_correlations: Array of peak correlations from super_fast_results
        n_neurons: Number of neurons (if None, will be estimated from number of pairs)
        max_neurons: Maximum number of neurons to show (for readability)
        value_range: Tuple of (min, max) values for color scale
        output_folder: Where to save the plot (optional)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # If n_neurons not provided, estimate from number of pairs
    if n_neurons is None:
        n_pairs = len(peak_correlations)
        n_neurons = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
        print(f"Estimated number of neurons: {n_neurons}")
    
    # Create the correlation matrix using peak correlations
    corr_matrix = np.zeros((n_neurons, n_neurons))
    np.fill_diagonal(corr_matrix, np.nan)  # Set diagonal to NaN for visualization
    
    # Fill the matrix with correlation values
    pair_idx = 0
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if pair_idx < len(peak_correlations):
                corr_matrix[i, j] = peak_correlations[pair_idx]
                corr_matrix[j, i] = peak_correlations[pair_idx]  # Symmetric
                pair_idx += 1
            else:
                break
        if pair_idx >= len(peak_correlations):
            break
    
    # Limit visualization to max_neurons for clarity
    if n_neurons > max_neurons:
        print(f"Limiting visualization to {max_neurons} neurons (out of {n_neurons})")
        # Select neurons with strongest correlations
        mean_abs_corr = np.nanmean(np.abs(corr_matrix), axis=1)
        top_indices = np.argsort(-mean_abs_corr)[:max_neurons]
        top_indices = np.sort(top_indices)  # Keep in original order
        corr_matrix = corr_matrix[np.ix_(top_indices, top_indices)]
    
    # Create a figure with two subplots: one for the matrix, one for histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # Plot the correlation matrix with custom color range
    np.fill_diagonal(corr_matrix, 0)  # Set diagonal to zero for visualization
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap (reversed)
    
    # Use custom value range for better color visualization
    vmin, vmax = value_range
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    im = ax1.matshow(corr_matrix, cmap=cmap, norm=norm)
    ax1.set_title('Peak Correlations Across All Lags', fontsize=14)
    
    # Add colorbar with custom range
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Peak Correlation Coefficient')
    
    # Add grid lines to separate neurons
    ax1.set_xticks(np.arange(-.5, corr_matrix.shape[0], 1), minor=True)
    ax1.set_yticks(np.arange(-.5, corr_matrix.shape[1], 1), minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set major ticks and labels
    tick_step = max(1, corr_matrix.shape[0] // 10)
    ax1.set_xticks(np.arange(0, corr_matrix.shape[0], tick_step))
    ax1.set_yticks(np.arange(0, corr_matrix.shape[1], tick_step))
    ax1.set_xticklabels(np.arange(0, corr_matrix.shape[0], tick_step))
    ax1.set_yticklabels(np.arange(0, corr_matrix.shape[1], tick_step))
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Neuron Index')
    
    # Plot the histogram of correlation values
    flattened_corr = corr_matrix[~np.isnan(corr_matrix) & ~np.eye(corr_matrix.shape[0], dtype=bool)]
    ax2.hist(flattened_corr, bins=50, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.set_ylim(vmin, vmax)  # Use same range as the heatmap
    ax2.set_xlabel('Count')
    ax2.set_ylabel('Peak Correlation Coefficient')
    ax2.set_title('Distribution of Peak Correlations')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    
    # Add text with statistics
    stats_text = (
        f"Peak Correlation Statistics:\n\n"
        f"Mean: {np.nanmean(flattened_corr):.3f}\n"
        f"Median: {np.nanmedian(flattened_corr):.3f}\n"
        f"Std Dev: {np.nanstd(flattened_corr):.3f}\n"
        f"Min: {np.nanmin(flattened_corr):.3f}\n"
        f"Max: {np.nanmax(flattened_corr):.3f}\n\n"
        f"Positive: {np.sum(flattened_corr > 0)} "
        f"({np.sum(flattened_corr > 0)/len(flattened_corr)*100:.1f}%)\n"
        f"Negative: {np.sum(flattened_corr < 0)} "
        f"({np.sum(flattened_corr < 0)/len(flattened_corr)*100:.1f}%)"
    )
    
    # Position the text below the histogram
    ax2.text(0.5, -0.15, stats_text, transform=ax2.transAxes, 
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)  # Make room for the text
    
    plt.suptitle(f'Neural Correlation Matrix (n={corr_matrix.shape[0]}) - Peak Values Across All Lags', 
                 fontsize=16, y=0.98)
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, "peak_correlation_matrix_enhanced.png"), 
                   dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return corr_matrix


def analyze_state_specific_correlations(combined_counts, time_bins, sleep_mask, wake_mask, sleep_periods,
                                       output_dir=None):
    """
    Calculate pairwise correlations between neurons separately for sleep and wake states.
    
    Parameters:
        combined_counts: Neural activity matrix (neurons x time bins)
        time_bins: Array of time values corresponding to columns in combined_counts
        sleep_mask: Boolean mask indicating sleep time bins
        wake_mask: Boolean mask indicating wake time bins 
        sleep_periods: List of (start_time, end_time) tuples for each sleep bout
        output_dir: Directory to save output plots
        
    Returns:
        Dictionary containing correlation matrices and statistics for both states
    """
    
    n_neurons, n_time_bins = combined_counts.shape
    print(f"Input data: {n_neurons} neurons, {n_time_bins} time bins")
    print(f"Sleep bins: {np.sum(sleep_mask)} ({np.sum(sleep_mask)/n_time_bins*100:.1f}%)")
    print(f"Wake bins: {np.sum(wake_mask)} ({np.sum(wake_mask)/n_time_bins*100:.1f}%)")
    
    # Split data by state
    sleep_data = combined_counts[:, sleep_mask]
    wake_data = combined_counts[:, wake_mask]
    
    print(f"Sleep data shape: {sleep_data.shape}")
    print(f"Wake data shape: {wake_data.shape}")
    
    # Function to calculate correlation matrix efficiently
    def calculate_state_correlations(state_data, state_name):
        n_neurons, n_time_bins = state_data.shape
        
        # Check for sufficient data
        if n_time_bins < 10:
            print(f"Warning: Very few time bins for {state_name} state ({n_time_bins}), correlations may be unreliable")
            
        # Filter neurons with zero variance (flat activity)
        neuron_var = np.var(state_data, axis=1)
        active_neurons = neuron_var > 0
        
        if np.sum(~active_neurons) > 0:
            print(f"Removing {np.sum(~active_neurons)} neurons with zero variance in {state_name} state")
            filtered_data = state_data[active_neurons]
        else:
            filtered_data = state_data
            
        # Use np.corrcoef for efficient correlation calculation
        print(f"Calculating correlation matrix for {state_name} state ({filtered_data.shape[0]} neurons)...")
        corr_matrix = np.corrcoef(filtered_data)
        
        # Extract upper triangular values (unique pairs only)
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        pair_corrs = corr_matrix[mask.astype(bool)]
        
        # Calculate basic statistics
        stats = {
            "mean": np.mean(pair_corrs),
            "median": np.median(pair_corrs),
            "std": np.std(pair_corrs),
            "min": np.min(pair_corrs), 
            "max": np.max(pair_corrs),
            "n_positive": np.sum(pair_corrs > 0),
            "n_negative": np.sum(pair_corrs < 0),
            "n_total": len(pair_corrs)
        }
        
        print(f"  {state_name} correlations: mean={stats['mean']:.3f}, median={stats['median']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Restore to original dimensions by padding with NaN
        if np.sum(~active_neurons) > 0:
            full_corr = np.full((n_neurons, n_neurons), np.nan)
            # Create mapping from filtered index to original index
            orig_idx = np.where(active_neurons)[0]
            for i, oi in enumerate(orig_idx):
                for j, oj in enumerate(orig_idx):
                    full_corr[oi, oj] = corr_matrix[i, j]
            return full_corr, stats
        else:
            return corr_matrix, stats
            
    # Calculate correlations for each state
    sleep_corr, sleep_stats = calculate_state_correlations(sleep_data, "Sleep")
    wake_corr, wake_stats = calculate_state_correlations(wake_data, "Wake")
    
    # Calculate difference matrix
    diff_corr = sleep_corr - wake_corr
    
    # Extract upper triangular values for difference
    mask = np.triu(np.ones_like(diff_corr), k=1)
    diff_values = diff_corr[np.logical_and(mask.astype(bool), ~np.isnan(diff_corr))]
    
    diff_stats = {
        "mean": np.nanmean(diff_values),
        "median": np.nanmedian(diff_values),
        "std": np.nanstd(diff_values),
        "min": np.nanmin(diff_values),
        "max": np.nanmax(diff_values),
        "n_positive": np.sum(diff_values > 0),  # Sleep > Wake
        "n_negative": np.sum(diff_values < 0),  # Wake > Sleep
        "n_total": np.sum(~np.isnan(diff_values))
    }
    
    print(f"Difference (Sleep - Wake): mean={diff_stats['mean']:.3f}, median={diff_stats['median']:.3f}")
    print(f"Pairs stronger in sleep: {diff_stats['n_positive']} ({diff_stats['n_positive']/diff_stats['n_total']*100:.1f}%)")
    print(f"Pairs stronger in wake: {diff_stats['n_negative']} ({diff_stats['n_negative']/diff_stats['n_total']*100:.1f}%)")
    
    return {
        'sleep_correlation_matrix': sleep_corr,
        'wake_correlation_matrix': wake_corr,
        'difference_matrix': diff_corr,
        'sleep_stats': sleep_stats,
        'wake_stats': wake_stats,
        'difference_stats': diff_stats
    }

def plot_state_correlation_matrices(state_corr_results, np_results=None, max_neurons=100, 
                                   value_range=(-0.4, 0.4), sort_by_MI=True):
    """
    Plot correlation matrices for sleep and wake states using matshow
    
    Parameters:
        state_corr_results: Results dictionary from analyze_state_specific_correlations
        np_results: Results from analyze_sleep_wake_activity (for modulation index)
        max_neurons: Maximum number of neurons to show (for readability)
        value_range: Tuple of (min, max) values for color scale
        sort_by_MI: Whether to sort neurons by modulation index (True) or use original order (False)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    
    # Extract correlation matrices
    sleep_corr = state_corr_results['sleep_correlation_matrix'].copy()
    wake_corr = state_corr_results['wake_correlation_matrix'].copy()
    diff_corr = state_corr_results['difference_matrix'].copy()
    
    # Extract stats
    sleep_stats = state_corr_results['sleep_stats']
    wake_stats = state_corr_results['wake_stats']
    diff_stats = state_corr_results['difference_stats']
    
    # Limit number of neurons for visualization
    n_neurons = sleep_corr.shape[0]
    neurons_to_use = min(n_neurons, max_neurons)
    
    # Sort by modulation index if requested and available
    if sort_by_MI and np_results is not None and 'merged' in np_results:
        print("Sorting neurons by sleep/wake modulation index")
        
        # Get modulation index
        modulation_index = np_results['merged']['modulation_index']
        
        # Make sure sizes match
        if len(modulation_index) != n_neurons:
            print(f"Warning: Modulation index has {len(modulation_index)} neurons but correlation matrix has {n_neurons}")
            print("Using original neuron ordering instead")
            sort_indices = np.arange(n_neurons)
        else:
            # Sort from most sleep-selective to most wake-selective
            sort_indices = np.argsort(modulation_index)
            
            # Apply sorting to matrices
            sleep_corr = sleep_corr[np.ix_(sort_indices, sort_indices)]
            wake_corr = wake_corr[np.ix_(sort_indices, sort_indices)]
            diff_corr = diff_corr[np.ix_(sort_indices, sort_indices)]
            
            # Create a modulation index color bar for y-axis
            mi_cmap = plt.cm.get_cmap('coolwarm')
            mi_colors = mi_cmap((modulation_index[sort_indices[:neurons_to_use]] + 1) / 2)  # Normalize to 0-1
    else:
        # Use original order
        sort_indices = np.arange(n_neurons)
        
    # Limit to max_neurons
    if n_neurons > neurons_to_use:
        print(f"Limiting display to {neurons_to_use} neurons (out of {n_neurons})")
        if sort_by_MI and np_results is not None and 'merged' in np_results:
            # Already sorted, just take first N
            sleep_corr = sleep_corr[:neurons_to_use, :neurons_to_use]
            wake_corr = wake_corr[:neurons_to_use, :neurons_to_use]
            diff_corr = diff_corr[:neurons_to_use, :neurons_to_use]
        else:
            # Take first N from original ordering
            sleep_corr = sleep_corr[:neurons_to_use, :neurons_to_use]
            wake_corr = wake_corr[:neurons_to_use, :neurons_to_use]
            diff_corr = diff_corr[:neurons_to_use, :neurons_to_use]
    
    # Create figure with three subplots
    if sort_by_MI and np_results is not None and 'merged' in np_results:
        # Create figure with space for MI color bar
        fig = plt.figure(figsize=(22, 7))
        gs = plt.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
        ax_mi = plt.subplot(gs[3])
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        ax1, ax2, ax3 = axes
    
    # Define color range
    vmin, vmax = value_range
    
    # For difference, calculate symmetric range based on data
    diff_max = max(abs(np.nanmin(diff_corr)), abs(np.nanmax(diff_corr)))
    diff_range = (-diff_max, diff_max)
    
    # Plot sleep correlation matrix
    im1 = ax1.matshow(sleep_corr, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Sleep State Correlations\nMean: {sleep_stats["mean"]:.3f}', fontsize=14)
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Neuron Index')
    
    # Add grid lines
    ax1.set_xticks(np.arange(-.5, sleep_corr.shape[0], 10), minor=True)
    ax1.set_yticks(np.arange(-.5, sleep_corr.shape[1], 10), minor=True)
    ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Correlation')
    
    # Plot wake correlation matrix
    im2 = ax2.matshow(wake_corr, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax2.set_title(f'Wake State Correlations\nMean: {wake_stats["mean"]:.3f}', fontsize=14)
    ax2.set_xlabel('Neuron Index')
    
    # Add grid lines
    ax2.set_xticks(np.arange(-.5, wake_corr.shape[0], 10), minor=True)
    ax2.set_yticks(np.arange(-.5, wake_corr.shape[1], 10), minor=True)
    ax2.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Correlation')
    
    # Plot difference matrix
    im3 = ax3.matshow(diff_corr, cmap='RdBu_r', vmin=diff_range[0], vmax=diff_range[1])
    ax3.set_title(f'Difference (Sleep - Wake)\nMean: {diff_stats["mean"]:.3f}', fontsize=14)
    ax3.set_xlabel('Neuron Index')
    
    # Add grid lines
    ax3.set_xticks(np.arange(-.5, diff_corr.shape[0], 10), minor=True)
    ax3.set_yticks(np.arange(-.5, diff_corr.shape[1], 10), minor=True)
    ax3.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Correlation Difference')
    
    # Add modulation index color bar if sorted by MI
    if sort_by_MI and np_results is not None and 'merged' in np_results and len(modulation_index) == n_neurons:
        # Create modulation index color bar
        mi_sorted = modulation_index[sort_indices[:neurons_to_use]]
        mi_plot = np.expand_dims(mi_sorted, axis=1)
        
        # Plot the modulation index as a vertical colorbar
        im_mi = ax_mi.imshow(mi_plot, cmap='coolwarm', aspect='auto',
                             vmin=-1, vmax=1)
        ax_mi.set_title('MI')
        ax_mi.set_xticks([])
        
        # Only show a few y-tick labels to avoid crowding
        step = max(1, neurons_to_use // 10)
        ax_mi.set_yticks(np.arange(0, neurons_to_use, step))
        ax_mi.set_yticklabels([])  # Remove tick labels
        
        # Add a colorbar to explain the modulation index
        cbar_mi = plt.colorbar(im_mi, ax=ax_mi, shrink=0.8)
        cbar_mi.set_label('Modulation Index\n(Wake - Sleep)/(Wake + Sleep)')
        
        # Add annotations for wake and sleep selectivity
        ax_mi.text(0.5, -0.05, 'Sleep-selective', transform=ax_mi.transAxes, 
                  ha='center', va='top', color='blue')
        ax_mi.text(0.5, 1.05, 'Wake-selective', transform=ax_mi.transAxes, 
                  ha='center', va='bottom', color='red')
        
    # Set common title based on sorting method
    if sort_by_MI and np_results is not None and 'merged' in np_results and len(modulation_index) == n_neurons:
        plt.suptitle('Neural Correlation Matrices by Behavioral State\n(Sorted by Sleep/Wake Modulation Index)', 
                    fontsize=16, y=1.05)
    else:
        plt.suptitle('Neural Correlation Matrices by Behavioral State', 
                    fontsize=16, y=1.05)
    
    # Improve spacing
    plt.tight_layout()
    
    return fig


def calculate_keeper_ranking_score(sleep_corr, wake_corr, n_top=150, position_weight=0.9, max_rank_shift=30):
    """
    Calculate a definitive keeper-to-switcher score for ranking all neurons
    
    Parameters:
        sleep_corr: Sleep correlation matrix
        wake_corr: Wake correlation matrix
        n_top: Number of top partners to consider
        position_weight: Weight parameter (0-1, higher means top positions matter more)
        max_rank_shift: Maximum rank shift to consider as "good" (beyond this is penalized heavily)
    
    Returns:
        Array of keeper scores for each neuron (higher = better keeper)
    """
    import numpy as np
    
    n_neurons = sleep_corr.shape[0]
    keeper_scores = np.zeros(n_neurons)
    
    for neuron_i in range(n_neurons):
        # Get correlations with all other neurons
        sleep_corrs = sleep_corr[neuron_i, :]
        wake_corrs = wake_corr[neuron_i, :]
        
        # Exclude self-correlation
        mask = np.ones(n_neurons, dtype=bool)
        mask[neuron_i] = False
        sleep_corrs = sleep_corrs[mask]
        wake_corrs = wake_corrs[mask]
        other_neurons = np.where(mask)[0]
        
        # Sort correlation partners by strength
        sleep_order = np.argsort(-sleep_corrs)
        wake_order = np.argsort(-wake_corrs)
        
        # Get top partners by original indices
        sleep_top_partners = other_neurons[sleep_order[:n_top]]
        wake_top_partners = other_neurons[wake_order[:n_top]]
        
        # Create rank lookup dictionaries (1-based ranking for readability)
        sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
        wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
        
        # Find common partners in top rankings
        common_partners = set(sleep_top_partners) & set(wake_top_partners)
        
        # Base score component from overlap ratio
        overlap_ratio = len(common_partners) / n_top
        
        # Position-weighted rank stability component
        stability_score = 0
        total_weight = 0
        
        # Analyze each top sleep partner
        for rank, partner in enumerate(sleep_top_partners):
            # Weight by position importance (exponential decay)
            pos_weight = position_weight ** rank
            total_weight += pos_weight
            
            # If this partner is also a top wake partner
            if partner in common_partners:
                # Calculate rank shift penalty
                sleep_rank = rank + 1  # Convert to 1-based
                wake_rank = wake_ranks[partner]
                rank_shift = abs(sleep_rank - wake_rank)
                
                # Score this partner's stability
                if rank_shift <= max_rank_shift:
                    # Small rank shifts are penalized linearly
                    rank_factor = 1 - (rank_shift / (2 * max_rank_shift))
                else:
                    # Large rank shifts are penalized more heavily
                    rank_factor = 0.5 * (max_rank_shift / rank_shift)
                
                # Add this partner's contribution to the score
                stability_score += pos_weight * rank_factor
        
        # Normalize stability score
        if total_weight > 0:
            normalized_stability = stability_score / total_weight
        else:
            normalized_stability = 0
            
        # Final keeper score combines stability and overlap with appropriate weights
        # Give more weight to stability (70%) than raw overlap (30%)
        keeper_scores[neuron_i] = (0.7 * normalized_stability) + (0.3 * overlap_ratio)
    
    return keeper_scores

def analyze_correlation_partner_stability(state_corr_results, np_results=None, output_dir=None):
    """
    Analyze how neurons maintain correlation partners between sleep and wake states,
    with a definitive ranking from top keepers to top switchers.
    
    Parameters:
        state_corr_results: Results from analyze_state_specific_correlations
        np_results: Results from analyze_sleep_wake_activity (for modulation index)
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with stability metrics for each neuron
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Extract correlation matrices
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    
    n_neurons = sleep_corr.shape[0]
    print(f"Analyzing correlation partner stability for {n_neurons} neurons")
    
    # Get modulation index if available
    if np_results is not None and 'merged' in np_results:
        modulation_index = np_results['merged']['modulation_index']
        print("Using sleep/wake modulation index for analysis")
    else:
        modulation_index = None
        print("No modulation index available")
    
    # Calculate the definitive keeper score
    n_top = min(150, n_neurons - 2)  # Use top 150 partners or all if fewer
    keeper_scores = calculate_keeper_ranking_score(
        sleep_corr, wake_corr, 
        n_top=n_top,
        position_weight=0.9, 
        max_rank_shift=30
    )
    
    # Get the ranking from keeper to switcher
    keeper_ranking = np.argsort(-keeper_scores)  # Highest to lowest
    
    # Identify top keepers and top switchers (25% each)
    n_top_examples = max(3, n_neurons // 10)  # At least 3, up to 10% of neurons
    top_keepers = keeper_ranking[:n_top_examples]
    top_switchers = keeper_ranking[-n_top_examples:]
    
    print(f"Top keeper neurons: {top_keepers}")
    print(f"Top switcher neurons: {top_switchers}")
    
    # Create visualizations
    if output_dir:
        # 1. Plot the distribution of keeper scores
        plt.figure(figsize=(10, 6))
        sns.histplot(keeper_scores, bins=30, kde=True)
        plt.axvline(np.mean(keeper_scores), color='k', linestyle='--',
                   label=f'Mean: {np.mean(keeper_scores):.3f}')
        plt.axvline(np.percentile(keeper_scores, 75), color='g', linestyle='-.',
                   label=f'75th percentile')
        plt.axvline(np.percentile(keeper_scores, 25), color='r', linestyle='-.',
                   label=f'25th percentile')
        plt.xlabel('Keeper Score')
        plt.ylabel('Count')
        plt.title('Distribution of Keeper-to-Switcher Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keeper_score_distribution.png'),
                   dpi=300, bbox_inches='tight')
        
        # 2. Plot relationship between stability and modulation index
        if modulation_index is not None:
            plt.figure(figsize=(10, 8))
            
            # Create a scatter plot with color gradient based on keeper score
            plt.scatter(modulation_index, keeper_scores, 
                       c=keeper_scores, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Keeper Score')
            
            # Highlight top keepers and switchers
            plt.scatter(modulation_index[top_keepers], keeper_scores[top_keepers], 
                       color='green', s=100, alpha=0.8, label='Top Keepers')
            plt.scatter(modulation_index[top_switchers], keeper_scores[top_switchers], 
                       color='red', s=100, alpha=0.8, label='Top Switchers')
            
            plt.xlabel('Sleep/Wake Modulation Index')
            plt.ylabel('Keeper Score')
            plt.title('Relationship Between State Preference and Partner Stability')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'keeper_score_vs_modulation.png'),
                       dpi=300, bbox_inches='tight')
        
        # 3. Create detailed plots showing examples of top keepers vs top switchers
        # Function to analyze a single neuron's partners
        def analyze_neuron_partners(neuron_idx, ax, label):
            # Get this neuron's partners
            sleep_corrs = sleep_corr[neuron_idx, :]
            wake_corrs = wake_corr[neuron_idx, :]
            
            # Exclude self-correlation
            mask = np.ones(n_neurons, dtype=bool)
            mask[neuron_idx] = False
            sleep_corrs = sleep_corrs[mask]
            wake_corrs = wake_corrs[mask]
            other_neurons = np.where(mask)[0]
            
            # Sort correlation partners
            sleep_order = np.argsort(-sleep_corrs)
            wake_order = np.argsort(-wake_corrs)
            
            # Get top 20 partners in each state
            top_n = min(20, len(sleep_order))
            sleep_top_idx = sleep_order[:top_n]
            wake_top_idx = wake_order[:top_n]
            
            sleep_top_partners = other_neurons[sleep_top_idx]
            wake_top_partners = other_neurons[wake_top_idx]
            
            sleep_top_corrs = sleep_corrs[sleep_top_idx]
            wake_top_corrs = wake_corrs[wake_top_idx]
            
            # Find partners that appear in both lists
            common_partners = set(sleep_top_partners) & set(wake_top_partners)
            
            # Create ranks dictionary for both states
            all_sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
            all_wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
            
            # Calculate average rank shift for common partners
            rank_shifts = []
            for partner in common_partners:
                sleep_rank = all_sleep_ranks[partner]
                wake_rank = all_wake_ranks[partner]
                if sleep_rank <= top_n or wake_rank <= top_n:  # Only count if top in at least one state
                    rank_shifts.append(abs(sleep_rank - wake_rank))
            
            avg_rank_shift = np.mean(rank_shifts) if rank_shifts else np.nan
            
            # Plot sleep partners
            for j, (partner, corr) in enumerate(zip(sleep_top_partners, sleep_top_corrs)):
                color = 'blue' if partner in common_partners else 'lightblue'
                ax.bar(j, corr, color=color, alpha=0.7)
                
                # Label with rank in other state if common
                if partner in common_partners:
                    wake_rank = all_wake_ranks[partner]
                    ax.text(j, corr/2, f"{wake_rank}", ha='center', fontweight='bold')
            
            # Plot wake partners
            for j, (partner, corr) in enumerate(zip(wake_top_partners, wake_top_corrs)):
                color = 'red' if partner in common_partners else 'lightcoral'
                ax.bar(j + top_n + 5, corr, color=color, alpha=0.7)
                
                # Label with rank in other state if common
                if partner in common_partners:
                    sleep_rank = all_sleep_ranks[partner]
                    ax.text(j + top_n + 5, corr/2, f"{sleep_rank}", ha='center', fontweight='bold')
            
            # Add titles and labels
            subtitle = f"Score: {keeper_scores[neuron_idx]:.3f}, Common: {len(common_partners)}/{top_n}"
            if not np.isnan(avg_rank_shift):
                subtitle += f", Avg. Shift: {avg_rank_shift:.1f} ranks"
            
            ax.set_title(f"{label} Neuron {neuron_idx} - {subtitle}")
            ax.set_xticks([top_n//2, top_n + 5 + top_n//2])
            ax.set_xticklabels(['Sleep Partners', 'Wake Partners'])
            ax.set_ylabel('Correlation')
            
            # Add vertical separator
            ax.axvline(top_n + 2.5, color='k', linestyle='--', alpha=0.5)
        
        # Create a figure showing top 3 keepers and top 3 switchers
        n_examples = min(3, len(top_keepers))
        fig, axes = plt.subplots(2, n_examples, figsize=(n_examples*6, 10))
        
        # Plot top keepers
        for i in range(n_examples):
            analyze_neuron_partners(top_keepers[i], axes[0, i], f"Top Keeper #{i+1}")
        
        # Plot top switchers
        for i in range(n_examples):
            analyze_neuron_partners(top_switchers[i], axes[1, i], f"Top Switcher #{i+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_keepers_vs_switchers.png'),
                   dpi=300, bbox_inches='tight')
        
        # 4. Create a scatter plot of all neurons showing overlap vs stability
        # Calculate these separately for visualization
        overlap_ratios = np.zeros(n_neurons)
        avg_rank_shifts = np.zeros(n_neurons)
        
        for neuron_i in range(n_neurons):
            # Get correlations with all other neurons
            sleep_corrs = sleep_corr[neuron_i, :]
            wake_corrs = wake_corr[neuron_i, :]
            
            # Exclude self-correlation
            mask = np.ones(n_neurons, dtype=bool)
            mask[neuron_i] = False
            other_neurons = np.where(mask)[0]
            
            # Sort correlation partners
            sleep_order = np.argsort(-sleep_corrs[mask])
            wake_order = np.argsort(-wake_corrs[mask])
            
            # Get top partners
            sleep_top = other_neurons[sleep_order[:n_top]]
            wake_top = other_neurons[wake_order[:n_top]]
            
            # Calculate overlap ratio
            common = set(sleep_top) & set(wake_top)
            overlap_ratios[neuron_i] = len(common) / n_top
            
            # Create rank dictionaries
            sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
            wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
            
            # Calculate average rank shift for common partners
            if common:
                shifts = [abs(sleep_ranks[p] - wake_ranks[p]) for p in common]
                avg_rank_shifts[neuron_i] = np.mean(shifts)
            else:
                avg_rank_shifts[neuron_i] = np.nan
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with color based on keeper score
        scatter = plt.scatter(overlap_ratios, avg_rank_shifts, 
                            c=keeper_scores, cmap='viridis', alpha=0.7)
        
        # Highlight top keepers and switchers
        plt.scatter(overlap_ratios[top_keepers], avg_rank_shifts[top_keepers], 
                   color='green', s=100, alpha=0.8, label='Top Keepers')
        plt.scatter(overlap_ratios[top_switchers], avg_rank_shifts[top_switchers], 
                   color='red', s=100, alpha=0.8, label='Top Switchers')
        
        plt.colorbar(scatter, label='Keeper Score')
        plt.xlabel('Partner Overlap Ratio')
        plt.ylabel('Average Rank Shift')
        plt.title('Partner Overlap vs Rank Shifts')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overlap_vs_rank_shift.png'),
                   dpi=300, bbox_inches='tight')
        
        plt.close('all')
    
    # Return comprehensive results
    results = {
        'keeper_scores': keeper_scores,
        'keeper_ranking': keeper_ranking,
        'top_keepers': top_keepers,
        'top_switchers': top_switchers,
        'full_ranking': {
            'neuron_ids': np.arange(n_neurons)[keeper_ranking],
            'keeper_scores': keeper_scores[keeper_ranking]
        }
    }
    
    return results