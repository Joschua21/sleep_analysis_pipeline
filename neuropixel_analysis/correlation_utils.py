import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve, savgol_filter
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
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
from neuropixel_utils import filter_clusters_by_quality



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
    Fixed version with proper sleep period overlay and updated visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
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
    
    # === UPDATED VISUALIZATION ===
    fig, ax = plt.subplots(figsize=(14, 5))  # Made shorter in Y dimension like PC plot
    
    # Plot correlation over time with thinner line
    ax.plot(window_times, mean_correlations, 'k-', linewidth=0.5)
    
    # Add sleep periods with proper y-range handling
    if len(sleep_periods) > 0:
        print(f"Adding {len(sleep_periods)} sleep periods to plot...")
        
        for sleep_start, sleep_end in sleep_periods:
            ax.axvspan(sleep_start, sleep_end, 
                      ymin=0, ymax=1, color='blue', alpha=0.2)  # ymin/ymax relative to axes
        
        print(f"Sleep periods plotted: {len(sleep_periods)} spans")
    else:
        print("Warning: No sleep periods provided")
    
    # Apply formatting changes (matching PC plot style)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # Only 4 y labels
    ax.set_xlabel('Time (s)', fontsize=26)
    ax.set_ylabel('Mean Population\nCorrelation', fontsize=26)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(left=0)
    
    # Fix the X label cutoff issue
    plt.tight_layout()
    
    # Save with descriptive filename
    if output_dir:
        synchrony_filename = f"{output_dir}/population_synchrony_over_time.png"
        plt.savefig(synchrony_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {synchrony_filename}")
    
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
            npz_data = np.load(file_path, allow_pickle=True)
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


def correlate_with_population_average(spike_data, lag_range_ms=100, lag_resolution_ms=1, 
                                    batch_size=100, min_firing_rate=0.001):
    """
    Correlate each neuron's activity with the population average (excluding itself) across lags.
    
    Parameters:
        spike_data: Neural activity matrix (n_neurons x n_time_bins)
        lag_range_ms: Maximum lag in milliseconds (creates range from -lag_range to +lag_range)
        lag_resolution_ms: Step size for lags in milliseconds
        batch_size: Number of neurons to process at once (for memory efficiency)
        min_firing_rate: Minimum firing rate threshold to include neurons
    
    Returns:
        Dictionary with correlation results for each neuron
    """
    from scipy.signal import correlate
    import numpy as np
    from tqdm import tqdm
    
    print(f"Input data shape: {spike_data.shape}")
    n_neurons, n_time_bins = spike_data.shape
    
    # Filter by firing rate
    firing_rates = np.mean(spike_data, axis=1)
    active_mask = firing_rates > min_firing_rate
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    
    print(f"Using {n_active}/{n_neurons} neurons (firing rate > {min_firing_rate} Hz)")
    
    if n_active == 0:
        print("No active neurons found!")
        return {}
    
    # Pre-compute total population activity
    total_activity = np.sum(spike_data[active_indices], axis=0)
    print(f"Total population activity computed: {total_activity.shape}")
    
    # Setup lag parameters
    max_lag_bins = int(lag_range_ms / lag_resolution_ms)
    lag_offsets = np.arange(-max_lag_bins, max_lag_bins + 1) * int(lag_resolution_ms)
    n_lags = len(lag_offsets)
    
    print(f"Lag parameters: ±{lag_range_ms}ms range, {lag_resolution_ms}ms resolution")
    print(f"Total lags to compute: {n_lags}")
    
    # Initialize results storage
    neuron_correlations = np.zeros((n_active, n_lags))
    peak_correlations = np.zeros(n_active)
    peak_lags = np.zeros(n_active)
    
    # Process neurons in batches
    n_batches = int(np.ceil(n_active / batch_size))
    print(f"Processing {n_active} neurons in {n_batches} batches of {batch_size}")
    
    with tqdm(total=n_active, desc="Population correlation") as pbar:
        for batch_idx in range(n_batches):
            # Define batch boundaries
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_active)
            batch_neurons = active_indices[start_idx:end_idx]
            current_batch_size = end_idx - start_idx
            
            # Process each neuron in the batch
            for i, neuron_idx in enumerate(batch_neurons):
                global_neuron_idx = start_idx + i
                
                # Get this neuron's activity
                neuron_activity = spike_data[neuron_idx]
                
                # Compute population activity excluding this neuron
                excluded_population = total_activity - neuron_activity
                
                # Use scipy.signal.correlate for efficient lag correlation
                # 'full' mode gives all possible overlaps
                full_correlation = correlate(neuron_activity, excluded_population, mode='full')
                
                # Extract the lags we want from the full correlation
                # The center of full correlation corresponds to zero lag
                center_idx = len(full_correlation) // 2
                
                # Extract correlations for our desired lag range
                lag_correlations = np.zeros(n_lags)
                
                for lag_idx, lag_offset in enumerate(lag_offsets):
                    lag_bin_offset = int(lag_offset)  # Already in bins since lag_resolution_ms = bin_size
                    
                    # Calculate index in full correlation array
                    corr_idx = center_idx + lag_bin_offset
                    
                    # Check bounds
                    if 0 <= corr_idx < len(full_correlation):
                        # Normalize by the number of overlapping samples
                        if lag_bin_offset >= 0:
                            n_overlap = n_time_bins - lag_bin_offset
                        else:
                            n_overlap = n_time_bins + lag_bin_offset
                        
                        if n_overlap > 0:
                            # Convert to correlation coefficient by normalizing
                            # This is a simplified normalization - for proper correlation coefficient,
                            # we'd need to account for means and standard deviations
                            lag_correlations[lag_idx] = full_correlation[corr_idx] / n_overlap
                        else:
                            lag_correlations[lag_idx] = 0
                    else:
                        lag_correlations[lag_idx] = 0
                
                # Store results for this neuron
                neuron_correlations[global_neuron_idx] = lag_correlations
                
                # Find peak correlation and its lag
                peak_idx = np.argmax(np.abs(lag_correlations))
                peak_correlations[global_neuron_idx] = lag_correlations[peak_idx] 
                peak_lags[global_neuron_idx] = lag_offsets[peak_idx]
                
                pbar.update(1)
    
    # Convert back to normalized correlation coefficients
    # This is an approximation - for exact correlation coefficients we'd need more complex normalization
    print("Converting to correlation coefficients...")
    
    for i in range(n_active):
        neuron_idx = active_indices[i]
        neuron_activity = spike_data[neuron_idx]
        excluded_population = total_activity - neuron_activity
        
        # Calculate standard deviations for normalization
        neuron_std = np.std(neuron_activity)
        pop_std = np.std(excluded_population)
        
        if neuron_std > 0 and pop_std > 0:
            # Normalize the correlation values
            neuron_correlations[i] = neuron_correlations[i] / (neuron_std * pop_std)
        else:
            neuron_correlations[i] = 0
    
    # Update peak correlations after normalization
    for i in range(n_active):
        lag_correlations = neuron_correlations[i]
        peak_idx = np.argmax(np.abs(lag_correlations))
        peak_correlations[i] = lag_correlations[peak_idx]
        peak_lags[i] = lag_offsets[peak_idx]
    
    print(f"✅ Completed correlation analysis")
    print(f"Peak correlation range: {np.min(peak_correlations):.3f} to {np.max(peak_correlations):.3f}")
    print(f"Peak lag range: {np.min(peak_lags):.1f} to {np.max(peak_lags):.1f} ms")
    
    return {
        'neuron_correlations': neuron_correlations,  # Shape: (n_active_neurons, n_lags)
        'peak_correlations': peak_correlations,      # Shape: (n_active_neurons,)
        'peak_lags': peak_lags,                     # Shape: (n_active_neurons,)
        'lag_offsets': lag_offsets,                 # Shape: (n_lags,)
        'active_neuron_indices': active_indices,    # Original indices of active neurons
        'n_active_neurons': n_active,
        'n_lags': n_lags
    }


def correlate_with_population_average_by_state(spike_data, sleep_mask, wake_mask, 
                                              lag_range_ms=100, lag_resolution_ms=1, 
                                              batch_size=100, min_firing_rate=0.001):
    """
    Correlate each neuron's activity with the population average (excluding itself) across lags,
    calculated separately for sleep and wake states.
    
    Parameters:
        spike_data: Neural activity matrix (n_neurons x n_time_bins)
        sleep_mask: Boolean array indicating sleep time bins
        wake_mask: Boolean array indicating wake time bins
        lag_range_ms: Maximum lag in milliseconds (creates range from -lag_range to +lag_range)
        lag_resolution_ms: Step size for lags in milliseconds
        batch_size: Number of neurons to process at once (for memory efficiency)
        min_firing_rate: Minimum firing rate threshold to include neurons
    
    Returns:
        Dictionary with correlation results for each neuron, split by state
    """
    from scipy.signal import correlate
    import numpy as np
    from tqdm import tqdm
    
    print(f"Input data shape: {spike_data.shape}")
    print(f"Sleep bins: {np.sum(sleep_mask)}, Wake bins: {np.sum(wake_mask)}")
    
    n_neurons, n_time_bins = spike_data.shape
    
    # Validate masks
    if len(sleep_mask) != n_time_bins or len(wake_mask) != n_time_bins:
        raise ValueError("Mask lengths must match number of time bins")
    
    if np.sum(sleep_mask) == 0 or np.sum(wake_mask) == 0:
        raise ValueError("Both sleep and wake periods must have at least some time bins")
    
    # Filter by firing rate (using combined data)
    firing_rates = np.mean(spike_data, axis=1)
    active_mask = firing_rates > min_firing_rate
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    
    print(f"Using {n_active}/{n_neurons} neurons (firing rate > {min_firing_rate} Hz)")
    
    if n_active == 0:
        print("No active neurons found!")
        return {}
    
    # Extract sleep and wake data
    sleep_data = spike_data[:, sleep_mask]
    wake_data = spike_data[:, wake_mask]
    
    print(f"Sleep data shape: {sleep_data.shape}")
    print(f"Wake data shape: {wake_data.shape}")
    
    # Pre-compute total population activity for each state
    sleep_total_activity = np.sum(sleep_data[active_indices], axis=0)
    wake_total_activity = np.sum(wake_data[active_indices], axis=0)
    
    # Setup lag parameters
    max_lag_bins = int(lag_range_ms / lag_resolution_ms)
    lag_offsets = np.arange(-max_lag_bins, max_lag_bins + 1) * int(lag_resolution_ms)
    n_lags = len(lag_offsets)
    
    print(f"Lag parameters: ±{lag_range_ms}ms range, {lag_resolution_ms}ms resolution")
    print(f"Total lags to compute: {n_lags}")
    
    # Initialize results storage for both states
    sleep_correlations = np.zeros((n_active, n_lags))
    wake_correlations = np.zeros((n_active, n_lags))
    
    sleep_peak_correlations = np.zeros(n_active)
    wake_peak_correlations = np.zeros(n_active)
    sleep_peak_lags = np.zeros(n_active)
    wake_peak_lags = np.zeros(n_active)
    
    # Process neurons in batches
    n_batches = int(np.ceil(n_active / batch_size))
    print(f"Processing {n_active} neurons in {n_batches} batches of {batch_size}")
    
    def process_state_correlations(state_data, state_total, state_name):
        """Helper function to process correlations for one state"""
        correlations = np.zeros((n_active, n_lags))
        n_time_state = state_data.shape[1]
        
        with tqdm(total=n_active, desc=f"{state_name} correlation") as pbar:
            for batch_idx in range(n_batches):
                # Define batch boundaries
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_active)
                batch_neurons = active_indices[start_idx:end_idx]
                
                # Process each neuron in the batch
                for i, neuron_idx in enumerate(batch_neurons):
                    global_neuron_idx = start_idx + i
                    
                    # Get this neuron's activity for this state
                    neuron_activity = state_data[neuron_idx]
                    
                    # Compute population activity excluding this neuron
                    excluded_population = state_total - neuron_activity
                    
                    # Use scipy.signal.correlate for efficient lag correlation
                    full_correlation = correlate(neuron_activity, excluded_population, mode='full')
                    
                    # Extract the lags we want from the full correlation
                    center_idx = len(full_correlation) // 2
                    
                    # Extract correlations for our desired lag range
                    lag_correlations = np.zeros(n_lags)
                    
                    for lag_idx, lag_offset in enumerate(lag_offsets):
                        lag_bin_offset = int(lag_offset)
                        corr_idx = center_idx + lag_bin_offset
                        
                        # Check bounds
                        if 0 <= corr_idx < len(full_correlation):
                            # Normalize by the number of overlapping samples
                            if lag_bin_offset >= 0:
                                n_overlap = n_time_state - lag_bin_offset
                            else:
                                n_overlap = n_time_state + lag_bin_offset
                            
                            if n_overlap > 0:
                                lag_correlations[lag_idx] = full_correlation[corr_idx] / n_overlap
                            else:
                                lag_correlations[lag_idx] = 0
                        else:
                            lag_correlations[lag_idx] = 0
                    
                    # Store results for this neuron
                    correlations[global_neuron_idx] = lag_correlations
                    pbar.update(1)
        
        # Convert to normalized correlation coefficients
        print(f"Converting {state_name} correlations to correlation coefficients...")
        
        for i in range(n_active):
            neuron_idx = active_indices[i]
            neuron_activity = state_data[neuron_idx]
            excluded_population = state_total - neuron_activity
            
            # Calculate standard deviations for normalization
            neuron_std = np.std(neuron_activity)
            pop_std = np.std(excluded_population)
            
            if neuron_std > 0 and pop_std > 0:
                # Normalize the correlation values
                correlations[i] = correlations[i] / (neuron_std * pop_std)
            else:
                correlations[i] = 0
        
        return correlations
    
    # Process sleep and wake states
    print("Processing SLEEP state...")
    sleep_correlations = process_state_correlations(sleep_data, sleep_total_activity, "Sleep")
    
    print("Processing WAKE state...")
    wake_correlations = process_state_correlations(wake_data, wake_total_activity, "Wake")
    
    # Calculate peak correlations and lags for both states
    for i in range(n_active):
        # Sleep state peaks
        sleep_lag_correlations = sleep_correlations[i]
        sleep_peak_idx = np.argmax(np.abs(sleep_lag_correlations))
        sleep_peak_correlations[i] = sleep_lag_correlations[sleep_peak_idx]
        sleep_peak_lags[i] = lag_offsets[sleep_peak_idx]
        
        # Wake state peaks
        wake_lag_correlations = wake_correlations[i]
        wake_peak_idx = np.argmax(np.abs(wake_lag_correlations))
        wake_peak_correlations[i] = wake_lag_correlations[wake_peak_idx]
        wake_peak_lags[i] = lag_offsets[wake_peak_idx]
    
    print(f"✅ Completed state-specific correlation analysis")
    print(f"Sleep - Peak correlation range: {np.min(sleep_peak_correlations):.3f} to {np.max(sleep_peak_correlations):.3f}")
    print(f"Sleep - Peak lag range: {np.min(sleep_peak_lags):.1f} to {np.max(sleep_peak_lags):.1f} ms")
    print(f"Wake - Peak correlation range: {np.min(wake_peak_correlations):.3f} to {np.max(wake_peak_correlations):.3f}")
    print(f"Wake - Peak lag range: {np.min(wake_peak_lags):.1f} to {np.max(wake_peak_lags):.1f} ms")
    
    return {
        'sleep_correlations': sleep_correlations,
        'wake_correlations': wake_correlations,
        'sleep_peak_correlations': sleep_peak_correlations,
        'wake_peak_correlations': wake_peak_correlations,
        'sleep_peak_lags': sleep_peak_lags,
        'wake_peak_lags': wake_peak_lags,
        'lag_offsets': lag_offsets,
        'active_neuron_indices': active_indices,
        'n_active_neurons': n_active,
        'n_lags': n_lags
    }


# ------------------------VISUALIZATION FUNCTIONS------------------------


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


def plot_state_correlation_matrices(state_corr_results, np_results, max_neurons=None, 
                                  value_range=(-0.7, 0.7), sorting='default', 
                                  save_plots=True, output_folder=None, pca_results=None,
                                  population_corr_results=None):
    """
    Plot correlation matrices for sleep and wake states with different sorting options.
    
    Parameters:
    -----------
    state_corr_results : dict
        Results from analyze_state_specific_correlations
    np_results : dict
        Results from analyze_sleep_wake_activity containing modulation indices
    max_neurons : int, optional
        Maximum number of neurons to plot
    value_range : tuple
        Range for colormap (vmin, vmax)
    sorting : str
        Sorting method: 'default' (by cluster ID/depth), 'MI' (by modulation index), 
        'PC' (by PC1), or 'corr' (by population correlation)
    save_plots : bool
        Whether to save the plots
    output_folder : str
        Directory to save plots
    pca_results : dict, optional
        PCA results containing components for PC sorting
    population_corr_results : dict, optional
        Results from correlate_with_population_average for correlation sorting
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    
    # Determine number of neurons to plot
    n_neurons = sleep_corr.shape[0]
    if max_neurons is not None and n_neurons > max_neurons:
        n_neurons = max_neurons
        print(f"Limiting display to first {max_neurons} neurons")
      # Get sorting indices based on sorting parameter
    if sorting == 'default':
        # Default sorting by cluster ID (depth)
        sort_indices = np.arange(n_neurons)
        sort_label = "Default (Depth)"
        
    elif sorting == 'MI':
        # Sort by modulation index
        if 'merged' in np_results:
            modulation_indices = np_results['merged']['modulation_index'][:n_neurons]
            sort_indices = np.argsort(modulation_indices)  # Low to high MI
            sort_label = "Modulation Index"
        else:
            print("Warning: Modulation indices not found, using default sorting")
            sort_indices = np.arange(n_neurons)
            sort_label = "Default (MI not available)"
            
    elif sorting == 'PC':
        # Sort by PC1 scores (not loadings)
        if pca_results is not None:
            if 'pca_result' in pca_results:
                pc1_scores = pca_results['pca_result'][:, 0]  # PC1 scores for all time points
                # Use mean PC1 score for each neuron (across time)
                # This assumes neurons are rows in the correlation matrix, same order as in combined_matrix
                # We need the mean PC1 activity pattern to sort neurons
                # Actually, let's sort by the first PC1 score as a proxy
                sort_indices = np.argsort(pc1_scores[:n_neurons] if len(pc1_scores) >= n_neurons else pc1_scores)
                sort_label = "PC1 Score"
            else:
                sort_indices = np.arange(n_neurons)
                sort_label = "Default (PCA not available)"
        else:
            print("Warning: PCA results not found, using default sorting")
            sort_indices = np.arange(n_neurons)
            sort_label = "Default (PCA not available)"
            
    elif sorting == 'corr':
        # Sort by population correlation (highest correlation first)
        if population_corr_results is not None:
            peak_correlations = population_corr_results['peak_correlations']
            active_indices = population_corr_results['active_neuron_indices']
            
            # Create a mapping from active neuron indices to their correlation values
            corr_mapping = dict(zip(active_indices, peak_correlations))
            
            # Get correlation values for the neurons we're plotting
            # Assume correlation matrix rows correspond to neurons 0, 1, 2, ... n_neurons-1
            neuron_correlations = np.zeros(n_neurons)
            for i in range(n_neurons):
                if i in corr_mapping:
                    neuron_correlations[i] = corr_mapping[i]
                else:
                    neuron_correlations[i] = 0  # Neurons not in active set get 0 correlation
            
            # Sort by correlation (highest first, so descending order)
            sort_indices = np.argsort(-neuron_correlations)  # Negative for descending
            sort_label = "Population Correlation"
        else:
            print("Warning: Population correlation results not found, using default sorting")
            sort_indices = np.arange(n_neurons)
            sort_label = "Default (Corr not available)"
    else:
        print(f"Warning: Unknown sorting method '{sorting}', using default sorting")
        sort_indices = np.arange(n_neurons)
        sort_label = "Default"
      # Apply sorting to correlation matrices
    sleep_corr_sorted = sleep_corr[sort_indices][:, sort_indices][:n_neurons, :n_neurons]
    wake_corr_sorted = wake_corr[sort_indices][:, sort_indices][:n_neurons, :n_neurons]
    
    # Create the plot - 2-MATRIX LAYOUT (removed difference matrix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Sleep correlation matrix
    im1 = axes[0].imshow(sleep_corr_sorted, cmap='RdBu_r', 
                        vmin=value_range[0], vmax=value_range[1], aspect='equal')
    axes[0].set_title(f'Sleep State Correlations\n(Sorted by {sort_label})')
    axes[0].set_xlabel('Neuron Index (Sorted)')
    axes[0].set_ylabel('Neuron Index (Sorted)')
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    
    # Wake correlation matrix
    im2 = axes[1].imshow(wake_corr_sorted, cmap='RdBu_r', 
                        vmin=value_range[0], vmax=value_range[1], aspect='equal')
    axes[1].set_title(f'Wake State Correlations\n(Sorted by {sort_label})')
    axes[1].set_xlabel('Neuron Index (Sorted)')
    axes[1].set_ylabel('Neuron Index (Sorted)')
    plt.colorbar(im2, ax=axes[1], label='Correlation')

    
    # Print some statistics about the sorting
    if sorting == 'PC':
        if 'pc1_scores' in locals():
            print(f"PC1 score range: {np.min(pc1_scores[:n_neurons]):.3f} to {np.max(pc1_scores[:n_neurons]):.3f}")
    elif sorting == 'MI':
        if 'modulation_indices' in locals():
            print(f"Modulation index range: {np.min(modulation_indices):.3f} to {np.max(modulation_indices):.3f}")
    
    # Overall title
    fig.suptitle(f'State-Specific Correlation Matrices ({n_neurons} neurons, sorted by {sort_label})', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the plot
    if save_plots and output_folder:
        plot_filename = f'state_correlation_matrices_sorted_{sorting.lower()}_{n_neurons}neurons.png'
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"State correlation matrices plot saved: {plot_path}")
    
    plt.show()
    
    return fig


def calculate_modified_rrf_keeper_score(sleep_corr, wake_corr, n_top=150, k=60):
    """
    Calculate keeper score using a modified RRF approach that prioritizes 
    partners that are highly ranked in both states.
    
    Parameters:
        sleep_corr: Sleep correlation matrix
        wake_corr: Wake correlation matrix
        n_top: Number of top partners to consider
        k: Constant in RRF formula (typically 60)
    
    Returns:
        Array of normalized RRF-based keeper scores (0-1 range)
    """
    
    n_neurons = sleep_corr.shape[0]
    rrf_scores = np.zeros(n_neurons)
    
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
        
        # Sort correlation partners by strength (get indices)
        sleep_order = np.argsort(-sleep_corrs)
        wake_order = np.argsort(-wake_corrs)
        
        # Get top partners
        sleep_top_idx = sleep_order[:n_top]
        wake_top_idx = wake_order[:n_top]
        
        sleep_top_partners = other_neurons[sleep_top_idx]
        wake_top_partners = other_neurons[wake_top_idx]
        
        # Find shared partners
        shared_partners = set(sleep_top_partners) & set(wake_top_partners)
        
        # Create rank dictionaries (1-based ranking)
        sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
        wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
        
        # Calculate RRF score using your formula
        score = 0
        for partner in shared_partners:
            # Your formula: 1/(sleep_rank + wake_rank + k)
            sleep_rank = sleep_ranks[partner]
            wake_rank = wake_ranks[partner]
            
            # Only include if in top n_top for both states
            if sleep_rank <= n_top and wake_rank <= n_top:
                score += 1/(sleep_rank + wake_rank + k)
        
        rrf_scores[neuron_i] = score
    
    # Normalization to 0-1 range
    if np.max(rrf_scores) > 0:  # Avoid division by zero
        rrf_scores = rrf_scores / np.max(rrf_scores)
    
    return rrf_scores


def analyze_correlation_partner_stability_rrf(state_corr_results, np_results=None, output_dir=None):
    """
    Analyze partner stability using modified RRF-based keeper scoring.
    
    Formula: sum(1/(sleep_rank + wake_rank + k)) for shared partners
    
    Parameters:
        state_corr_results: Results from analyze_state_specific_correlations
        np_results: Results from analyze_sleep_wake_activity (for modulation index)
        output_dir: Directory to save output plots
    
    Returns:
        Dictionary with RRF-based stability metrics
    """
    
    # Extract correlation matrices
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    
    n_neurons = sleep_corr.shape[0]
    print(f"Analyzing correlation partner stability for {n_neurons} neurons using modified RRF")
    
    # Get modulation index if available
    if np_results is not None and 'merged' in np_results:
        modulation_index = np_results['merged']['modulation_index']
        print("Using sleep/wake modulation index for analysis")
    else:
        modulation_index = None
        print("No modulation index available")
    
    # Calculate the RRF-based keeper score
    n_top = min(150, n_neurons - 2)  # Use top 150 partners or all if fewer
    rrf_scores = calculate_modified_rrf_keeper_score(
        sleep_corr, wake_corr, 
        n_top=n_top,
        k=60  # Standard value in RRF
    )
    
    # Calculate overlap statistics for reference
    overlap_counts = np.zeros(n_neurons)
    avg_rank_sums = np.zeros(n_neurons)
    
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
        
        # Sort correlation partners
        sleep_order = np.argsort(-sleep_corrs)
        wake_order = np.argsort(-wake_corrs)
        
        # Get top partners
        sleep_top_idx = sleep_order[:n_top]
        wake_top_idx = wake_order[:n_top]
        
        sleep_top_partners = other_neurons[sleep_top_idx]
        wake_top_partners = other_neurons[wake_top_idx]
        
        # Find shared partners
        shared_partners = set(sleep_top_partners) & set(wake_top_partners)
        overlap_counts[neuron_i] = len(shared_partners)
        
        # Create rank dictionaries
        sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
        wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
        
        # Calculate average rank sum for shared partners
        if shared_partners:
            rank_sums = [sleep_ranks[p] + wake_ranks[p] for p in shared_partners 
                         if sleep_ranks[p] <= n_top and wake_ranks[p] <= n_top]
            avg_rank_sums[neuron_i] = np.mean(rank_sums) if rank_sums else np.nan
        else:
            avg_rank_sums[neuron_i] = np.nan
    
    # Get the ranking from keeper to switcher
    keeper_ranking = np.argsort(-rrf_scores)  # Highest to lowest
    
    # Identify top keepers and top switchers
    n_top_examples = max(3, min(5, n_neurons // 20))  # Between 3 and 5 examples
    top_keepers = keeper_ranking[:n_top_examples]
    top_switchers = keeper_ranking[-n_top_examples:]
    
    print(f"Top keeper neurons: {top_keepers}")
    print(f"Top switcher neurons: {top_switchers}")
    
    
    # Return results
    results = {
        'rrf_scores': rrf_scores,
        'keeper_ranking': keeper_ranking,
        'top_keepers': top_keepers,
        'top_switchers': top_switchers,
        'overlap_counts': overlap_counts,
        'overlap_ratios': overlap_counts/n_top,
        'avg_rank_sums': avg_rank_sums,
        'full_ranking': {
            'neuron_ids': np.arange(n_neurons)[keeper_ranking],
            'rrf_scores': rrf_scores[keeper_ranking]
        }
    }
    
    return results

def visualize_partner_stability_results(rrf_results, state_corr_results, np_results, results, smoothed_results=None, output_dir=None):
    """
    Create enhanced visualizations for partner stability analysis results.
    
    Parameters:
        rrf_results: Results from analyze_correlation_partner_stability_rrf
        state_corr_results: Results from analyze_state_specific_correlations
        np_results: Results from analyze_sleep_wake_activity
        results: Original results with probe data
        smoothed_results: Optional smoothed band power results
        output_dir: Directory to save output plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from matplotlib.colors import Normalize
    import matplotlib.gridspec as gridspec
    
    # Extract necessary data
    rrf_scores = rrf_results['rrf_scores']
    overlap_ratios = rrf_results['overlap_ratios']
    top_keepers = rrf_results['top_keepers']
    top_switchers = rrf_results['top_switchers']
    
    # Get modulation index
    if 'merged' in np_results and 'modulation_index' in np_results['merged']:
        modulation_index = np_results['merged']['modulation_index']
        sleep_rates = np_results['merged']['sleep_rates']
        wake_rates = np_results['merged']['wake_rates']
        
        # Calculate mean firing rate across session
        if smoothed_results is not None and 'sleep_percentage_ma' in smoothed_results:
            # Use sleep percentage to weight the average
            sleep_percentage = np.mean(smoothed_results['sleep_percentage_ma'])
            mean_rates = (sleep_rates * sleep_percentage) + (wake_rates * (1 - sleep_percentage))
        else:
            # Simple average if sleep percentage not available
            mean_rates = (sleep_rates + wake_rates) / 2
    else:
        modulation_index = None
        mean_rates = None
    
    # Extract correlation matrices for rank comparison
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    
    if output_dir:
        # 1. Plot the histogram of keeper scores
        plt.figure(figsize=(10, 6))
        sns.histplot(rrf_scores, bins=30, kde=True)
        plt.axvline(np.mean(rrf_scores), color='k', linestyle='--',
                   label=f'Mean: {np.mean(rrf_scores):.3f}')
        plt.xlabel('Partner Stability Score')
        plt.ylabel('Count')
        plt.title('Distribution of Partner Stability Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keeper_score_distribution.png'),
                   dpi=300, bbox_inches='tight')
        
        # 2. Scatter plot of stability score vs partner overlap ratio
        plt.figure(figsize=(10, 8))
        
        # Color by state preference
        if modulation_index is not None:
            colors = np.where(modulation_index > 0, 'red', 'blue')
            labels = {
                'red': 'Wake-active neurons',
                'blue': 'Sleep-active neurons'
            }
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                markersize=10, label=label)
                    for color, label in labels.items()]
        else:
            colors = 'gray'
            handles = []
        
        plt.scatter(overlap_ratios, rrf_scores, c=colors, alpha=0.7)
        
        # Highlight top keepers and switchers
        plt.scatter(overlap_ratios[top_keepers], rrf_scores[top_keepers], 
                   color='green', s=100, alpha=0.8, label='Top Keepers')
        plt.scatter(overlap_ratios[top_switchers], rrf_scores[top_switchers], 
                   color='red', s=100, alpha=0.8, label='Top Switchers')
        
        # Set fixed axis limits from 0 to 1
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.xlabel('Partner Overlap Ratio')
        plt.ylabel('Partner Stability Score')
        plt.title('Stability Score vs Partner Overlap')
        plt.legend(handles=handles + [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Top Keepers'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Top Switchers')
        ] if handles else [])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stability_vs_overlap.png'),
                   dpi=300, bbox_inches='tight')
        
        # 3. Scatter plot of keeper score vs modulation index, colored by mean firing rate
        if modulation_index is not None and mean_rates is not None:
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot colored by mean firing rate
            scatter = plt.scatter(modulation_index, rrf_scores, 
                                c=mean_rates, cmap='viridis', alpha=0.7,
                                norm=Normalize(vmin=0, vmax=np.percentile(mean_rates, 95)))
            
            # Highlight top keepers and switchers
            plt.scatter(modulation_index[top_keepers], rrf_scores[top_keepers], 
                       edgecolor='green', s=100, linewidth=2, facecolors='none')
            plt.scatter(modulation_index[top_switchers], rrf_scores[top_switchers], 
                       edgecolor='red', s=100, linewidth=2, facecolors='none')
            
            plt.colorbar(scatter, label='Mean Firing Rate (Hz)')
            plt.xlabel('Sleep/Wake Modulation Index')
            plt.ylabel('Partner Stability Score')
            plt.title('Partner Stability vs Sleep/Wake Modulation')
            plt.xlim(-1, 1)
            plt.ylim(0, 1)
            plt.axvline(0, color='k', linestyle='--', alpha=0.3)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'stability_vs_modulation.png'),
                       dpi=300, bbox_inches='tight')
        
        # 4. Create partner rank comparison plots for top 5 keepers and switchers
        # Set up a 5x2 grid for individual neuron comparisons
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        
        n_neurons = sleep_corr.shape[0]
        n_top = 150  # Number of top partners to consider
        
        # Function to get partners and plot them for one neuron
        def plot_neuron_partners(neuron_idx, ax, title_prefix):
            # Get correlation vectors
            sleep_corrs = sleep_corr[neuron_idx, :]
            wake_corrs = wake_corr[neuron_idx, :]
            
            # Exclude self-correlation
            mask = np.ones(n_neurons, dtype=bool)
            mask[neuron_idx] = False
            sleep_corrs = sleep_corrs[mask]
            wake_corrs = wake_corrs[mask]
            other_neurons = np.where(mask)[0]
            
            # Get partner ranks
            sleep_order = np.argsort(-sleep_corrs)
            wake_order = np.argsort(-wake_corrs)
            
            # Get top partners
            sleep_top = other_neurons[sleep_order[:n_top]]
            wake_top = other_neurons[wake_order[:n_top]]
            
            # Find shared partners
            shared_partners = set(sleep_top) & set(wake_top)
            
            # Create rank dictionaries (1-based ranking)
            sleep_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[sleep_order])}
            wake_ranks = {neuron: rank+1 for rank, neuron in enumerate(other_neurons[wake_order])}
            
            # Get ranks for shared partners
            shared_sleep_ranks = []
            shared_wake_ranks = []
            
            for partner in shared_partners:
                if sleep_ranks[partner] <= n_top and wake_ranks[partner] <= n_top:
                    shared_sleep_ranks.append(sleep_ranks[partner])
                    shared_wake_ranks.append(wake_ranks[partner])
            
            # Plot scatter of ranks
            ax.scatter(shared_sleep_ranks, shared_wake_ranks, color='gray', alpha=0.5)
            
            # Add diagonal line
            ax.plot([1, n_top], [1, n_top], 'k--', alpha=0.5)
            
            # Add limits and labels
            ax.set_xlim(0, n_top)
            ax.set_ylim(0, n_top)
            ax.set_xlabel('Sleep Partner Rank')
            ax.set_ylabel('Wake Partner Rank')
            
            # Add title with neuron info
            if 'merged' in np_results:
                mod_idx = np_results['merged']['modulation_index'][neuron_idx]
                mod_type = "Wake" if mod_idx > 0 else "Sleep"
                mod_str = f", {mod_type}-active (MI={mod_idx:.2f})"
            else:
                mod_str = ""
                
            ax.set_title(f"{title_prefix} #{neuron_idx}{mod_str}\nScore: {rrf_scores[neuron_idx]:.3f}, Shared: {len(shared_partners)}/{n_top}")
            
            return shared_partners, sleep_top, wake_top
        
        # Plot top 5 keepers
        for i in range(5):
            if i < len(top_keepers):
                keeper_idx = top_keepers[i]
                plot_neuron_partners(keeper_idx, axes[i, 0], f"Keeper {i+1}")
            else:
                axes[i, 0].set_visible(False)
        
        # Plot top 5 switchers
        for i in range(5):
            if i < len(top_switchers):
                switcher_idx = top_switchers[i]
                plot_neuron_partners(switcher_idx, axes[i, 1], f"Switcher {i+1}")
            else:
                axes[i, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rank_comparison.png'),
                   dpi=300, bbox_inches='tight')
        
## ------------------------------ PLOT 5 --------------------------------
    # 5. Create activity traces for top keepers and switchers with their partners
    if results is not None:
        # Import the function from neuropixel_utils
        from neuropixel_utils import filter_clusters_by_quality
        
        # Extract neural data and time information from results
        valid_probes = []
        for probe in results:
            if 'sleep_bout_mapping' not in results[probe]:
                continue
            if 'cluster_quality' not in results[probe]:
                continue
            valid_probes.append(probe)
        
        if valid_probes:
            # Use first probe for time bins and sleep mapping
            reference_probe = valid_probes[0]
            neural_time_bins = results[reference_probe]['time_bins']
            sleep_bout_mapping = results[reference_probe]['sleep_bout_mapping']
            
            # Collect and merge neural data (same as analyze_sleep_wake_activity)
            all_counts = []
            all_cluster_ids = []
            all_min_length = float('inf')
            
            # First pass: determine minimum length
            for probe in valid_probes:
                counts, cluster_ids, quality_mask = filter_clusters_by_quality(
                    results, probe, include_qualities=['good', 'mua']
                )
                if counts.shape[0] > 0:
                    all_min_length = min(all_min_length, counts.shape[1])
                    all_counts.append(counts)
                    all_cluster_ids.extend([(probe, cid) for cid in cluster_ids])
            
            # Second pass: truncate and merge
            if all_counts:
                for i in range(len(all_counts)):
                    all_counts[i] = all_counts[i][:, :all_min_length]
                
                merged_neural_counts = np.vstack(all_counts)
                neural_time_bins = neural_time_bins[:all_min_length]
                
                # Get correlation matrices
                sleep_corr = state_corr_results['sleep_correlation_matrix']
                wake_corr = state_corr_results['wake_correlation_matrix']
                
                # Get modulation indices
                modulation_indices = np_results['merged']['modulation_index']
                
                # Create 2x2 subplot grid
                fig, axes = plt.subplots(2, 2, figsize=(20, 12))
                axes = axes.flatten()
                
                # Function to plot activity heatmap
                def plot_activity_heatmap(ax, neuron_indices, title_prefix, main_neuron_idx, 
                        sleep_partners=None, wake_partners=None):
                    # Create activity matrix for selected neurons
                    activity_matrix = merged_neural_counts[neuron_indices, :]
                    
                    # Normalize each neuron's activity by its 95th percentile
                    normalized_matrix = np.zeros_like(activity_matrix, dtype=float)
                    for i, neuron_idx in enumerate(neuron_indices):
                        activity = activity_matrix[i, :]
                        p95 = np.percentile(activity, 95)
                        if p95 > 0:
                            normalized_matrix[i, :] = activity / p95
                        else:
                            normalized_matrix[i, :] = activity
                    
                    # Plot heatmap
                    im = ax.matshow(normalized_matrix, aspect='auto', cmap='Greys', 
                                interpolation='nearest', vmin=0, vmax=1)
                    
                    # Add sleep indicator line ABOVE the heatmap
                    # Only plot blue line during sleep periods
                    for _, bout in sleep_bout_mapping.iterrows():
                        if bout['in_range']:
                            start_bin = np.searchsorted(neural_time_bins, bout['start_bin_time'])
                            end_bin = np.searchsorted(neural_time_bins, bout['end_bin_time'])
                            # Plot blue line above the heatmap (negative y values are above when y-axis is inverted)
                            ax.plot(range(start_bin, end_bin), 
                                [-0.65] * (end_bin - start_bin),  # Position well above heatmap
                                color='blue', linewidth=4, solid_capstyle='butt')
                    
                    # Set time axis
                    time_ticks = np.linspace(0, len(neural_time_bins)-1, 6).astype(int)
                    time_labels = [f"{neural_time_bins[i]:.0f}s" for i in time_ticks]
                    ax.set_xticks(time_ticks)
                    ax.set_xticklabels(time_labels)
                    
                    # Set neuron labels with SMI and color coding
                    neuron_labels = []
                    label_colors = []
                    
                    for i, neuron_idx in enumerate(neuron_indices):
                        smi = modulation_indices[neuron_idx]
                        
                        if neuron_idx == main_neuron_idx:
                            label = f"#{neuron_idx} (main)\nSMI={smi:.2f}"
                            label_colors.append('black')
                        else:
                            label = f"#{neuron_idx}\nSMI={smi:.2f}"
                            
                            # Color code for switcher partners
                            if sleep_partners is not None and wake_partners is not None:
                                if neuron_idx in sleep_partners and neuron_idx in wake_partners:
                                    label_colors.append('purple')  # Both sleep and wake partner
                                elif neuron_idx in sleep_partners:
                                    label_colors.append('blue')    # Sleep partner
                                elif neuron_idx in wake_partners:
                                    label_colors.append('orange')  # Wake partner
                                else:
                                    label_colors.append('black')   # Other
                            else:
                                label_colors.append('black')       # Keeper partners
                        
                        neuron_labels.append(label)
                    
                    ax.set_yticks(range(len(neuron_indices)))
                    ax.set_yticklabels(neuron_labels, fontsize=8)
                    
                    # Color the y-axis labels
                    for i, (tick, color) in enumerate(zip(ax.get_yticklabels(), label_colors)):
                        tick.set_color(color)
                    
                    ax.set_title(title_prefix, fontsize=10)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Neurons')
                    
                    # Adjust y-axis limits to accommodate the sleep indicator above the heatmap
                    ax.set_ylim(-0.8, len(neuron_indices) - 0.5)  # Extend upper limit for sleep line
                    
                    # Invert y-axis so sleep line is at top, main neuron next, then partners
                    ax.invert_yaxis()
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Normalized Activity', fontsize=8)
                    
                    return im
                
                # Plot top 2 keepers with their top 4 partners
                n_keepers_to_plot = min(2, len(top_keepers))
                for i in range(n_keepers_to_plot):
                    keeper_idx = top_keepers[i]
                    
                    # Get top 4 correlation partners (using average of sleep and wake correlations)
                    avg_corr = (sleep_corr[keeper_idx, :] + wake_corr[keeper_idx, :]) / 2
                    avg_corr[keeper_idx] = -np.inf  # Exclude self
                    top_partner_indices = np.argsort(-avg_corr)[:4]
                    
                    # Put keeper neuron first, then partners
                    neurons_to_plot = [keeper_idx] + list(top_partner_indices)
                    
                    plot_activity_heatmap(
                        axes[i], 
                        neurons_to_plot, 
                        f"Keeper #{i+1} (Neuron {keeper_idx})\nwith Top 4 Partners\nScore: {rrf_scores[keeper_idx]:.3f}",
                        keeper_idx
                    )
                
                # Plot top 2 switchers with their state-specific partners
                n_switchers_to_plot = min(2, len(top_switchers))
                for i in range(n_switchers_to_plot):
                    switcher_idx = top_switchers[i]
                    
                    # Get top 2 sleep partners and top 2 wake partners
                    sleep_corr_switcher = sleep_corr[switcher_idx, :].copy()
                    wake_corr_switcher = wake_corr[switcher_idx, :].copy()
                    
                    # Exclude self-correlation
                    sleep_corr_switcher[switcher_idx] = -np.inf
                    wake_corr_switcher[switcher_idx] = -np.inf
                    
                    # Get top partners for each state
                    top_sleep_partners = np.argsort(-sleep_corr_switcher)[:2]
                    top_wake_partners = np.argsort(-wake_corr_switcher)[:2]
                    
                    # Combine all partners (remove duplicates but preserve order)
                    all_partners = []
                    all_partners.extend(top_sleep_partners)
                    for partner in top_wake_partners:
                        if partner not in all_partners:
                            all_partners.append(partner)
                    
                    # Put switcher neuron first, then partners
                    neurons_to_plot = [switcher_idx] + all_partners
                    
                    # Create title with partner information
                    sleep_partner_str = ", ".join([str(p) for p in top_sleep_partners])
                    wake_partner_str = ", ".join([str(p) for p in top_wake_partners])
                    title = (f"Switcher #{i+1} (Neuron {switcher_idx})\n"
                            f"Sleep partners: {sleep_partner_str}\nWake partners: {wake_partner_str}\n"
                            f"Score: {rrf_scores[switcher_idx]:.3f}")
                    
                    plot_activity_heatmap(
                        axes[i + 2], 
                        neurons_to_plot, 
                        title,
                        switcher_idx,
                        sleep_partners=top_sleep_partners,
                        wake_partners=top_wake_partners
                    )
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'neural_activity_heatmaps_keepers_switchers.png'),
                        dpi=300, bbox_inches='tight')
                plt.show()
                
            else:
                print("Warning: No valid neural data found for activity traces")
        else:
            print("Warning: No valid probes found for activity traces")
    else:
        print("Warning: results parameter not provided for activity traces")
    
    return {
        'completed': True,
        'output_dir': output_dir
    }
def plot_individual_correlograms(population_corr_results, n_random=10, specific_neuron_ids=None, 
                               output_folder=None, filter_hz=None, figsize=(15, 10)):
    """
    Plot individual cross-correlograms with z-scored center of mass marked.
    
    Parameters:
        population_corr_results: Results from correlate_with_population_average
        n_random: Number of random neurons to plot (if specific_neuron_ids not provided)
        specific_neuron_ids: List of specific neuron IDs to plot
        output_folder: Directory to save plot
        filter_hz: Low-pass filter frequency in Hz (None for no filtering)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import butter, filtfilt
    
    # Extract data
    neuron_correlations = population_corr_results['neuron_correlations']  # (n_neurons, n_lags)
    lag_offsets = population_corr_results['lag_offsets']
    peak_correlations = population_corr_results['peak_correlations']
    peak_lags = population_corr_results['peak_lags']
    active_indices = population_corr_results['active_neuron_indices']
    
    n_neurons, n_lags = neuron_correlations.shape
    
    # Apply low-pass filtering if requested
    filtered_correlations = neuron_correlations.copy()
    
    if filter_hz is not None:
        print(f"Applying {filter_hz} Hz low-pass filter to correlograms...")
        
        # Calculate sampling rate from lag_offsets
        lag_step = lag_offsets[1] - lag_offsets[0]  # Step size in ms
        fs = 1000.0 / lag_step  # Sampling rate in Hz
        
        # Design low-pass filter
        nyquist = fs / 2
        if filter_hz >= nyquist:
            print(f"Warning: Filter frequency ({filter_hz} Hz) >= Nyquist frequency ({nyquist:.1f} Hz). No filtering applied.")
            filter_hz = None  # Disable filtering
        else:
            # Butterworth low-pass filter
            b, a = butter(N=4, Wn=filter_hz/nyquist, btype='low')
            
            # Apply filter to each neuron's correlogram
            for i in range(n_neurons):
                try:
                    filtered_correlations[i, :] = filtfilt(b, a, neuron_correlations[i, :])
                except Exception as e:
                    print(f"Warning: Filtering failed for neuron {i}: {e}")
                    filtered_correlations[i, :] = neuron_correlations[i, :]
      # Apply z-scoring to each neuron's correlogram and calculate center of mass
    z_scored_correlations = np.zeros_like(filtered_correlations)
    center_of_mass = np.zeros(n_neurons)
    
    for i in range(n_neurons):
        correlogram = filtered_correlations[i, :]
        
        # Apply z-scoring
        mean_corr = np.mean(correlogram)
        std_corr = np.std(correlogram)
        
        if std_corr > 0:
            z_scored_correlations[i, :] = (correlogram - mean_corr) / std_corr
        else:
            # If std is 0 (flat correlogram), keep as zeros
            z_scored_correlations[i, :] = np.zeros_like(correlogram)
        
        # Calculate center of mass using z-scored correlogram
        z_correlogram = z_scored_correlations[i, :]
        
        # Use all positive z-scores for center of mass calculation
        positive_mask = z_correlogram > 0
        
        if np.sum(positive_mask) > 0 and np.sum(z_correlogram[positive_mask]) > 0:
            center_of_mass[i] = np.average(lag_offsets[positive_mask], weights=z_correlogram[positive_mask])
        else:
            # Fallback to peak position of original correlogram
            peak_idx = np.argmax(correlogram)
            center_of_mass[i] = lag_offsets[peak_idx]
    
    # Select neurons to plot
    if specific_neuron_ids is not None:
        # Find indices for specific neuron IDs
        plot_indices = []
        plot_neuron_ids = []
        for neuron_id in specific_neuron_ids:
            idx = np.where(active_indices == neuron_id)[0]
            if len(idx) > 0:
                plot_indices.append(idx[0])
                plot_neuron_ids.append(neuron_id)
            else:
                print(f"Warning: Neuron ID {neuron_id} not found in active neurons")
        
        if len(plot_indices) == 0:
            print("Error: No valid neuron IDs found")
            return
            
        plot_indices = np.array(plot_indices)
        plot_neuron_ids = np.array(plot_neuron_ids)
        
    else:
        # Select random neurons
        if n_random > n_neurons:
            n_random = n_neurons
            print(f"Reduced n_random to {n_random} (total available neurons)")
        
        plot_indices = np.random.choice(n_neurons, size=n_random, replace=False)
        plot_neuron_ids = active_indices[plot_indices]
    
    n_to_plot = len(plot_indices)
    
    # Calculate subplot layout
    n_cols = min(5, n_to_plot)
    n_rows = int(np.ceil(n_to_plot / n_cols))
    
    # Create the plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Plot each selected neuron
    for i, (idx, neuron_id) in enumerate(zip(plot_indices, plot_neuron_ids)):
        ax = axes[i]
          # Get the correlogram (original, filtered, and z-scored)
        correlogram = filtered_correlations[idx, :]
        z_correlogram = z_scored_correlations[idx, :]
        original_correlogram = neuron_correlations[idx, :]
        
        # Plot the correlogram and z-scored version
        if filter_hz is not None:
            # Plot original (light), filtered (medium), and z-scored (dark)
            ax.plot(lag_offsets, original_correlogram, 'lightgray', alpha=0.7, linewidth=1, label='Original')
            ax.plot(lag_offsets, correlogram, 'blue', alpha=0.7, linewidth=1.5, label=f'Filtered ({filter_hz}Hz)')
            
            # Plot z-scored on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(lag_offsets, z_correlogram, 'red', linewidth=2, label='Z-scored')
            ax2.set_ylabel('Z-scored Correlation', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        else:
            # Plot original and z-scored
            ax.plot(lag_offsets, correlogram, 'blue', alpha=0.7, linewidth=1.5, label='Original')
            
            # Plot z-scored on secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(lag_offsets, z_correlogram, 'red', linewidth=2, label='Z-scored')
            ax2.set_ylabel('Z-scored Correlation', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Mark z-scored center of mass with red dot on z-scored curve
        com_value = center_of_mass[idx]
        com_idx = np.argmin(np.abs(lag_offsets - com_value))
        com_z_correlation = z_correlogram[com_idx]
        ax2.plot(com_value, com_z_correlation, 'ro', markersize=6, label=f'Z-scored CoM', zorder=10)
        
        # Mark peak correlation with smaller marker for comparison on original scale
        peak_corr = peak_correlations[idx]
        peak_lag = peak_lags[idx]
        peak_idx = np.argmin(np.abs(lag_offsets - peak_lag))
        peak_correlation_value = original_correlogram[peak_idx]
        ax.plot(peak_lag, peak_correlation_value, '^k', markersize=4, label='Peak', zorder=10)        
        # Add zero lag line and reference lines
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.3)  # Zero line for z-scored data
        
        # Formatting
        ax.set_xlabel('Lag (ms)')
        ax.set_ylabel('Correlation')
        
        # Create title with both CoM and peak info
        title = f'Neuron {neuron_id}\n'
        title += f'Z-scored CoM: {com_value:.1f}ms ({com_z_correlation:.3f})\n'
        title += f'Peak: {peak_lag:.1f}ms ({peak_corr:.3f})'
        ax.set_title(title, fontsize=10)
        
        # Add legends for first plot
        if i == 0:
            ax.legend(fontsize=8, loc='upper left')
            ax2.legend(fontsize=8, loc='upper right')        
        # Set consistent y-limits across all plots
        ax.set_ylim([np.min(neuron_correlations) * 1.1, np.max(neuron_correlations) * 1.1])
        ax2.set_ylim([np.min(z_scored_correlations) * 1.1, np.max(z_scored_correlations) * 1.1])
    
    # Hide unused subplots
    for i in range(n_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    # Overall title
    if filter_hz is not None:
        suptitle = f'Individual Cross-Correlograms with Population\n(Low-pass filtered at {filter_hz} Hz)\nRed dot = Z-scored Center of Mass, Black triangle = Peak'
    else:
        suptitle = f'Individual Cross-Correlograms with Population\nRed dot = Z-scored Center of Mass, Black triangle = Peak'
    
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for suptitle
    
    if output_folder:
        if specific_neuron_ids is not None:
            neuron_suffix = f"_neurons_{'_'.join(map(str, specific_neuron_ids))}"
        else:
            neuron_suffix = f"_random_{n_random}"
        
        filter_suffix = f"_filtered_{filter_hz}Hz" if filter_hz is not None else "_unfiltered"
        filename = f"individual_correlograms{neuron_suffix}{filter_suffix}.png"
        plt.savefig(f"{output_folder}/{filename}", dpi=300, bbox_inches='tight')
        print(f"Saved individual correlograms to {filename}")
    
    plt.show()
    
    # Print summary for plotted neurons
    print(f"\n=== INDIVIDUAL CORRELOGRAM ANALYSIS ===")
    print(f"Plotted {n_to_plot} neurons")
    if filter_hz is not None:
        print(f"Applied {filter_hz} Hz low-pass filter")
    
    print(f"\nNeuron details (z-scored center of mass):")
    for i, (idx, neuron_id) in enumerate(zip(plot_indices, plot_neuron_ids)):
        com_value = center_of_mass[idx]
        peak_corr = peak_correlations[idx]
        peak_lag = peak_lags[idx]
        com_peak_diff = abs(com_value - peak_lag)
        
        # Get z-scored statistics
        z_correlogram = z_scored_correlations[idx, :]
        com_idx = np.argmin(np.abs(lag_offsets - com_value))
        com_z_score = z_correlogram[com_idx]
        mean_z_score = np.mean(z_correlogram)
        std_z_score = np.std(z_correlogram)
        
        print(f"  Neuron {neuron_id}: Z-scored CoM = {com_value:.1f}ms (z={com_z_score:.3f}), Peak = {peak_lag:.1f}ms ({peak_corr:.3f}), |CoM-Peak| = {com_peak_diff:.1f}ms")
    
    # Calculate and report average differences
    plotted_com_values = center_of_mass[plot_indices]
    plotted_peak_lags = peak_lags[plot_indices]
    avg_com_peak_diff = np.mean(np.abs(plotted_com_values - plotted_peak_lags))
    
    print(f"\nAverage |Z-scored CoM - Peak Lag| difference: {avg_com_peak_diff:.1f}ms")
    
    return {
        'plotted_neuron_ids': plot_neuron_ids,
        'plotted_indices': plot_indices,
        'center_of_mass': center_of_mass[plot_indices],
        'z_scored_correlations': z_scored_correlations[plot_indices],
        'filtered_correlations': filtered_correlations[plot_indices],
        'peak_correlations': peak_correlations[plot_indices],
        'peak_lags': peak_lags[plot_indices],
        'filter_hz': filter_hz
    }


def plot_peak_lag_stripe_map(population_corr_results, filter_hz=None, output_folder=None, 
                            figsize=(12, 8), max_neurons=None, split_states=False):
    """
    Plot neurons as rows with full cross-correlograms colored by correlation strength.
    Neurons are sorted by center of mass of their z-scored correlation function.
    
    Parameters:
        population_corr_results: Results from correlate_with_population_average OR 
                               correlate_with_population_average_by_state
        filter_hz: Low-pass filter frequency in Hz (None for no filtering)
        output_folder: Directory to save plot
        figsize: Figure size
        max_neurons: Maximum number of neurons to show (for readability)
        split_states: If True, expects state-specific results and plots sleep/wake separately
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from scipy.signal import butter, filtfilt
      # Extract data based on split_states parameter
    if split_states:
        # State-specific data
        sleep_correlations = population_corr_results['sleep_correlations']
        wake_correlations = population_corr_results['wake_correlations']
        lag_offsets = population_corr_results['lag_offsets']
        sleep_peak_correlations = population_corr_results['sleep_peak_correlations']
        wake_peak_correlations = population_corr_results['wake_peak_correlations']
        sleep_peak_lags = population_corr_results['sleep_peak_lags']
        wake_peak_lags = population_corr_results['wake_peak_lags']
        active_indices = population_corr_results['active_neuron_indices']
        
        n_neurons, n_lags = sleep_correlations.shape
        print(f"Processing {n_neurons} neurons with {n_lags} lag points (state-specific)")
        
        # We'll process both sleep and wake data
        states_data = {
            'Sleep': {
                'correlations': sleep_correlations,
                'peak_correlations': sleep_peak_correlations,
                'peak_lags': sleep_peak_lags
            },
            'Wake': {
                'correlations': wake_correlations,
                'peak_correlations': wake_peak_correlations,
                'peak_lags': wake_peak_lags
            }
        }
    else:
        # Regular combined data
        neuron_correlations = population_corr_results['neuron_correlations']  # (n_neurons, n_lags)
        lag_offsets = population_corr_results['lag_offsets']
        peak_correlations = population_corr_results['peak_correlations']
        peak_lags = population_corr_results['peak_lags']
        active_indices = population_corr_results['active_neuron_indices']
        
        n_neurons, n_lags = neuron_correlations.shape
        print(f"Processing {n_neurons} neurons with {n_lags} lag points")
        
        # Wrap in similar structure for consistent processing
        states_data = {
            'Combined': {
                'correlations': neuron_correlations,
                'peak_correlations': peak_correlations,
                'peak_lags': peak_lags
            }
        }
      # Apply low-pass filtering if requested
    filtered_states_data = {}
    
    if filter_hz is not None:
        print(f"Applying {filter_hz} Hz low-pass filter...")
        
        # Calculate sampling rate from lag_offsets
        lag_step = lag_offsets[1] - lag_offsets[0]  # Step size in ms
        fs = 1000.0 / lag_step  # Sampling rate in Hz
        
        # Design low-pass filter
        nyquist = fs / 2
        if filter_hz >= nyquist:
            print(f"Warning: Filter frequency ({filter_hz} Hz) >= Nyquist frequency ({nyquist:.1f} Hz). No filtering applied.")
            filtered_states_data = states_data
        else:
            # Butterworth low-pass filter
            b, a = butter(N=4, Wn=filter_hz/nyquist, btype='low')
            
            # Apply filter to each state's data
            for state_name, state_info in states_data.items():
                filtered_correlations = state_info['correlations'].copy()
                
                # Apply filter to each neuron's correlogram
                for i in range(n_neurons):
                    try:
                        filtered_correlations[i, :] = filtfilt(b, a, state_info['correlations'][i, :])
                    except Exception as e:
                        print(f"Warning: Filtering failed for {state_name} neuron {i}: {e}")
                        # Keep original if filtering fails
                        filtered_correlations[i, :] = state_info['correlations'][i, :]
                
                filtered_states_data[state_name] = {
                    'correlations': filtered_correlations,
                    'peak_correlations': state_info['peak_correlations'],
                    'peak_lags': state_info['peak_lags']
                }
            
            print(f"Low-pass filtering completed at {filter_hz} Hz")
    else:
        filtered_states_data = states_data    
    # Process each state separately (z-scoring and center of mass)
    processed_states = {}
    
    for state_name, state_info in filtered_states_data.items():
        print(f"Processing {state_name} state...")
        
        # Apply z-scoring to each neuron's correlogram for this state
        z_scored_correlations = np.zeros_like(state_info['correlations'])
        
        for i in range(n_neurons):
            correlogram = state_info['correlations'][i, :]
            mean_corr = np.mean(correlogram)
            std_corr = np.std(correlogram)
            
            if std_corr > 0:
                z_scored_correlations[i, :] = (correlogram - mean_corr) / std_corr
            else:
                # If std is 0 (flat correlogram), keep as zeros
                z_scored_correlations[i, :] = np.zeros_like(correlogram)
        
        # Calculate center of mass for each neuron's z-scored correlogram
        center_of_mass = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            z_correlogram = z_scored_correlations[i, :]
            
            # Use all positive z-scores for center of mass calculation
            positive_mask = z_correlogram > 0
            
            if np.sum(positive_mask) > 0 and np.sum(z_correlogram[positive_mask]) > 0:
                center_of_mass[i] = np.average(lag_offsets[positive_mask], weights=z_correlogram[positive_mask])
            else:
                # Fallback to peak position of filtered correlogram
                peak_idx = np.argmax(state_info['correlations'][i, :])
                center_of_mass[i] = lag_offsets[peak_idx]
        
        processed_states[state_name] = {
            'z_scored_correlations': z_scored_correlations,
            'center_of_mass': center_of_mass,
            'peak_correlations': state_info['peak_correlations'],
            'peak_lags': state_info['peak_lags'],
            'filtered_correlations': state_info['correlations']
        }
        
        print(f"{state_name} - Center of mass range (z-scored): {np.min(center_of_mass):.1f} to {np.max(center_of_mass):.1f} ms")    
    # Handle neuron selection and sorting based on split_states parameter
    if split_states:
        # For split states: sort each state independently by its own center of mass
        # But first handle max_neurons selection using the first state
        primary_state = list(processed_states.keys())[0]
        primary_center_of_mass = processed_states[primary_state]['center_of_mass']
        
        # Limit number of neurons for readability (using primary state for selection)
        if max_neurons and n_neurons > max_neurons:
            # Sample neurons evenly across the center of mass range
            sort_indices_temp = np.argsort(primary_center_of_mass)
            step = n_neurons // max_neurons
            selected_indices = sort_indices_temp[::step][:max_neurons]
            
            # Apply selection to all states
            for state_name in processed_states:
                processed_states[state_name]['z_scored_correlations'] = processed_states[state_name]['z_scored_correlations'][selected_indices]
                processed_states[state_name]['center_of_mass'] = processed_states[state_name]['center_of_mass'][selected_indices]
                processed_states[state_name]['peak_correlations'] = processed_states[state_name]['peak_correlations'][selected_indices]
                processed_states[state_name]['peak_lags'] = processed_states[state_name]['peak_lags'][selected_indices]
            
            active_indices = active_indices[selected_indices]
            n_neurons = len(selected_indices)
            print(f"Showing {n_neurons} neurons (sampled from {len(sort_indices_temp)})")
        
        # Sort each state independently by its own center of mass
        for state_name in processed_states:
            state_center_of_mass = processed_states[state_name]['center_of_mass']
            state_sort_indices = np.argsort(state_center_of_mass)  # Most negative first
            
            # Apply state-specific sorting
            processed_states[state_name]['z_scored_correlations'] = processed_states[state_name]['z_scored_correlations'][state_sort_indices]
            processed_states[state_name]['center_of_mass'] = processed_states[state_name]['center_of_mass'][state_sort_indices]
            processed_states[state_name]['peak_correlations'] = processed_states[state_name]['peak_correlations'][state_sort_indices]
            processed_states[state_name]['peak_lags'] = processed_states[state_name]['peak_lags'][state_sort_indices]
            processed_states[state_name]['sorted_neuron_ids'] = active_indices[state_sort_indices]
        
        print("Each state sorted independently by its own center of mass")
        
    else:
        # For combined state: use single consistent sorting (existing behavior)
        primary_state = list(processed_states.keys())[0]
        primary_center_of_mass = processed_states[primary_state]['center_of_mass']
        
        # Limit number of neurons for readability
        if max_neurons and n_neurons > max_neurons:
            # Sample neurons evenly across the center of mass range
            sort_indices_temp = np.argsort(primary_center_of_mass)
            step = n_neurons // max_neurons
            selected_indices = sort_indices_temp[::step][:max_neurons]
            
            # Apply selection to all states
            for state_name in processed_states:
                processed_states[state_name]['z_scored_correlations'] = processed_states[state_name]['z_scored_correlations'][selected_indices]
                processed_states[state_name]['center_of_mass'] = processed_states[state_name]['center_of_mass'][selected_indices]
                processed_states[state_name]['peak_correlations'] = processed_states[state_name]['peak_correlations'][selected_indices]
                processed_states[state_name]['peak_lags'] = processed_states[state_name]['peak_lags'][selected_indices]
            
            active_indices = active_indices[selected_indices]
            n_neurons = len(selected_indices)
            primary_center_of_mass = primary_center_of_mass[selected_indices]
            print(f"Showing {n_neurons} neurons (sampled from {len(sort_indices_temp)})")
        
        # Sort neurons by center of mass (ascending: most negative first)
        sort_indices = np.argsort(primary_center_of_mass)
        sorted_neuron_ids = active_indices[sort_indices]
        
        # Apply sorting to all states
        for state_name in processed_states:
            processed_states[state_name]['z_scored_correlations'] = processed_states[state_name]['z_scored_correlations'][sort_indices]
            processed_states[state_name]['center_of_mass'] = processed_states[state_name]['center_of_mass'][sort_indices]
            processed_states[state_name]['peak_correlations'] = processed_states[state_name]['peak_correlations'][sort_indices]
            processed_states[state_name]['peak_lags'] = processed_states[state_name]['peak_lags'][sort_indices]
            processed_states[state_name]['sorted_neuron_ids'] = sorted_neuron_ids
      # Create the plot
    n_states = len(processed_states)
    if split_states:
        # For split states: just show the state plots, no side panel
        fig, axes = plt.subplots(1, n_states, figsize=(figsize[0] * 1.2, figsize[1]))
        if n_states == 1:
            main_axes = [axes]
        else:
            main_axes = axes
        side_ax = None
    else:
        # For combined: show main plot + side panel
        fig, (main_ax, side_ax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 1]})
        main_axes = [main_ax]
    
    # Plot each state
    for i, (state_name, state_data) in enumerate(processed_states.items()):
        ax = main_axes[i] if len(main_axes) > 1 else main_axes[0]
        
        correlation_matrix = state_data['z_scored_correlations']
        sorted_center_of_mass = state_data['center_of_mass']
        sorted_peak_correlations = state_data['peak_correlations']
        
        # Calculate color scaling based on the range of z-scored correlation values
        max_abs_z = np.max(np.abs(correlation_matrix))
        min_z = np.min(correlation_matrix)
        max_z = np.max(correlation_matrix)
        
        print(f"{state_name} - Z-scored correlation range: {min_z:.3f} to {max_z:.3f}")
        print(f"{state_name} - Max absolute z-score: {max_abs_z:.3f}")
        
        # Main correlogram heatmap (using z-scored data)
        im = ax.matshow(correlation_matrix, aspect='auto', cmap='inferno', 
                       extent=[lag_offsets[0], lag_offsets[-1], n_neurons, 0],
                       vmin=-max_abs_z, vmax=max_abs_z, interpolation='nearest')
          # Formatting for main plot
        ax.set_xlabel('Lag (ms)')
        if i == 0:  # Only label y-axis for first plot
            ax.set_ylabel('Neurons (z-scored)')
        
        # Create title based on state and filtering
        if split_states:
            title = f'{state_name} Cross-Correlograms'
        else:
            title = 'Cross-Correlograms'
        
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Z-scored Correlation')        
        # Add zero lag line
        ax.axvline(0, color='black', linestyle='-', alpha=0.7, linewidth=1)
    
    # Side plot: Center of mass distribution (only for combined state)
    if not split_states and side_ax is not None:
        primary_state = list(processed_states.keys())[0]
        primary_state_data = processed_states[primary_state]
        colors = ['blue' if corr < 0 else 'red' for corr in primary_state_data['peak_correlations']]
        bars = side_ax.barh(range(n_neurons), primary_state_data['center_of_mass'], height=1, 
                           color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        side_ax.axvline(0, color='black', linestyle='-', alpha=0.8)
        side_ax.set_xlabel('Peak lag (ms)')
        side_ax.set_ylabel('Neurons')
        side_ax.set_title('Center of Mass')
        side_ax.set_ylim(n_neurons, 0)  # Match main plot
    
    plt.tight_layout()
    
    if output_folder:
        filter_suffix = f"_filtered_{filter_hz}Hz" if filter_hz is not None else "_unfiltered"
        state_suffix = "_split_states" if split_states else "_combined"
        plt.savefig(f"{output_folder}/z_scored_cross_correlogram_heatmap{state_suffix}{filter_suffix}.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Z-SCORED CROSS-CORRELOGRAM HEATMAP ANALYSIS ===")
    print(f"Neurons analyzed: {n_neurons}")
    
    for state_name, state_data in processed_states.items():
        sorted_center_of_mass = state_data['center_of_mass']
        sorted_peak_correlations = state_data['peak_correlations']
        correlation_matrix = state_data['z_scored_correlations']
        
        min_z = np.min(correlation_matrix)
        max_z = np.max(correlation_matrix)
        
        print(f"\n{state_name} State:")
        print(f"  Z-scored center of mass range: {np.min(sorted_center_of_mass):.1f} to {np.max(sorted_center_of_mass):.1f} ms")
        print(f"  Mean center of mass: {np.mean(sorted_center_of_mass):.1f} ± {np.std(sorted_center_of_mass):.1f} ms")
        print(f"  Z-scored correlation range: {min_z:.3f} to {max_z:.3f}")
        print(f"  Peak correlation range: {np.min(sorted_peak_correlations):.3f} to {np.max(sorted_peak_correlations):.3f}")
        
        # Count neurons at different center of mass positions
        zero_com_neurons = np.sum(np.abs(sorted_center_of_mass) < 1)  # Within ±1ms of zero
        leading_com_neurons = np.sum(sorted_center_of_mass < -1)  # More than 1ms leading
        lagging_com_neurons = np.sum(sorted_center_of_mass > 1)   # More than 1ms lagging
        
        print(f"  Temporal distribution (z-scored center of mass):")
        print(f"    Leading (< -1ms): {leading_com_neurons} ({leading_com_neurons/n_neurons*100:.1f}%)")
        print(f"    Centered (±1ms): {zero_com_neurons} ({zero_com_neurons/n_neurons*100:.1f}%)")
        print(f"    Lagging (> +1ms): {lagging_com_neurons} ({lagging_com_neurons/n_neurons*100:.1f}%)")
    
    if filter_hz is not None:
        print(f"\nApplied {filter_hz} Hz low-pass filter before analysis")
    
    if split_states:
        print(f"Split states: Yes - Each state sorted independently by its own center of mass")
    else:
        print(f"Split states: No - Combined data sorted by center of mass")


def plot_sleep_vs_wake_correlation_scatter(state_corr_results, max_pairs=None, 
                                         save_plots=True, output_folder=None,
                                         color_by='density', figsize=(10, 8)):
    """
    Create a scatter plot of sleep vs wake correlations for all neuron pairs.
    
    Parameters:
    -----------
    state_corr_results : dict
        Results from analyze_state_specific_correlations containing correlation matrices
    max_pairs : int, optional
        Maximum number of pairs to plot (for performance with large matrices)
    save_plots : bool
        Whether to save the plot
    output_folder : str
        Directory to save the plot
    color_by : str
        'density' for density-based coloring, 'simple' for single color, 
        'magnitude' for correlation magnitude
    figsize : tuple
        Figure size
        
    Returns:
    --------
    dict
        Dictionary containing extracted correlation values and statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import LogNorm
    import os
    
    # Extract correlation matrices
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    
    n_neurons = sleep_corr.shape[0]
    print(f"Extracting correlations for {n_neurons} neurons")
    
    # Extract upper triangular values (unique pairs only)
    upper_tri_mask = np.triu(np.ones_like(sleep_corr, dtype=bool), k=1)
    
    # Get correlation values for each pair
    sleep_correlations = sleep_corr[upper_tri_mask]
    wake_correlations = wake_corr[upper_tri_mask]
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sleep_correlations) | np.isnan(wake_correlations))
    sleep_correlations = sleep_correlations[valid_mask]
    wake_correlations = wake_correlations[valid_mask]
    
    n_valid_pairs = len(sleep_correlations)
    print(f"Valid correlation pairs: {n_valid_pairs}")
    
    # Keep full data for density plot
    sleep_correlations_full = sleep_correlations.copy()
    wake_correlations_full = wake_correlations.copy()
    
    # Subsample if too many pairs for visualization (only for scatter plot)
    if max_pairs and n_valid_pairs > max_pairs:
        sample_indices = np.random.choice(n_valid_pairs, max_pairs, replace=False)
        sleep_correlations = sleep_correlations[sample_indices]
        wake_correlations = wake_correlations[sample_indices]
        print(f"Subsampled to {max_pairs} pairs for scatter plot visualization")
    
    # Calculate difference (sleep - wake) for statistics
    correlation_difference = sleep_correlations_full - wake_correlations_full
    
    # Calculate axis limits and create nice tick values
    min_corr = min(np.min(wake_correlations_full), np.min(sleep_correlations_full))
    max_corr = max(np.max(wake_correlations_full), np.max(sleep_correlations_full))
    
    # Create nice, round tick values
    # Round limits to nearest 0.1 and extend slightly
    min_corr_rounded = np.floor(min_corr * 10) / 10
    max_corr_rounded = np.ceil(max_corr * 10) / 10
    
    # Create exactly 4 nice tick values
    tick_range = max_corr_rounded - min_corr_rounded
    tick_step = tick_range / 3  # 3 intervals = 4 ticks
    
    # Round tick step to nearest nice value
    if tick_step <= 0.2:
        tick_step = 0.2
    elif tick_step <= 0.3:
        tick_step = 0.3
    elif tick_step <= 0.4:
        tick_step = 0.4
    elif tick_step <= 0.5:
        tick_step = 0.5
    else:
        tick_step = np.ceil(tick_step * 10) / 10
    
    # Generate nice tick values
    nice_ticks = []
    current_tick = min_corr_rounded
    while current_tick <= max_corr_rounded:
        nice_ticks.append(current_tick)
        current_tick += tick_step
    
    # Ensure we have exactly 4 ticks, adjust if needed
    if len(nice_ticks) > 4:
        nice_ticks = nice_ticks[:4]
    elif len(nice_ticks) < 4:
        nice_ticks.append(nice_ticks[-1] + tick_step)
    
    nice_ticks = np.array(nice_ticks)
    
    # ======================== FIGURE 1: Scatter Plot (no colorbar) ========================
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    if color_by == 'simple' or color_by == 'density':
        # Use simple blue dots (no colorbar for first figure as requested)
        ax1.scatter(wake_correlations, sleep_correlations, 
                  alpha=0.6, s=1, color='blue')
        
    elif color_by == 'magnitude':
        # Color by average correlation magnitude (but no colorbar)
        avg_magnitude = (np.abs(sleep_correlations) + np.abs(wake_correlations)) / 2
        ax1.scatter(wake_correlations, sleep_correlations, 
                   c=avg_magnitude, cmap='plasma', alpha=0.6, s=1)
    
    # Add diagonal line (x = y, where sleep = wake)
    ax1.plot([min_corr, max_corr], [min_corr, max_corr], 
            'r--', alpha=0.7, linewidth=2, label='Sleep = Wake')
    
    # Clean formatting - no title, no grid, remove top and right spines
    ax1.set_xlabel('Wake Correlation', fontsize=24)
    ax1.set_ylabel('Sleep Correlation', fontsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Set nice tick values
    ax1.set_xticks(nice_ticks)
    ax1.set_yticks(nice_ticks)
    
    # Format tick labels to 1 decimal place
    ax1.set_xticklabels([f'{x:.1f}' for x in nice_ticks])
    ax1.set_yticklabels([f'{y:.1f}' for y in nice_ticks])
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Larger legend font
    ax1.legend(fontsize=14)
    ax1.set_xlim(min_corr, max_corr)
    ax1.set_ylim(min_corr, max_corr)
    ax1.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save first plot if requested
    if save_plots and output_folder:
        plot_filename1 = 'sleep_vs_wake_correlation_scatter.png'
        plot_path1 = os.path.join(output_folder, plot_filename1)
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved: {plot_path1}")
    
    plt.show()
    
    # ======================== FIGURE 2: Hexbin Density Plot with Log Scale ========================
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
    
    # Create hexbin plot using the full dataset with logarithmic normalization
    gridsize = 50  # Adjust this for resolution vs performance
    
    # Create hexbin plot with logarithmic color scaling
    hexbin = ax2.hexbin(wake_correlations_full, sleep_correlations_full, 
                       gridsize=gridsize, cmap='viridis', mincnt=1,
                       extent=[min_corr, max_corr, min_corr, max_corr],
                       norm=LogNorm(vmin=1, vmax=None))  # Log normalization starting from 1
    
    # Add colorbar for hexbin plot
    cbar2 = plt.colorbar(hexbin, ax=ax2)
    cbar2.set_label('Number of Pairs', fontsize=22)
    cbar2.ax.tick_params(labelsize=18)
    
    # Add diagonal line
    ax2.plot([min_corr, max_corr], [min_corr, max_corr], 
            'r--', alpha=0.8, linewidth=2, label='Sleep = Wake')
    
    # Clean formatting - no title, no grid, remove top and right spines
    ax2.set_xlabel('Wake Correlation', fontsize=24)
    ax2.set_ylabel('Sleep Correlation', fontsize=24)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Set nice tick values (same as scatter plot)
    ax2.set_xticks(nice_ticks)
    ax2.set_yticks(nice_ticks)
    ax2.set_xticklabels([f'{x:.1f}' for x in nice_ticks])
    ax2.set_yticklabels([f'{y:.1f}' for y in nice_ticks])
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Larger legend font
    ax2.legend(fontsize=18)
    ax2.set_xlim(min_corr, max_corr)
    ax2.set_ylim(min_corr, max_corr)
    
    plt.tight_layout()
    
    # Save second plot if requested
    if save_plots and output_folder:
        plot_filename2 = 'sleep_vs_wake_correlation_density.png'
        plot_path2 = os.path.join(output_folder, plot_filename2)
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Density plot saved: {plot_path2}")
    
    plt.show()
    
    # Calculate statistics
    sleep_wake_corr = np.corrcoef(sleep_correlations_full, wake_correlations_full)[0, 1]
    
    # Count points above/below diagonal
    above_diagonal = np.sum(sleep_correlations_full > wake_correlations_full)  # Stronger in sleep
    below_diagonal = np.sum(sleep_correlations_full < wake_correlations_full)  # Stronger in wake
    on_diagonal = np.sum(sleep_correlations_full == wake_correlations_full)    # Equal
    
    # Print summary statistics
    print(f"\n=== CORRELATION SCATTER ANALYSIS ===")
    print(f"Total valid pairs: {len(sleep_correlations_full):,}")
    print(f"Correlation between sleep and wake correlations: {sleep_wake_corr:.3f}")
    print(f"Mean correlation difference (sleep - wake): {np.mean(correlation_difference):.4f}")
    print(f"Points above diagonal (stronger in sleep): {above_diagonal:,} ({above_diagonal/len(sleep_correlations_full)*100:.1f}%)")
    print(f"Points below diagonal (stronger in wake): {below_diagonal:,} ({below_diagonal/len(sleep_correlations_full)*100:.1f}%)")
    
    return {
        'sleep_correlations': sleep_correlations_full,
        'wake_correlations': wake_correlations_full,
        'correlation_difference': correlation_difference,
        'sleep_wake_correlation': sleep_wake_corr,
        'n_pairs': len(sleep_correlations_full),
        'above_diagonal': above_diagonal,
        'below_diagonal': below_diagonal,
        'on_diagonal': on_diagonal
    }

def analyze_neuron_correlation_stability(state_corr_results, rrf_results, n_random=10, 
                                         specific_neuron_ids=None, output_folder=None, 
                                         figsize=(15, 10)):
    """
    Analyze correlation stability between sleep and wake states for individual neurons.
    
    For each neuron, we extract its correlation vector with all other neurons in sleep
    and wake states, then calculate the Spearman correlation between these vectors.
    This gives an intuitive measure of how stable a neuron's correlation partners
    are across states, which should correlate with RRF keeper scores.
    
    Parameters:
    -----------
    state_corr_results : dict
        Results from analyze_state_specific_correlations
    rrf_results : dict
        Results from analyze_correlation_partner_stability_rrf
    n_random : int
        Number of random neurons to show in scatter plots (ignored if specific_neuron_ids provided)
    specific_neuron_ids : list or None
        List of specific neuron indices to plot (overrides n_random if provided)
    output_folder : str
        Directory to save plots
    figsize : tuple
        Figure size for plots
        
    Returns:
    --------
    dict
        Dictionary containing stability scores and analysis results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    from matplotlib.ticker import MaxNLocator
    
    # Extract correlation matrices
    sleep_corr = state_corr_results['sleep_correlation_matrix']
    wake_corr = state_corr_results['wake_correlation_matrix']
    rrf_scores = rrf_results['rrf_scores']
    
    n_neurons = sleep_corr.shape[0]
    print(f"Analyzing correlation stability for {n_neurons} neurons")
    
    # Calculate stability scores for each neuron
    stability_scores = np.zeros(n_neurons)
    
    for neuron_i in range(n_neurons):
        # Get correlation vector for this neuron (exclude self-correlation)
        sleep_vector = np.concatenate([sleep_corr[neuron_i, :neuron_i], 
                                     sleep_corr[neuron_i, neuron_i+1:]])
        wake_vector = np.concatenate([wake_corr[neuron_i, :neuron_i], 
                                    wake_corr[neuron_i, neuron_i+1:]])
        
        # Remove any NaN pairs
        valid_mask = ~(np.isnan(sleep_vector) | np.isnan(wake_vector))
        
        if np.sum(valid_mask) > 2:  # Need at least 3 valid pairs for correlation
            stability_scores[neuron_i] = spearmanr(sleep_vector[valid_mask], 
                                                 wake_vector[valid_mask])[0]
        else:
            stability_scores[neuron_i] = np.nan
    
    print(f"Stability scores range: {np.nanmin(stability_scores):.3f} to {np.nanmax(stability_scores):.3f}")
    
    # Select neurons for visualization
    if specific_neuron_ids is not None:
        neurons_to_plot = specific_neuron_ids
        print(f"Plotting specific neurons: {neurons_to_plot}")
    else:
        # Select random neurons for visualization
        valid_neurons = np.where(~np.isnan(stability_scores))[0]
        neurons_to_plot = np.random.choice(valid_neurons, 
                                         size=min(n_random, len(valid_neurons)), 
                                         replace=False)
        print(f"Plotting {len(neurons_to_plot)} random neurons")
    
    # Create two separate figures: 1) Individual plots, 2) Summary plots
    
    # FIGURE 1: Individual neuron correlation plots
    n_individual_plots = len(neurons_to_plot)
    n_cols = min(5, n_individual_plots)  # Max 5 columns for individual plots
    n_rows_individual = int(np.ceil(n_individual_plots / n_cols))
    
    fig1 = plt.figure(figsize=(figsize[0], n_rows_individual * 3))
    
    # Plot individual neurons (one subplot per neuron)
    for i, neuron_idx in enumerate(neurons_to_plot):
        ax = plt.subplot(n_rows_individual, n_cols, i + 1)
        
        # Get this neuron's correlation vectors
        sleep_vector = np.concatenate([sleep_corr[neuron_idx, :neuron_idx], 
                                     sleep_corr[neuron_idx, neuron_idx+1:]])
        wake_vector = np.concatenate([wake_corr[neuron_idx, :neuron_idx], 
                                    wake_corr[neuron_idx, neuron_idx+1:]])
        
        # Remove NaN pairs for plotting
        valid_mask = ~(np.isnan(sleep_vector) | np.isnan(wake_vector))
        sleep_clean = sleep_vector[valid_mask]
        wake_clean = wake_vector[valid_mask]
        
        if len(sleep_clean) > 0:
            # Scatter plot
            ax.scatter(wake_clean, sleep_clean, alpha=0.6, s=8, color='blue')
            
            # Add x=y line (more visible)
            min_val = min(np.min(wake_clean), np.min(sleep_clean))
            max_val = max(np.max(wake_clean), np.max(sleep_clean))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   color='darkgray', linewidth=2, alpha=0.8, linestyle='--')  # More visible line
            
            # Set equal aspect and limits
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect('equal')
            
            # Limit ticks to 3 each
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            
            # Larger tick and axis label fonts
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xlabel('Wake', fontsize=18)
            ax.set_ylabel('Sleep', fontsize=18)
            
            # Remove grid, title, top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Add overall title with proper spacing
    fig1.suptitle('Correlation Preservation across States', 
                  fontsize=14, y=0.98)
    
    # Adjust spacing to prevent overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.0, h_pad=5.0)  # Leave space for suptitle
    
    # Save individual plots if requested
    if output_folder:
        individual_filename = f"{output_folder}/correlation_stability_individual_neurons.png"
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
        print(f"Individual neuron plots saved: {individual_filename}")
    
    plt.show()
    
    # FIGURE 2: Histogram of stability scores only
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Remove NaN values for histogram
    valid_stability = stability_scores[~np.isnan(stability_scores)]
    
    # Create histogram
    ax.hist(valid_stability, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Set x-axis ticks every 0.2
    x_ticks = np.arange(-1.0, 1.2, 0.2)  # From -1 to 1 every 0.2
    ax.set_xticks(x_ticks)
    
    # Set y-axis ticks to multiples of 20
    y_max = ax.get_ylim()[1]
    y_ticks = np.arange(0, int(y_max) + 20, 20)
    ax.set_yticks(y_ticks)
    
    # Larger fonts for ticks and axis labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('Score', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save histogram if requested
    if output_folder:
        hist_filename = f"{output_folder}/correlation_stability_histogram.png"
        plt.savefig(hist_filename, dpi=300, bbox_inches='tight')
        print(f"Stability histogram saved: {hist_filename}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== CORRELATION STABILITY ANALYSIS ===")
    print(f"Neurons analyzed: {n_neurons}")
    print(f"Valid stability scores: {len(valid_stability)}")
    print(f"Stability score statistics:")
    print(f"  Mean: {np.mean(valid_stability):.3f}")
    print(f"  Median: {np.median(valid_stability):.3f}")
    print(f"  Std: {np.std(valid_stability):.3f}")
    print(f"  Range: {np.min(valid_stability):.3f} to {np.max(valid_stability):.3f}")
    
    # Identify most and least stable neurons
    valid_indices = np.where(~np.isnan(stability_scores))[0]
    most_stable_idx = valid_indices[np.argmax(stability_scores[valid_indices])]
    least_stable_idx = valid_indices[np.argmin(stability_scores[valid_indices])]
    
    print(f"\nMost stable neuron:")
    print(f"  Neuron {most_stable_idx}: Stability = {stability_scores[most_stable_idx]:.3f}")
    
    print(f"\nLeast stable neuron:")
    print(f"  Neuron {least_stable_idx}: Stability = {stability_scores[least_stable_idx]:.3f}")
    
    # Count neurons with high stability (top quartile)
    high_stability_threshold = np.percentile(valid_stability, 75)
    high_stability_neurons = np.sum(valid_stability >= high_stability_threshold)
    
    print(f"\nNeurons with high stability (≥75th percentile, {high_stability_threshold:.3f}):")
    print(f"  {high_stability_neurons} neurons ({high_stability_neurons/len(valid_stability)*100:.1f}%)")
    
    # Print info about plotted neurons
    print(f"\nPlotted neurons and their stability scores:")
    for neuron_idx in neurons_to_plot:
        if not np.isnan(stability_scores[neuron_idx]):
            print(f"  Neuron {neuron_idx}: {stability_scores[neuron_idx]:.3f}")
        else:
            print(f"  Neuron {neuron_idx}: NaN (insufficient data)")
    
    return {
        'stability_scores': stability_scores,
        'valid_stability_scores': valid_stability,
        'most_stable_neuron': most_stable_idx,
        'least_stable_neuron': least_stable_idx,
        'high_stability_threshold': high_stability_threshold,
        'n_high_stability': high_stability_neurons,
        'plotted_neurons': neurons_to_plot
    }

def plot_sleep_vs_wake_firing_rate_scatter(np_results, max_neurons=None, 
                                         save_plots=True, output_folder=None,
                                         color_by='density', figsize=(10, 8)):
    """
    Create separate scatter and density plots of sleep vs wake firing rates for all neurons.
    
    Parameters:
    -----------
    np_results : dict
        Results from analyze_sleep_wake_activity containing firing rates
    max_neurons : int, optional
        Maximum number of neurons to plot (for performance with large datasets)
    save_plots : bool
        Whether to save the plot
    output_folder : str
        Directory to save the plot
    color_by : str
        'density' for density-based coloring, 'simple' for single color, 
        'magnitude' for average firing rate
    figsize : tuple
        Figure size
        
    Returns:
    --------
    dict
        Dictionary containing extracted firing rate values and statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    
    # Extract firing rates from the merged results
    if 'merged' not in np_results:
        raise ValueError("No 'merged' results found in np_results. Make sure analyze_sleep_wake_activity was run.")
    
    merged_results = np_results['merged']
    sleep_rates = merged_results['sleep_rates']
    wake_rates = merged_results['wake_rates']
    
    n_neurons = len(sleep_rates)
    print(f"Plotting firing rates for {n_neurons} neurons")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(sleep_rates) | np.isnan(wake_rates))
    sleep_rates_valid = sleep_rates[valid_mask]
    wake_rates_valid = wake_rates[valid_mask]
    
    n_valid_neurons = len(sleep_rates_valid)
    print(f"Valid neurons: {n_valid_neurons}")
    
    # Keep full data for density plot
    sleep_rates_full = sleep_rates_valid.copy()
    wake_rates_full = wake_rates_valid.copy()
    
    # Subsample if too many neurons for visualization (only for scatter plot)
    if max_neurons and n_valid_neurons > max_neurons:
        sample_indices = np.random.choice(n_valid_neurons, max_neurons, replace=False)
        sleep_rates_scatter = sleep_rates_valid[sample_indices]
        wake_rates_scatter = wake_rates_valid[sample_indices]
        print(f"Subsampled to {max_neurons} neurons for scatter plot visualization")
    else:
        sleep_rates_scatter = sleep_rates_valid.copy()
        wake_rates_scatter = wake_rates_valid.copy()
    
    # Calculate difference (sleep - wake) for statistics
    rate_difference = sleep_rates_full - wake_rates_full
    
    # ======================== FIGURE 1: Scatter Plot ========================
    fig1, ax1 = plt.subplots(1, 1, figsize=figsize)
    
    if color_by == 'density':
        # Color by point density (useful for large datasets)
        try:
            # Calculate point density
            xy = np.vstack([wake_rates_scatter, sleep_rates_scatter])
            kde = gaussian_kde(xy)
            density = kde(xy)
            
            scatter = ax1.scatter(wake_rates_scatter, sleep_rates_scatter, 
                               c=density, cmap='viridis', alpha=0.6, s=15)
            cbar1 = plt.colorbar(scatter, ax=ax1)
            cbar1.set_label('Firing Rate (Hz)', fontsize=24)
            cbar1.ax.tick_params(labelsize=20)
        except:
            # Fallback to simple coloring if density calculation fails
            ax1.scatter(wake_rates_scatter, sleep_rates_scatter, 
                      alpha=0.6, s=15, color='blue')
            
    elif color_by == 'magnitude':
        # Color by average firing rate
        avg_rate = (sleep_rates_scatter + wake_rates_scatter) / 2
        scatter = ax1.scatter(wake_rates_scatter, sleep_rates_scatter, 
                           c=avg_rate, cmap='plasma', alpha=0.6, s=15)
        #cbar1 = plt.colorbar(scatter, ax=ax1)
        #cbar1.set_label('Firing Rate (Hz)', fontsize=24)
        #cbar1.ax.tick_params(labelsize=20)
        
    else:  # simple
        ax1.scatter(wake_rates_scatter, sleep_rates_scatter, 
                  alpha=0.6, s=15, color='blue')
    
    # Add diagonal line (x = y, where sleep = wake) - changed to black
    min_rate = min(np.min(wake_rates_full), np.min(sleep_rates_full))
    max_rate = max(np.max(wake_rates_full), np.max(sleep_rates_full))
    ax1.plot([min_rate, max_rate], [min_rate, max_rate], 
            'k-', alpha=0.7, linewidth=2)  # Changed to black ('k-')
    
    # Formatting for scatter plot
    ax1.set_xlabel('Firing Rate (Wake)', fontsize=24)
    ax1.set_ylabel('Firing Rate (Sleep)', fontsize=24)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    
    # Remove title, grid, and top/right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.set_xlim(min(-1, min_rate), max_rate)
    ax1.set_ylim(min(-1, min_rate), max_rate)
    ax1.set_aspect('equal')
    
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=3)) 
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))

    plt.tight_layout()
    
    # Save scatter plot if requested
    if save_plots and output_folder:
        plot_filename1 = 'sleep_vs_wake_firing_rate_scatter.png'
        plot_path1 = os.path.join(output_folder, plot_filename1)
        plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
        print(f"Firing rate scatter plot saved: {plot_path1}")
    
    plt.show()
    
    # ======================== FIGURE 2: Hexbin Density Plot ========================
    fig2, ax2 = plt.subplots(1, 1, figsize=figsize)
    
    # Create hexbin plot using the full dataset
    # gridsize controls resolution (higher = more detail, but slower)
    gridsize = 50  # Adjust this for resolution vs performance
    
    # Create hexbin plot - automatically handles density calculation
    hexbin = ax2.hexbin(wake_rates_full, sleep_rates_full, 
                   gridsize=gridsize, cmap='viridis', mincnt=1,
                   extent=[min_rate, max_rate, min_rate, max_rate])
    
    # Add colorbar for hexbin plot
    cbar2 = plt.colorbar(hexbin, ax=ax2)
    cbar2.set_label('Number of Neurons', fontsize=24)
    cbar2.ax.tick_params(labelsize=20)
    
    # Add diagonal line - changed to black
    ax2.plot([min_rate, max_rate], [min_rate, max_rate], 
            'k-', alpha=0.8, linewidth=2)  # Changed to black ('k-')
    
    # Formatting for density plot
    ax2.set_xlabel('Firing Rate (Wake)', fontsize=24)
    ax2.set_ylabel('Firing Rate (Sleep)', fontsize=24)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    
    # Remove title, grid, and top/right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax2.set_xlim(min(-1, min_rate), max_rate)
    ax2.set_ylim(min(-1, min_rate), max_rate)
    
    plt.tight_layout()
    
    # Save density plot if requested
    if save_plots and output_folder:
        plot_filename2 = 'sleep_vs_wake_firing_rate_density.png'
        plot_path2 = os.path.join(output_folder, plot_filename2)
        plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
        print(f"Firing rate density plot saved: {plot_path2}")
    
    plt.show()
    
    # Calculate statistics
    sleep_wake_corr = np.corrcoef(sleep_rates_full, wake_rates_full)[0, 1]
    
    # Count points above/below diagonal
    above_diagonal = np.sum(sleep_rates_full > wake_rates_full)  # Higher in sleep
    below_diagonal = np.sum(sleep_rates_full < wake_rates_full)  # Higher in wake
    on_diagonal = np.sum(sleep_rates_full == wake_rates_full)    # Equal
    
    # Print summary statistics
    print(f"\n=== FIRING RATE SCATTER ANALYSIS ===")
    print(f"Total valid neurons: {len(sleep_rates_full):,}")
    print(f"Correlation between sleep and wake firing rates: {sleep_wake_corr:.3f}")
    print(f"Mean firing rate difference (sleep - wake): {np.mean(rate_difference):.4f} Hz")
    print(f"Neurons with higher sleep rates: {above_diagonal:,} ({above_diagonal/len(sleep_rates_full)*100:.1f}%)")
    print(f"Neurons with higher wake rates: {below_diagonal:,} ({below_diagonal/len(sleep_rates_full)*100:.1f}%)")
    print(f"Mean sleep firing rate: {np.mean(sleep_rates_full):.4f} Hz")
    print(f"Mean wake firing rate: {np.mean(wake_rates_full):.4f} Hz")
    
    return {
        'sleep_rates': sleep_rates_full,
        'wake_rates': wake_rates_full,
        'rate_difference': rate_difference,
        'sleep_wake_correlation': sleep_wake_corr,
        'n_neurons': len(sleep_rates_full),
        'above_diagonal': above_diagonal,
        'below_diagonal': below_diagonal,
        'on_diagonal': on_diagonal,
        'mean_sleep_rate': np.mean(sleep_rates_full),
        'mean_wake_rate': np.mean(wake_rates_full)
    }


def plot_stability_vs_firing_rate(stability_results, state_corr_results, results, 
                                 output_folder=None, save_plots=True, figsize=(8, 6)):
    """
    Plot stability scores vs average firing rates for entire session, sleep only, and wake only.
    
    Parameters:
    -----------
    stability_results : dict
        Results from analyze_neuron_correlation_stability containing stability scores
    state_corr_results : dict
        Results from analyze_state_specific_correlations 
    results : dict
        Results from process_spike_data containing neural activity data
    output_folder : str, optional
        Directory to save plots
    save_plots : bool
        Whether to save the plots
    figsize : tuple
        Figure size for each plot
        
    Returns:
    --------
    dict
        Dictionary containing firing rate data and correlation statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from matplotlib.ticker import MaxNLocator
    import os
    
    # Extract stability scores
    stability_scores = stability_results['stability_scores']
    
    # Remove NaN values from stability scores
    valid_stability_mask = ~np.isnan(stability_scores)
    valid_stability_scores = stability_scores[valid_stability_mask]
    
    print(f"Valid stability scores: {len(valid_stability_scores)} out of {len(stability_scores)}")
    
    # Collect and merge neural data from all probes
    all_counts = []
    all_time_bins = None
    all_sleep_bouts = None
    
    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if (probe in results and 
            'counts' in results[probe] and 
            'time_bins' in results[probe] and
            'sleep_bout_mapping' in results[probe] and
            'good_mua_cluster_mask' in results[probe]):
            valid_probes.append(probe)
    
    if not valid_probes:
        print("Error: No valid probes found with required data")
        return None
    
    # Use the first probe's time bins and sleep bout info as reference
    reference_probe = valid_probes[0]
    all_time_bins = results[reference_probe]['time_bins']
    all_sleep_bouts = results[reference_probe]['sleep_bout_mapping']
    
    # Find minimum length across probes
    min_length = float('inf')
    for probe in valid_probes:
        probe_length = results[probe]['counts'].shape[1]
        min_length = min(min_length, probe_length)
    
    # Collect data from each probe with quality filtering
    for probe in valid_probes:
        probe_counts = results[probe]['counts']
        quality_mask = results[probe]['good_mua_cluster_mask']
        
        # Apply quality filter and truncate to minimum length
        filtered_counts = probe_counts[quality_mask][:, :min_length]
        all_counts.append(filtered_counts)
    
    # Merge counts from all probes
    if not all_counts:
        print("Error: No neural data found")
        return None
    
    merged_counts = np.vstack(all_counts)
    
    # Truncate time bins and create sleep mask
    time_bins = all_time_bins[:min_length]
    
    # Create sleep mask
    sleep_mask = np.zeros(len(time_bins), dtype=bool)
    for _, bout in all_sleep_bouts.iterrows():
        start_idx = np.searchsorted(time_bins, bout['start_timestamp_s'])
        end_idx = np.searchsorted(time_bins, bout['end_timestamp_s'])
        if start_idx < len(time_bins) and end_idx <= len(time_bins):
            sleep_mask[start_idx:end_idx] = True
    
    wake_mask = ~sleep_mask
    
    print(f"Neural data shape: {merged_counts.shape}")
    print(f"Sleep bins: {np.sum(sleep_mask)}, Wake bins: {np.sum(wake_mask)}")
    
    # Apply validity mask to neural data
    valid_merged_counts = merged_counts[valid_stability_mask]
    
    # Calculate firing rates
    # 1. Entire session average firing rate (Hz)
    bin_size_s = time_bins[1] - time_bins[0]  # Get bin size in seconds
    total_firing_rates = np.mean(valid_merged_counts, axis=1) / bin_size_s
    
    # 2. Sleep-only average firing rate (Hz)
    if np.sum(sleep_mask) > 0:
        sleep_firing_rates = np.mean(valid_merged_counts[:, sleep_mask], axis=1) / bin_size_s
    else:
        sleep_firing_rates = np.zeros(len(valid_stability_scores))
    
    # 3. Wake-only average firing rate (Hz)
    if np.sum(wake_mask) > 0:
        wake_firing_rates = np.mean(valid_merged_counts[:, wake_mask], axis=1) / bin_size_s
    else:
        wake_firing_rates = np.zeros(len(valid_stability_scores))
    
    print(f"Firing rate ranges:")
    print(f"  Total: {np.min(total_firing_rates):.3f} - {np.max(total_firing_rates):.3f} Hz")
    print(f"  Sleep: {np.min(sleep_firing_rates):.3f} - {np.max(sleep_firing_rates):.3f} Hz") 
    print(f"  Wake: {np.min(wake_firing_rates):.3f} - {np.max(wake_firing_rates):.3f} Hz")
    
    # Create the three plots
    plot_data = [
        (total_firing_rates, "Average Firing Rate", "total"),
        (sleep_firing_rates, "Average Firing Rate (Sleep)", "sleep"), 
        (wake_firing_rates, "Average Firing Rate (Wake)", "wake")
    ]
    
    results_dict = {}
    
    for firing_rates, xlabel, condition in plot_data:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create scatter plot
        ax.scatter(firing_rates, valid_stability_scores, alpha=0.6, s=20, color='blue')
        
        # Calculate correlations
        pearson_r, pearson_p = pearsonr(firing_rates, valid_stability_scores)
        spearman_r, spearman_p = spearmanr(firing_rates, valid_stability_scores)
        
        # Set axis labels with large font
        ax.set_xlabel(xlabel, fontsize=24)
        ax.set_ylabel('Stability', fontsize=24)
        
        # Set exactly 4 ticks on each axis
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        
        # Large tick font size
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plots and output_folder:
            plot_filename = f'stability_vs_firing_rate_{condition}.png'
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()
        
        # Store results
        results_dict[condition] = {
            'firing_rates': firing_rates,
            'stability_scores': valid_stability_scores,
            'pearson_correlation': pearson_r,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_r,
            'spearman_p_value': spearman_p,
            'n_neurons': len(valid_stability_scores)
        }
        
        # Print correlation statistics
        print(f"\n=== STABILITY vs FIRING RATE ({condition.upper()}) ===")
        print(f"Neurons: {len(valid_stability_scores)}")
        print(f"Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.1e}")
        print(f"Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.1e}")
    
    return results_dict