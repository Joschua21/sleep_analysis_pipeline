import numpy as np
from scipy.signal.windows import gaussian
from scipy.signal import convolve, savgol_filter
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from pinkrigs_tools.dataset.query import load_data, queryCSV
import os
from datetime import datetime
import seaborn as sns
import pandas as pd
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from glob import glob
from scipy.ndimage import gaussian_filter1d

def moving_average_with_padding(data, window_size, session_average=None):

    if session_average is None:
        session_average = np.mean(data)
    
    # Calculate half window size
    half_window = window_size // 2
    
    # Create padded array
    padded_data = np.concatenate([
        np.full(half_window, session_average),  # Pad start with session average
        data,
        np.full(half_window, session_average)   # Pad end with session average
    ])
    
    # Apply moving average
    kernel = np.ones(window_size) / window_size
    smoothed_padded = np.convolve(padded_data, kernel, mode='valid')
    
    # Ensure we return exactly the same length as input
    expected_length = len(data)
    actual_length = len(smoothed_padded)
    
    if actual_length != expected_length:
        # Trim or pad to match exact input length
        if actual_length > expected_length:
            # Trim from both ends equally
            excess = actual_length - expected_length
            start_trim = excess // 2
            end_trim = excess - start_trim
            if end_trim > 0:
                smoothed_padded = smoothed_padded[start_trim:-end_trim]
            else:
                smoothed_padded = smoothed_padded[start_trim:]
        else:
            # This shouldn't happen with our padding, but just in case
            smoothed_padded = np.pad(smoothed_padded, 
                                   (0, expected_length - actual_length), 
                                   'edge')
    
    return smoothed_padded

def process_spike_data(exp_kwargs, bin_size=0.3, show_plots=True):
    """
    Process spike data for all probes, bin the spikes, and optionally display plots
    
    Parameters:
    -----------
    exp_kwargs : dict
        Dictionary with experiment parameters (subject, expDate, expNum)
    bin_size : float
        Size of time bins in seconds
    show_plots : bool
        Whether to display plots of the binned spike data
        
    Returns:
    --------
    dict
        Dictionary containing binned spike data for each probe
    """
    
    # Load spike data for both probes
    ephys_dict = {'spikes':'all','clusters':'all'}
    data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
    spike_recordings = load_data(data_name_dict=data_name_dict, **exp_kwargs)
    
    # Process data from each probe
    results = {}
    for probe in ['probe0', 'probe1']:
        if probe in spike_recordings and len(spike_recordings[probe]) > 0:
            # Process the data
            probe_data = spike_recordings[probe][0]
            counts, time_bins, cluster_ids = bin_spikes(probe_data, bin_size=bin_size)
            
            # Store the results
            results[probe] = {
                'counts': counts,
                'time_bins': time_bins,
                'cluster_ids': cluster_ids
            }
            
            print(f"\nProcessed {probe}:")
            print(f"Shape of counts matrix: {counts.shape}")
            print(f"Time bins: {len(time_bins)} bins from {time_bins[0]:.2f}s to {time_bins[-1]:.2f}s")
            print(f"Number of clusters: {len(cluster_ids)}")
            
            if show_plots:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Take all clusters
                n_clusters_to_show = len(results[probe]['cluster_ids'])
                
                # Use matshow with a grayscale colormap and no interpolation
                counts_subset = results[probe]['counts'][:n_clusters_to_show]
                time_bins = results[probe]['time_bins']
                
                # Adjust extent to match the time bin centers
                extent = [time_bins[0], time_bins[-1], 0, n_clusters_to_show]
                max_spike_count = counts_subset.max()
                vmax_threshold = max_spike_count * 0.3  # Adjust this number (lower = higher contrast)

                # Use binary colormap (black for high values) instead of gray_r
                cax = ax.matshow(counts_subset, cmap='binary', aspect='auto', extent=extent, 
                         vmin=0, vmax=vmax_threshold)
                
                # Add colorbar to show spike count scale
                cbar = plt.colorbar(cax, ax=ax)
                cbar.set_label('Spike Count')
                
                # Set labels and title
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Cluster ID')
                ax.set_title(f'{probe}: First {n_clusters_to_show} clusters activity')
                
                # Set x-axis labels at the bottom (more intuitive for time series)
                ax.xaxis.set_ticks_position('bottom')
                
                plt.tight_layout()
                plt.show()
    
    return results


def bin_spikes(probe_data, bin_size=0.05):  # 50ms bins
    """
    Bin spikes for a single probe
    
    Parameters:
    -----------
    probe_data : dict
        Dictionary containing spike data for a probe
    bin_size : float
        Size of time bins in seconds
        
    Returns:
    --------
    tuple
        (counts, time_bins, cluster_ids)
    """
    # Get spike times and clusters
    spike_times = probe_data['spikes']['times']
    spike_clusters = probe_data['spikes']['clusters']
    
    # Define the time range to cover the entire recording
    start_time = np.min(spike_times)
    end_time = np.max(spike_times)
    
    # Apply bincount2D to get the matrix
    counts, time_bins, cluster_ids = bincount2D(
        x=spike_times,
        y=spike_clusters,
        xbin=bin_size,
        ybin=0,  # Use unique values for clusters
        xlim=[start_time, end_time],
        # ylim can be omitted to use min/max of clusters
    )
    
    return counts, time_bins, cluster_ids

# Add to neuropixel_utils.py
def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None, xsmoothing=0):
    """
    Computes a 2D histogram by aggregating values in a 2D array. Used if you want a binned version of your spike/event data for example
    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param ybin:
        scalar: bin size along 2nd dimension
        0: aggregate according to unique values
        array: aggregate according to exact values (count reduce operation)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :param xsmoothing: (optional) smoothing along the x axis with a half-gaussian, with sigma given by this value
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    
    # if no bounds provided, use min/max of vectors
    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    def _get_scale_and_indices(v, bin, lim):
        # if bin is a nonzero scalar, this is a bin size: create scale and indices
        if np.isscalar(bin) and bin != 0:
            scale = np.arange(lim[0], lim[1] + bin / 2, bin)
            ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
        # if bin == 0, aggregate over unique values
        else:
            scale, ind = np.unique(v, return_inverse=True)
        return scale, ind

    xscale, xind = _get_scale_and_indices(x, xbin, xlim)
    yscale, yind = _get_scale_and_indices(y, ybin, ylim)
    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)

    # if a set of specific values is requested output an array matching the scale dimensions
    if not np.isscalar(xbin) and xbin.size > 1:
        _, iout, ir = np.intersect1d(xbin, xscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ny, xbin.size))
        r[:, iout] = _r[:, ir]
        xscale = xbin

    if not np.isscalar(ybin) and ybin.size > 1:
        _, iout, ir = np.intersect1d(ybin, yscale, return_indices=True)
        _r = r.copy()
        r = np.zeros((ybin.size, r.shape[1]))
        r[iout, :] = _r[ir, :]
        yscale = ybin

    if xsmoothing > 0:
        w = xscale.size  # [tscale.size - 1 if tscale.size % 2 == 0 else tscale.size]
        window = gaussian(w, std=xsmoothing / xbin)
        # half (causal) gaussian filter
        window[: int(np.ceil(w / 2))] = 0
        window /= np.sum(window)  #
        binned_spikes_conv = [
            convolve(r[j, :], window, mode="same", method="auto")[:, np.newaxis]
            for j in range(r.shape[0])
        ]  # [:-1]
        r = np.concatenate(binned_spikes_conv, axis=1).T

    return r, xscale, yscale

def map_sleep_bouts_to_neural_data(results, sleep_times_csv, cam_times):
    """
    Maps sleep bout timestamps to neural time bins for each probe,
    automatically handling alignment between camera and neural data
    
    Parameters:
    -----------
    results : dict
        Dictionary containing neural data for each probe, as returned by process_spike_data()
    sleep_times_csv : str
        Path to the CSV file containing sleep bout information
    cam_times : array-like
        Array of camera frame timestamps in seconds
        
    Returns:
    --------
    dict
        Updated results dictionary with sleep bout mapping for each probe
    """
    
    # Load sleep bout information from CSV
    df_sleep = pd.read_csv(sleep_times_csv)
    
    # Determine the time offset automatically
    camera_start_time = cam_times[0]
    camera_end_time = cam_times[-1]
    
    # For each probe's data, create a mapping
    for probe in results:
        if 'time_bins' not in results[probe]:
            print(f"No time bins found for {probe}, skipping.")
            continue
            
        time_bins = results[probe]['time_bins']
        neural_start_time = time_bins[0]
        neural_end_time = time_bins[-1]
        
        # Calculate offset: If camera_start_time is negative, camera started before neural recording
        time_offset = -camera_start_time if camera_start_time < 0 else 0
        
        print(f"\nAlignment information for {probe}:")
        print(f"- Camera time range: {camera_start_time:.2f}s to {camera_end_time:.2f}s")
        print(f"- Neural time range: {neural_start_time:.2f}s to {neural_end_time:.2f}s")
        print(f"- Automatic time offset: {time_offset:.2f}s")
        
        if time_offset != 0:
            print(f"  (Adding {time_offset:.2f}s to camera timestamps to align with neural data)")
        
        # Get sleep bout timestamps and apply calculated time offset
        start_timestamps = df_sleep['start_timestamp_s'].values + time_offset
        end_timestamps = df_sleep['end_timestamp_s'].values + time_offset
        
        # Check for sleep bouts outside neural recording range and warn
        out_of_range_bouts = np.sum((start_timestamps < neural_start_time) | 
                                    (end_timestamps > neural_end_time))
        if out_of_range_bouts > 0:
            print(f"Warning: {out_of_range_bouts} sleep bouts are outside the neural recording range")
        
        # Get the corresponding bin indices by finding the closest time bin for each timestamp
        # Clip to valid range to avoid index errors
        start_bin_indices = [np.abs(time_bins - min(max(t, neural_start_time), neural_end_time)).argmin() 
                            for t in start_timestamps]
        end_bin_indices = [np.abs(time_bins - min(max(t, neural_start_time), neural_end_time)).argmin() 
                          for t in end_timestamps]
        
        # Create a new dataframe with the mapped information
        mapped_df = pd.DataFrame({
            'start_timestamp_s': df_sleep['start_timestamp_s'].values,  # original timestamps
            'end_timestamp_s': df_sleep['end_timestamp_s'].values,      # original timestamps
            'aligned_start_timestamp_s': start_timestamps,  # timestamps after alignment offset
            'aligned_end_timestamp_s': end_timestamps,      # timestamps after alignment offset
            'start_bin_index': start_bin_indices,
            'end_bin_index': end_bin_indices,
            'start_bin_time': [time_bins[i] for i in start_bin_indices],
            'end_bin_time': [time_bins[i] for i in end_bin_indices],
            'duration_frames': df_sleep['end_frame'] - df_sleep['start_frame'] if 'end_frame' in df_sleep.columns else None,
            'duration_s': df_sleep['end_timestamp_s'] - df_sleep['start_timestamp_s'],
            'duration_bins': np.array(end_bin_indices) - np.array(start_bin_indices),
            'in_range': ((start_timestamps >= neural_start_time) & (end_timestamps <= neural_end_time))
        })
        
        # Print summary information
        print(f"\nSleep bout mapping for {probe}:")
        print(f"- Time bins range: {neural_start_time:.2f}s to {neural_end_time:.2f}s")
        print(f"- Bin size: {time_bins[1] - time_bins[0]:.2f}s")
        print(f"- Total bins: {len(time_bins)}")
        print(f"- Number of sleep bouts: {len(mapped_df)}")
        print(f"- Sleep bouts within neural recording range: {np.sum(mapped_df['in_range'])}")
        
        # Print a sample of the mapping
        print("\nSample mapping (first 5 bouts):")
        print(mapped_df[['start_timestamp_s', 'aligned_start_timestamp_s', 'start_bin_time', 
                        'end_timestamp_s', 'aligned_end_timestamp_s', 'end_bin_time', 
                        'duration_s', 'in_range']].head())
        
        # Store the mapping in results for later use
        results[probe]['sleep_bout_mapping'] = mapped_df
    
    return results


def filter_clusters_by_quality(results, probe, include_qualities=['good', 'mua']):
    """
    Filter clusters by quality label from bombcell classification
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from process_spike_data() with bombcell classifications
    probe : str
        Probe to filter clusters for (e.g., 'probe0', 'probe1')
    include_qualities : list
        List of quality labels to include (e.g., ['good', 'mua'])
        
    Returns:
    --------
    tuple
        (filtered_counts, filtered_cluster_ids, quality_mask)
    """
    
    if 'cluster_quality' not in results[probe]:
        raise ValueError(f"No quality labels found for {probe}. Run bombcell classification first.")
    
    # Create mask for clusters matching requested qualities
    quality_mask = np.isin(results[probe]['cluster_quality'], include_qualities)
    
    # Filter clusters
    filtered_counts = results[probe]['counts'][quality_mask]
    filtered_cluster_ids = results[probe]['cluster_ids'][quality_mask]
    
    print(f"Filtered {probe}: kept {np.sum(quality_mask)} of {len(quality_mask)} clusters")
    
    return filtered_counts, filtered_cluster_ids, quality_mask



def analyze_sleep_wake_activity(results, output_dir=None, save_plots=False, num_top_clusters=10):
    """
    Analyze neural activity during sleep and wake periods, and plot the results
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from process_spike_data(), with sleep bout mapping and bombcell classification
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
    num_top_clusters : int, optional
        Number of top sleep and wake clusters to display (default: 10)
        
    Returns:
    --------
    dict
        Dictionary containing sleep-wake modulation values for each probe
    """
       
    modulation_results = {}
    
    # Collect data from all probes to merge
    all_counts = []
    all_cluster_ids = []
    all_probe_labels = []
    all_time_bins = None
    all_sleep_bouts = None
    
    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if 'sleep_bout_mapping' not in results[probe]:
            print(f"No sleep bout mapping found for {probe}. Skipping.")
            continue
            
        if 'cluster_quality' not in results[probe]:
            print(f"No quality labels found for {probe}. Skipping.")
            continue
            
        valid_probes.append(probe)
    
    if not valid_probes:
        print("No valid probes found with required data.")
        return modulation_results
    
    # Use the first probe's time bins as reference
    reference_probe = valid_probes[0]
    all_time_bins = results[reference_probe]['time_bins']
    all_sleep_bouts = results[reference_probe]['sleep_bout_mapping']
    
    # Create a mask for time bins that fall within any sleep period
    sleep_mask = np.zeros(len(all_time_bins), dtype=bool)
    for _, bout in all_sleep_bouts.iterrows():
        start_idx = bout['start_bin_index']
        end_idx = bout['end_bin_index']
        sleep_mask[start_idx:end_idx+1] = True
    
    wake_mask = ~sleep_mask
    
    # Process and collect data from each probe
    for probe in valid_probes:
        # Filter out noise clusters, keep good and mua
        counts, cluster_ids, quality_mask = filter_clusters_by_quality(
            results, probe, include_qualities=['good', 'mua']
        )
        
        # Append data
        all_counts.append(counts)
        all_cluster_ids.extend([(probe, cid) for cid in cluster_ids])
        all_probe_labels.extend([probe] * len(cluster_ids))
    
    # Merge counts from all probes
    if not all_counts:
        print("No valid clusters found after filtering.")
        return modulation_results
        
    merged_counts = np.vstack(all_counts)
    
    # Normalize each cluster's firing rate by its 95th percentile
    normalized_counts = np.zeros_like(merged_counts, dtype=float)
    for i in range(merged_counts.shape[0]):
        cluster_counts = merged_counts[i, :]
        p95 = np.percentile(cluster_counts, 95)
        
        # Avoid division by zero
        if p95 > 0:
            normalized_counts[i, :] = cluster_counts / p95
        else:
            normalized_counts[i, :] = cluster_counts
    
    # Calculate firing rate during sleep and wake for each cluster
    sleep_rates = []
    wake_rates = []
    
    # For each cluster, calculate average firing rate during sleep and wake periods
    for i in range(merged_counts.shape[0]):
        cluster_counts = merged_counts[i, :]
        
        # Calculate total spikes and time for sleep and wake periods
        sleep_spikes = np.sum(cluster_counts[sleep_mask])
        wake_spikes = np.sum(cluster_counts[wake_mask])
        
        # Calculate total time (in bins)
        sleep_time = np.sum(sleep_mask)
        wake_time = np.sum(wake_mask)
        
        # Bin size in seconds
        bin_size = all_time_bins[1] - all_time_bins[0]
        
        # Calculate rates (spikes per second)
        sleep_rate = sleep_spikes / (sleep_time * bin_size) if sleep_time > 0 else 0
        wake_rate = wake_spikes / (wake_time * bin_size) if wake_time > 0 else 0
        
        sleep_rates.append(sleep_rate)
        wake_rates.append(wake_rate)
    
    sleep_rates = np.array(sleep_rates)
    wake_rates = np.array(wake_rates)
    
    # Calculate modulation index: (wake - sleep) / (wake + sleep)
    epsilon = 1e-10  # Small value to prevent division by zero
    modulation_index = np.zeros_like(sleep_rates)
    for i in range(len(sleep_rates)):
        if sleep_rates[i] + wake_rates[i] > epsilon:
            modulation_index[i] = (wake_rates[i] - sleep_rates[i]) / (wake_rates[i] + sleep_rates[i] + epsilon)
        else:
            modulation_index[i] = 0
    
    # Sort clusters by modulation index (sleep-preferring clusters first)
    sorted_indices = np.argsort(modulation_index)
    
    # Select top clusters by modulation index
    sleep_selective = sorted_indices[:num_top_clusters]  # Most negative values (sleep-selective)
    wake_selective = sorted_indices[-num_top_clusters:][::-1]  # Most positive values (wake-selective)
    
    # Store results for both probes combined
    modulation_results['merged'] = {
        'cluster_ids': all_cluster_ids,
        'probe_labels': all_probe_labels,
        'sleep_rates': sleep_rates,
        'wake_rates': wake_rates,
        'modulation_index': modulation_index,
        'wake_selective_indices': wake_selective,
        'sleep_selective_indices': sleep_selective,
    }
    
    # Rearrange data based on modulation index for plotting
    modulation_sorted_indices = np.argsort(modulation_index)
    sorted_counts = merged_counts[modulation_sorted_indices]
    sorted_normalized_counts = normalized_counts[modulation_sorted_indices]
    
    # Create figure with 4 subplots with custom height ratios
    fig, axs = plt.subplots(4, 1, figsize=(14, 20), 
                           gridspec_kw={'height_ratios': [4, 1, 3, 3]},
                           sharex=True)
    
    # Common time extent for all plots
    time_extent = [all_time_bins[0], all_time_bins[-1]]
    
    # Common function to set vmax threshold based on percentile for better contrast
    def get_vmax_threshold(data, percentile=95):
        vmax = np.percentile(data, percentile)
        return vmax if vmax > 0 else 1.0  # Avoid zero vmax
    
    # 1. Plot all clusters sorted by modulation index
    extent = [time_extent[0], time_extent[1], 0, sorted_counts.shape[0]]
    vmax_threshold = get_vmax_threshold(sorted_normalized_counts)
    
    # Use binary colormap for better visualization like in process_spike_data
    im1 = axs[0].matshow(sorted_normalized_counts, aspect='auto', extent=extent, cmap='binary', 
                       interpolation='none', origin='lower', vmin=0, vmax=vmax_threshold)
    
    # Highlight sleep periods with semi-transparent overlay (light blue to be visible on black/white)
    for _, bout in all_sleep_bouts.iterrows():
        axs[0].axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')
    
    axs[0].set_title(f'All clusters from all probes, sorted by sleep-wake modulation (n={sorted_counts.shape[0]})')
    axs[0].set_ylabel('Cluster rank\n(sleep-selective → wake-selective)')
    plt.colorbar(im1, ax=axs[0], label='Normalized spike count')
    
    # 2. Average activity across all clusters
    mean_activity = np.mean(merged_counts, axis=0)

    # Create a separate axes for the average plot with precise positioning
    # First get the position of the existing second subplot
    pos = axs[1].get_position()

    # Remove the existing subplot
    axs[1].remove()

    # Create a new axes with exactly the same position as other plots horizontally
    # but maintain the smaller height
    axs[1] = fig.add_axes([axs[0].get_position().x0,  # Match x0 from first plot 
                        pos.y0, 
                        axs[0].get_position().width,  # Match width from first plot
                        pos.height])

    # Calculate reasonable y-limits with stronger padding (25%)
    buffer_factor = 0.15  # 15% buffer
    data_range = np.max(mean_activity) - np.min(mean_activity)
    buffer_amount = data_range * buffer_factor

    y_max = np.max(mean_activity) + buffer_amount  # Add buffer on top
    y_min = np.min(mean_activity) - buffer_amount  # Subtract buffer below

    # Plot with same x limits as other plots
    axs[1].plot(all_time_bins, mean_activity, color='black', linewidth=0.6)
    axs[1].set_xlim(time_extent)
    axs[1].set_ylim(y_min, y_max)
    
    # Add light colored vertical spans for sleep periods in the average plot
    for _, bout in all_sleep_bouts.iterrows():
        axs[1].axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                    color='lightblue', alpha=0.3, ec='none')

    # Add grid lines to make it easier to read values
    axs[1].grid(True, alpha=0.2)
    axs[1].set_title('Average activity across all filtered clusters')
    axs[1].set_ylabel('Mean spike count')
    
    # 3. Plot top sleep-selective clusters (using normalized counts)
    # IMPORTANT FIX: Use the actual sleep-selective indices to get their data
    sleep_selective_counts = normalized_counts[sleep_selective]
    vmax_sleep = get_vmax_threshold(sleep_selective_counts)
    
    # Display these clusters in order of their sleep selectivity (most to least)
    im2 = axs[2].matshow(sleep_selective_counts, aspect='auto', 
                       extent=[time_extent[0], time_extent[1], 0, sleep_selective_counts.shape[0]],
                       cmap='binary', interpolation='none', origin='lower', vmin=0, vmax=vmax_sleep)
    
    # Highlight sleep periods
    for _, bout in all_sleep_bouts.iterrows():
        axs[2].axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')
    
    axs[2].set_title(f'Top {num_top_clusters} Sleep-selective clusters')
    axs[2].set_ylabel('Rank (most sleep-selective first)')
    plt.colorbar(im2, ax=axs[2], label='Normalized spike count')
    
    # 4. Plot top wake-selective clusters (using normalized counts)
    # IMPORTANT FIX: Use the actual wake-selective indices to get their data
    wake_selective_counts = normalized_counts[wake_selective]
    vmax_wake = get_vmax_threshold(wake_selective_counts)
    
    # Display these clusters in order of their wake selectivity (most to least)
    im3 = axs[3].matshow(wake_selective_counts, aspect='auto', 
                       extent=[time_extent[0], time_extent[1], 0, wake_selective_counts.shape[0]],
                       cmap='binary', interpolation='none', origin='lower', vmin=0, vmax=vmax_wake)
    
    # Highlight sleep periods
    for _, bout in all_sleep_bouts.iterrows():
        axs[3].axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')
    
    axs[3].set_title(f'Top {num_top_clusters} Wake-selective clusters')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Rank (most wake-selective first)')
    plt.colorbar(im3, ax=axs[3], label='Normalized spike count')
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Print modulation values
    print(f"\nResults for merged probes:")
    print(f"\nTop {num_top_clusters} sleep-selective clusters:")
    for i, idx in enumerate(sleep_selective):
        cluster_info = all_cluster_ids[idx]
        probe_label = all_probe_labels[idx]
        mod = modulation_index[idx]
        print(f"  {i+1}. {probe_label} Cluster {cluster_info[1]}: Modulation = {mod:.4f} (Sleep: {sleep_rates[idx]:.4f} Hz, Wake: {wake_rates[idx]:.4f} Hz)")
        
    print(f"\nTop {num_top_clusters} wake-selective clusters:")
    for i, idx in enumerate(wake_selective):
        cluster_info = all_cluster_ids[idx]
        probe_label = all_probe_labels[idx]
        mod = modulation_index[idx]
        print(f"  {i+1}. {probe_label} Cluster {cluster_info[1]}: Modulation = {mod:.4f} (Sleep: {sleep_rates[idx]:.4f} Hz, Wake: {wake_rates[idx]:.4f} Hz)")
    
    # Save plot if requested
    if save_plots and output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"merged_probes_sleep_wake_analysis_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    
    plt.show()
    
    return modulation_results

def analyze_cluster_state_distribution(results, output_dir=None, save_plots=False, bin_size_s=120, state_threshold=0.9):
    """
    Analyze the distribution of cluster firing rates during predominantly sleep or wake states
    using pooled and normalized data from all probes
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from process_spike_data(), with sleep bout mapping and bombcell classification
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
    bin_size_s : float, optional
        Size of time bins in seconds for state analysis (default: 120s = 2 minutes)
    state_threshold : float, optional
        Threshold for classifying a bin as sleep/wake (default: 0.9 or 90%)
        
    Returns:
    --------
    dict
        Dictionary containing state-dependent firing rates for merged probes
    """
    if state_threshold > 1:
        state_threshold = state_threshold / 100

    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if 'sleep_bout_mapping' not in results[probe]:
            print(f"No sleep bout mapping found for {probe}. Skipping.")
            continue
        if 'cluster_quality' not in results[probe]:
            print(f"No quality labels found for {probe}. Skipping.")
            continue
        valid_probes.append(probe)
    
    if not valid_probes:
        print("No valid probes found with required data.")
        return {}
    
    # Use the first probe's time bins and sleep bout info as reference
    reference_probe = valid_probes[0]
    time_bins = results[reference_probe]['time_bins']
    sleep_bouts = results[reference_probe]['sleep_bout_mapping']
    original_bin_size = time_bins[1] - time_bins[0]
    
    # Create a mask for time bins that fall within any sleep period
    sleep_mask = np.zeros(len(time_bins), dtype=bool)
    for _, bout in sleep_bouts.iterrows():
        start_idx = bout['start_bin_index']
        end_idx = bout['end_bin_index']
        sleep_mask[start_idx:end_idx+1] = True

    wake_mask = ~sleep_mask
    
    # Collect and merge data from all probes
    all_counts = []
    all_cluster_ids = []
    all_min_length = float('inf')
    
    # First pass: determine minimum length and collect data
    for probe in valid_probes:
        # Filter out noise clusters, keep good and mua
        counts, cluster_ids, quality_mask = filter_clusters_by_quality(
            results, probe, include_qualities=['good', 'mua']
        )
        
        if counts.shape[0] > 0:
            all_min_length = min(all_min_length, counts.shape[1])
            all_counts.append(counts)
            all_cluster_ids.extend([(probe, cid) for cid in cluster_ids])
    
    if not all_counts:
        print("No valid clusters found after filtering.")
        return {}
    
    # Second pass: truncate all arrays to minimum length
    for i in range(len(all_counts)):
        all_counts[i] = all_counts[i][:, :all_min_length]
    
    # Merge counts from all probes
    merged_counts = np.vstack(all_counts)
    
    # Adjust time_bins and sleep_mask to match truncated data
    time_bins = time_bins[:all_min_length]
    sleep_mask = sleep_mask[:all_min_length]
    wake_mask = ~sleep_mask
    
    # Normalize each cluster's firing rate by its 95th percentile (same as analyze_sleep_wake_activity)
    normalized_counts = np.zeros_like(merged_counts, dtype=float)
    for i in range(merged_counts.shape[0]):
        cluster_counts = merged_counts[i, :]
        p95 = np.percentile(cluster_counts, 95)
        
        # Avoid division by zero
        if p95 > 0:
            normalized_counts[i, :] = cluster_counts / p95
        else:
            normalized_counts[i, :] = cluster_counts
    
    # Calculate how many original bins fit into our new larger bins
    bins_per_large_bin = int(bin_size_s / original_bin_size)
    num_large_bins = len(time_bins) // bins_per_large_bin
    
    if num_large_bins == 0:
        print(f"Warning: Recording too short for {bin_size_s}s bins. Try a smaller bin size.")
        return {}
    
    # Initialize arrays to store state and firing rates for each large bin
    large_bin_states = []
    large_bin_firing_rates = np.zeros((normalized_counts.shape[0], num_large_bins))
    large_bin_centers = np.zeros(num_large_bins)
    
    # Process each large bin
    for i in range(num_large_bins):
        start_idx = i * bins_per_large_bin
        end_idx = min((i + 1) * bins_per_large_bin, len(time_bins))
        
        # Calculate bin center time
        large_bin_centers[i] = np.mean(time_bins[start_idx:end_idx])
        
        # Calculate state for this bin
        sleep_fraction = np.sum(sleep_mask[start_idx:end_idx]) / (end_idx - start_idx)
        
        if sleep_fraction >= state_threshold:
            large_bin_states.append('Sleep')
        elif sleep_fraction <= (1 - state_threshold):
            large_bin_states.append('Wake')
        else:
            large_bin_states.append('Mixed')
        
        # Calculate normalized firing rate for each cluster in this bin
        for j in range(normalized_counts.shape[0]):
            # Use normalized counts instead of raw counts
            normalized_activity_in_bin = np.mean(normalized_counts[j, start_idx:end_idx])
            large_bin_firing_rates[j, i] = normalized_activity_in_bin
    
    # Collect data for plotting
    plot_data = []
    for i, cluster_id in enumerate(all_cluster_ids):
        sleep_rates = large_bin_firing_rates[i, [idx for idx, state in enumerate(large_bin_states) if state == 'Sleep']]
        wake_rates = large_bin_firing_rates[i, [idx for idx, state in enumerate(large_bin_states) if state == 'Wake']]
        
        # Calculate average normalized firing rate across bins for each state
        if len(sleep_rates) > 0:
            sleep_avg_rate = np.mean(sleep_rates)
            plot_data.append({
                'cluster_id': cluster_id,
                'state': 'Sleep',
                'firing_rate': sleep_avg_rate
            })
            
        if len(wake_rates) > 0:
            wake_avg_rate = np.mean(wake_rates)
            plot_data.append({
                'cluster_id': cluster_id,
                'state': 'Wake',
                'firing_rate': wake_avg_rate
            })
    
    # Convert to DataFrame for seaborn
    df_plot = pd.DataFrame(plot_data)
    
    # Check if DataFrame is empty
    if df_plot.empty:
        print("No data to plot. Try adjusting the bin size or threshold.")
        return {}
    
    # Store results
    state_results = {
        'merged': {
            'cluster_ids': all_cluster_ids,
            'large_bin_states': large_bin_states,
            'large_bin_firing_rates': large_bin_firing_rates,
            'large_bin_centers': large_bin_centers,
            'plot_data': df_plot
        }
    }
    
    # Create figure for swarm plot
    plt.figure(figsize=(10, 8))
    
    # Debug: print first few rows of DataFrame
    print(f"\nFirst few rows of DataFrame for merged probes:")
    print(df_plot.head())
    
    # Create swarm plot
    ax = sns.swarmplot(data=df_plot, x='state', y='firing_rate', color='gray', alpha=0.7, size=5)
    
    # Add box plot over swarm plot to show distribution
    sns.boxplot(data=df_plot, x='state', y='firing_rate', color='white', fliersize=0, width=0.5, 
               boxprops={"facecolor": (.9, .9, .9, 0.5), "edgecolor": "black"})
    
    # Perform statistical test
    sleep_rates = df_plot[df_plot['state'] == 'Sleep']['firing_rate'].values
    wake_rates = df_plot[df_plot['state'] == 'Wake']['firing_rate'].values
    
    if len(sleep_rates) > 0 and len(wake_rates) > 0:
        stat, p_value = stats.mannwhitneyu(sleep_rates, wake_rates)
        
        # Add statistics to plot
        plt.title(f'Merged Probes: Normalized Cluster Firing Rates by State (p={p_value:.4f})')
        
        # Add number of observations
        plt.annotate(f'n={len(sleep_rates)} observations\n{len(np.unique(df_plot[df_plot["state"] == "Sleep"]["cluster_id"]))} clusters', 
                    xy=(0, 0), xytext=(0.1, 0.95), textcoords='figure fraction',
                    horizontalalignment='left', verticalalignment='top')
        
        plt.annotate(f'n={len(wake_rates)} observations\n{len(np.unique(df_plot[df_plot["state"] == "Wake"]["cluster_id"]))} clusters', 
                    xy=(0, 0), xytext=(0.9, 0.95), textcoords='figure fraction',
                    horizontalalignment='right', verticalalignment='top')
    else:
        plt.title('Merged Probes: Normalized Cluster Firing Rates by State')
    
    # Set axis labels
    plt.xlabel('State')
    plt.ylabel('Normalized Firing Rate')
    
    # Apply gridlines
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save plot if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"merged_probes_state_firing_rates_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    
    # Display summary statistics
    print(f"\nResults for merged probes using {bin_size_s}s bins and {state_threshold*100}% state threshold:")
    print(f"Total large bins: {num_large_bins}")
    print(f"Sleep bins: {large_bin_states.count('Sleep')}")
    print(f"Wake bins: {large_bin_states.count('Wake')}")
    print(f"Mixed bins: {large_bin_states.count('Mixed')}")
    print(f"Total clusters analyzed: {len(all_cluster_ids)}")
    
    if len(sleep_rates) > 0 and len(wake_rates) > 0:
        print("\nNormalized firing rate statistics:")
        print(f"Sleep: median={np.median(sleep_rates):.4f}, mean={np.mean(sleep_rates):.4f}")
        print(f"Wake: median={np.median(wake_rates):.4f}, mean={np.mean(wake_rates):.4f}")
        print(f"Mann-Whitney U test: p={p_value:.4f}")
    
    plt.show()
    
    return state_results

def analyze_neuronal_stability(results, output_dir=None, save_plots=False, bin_size_s=120, state_threshold=0.9, max_iterations=1000):
    """
    Analyze neuronal stability by splitting bins into two categories and comparing normalized firing rates
    using pooled data from all probes
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from process_spike_data(), with sleep bout mapping and bombcell classification
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
    bin_size_s : float, optional
        Size of time bins in seconds for state analysis (default: 120s = 2 minutes)
    state_threshold : float, optional
        Threshold for classifying a bin as sleep/wake (default: 0.9 or 90%)
    max_iterations : int, optional
        Maximum number of iterations for random assignment (default: 1000)
        
    Returns:
    --------
    dict
        Dictionary containing stability metrics for merged probes
    """
    
    # Convert threshold from percentage to fraction if needed
    if state_threshold > 1:
        state_threshold = state_threshold / 100
    
    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if 'sleep_bout_mapping' not in results[probe]:
            print(f"No sleep bout mapping found for {probe}. Skipping.")
            continue
        if 'cluster_quality' not in results[probe]:
            print(f"No quality labels found for {probe}. Skipping.")
            continue
        valid_probes.append(probe)
    
    if not valid_probes:
        print("No valid probes found with required data.")
        return {}
    
    # Use the first probe's time bins and sleep bout info as reference
    reference_probe = valid_probes[0]
    time_bins = results[reference_probe]['time_bins']
    sleep_bouts = results[reference_probe]['sleep_bout_mapping']
    original_bin_size = time_bins[1] - time_bins[0]
    
    # Create a mask for time bins that fall within any sleep period
    sleep_mask = np.zeros(len(time_bins), dtype=bool)
    for _, bout in sleep_bouts.iterrows():
        start_idx = bout['start_bin_index']
        end_idx = bout['end_bin_index']
        sleep_mask[start_idx:end_idx+1] = True

    wake_mask = ~sleep_mask
    
    # Collect and merge data from all probes
    all_counts = []
    all_cluster_ids = []
    all_min_length = float('inf')
    
    # First pass: determine minimum length and collect data
    for probe in valid_probes:
        # Filter out noise clusters, keep good and mua
        counts, cluster_ids, quality_mask = filter_clusters_by_quality(
            results, probe, include_qualities=['good', 'mua']
        )
        
        if counts.shape[0] > 0:
            all_min_length = min(all_min_length, counts.shape[1])
            all_counts.append(counts)
            all_cluster_ids.extend([(probe, cid) for cid in cluster_ids])
    
    if not all_counts:
        print("No valid clusters found after filtering.")
        return {}
    
    # Second pass: truncate all arrays to minimum length
    for i in range(len(all_counts)):
        all_counts[i] = all_counts[i][:, :all_min_length]
    
    # Merge counts from all probes
    merged_counts = np.vstack(all_counts)
    
    # Adjust time_bins and sleep_mask to match truncated data
    time_bins = time_bins[:all_min_length]
    sleep_mask = sleep_mask[:all_min_length]
    wake_mask = ~sleep_mask
    
    # Normalize each cluster's firing rate by its 95th percentile (same as analyze_sleep_wake_activity)
    normalized_counts = np.zeros_like(merged_counts, dtype=float)
    for i in range(merged_counts.shape[0]):
        cluster_counts = merged_counts[i, :]
        p95 = np.percentile(cluster_counts, 95)
        
        # Avoid division by zero
        if p95 > 0:
            normalized_counts[i, :] = cluster_counts / p95
        else:
            normalized_counts[i, :] = cluster_counts
    
    # Calculate how many original bins fit into our new larger bins
    bins_per_large_bin = int(bin_size_s / original_bin_size)
    num_large_bins = len(time_bins) // bins_per_large_bin
    
    if num_large_bins == 0:
        print(f"Warning: Recording too short for {bin_size_s}s bins. Try a smaller bin size.")
        return {}
    
    # Initialize arrays to store state and firing rates for each large bin
    large_bin_states = []
    large_bin_times = []
    large_bin_firing_rates = np.zeros((normalized_counts.shape[0], num_large_bins))
    
    # Process each large bin
    for i in range(num_large_bins):
        start_idx = i * bins_per_large_bin
        end_idx = min((i + 1) * bins_per_large_bin, len(time_bins))
        
        # Calculate bin center time
        bin_center = np.mean(time_bins[start_idx:end_idx])
        large_bin_times.append(bin_center)
        
        # Calculate state for this bin
        sleep_fraction = np.sum(sleep_mask[start_idx:end_idx]) / (end_idx - start_idx)
        
        if sleep_fraction >= state_threshold:
            large_bin_states.append('Sleep')
        elif sleep_fraction <= (1 - state_threshold):
            large_bin_states.append('Wake')
        else:
            large_bin_states.append('Mixed')
        
        # Calculate normalized firing rate for each cluster in this bin
        for j in range(normalized_counts.shape[0]):
            # Use normalized counts instead of raw counts
            normalized_activity_in_bin = np.mean(normalized_counts[j, start_idx:end_idx])
            large_bin_firing_rates[j, i] = normalized_activity_in_bin
    
    # Convert to arrays for easier manipulation
    large_bin_times = np.array(large_bin_times)
    large_bin_states = np.array(large_bin_states)
    
    # Identify recording midpoint
    midpoint_time = (time_bins[0] + time_bins[-1]) / 2
    
    # Create bin groups
    first_half_sleep_bins = np.where((large_bin_times < midpoint_time) & (large_bin_states == 'Sleep'))[0]
    first_half_wake_bins = np.where((large_bin_times < midpoint_time) & (large_bin_states == 'Wake'))[0]
    second_half_sleep_bins = np.where((large_bin_times >= midpoint_time) & (large_bin_states == 'Sleep'))[0]
    second_half_wake_bins = np.where((large_bin_times >= midpoint_time) & (large_bin_states == 'Wake'))[0]
    
    # Print bin distribution
    print(f"\nBin distribution for merged probes:")
    print(f"First half sleep bins: {len(first_half_sleep_bins)}")
    print(f"First half wake bins: {len(first_half_wake_bins)}")
    print(f"Second half sleep bins: {len(second_half_sleep_bins)}")
    print(f"Second half wake bins: {len(second_half_wake_bins)}")
    
    # Check if we have enough bins for analysis
    total_usable_bins = len(first_half_sleep_bins) + len(first_half_wake_bins) + len(second_half_sleep_bins) + len(second_half_wake_bins)
    if total_usable_bins < 4:
        print("Not enough usable bins to perform stability analysis. Need at least 4 bins.")
        return {}
    
    # Try to find a balanced split
    all_first_half_bins = np.concatenate([first_half_sleep_bins, first_half_wake_bins])
    all_second_half_bins = np.concatenate([second_half_sleep_bins, second_half_wake_bins])
    all_sleep_bins = np.concatenate([first_half_sleep_bins, second_half_sleep_bins])
    all_wake_bins = np.concatenate([first_half_wake_bins, second_half_wake_bins])
    
    # Define constraints
    success = False
    iteration = 0
    
    start_time = time.time()
    while not success and iteration < max_iterations:
        # Create a random assignment (1 = C1, 0 = C2)
        bin_assignment = np.zeros(num_large_bins, dtype=int)
        
        # Randomly assign bins to C1 (1) or C2 (0)
        all_bins = np.arange(num_large_bins)
        usable_bins = np.where(np.isin(all_bins, np.concatenate([all_first_half_bins, all_second_half_bins])))[0]
        
        # Randomly select ~50% of bins for C1
        c1_bins = np.random.choice(usable_bins, size=len(usable_bins)//2, replace=False)
        bin_assignment[c1_bins] = 1
        
        # Check balance criteria
        if len(all_first_half_bins) > 0 and len(all_second_half_bins) > 0:
            first_half_in_c1 = np.sum(bin_assignment[all_first_half_bins])
            first_half_in_c2 = len(all_first_half_bins) - first_half_in_c1
            
            second_half_in_c1 = np.sum(bin_assignment[all_second_half_bins])
            second_half_in_c2 = len(all_second_half_bins) - second_half_in_c1
            
            # Check temporal balance (no more than 70% from one half)
            temporal_balance_c1 = (first_half_in_c1 / (first_half_in_c1 + second_half_in_c1) <= 0.7 and
                                  second_half_in_c1 / (first_half_in_c1 + second_half_in_c1) <= 0.7)
            
            temporal_balance_c2 = (first_half_in_c2 / (first_half_in_c2 + second_half_in_c2) <= 0.7 and
                                  second_half_in_c2 / (first_half_in_c2 + second_half_in_c2) <= 0.7)
            
            # Additional check: try to have at least one sleep/wake bin from each half
            preferred_constraint = True
            
            if len(all_sleep_bins) >= 2 and len(all_wake_bins) >= 2:
                # This is a preferred constraint, not mandatory
                pass
            
            # If both criteria are met, we're successful
            if temporal_balance_c1 and temporal_balance_c2 and preferred_constraint:
                success = True
            elif temporal_balance_c1 and temporal_balance_c2 and iteration > max_iterations//2:
                success = True
        
        iteration += 1
    
    end_time = time.time()
    
    if not success:
        print(f"Could not find a balanced split after {max_iterations} iterations. Using best available split.")
    else:
        print(f"Found balanced split after {iteration} iterations ({end_time - start_time:.2f}s)")
    
    # Calculate firing rates for each category
    c1_mask = bin_assignment == 1
    c2_mask = bin_assignment == 0
    
    # For debugging
    c1_sleep = np.sum(c1_mask & np.isin(np.arange(num_large_bins), all_sleep_bins))
    c1_wake = np.sum(c1_mask & np.isin(np.arange(num_large_bins), all_wake_bins))
    c2_sleep = np.sum(c2_mask & np.isin(np.arange(num_large_bins), all_sleep_bins))
    c2_wake = np.sum(c2_mask & np.isin(np.arange(num_large_bins), all_wake_bins))
    
    print(f"\nFinal bin distribution:")
    print(f"C1: {np.sum(c1_mask)} bins total, {c1_sleep} sleep, {c1_wake} wake")
    print(f"C2: {np.sum(c2_mask)} bins total, {c2_sleep} sleep, {c2_wake} wake")
    
    # Calculate average normalized firing rates in C1 and C2 for each cluster
    c1_rates = np.zeros(normalized_counts.shape[0])
    c2_rates = np.zeros(normalized_counts.shape[0])
    
    # Also calculate sleep-wake difference for each category
    c1_sleep_rates = np.zeros(normalized_counts.shape[0])
    c1_wake_rates = np.zeros(normalized_counts.shape[0])
    c2_sleep_rates = np.zeros(normalized_counts.shape[0])
    c2_wake_rates = np.zeros(normalized_counts.shape[0])
    
    for i in range(normalized_counts.shape[0]):
        # Average normalized rates in each category
        c1_rates[i] = np.mean(large_bin_firing_rates[i, c1_mask]) if np.sum(c1_mask) > 0 else np.nan
        c2_rates[i] = np.mean(large_bin_firing_rates[i, c2_mask]) if np.sum(c2_mask) > 0 else np.nan
        
        # Sleep/wake rates in C1
        c1_sleep_mask = c1_mask & np.isin(np.arange(num_large_bins), all_sleep_bins)
        c1_wake_mask = c1_mask & np.isin(np.arange(num_large_bins), all_wake_bins)
        
        c1_sleep_rates[i] = np.mean(large_bin_firing_rates[i, c1_sleep_mask]) if np.sum(c1_sleep_mask) > 0 else np.nan
        c1_wake_rates[i] = np.mean(large_bin_firing_rates[i, c1_wake_mask]) if np.sum(c1_wake_mask) > 0 else np.nan
        
        # Sleep/wake rates in C2
        c2_sleep_mask = c2_mask & np.isin(np.arange(num_large_bins), all_sleep_bins)
        c2_wake_mask = c2_mask & np.isin(np.arange(num_large_bins), all_wake_bins)
        
        c2_sleep_rates[i] = np.mean(large_bin_firing_rates[i, c2_sleep_mask]) if np.sum(c2_sleep_mask) > 0 else np.nan
        c2_wake_rates[i] = np.mean(large_bin_firing_rates[i, c2_wake_mask]) if np.sum(c2_wake_mask) > 0 else np.nan
    
    # Calculate modulation
    c1_modulation = c1_wake_rates - c1_sleep_rates
    c2_modulation = c2_wake_rates - c2_sleep_rates
    
    # Store results
    stability_results = {
        'merged': {
            'cluster_ids': all_cluster_ids,
            'c1_rates': c1_rates,
            'c2_rates': c2_rates,
            'c1_sleep_rates': c1_sleep_rates,
            'c1_wake_rates': c1_wake_rates,
            'c2_sleep_rates': c2_sleep_rates,
            'c2_wake_rates': c2_wake_rates,
            'c1_modulation': c1_modulation,
            'c2_modulation': c2_modulation
        }
    }
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Average normalized firing rates C1 vs C2
    valid_mask = ~np.isnan(c1_rates) & ~np.isnan(c2_rates)
    
    if np.sum(valid_mask) > 1:
        # Get max value for axis scaling
        max_val = max(np.nanmax(c1_rates), np.nanmax(c2_rates))
        
        # Create scatter plot
        axes[0].scatter(c1_rates[valid_mask], c2_rates[valid_mask], alpha=0.7)
        
        # Add identity line
        axes[0].plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.7)
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            c1_rates[valid_mask], c2_rates[valid_mask]
        )
        
        x_vals = np.array([0, max_val*1.1])
        axes[0].plot(x_vals, intercept + slope * x_vals, 'r-', alpha=0.7,
                    label=f'y = {slope:.2f}x + {intercept:.2f} (r²={r_value**2:.2f}, p={p_value:.4f})')
        
        axes[0].set_xlabel('Category 1 - Average Normalized Firing Rate')
        axes[0].set_ylabel('Category 2 - Average Normalized Firing Rate')
        axes[0].set_title('Merged Probes: Neuronal Stability - Normalized Firing Rate Consistency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Insufficient data for analysis',
                   ha='center', va='center', transform=axes[0].transAxes)
    
    # Plot 2: Sleep-Wake modulation consistency
    valid_mod_mask = (~np.isnan(c1_modulation) & ~np.isnan(c2_modulation))
    
    if np.sum(valid_mod_mask) > 1:
        # Get max absolute value for axis scaling
        max_mod = max(
            np.nanmax(np.abs(c1_modulation)), 
            np.nanmax(np.abs(c2_modulation))
        ) * 1.1
        
        # Create scatter plot
        axes[1].scatter(c1_modulation[valid_mod_mask], c2_modulation[valid_mod_mask], alpha=0.7)
        
        # Add identity line
        axes[1].plot([-max_mod, max_mod], [-max_mod, max_mod], 'k--', alpha=0.7)
        
        # Add regression line
        mod_slope, mod_intercept, mod_r, mod_p, mod_err = stats.linregress(
            c1_modulation[valid_mod_mask], c2_modulation[valid_mod_mask]
        )
        
        x_mod_vals = np.array([-max_mod, max_mod])
        axes[1].plot(x_mod_vals, mod_intercept + mod_slope * x_mod_vals, 'r-', alpha=0.7,
                    label=f'y = {mod_slope:.2f}x + {mod_intercept:.2f} (r²={mod_r**2:.2f}, p={mod_p:.4f})')
        
        # Add quadrant lines
        axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Count points in each quadrant
        q1 = np.sum((c1_modulation > 0) & (c2_modulation > 0) & valid_mod_mask)
        q2 = np.sum((c1_modulation < 0) & (c2_modulation > 0) & valid_mod_mask)
        q3 = np.sum((c1_modulation < 0) & (c2_modulation < 0) & valid_mod_mask)
        q4 = np.sum((c1_modulation > 0) & (c2_modulation < 0) & valid_mod_mask)
        
        print(f"\nModulation quadrant counts:")
        print(f"Q1 (Wake selective in both): {q1}")
        print(f"Q2 (Sleep in C1, Wake in C2): {q2}")
        print(f"Q3 (Sleep selective in both): {q3}")
        print(f"Q4 (Wake in C1, Sleep in C2): {q4}")
        
        axes[1].set_xlabel('Category 1 - Wake-Sleep Modulation (Normalized)')
        axes[1].set_ylabel('Category 2 - Wake-Sleep Modulation (Normalized)')
        axes[1].set_title('Merged Probes: Sleep-Wake Modulation Consistency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Set equal x and y limits
        axes[1].set_xlim(-max_mod, max_mod)
        axes[1].set_ylim(-max_mod, max_mod)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data for modulation analysis',
                   ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"merged_probes_neuronal_stability_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    
    plt.show()
    
    # Calculate stability metrics
    if np.sum(valid_mask) > 1:
        print(f"\nStability metrics for merged probes:")
        print(f"Total clusters analyzed: {len(all_cluster_ids)}")
        print(f"Normalized firing rate correlation: r = {r_value:.4f}, p = {p_value:.4f}")
        print(f"Modulation correlation: r = {mod_r:.4f}, p = {mod_p:.4f}")
        
        # Calculate percentage of neurons that maintain consistent sleep/wake preference
        consistent = (q1 + q3) / np.sum(valid_mod_mask) if np.sum(valid_mod_mask) > 0 else np.nan
        print(f"Neurons with consistent sleep/wake preference: {consistent*100:.1f}%")
    
    return stability_results


def analyze_power_spectrum(results, output_dir=None, save_plots=False, 
                          nperseg=1000, noverlap=500, freq_range=(0, 30), show_plots=True,
                          sleep_bouts=None, use_quality_filter=True):
    """
    Analyze and visualize power spectrum across the recording, focusing on neural oscillations
    like theta (4-8 Hz) and delta (1-4 Hz) rhythms.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results from process_spike_data() with high temporal resolution
        Recommended: use bin_size=0.005 (200Hz) when calling process_spike_data()
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
    nperseg : int, optional
        Length of each segment for spectrogram (default: 1000)
    noverlap : int, optional
        Number of points to overlap between segments (default: 500)
    freq_range : tuple, optional
        Frequency range to analyze in Hz (default: (0, 30))
    show_plots : bool, optional
        Whether to display plots (default: True)
    sleep_bouts : pd.DataFrame, optional
        DataFrame containing sleep bout information for visualization (default: None)
    use_quality_filter : bool, optional
        Whether to use only clusters classified as 'good' or 'mua' by bombcell (default: True)
        
    Returns:
    --------
    dict
        Dictionary containing power spectrum analysis results for each probe
    """
     
    spectrum_results = {}
    
    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if use_quality_filter and 'cluster_quality' not in results[probe]:
            print(f"No quality labels found for {probe}. Include all clusters.")
            # Still include the probe but will use all clusters
        valid_probes.append(probe)
    
    if not valid_probes:
        print("No valid probes found.")
        return spectrum_results
    
    # Use the first probe's time bins
    reference_probe = valid_probes[0]
    time_bins = results[reference_probe]['time_bins']
    
    # Check temporal resolution
    bin_size = time_bins[1] - time_bins[0]
    sampling_rate = 1 / bin_size
    nyquist_freq = sampling_rate / 2
    
    if nyquist_freq < 15:
        print(f"Warning: Temporal resolution too low (bin_size={bin_size:.4f}s).")
        print(f"Maximum detectable frequency is {nyquist_freq:.1f}Hz. Consider using bin_size=0.005s.")
    else:
        print(f"Sampling rate: {sampling_rate:.1f}Hz (bin_size={bin_size:.4f}s)")
        print(f"Maximum detectable frequency: {nyquist_freq:.1f}Hz")
    
    # Initialize sleep mask if sleep_bouts is provided
    sleep_mask = None
    if sleep_bouts is not None and 'sleep_bout_mapping' in results[reference_probe]:
        sleep_bout_mapping = results[reference_probe]['sleep_bout_mapping']
        
        # Create a mask for time bins that fall within any sleep period
        sleep_mask = np.zeros(len(time_bins), dtype=bool)
        for _, bout in sleep_bout_mapping.iterrows():
            start_idx = bout['start_bin_index']
            end_idx = bout['end_bin_index']
            sleep_mask[start_idx:end_idx+1] = True
    
    # Collect data from all probes
    all_counts = []
    all_min_length = float('inf')  # Track the minimum length across probes
    
    # First pass: determine the minimum length
    for probe in valid_probes:
        if use_quality_filter and 'cluster_quality' in results[probe]:
            # Filter out noise clusters, keep good and mua
            counts, cluster_ids, quality_mask = filter_clusters_by_quality(
                results, probe, include_qualities=['good', 'mua']
            )
            print(f"Probe {probe}: Selected {np.sum(quality_mask)} clusters ('good' and 'mua') out of {len(results[probe]['cluster_quality'])}")
        else:
            # Use all clusters
            counts = results[probe]['counts']
            print(f"Probe {probe}: Using all {counts.shape[0]} clusters (no quality filtering)")
        
        if counts.shape[0] > 0:
            # Track minimum length
            all_min_length = min(all_min_length, counts.shape[1])
            all_counts.append(counts)
    
    if not all_counts:
        print("No valid clusters found after filtering.")
        return spectrum_results
    
    # Second pass: truncate all arrays to the minimum length
    for i in range(len(all_counts)):
        all_counts[i] = all_counts[i][:, :all_min_length]
    
    # Merge counts from all probes
    merged_counts = np.vstack(all_counts)
    print(f"Merged {merged_counts.shape[0]} clusters from {len(valid_probes)} probes")
    print(f"Using {all_min_length} time bins (truncated to match across probes)")
    
    # Adjust time_bins to match the truncated data
    time_bins = time_bins[:all_min_length]
    
    # Calculate mean activity across all neurons
    mean_activity = np.mean(merged_counts, axis=0)
    
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        mean_activity,
        fs=sampling_rate,
        window=('hamming'),
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density',
        detrend='constant'
    )
    
    # Convert to dB scale and limit frequency range
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    frequencies_filtered = frequencies[freq_mask]
    Sxx_db_filtered = Sxx_db[freq_mask, :]
    
    # Initialize band_sleep_wake_power for use outside of plotting
    band_sleep_wake_power = {}
    
    if show_plots:
        # Create figure with just 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})
        
        power_min = np.min(Sxx_db_filtered)
        power_max = np.max(Sxx_db_filtered)
        power_mean = np.mean(Sxx_db_filtered)
        power_median = np.median(Sxx_db_filtered)
        power_5th = np.percentile(Sxx_db_filtered, 5)
        power_95th = np.percentile(Sxx_db_filtered, 95)

        print(f"Power spectrum statistics:")
        print(f"Min: {power_min:.1f} dB, Max: {power_max:.1f} dB")
        print(f"Mean: {power_mean:.1f} dB, Median: {power_median:.1f} dB")
        print(f"5th percentile: {power_5th:.1f} dB, 95th percentile: {power_95th:.1f} dB")

        # Use data-driven color limits - slightly wider than the 5-95% range
        vmin = power_5th - 5  # Go slightly below 5th percentile
        vmax = power_95th + 5  # Go slightly above 95th percentile
        
        # Calculate new extent to align with the time bins
        extent = [time_bins[0], time_bins[-1], frequencies_filtered[0], frequencies_filtered[-1]]

        im = axs[0].matshow(Sxx_db_filtered, aspect='auto', origin='lower', 
                      extent=extent, cmap='viridis',
                      vmin=vmin, vmax=vmax)
        
        # Add sleep bout outlines if available
        if sleep_mask is not None:
            for _, bout in sleep_bout_mapping.iterrows():
                if bout['start_bin_time'] <= time_bins[-1] and bout['end_bin_time'] >= time_bins[0]:
                    # Clip to our time range
                    start_time = max(bout['start_bin_time'], time_bins[0])
                    end_time = min(bout['end_bin_time'], time_bins[-1])
                    # Draw vertical lines at bout boundaries
                    axs[0].axvline(x=start_time, color='white', linestyle='--', alpha=0.7)
                    axs[0].axvline(x=end_time, color='white', linestyle='--', alpha=0.7)
        
        # Add horizontal lines for frequency bands (only delta and theta)
        axs[0].axhline(y=1, color='white', linestyle='-', alpha=0.5, label='Delta start (1Hz)')
        axs[0].axhline(y=4, color='white', linestyle='-', alpha=0.5, label='Delta end / Theta start (4Hz)')
        axs[0].axhline(y=8, color='white', linestyle='-', alpha=0.5, label='Theta end (8Hz)')
        
        plt.colorbar(im, ax=axs[0], label='Power (dB)')
        axs[0].set_title('Power Spectrum - All Probes Combined')
        axs[0].set_ylabel('Frequency (Hz)')
        axs[0].set_xlabel('Time (s)')
        axs[0].legend(loc='upper right', fontsize='small')
        
        # Plot 2: Power in specific frequency bands over time (only delta and theta)
        # Define frequency bands
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8)
        }
        
        # Calculate band powers for each time point
        band_powers = {}
        for band_name, (low, high) in bands.items():
            # Get indices for this band
            band_mask = (frequencies >= low) & (frequencies <= high)
            # Calculate mean power across band frequencies
            band_powers[band_name] = np.mean(Sxx_db[band_mask, :], axis=0)
        
        # Create time values that correspond to spectrogram time bins
        spec_times = np.linspace(time_bins[0], time_bins[-1], len(band_powers['Delta']))
        
        pos0 = axs[0].get_position()

        # Remove the existing second subplot
        axs[1].remove()

        # Create a new axes with exactly the same x-position and width as the first plot
        axs[1] = fig.add_axes([pos0.x0,
                            0.1,
                            pos0.width,
                            0.25])

        # Now plot the frequency bands on the correctly positioned axes
        # Plot each band
        for band_name, power in band_powers.items():
            axs[1].plot(spec_times, power, label=band_name, linewidth=1)

        # Set proper x limits to match the first plot
        axs[1].set_xlim(time_bins[0], time_bins[-1])
        
        # Add sleep bout highlights if available
        if sleep_mask is not None:
            for _, bout in sleep_bout_mapping.iterrows():
                if bout['start_bin_time'] <= time_bins[-1] and bout['end_bin_time'] >= time_bins[0]:
                    # Clip to our time range
                    start_time = max(bout['start_bin_time'], time_bins[0])
                    end_time = min(bout['end_bin_time'], time_bins[-1])
                    # Add the span
                    axs[1].axvspan(start_time, end_time, 
                                    color='lightblue', alpha=0.3, ec='none')
        
        axs[1].set_xlim(time_bins[0], time_bins[-1])
        axs[1].set_title('Power in Frequency Bands Over Time')
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_xlabel('Time (s)')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
            
        # Calculate average power in each band during sleep and wake (if sleep data available)
        if sleep_mask is not None:
            # Adjust sleep_mask to match truncated data
            sleep_mask = sleep_mask[:all_min_length]
            wake_mask = ~sleep_mask
            
            for band_name, power in band_powers.items():
                # Map spectrogram times to the closest time bins to use the sleep mask
                time_indices = np.searchsorted(time_bins, spec_times)
                time_indices = np.clip(time_indices, 0, len(sleep_mask) - 1)
                
                # Extract sleep and wake masks corresponding to spectrogram times
                spec_sleep_mask = sleep_mask[time_indices]
                spec_wake_mask = ~spec_sleep_mask
                
                # Calculate mean power during sleep and wake
                sleep_power = np.mean(power[spec_sleep_mask]) if np.any(spec_sleep_mask) else np.nan
                wake_power = np.mean(power[spec_wake_mask]) if np.any(spec_wake_mask) else np.nan
                
                band_sleep_wake_power[band_name] = {
                    'sleep': sleep_power,
                    'wake': wake_power
                }
                
            # Print summary statistics if sleep data is available
            print(f"\nPower spectrum analysis results for merged probes:")
            print("\nAverage power during sleep and wake periods (dB):")
            for band in bands:
                sleep_val = band_sleep_wake_power[band]['sleep']
                wake_val = band_sleep_wake_power[band]['wake']
                diff = sleep_val - wake_val
                print(f"{band}: Sleep: {sleep_val:.2f}, Wake: {wake_val:.2f}, Difference: {diff:.2f}")
        
        # Save plot if requested
        if save_plots and output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save spectrogram figure
            spec_filename = f"merged_probes_power_spectrum_{timestamp}.png"
            spec_filepath = os.path.join(output_dir, spec_filename)
            fig.savefig(spec_filepath, dpi=300, bbox_inches='tight')
            print(f"Saved spectrogram to: {spec_filepath}")
        
        plt.show()
    
    # Define bands for storing in results (do this even if not plotting)
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8)
    }
    
    # Calculate band powers for storage (regardless of plotting)
    band_powers = {}
    for band_name, (low, high) in bands.items():
        band_mask = (frequencies >= low) & (frequencies <= high)
        band_powers[band_name] = np.mean(Sxx_db[band_mask, :], axis=0)
    
    # Store results
    spectrum_results['merged'] = {
        'frequencies': frequencies,
        'times': times,
        'power_spectrum': Sxx,
        'band_powers': band_powers,
        'band_sleep_wake_power': band_sleep_wake_power,
        'time_bins': time_bins  # Store the time bins for later reference
    }
    
    return spectrum_results


def combined_visualization(results, freq_results, np_results, spectrum_results, 
                          dlc_folder, output_dir=None, save_plots=False, smoothed_results=None, pca_results=None):
    """
    Create a combined visualization of sleep-wake activity, power spectrum analysis,
    and behavioral data using pre-computed results.
    
    Parameters:
    -----------
    results : dict
        Original dictionary containing results from process_spike_data()
    freq_results : dict
        Original high temporal resolution data for frequency analysis
    np_results : dict
        Results from analyze_sleep_wake_activity()
    spectrum_results : dict
        Results from analyze_power_spectrum()
    dlc_folder : str
        Path to the DLC folder containing behavioral data
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
        
    Returns:
    --------
    None
    """
    
    # Check which probes have the required data
    valid_probes = []
    for probe in results:
        if 'sleep_bout_mapping' not in results[probe]:
            print(f"No sleep bout mapping found for {probe}. Skipping.")
            continue
        if 'cluster_quality' not in results[probe]:
            print(f"No quality labels found for {probe}. Skipping.")
            continue
        valid_probes.append(probe)
    
    if not valid_probes:
        print("No valid probes found with required data.")
        return
    
    # Use the first probe's time bins and sleep bout info as reference
    reference_probe = valid_probes[0]
    time_bins = results[reference_probe]['time_bins']
    sleep_bouts = results[reference_probe]['sleep_bout_mapping']
    
    # Create a mask for time bins that fall within any sleep period
    sleep_mask = np.zeros(len(time_bins), dtype=bool)
    for _, bout in sleep_bouts.iterrows():
        start_idx = bout['start_bin_index']
        end_idx = bout['end_bin_index']
        sleep_mask[start_idx:end_idx+1] = True

    # Time extent for all plots
    time_extent = [time_bins[0], time_bins[-1]]
    
    # Load behavioral data from DLC pixel_difference folder
    pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
    
    # Find the CSV file containing pixel differences
    pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                      if f.endswith('.csv') and 'pixel_differences' in f]
    
    if not pixel_diff_files:
        print(f"No pixel difference CSV found in {pixel_diff_path}")
        behavior_data = None
    else:
        # Use the first matching file
        pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
        try:
            behavior_data = pd.read_csv(pixel_diff_file)
            print(f"Loaded behavioral data from {pixel_diff_file}")
            
            # Check if the required column exists
            if 'smoothed_difference' not in behavior_data.columns or 'time_sec' not in behavior_data.columns:
                print(f"Required columns not found in {pixel_diff_file}")
                print(f"Available columns: {list(behavior_data.columns)}")
                behavior_data = None
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            behavior_data = None
    
    # Create figure with 5 subplots with custom height ratios (added behavior plot)
    fig = plt.figure(figsize=(16, 24))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1, 3, 1, 1, 3, 1.5], figure=fig)
    
    # ================ PLOT 1: BEHAVIOR DATA ================
    ax_behav = fig.add_subplot(gs[0])
    
    if behavior_data is not None:
        # Extract time and smoothed difference data
        time_sec = behavior_data['time_sec'].values
        smoothed_diff = behavior_data['smoothed_difference'].values
        
        # Handle missing values - create separate segments to plot
        valid_mask = ~np.isnan(smoothed_diff)
        gaps = np.where(np.diff(valid_mask.astype(int)) != 0)[0] + 1
        segments = np.split(np.arange(len(valid_mask)), gaps)
        
        # Plot each continuous segment separately
        for segment in segments:
            if len(segment) > 0 and valid_mask[segment[0]]:  # Only plot valid segments
                ax_behav.plot(time_sec[segment], smoothed_diff[segment], 'k-', linewidth=1)
        
        # Add light colored vertical spans for sleep periods
        for _, bout in sleep_bouts.iterrows():
            ax_behav.axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                            color='lightblue', alpha=0.3, ec='none')
        
        ax_behav.set_title('Motion Activity (Pixel Difference)')
        ax_behav.set_ylabel('Smoothed\nDifference')
        ax_behav.set_xlim(time_extent)
        ax_behav.grid(True, alpha=0.2)
    else:
        ax_behav.text(0.5, 0.5, 'Behavioral data not available', 
                    ha='center', va='center', transform=ax_behav.transAxes)
        ax_behav.set_xlim(time_extent)
    
    # ================ PLOT 2: CLUSTER ACTIVITY HEATMAP ================
    ax1 = fig.add_subplot(gs[1], sharex=ax_behav)
    
    # Get sorted normalized counts from np_results if available
    if 'merged' in np_results and 'modulation_index' in np_results['merged']:
        # Extract data from np_results
        merged_data = np_results['merged']
        modulation_index = merged_data['modulation_index']
        
        # Sort by modulation index
        sorted_indices = np.argsort(modulation_index)
        
        # Get or reconstruct normalized counts
        if 'counts' in merged_data:
            sorted_normalized_counts = merged_data['counts'][sorted_indices]
        else:
            # Need to reconstruct from original data
            # Collect data from all probes
            all_counts = []
            for probe in valid_probes:
                # Filter out noise clusters, keep good and mua
                counts, _, _ = filter_clusters_by_quality(
                    results, probe, include_qualities=['good', 'mua']
                )
                all_counts.append(counts)
                
            merged_counts = np.vstack(all_counts)
            
            # Normalize and sort
            normalized_counts = np.zeros_like(merged_counts, dtype=float)
            for i in range(merged_counts.shape[0]):
                cluster_counts = merged_counts[i, :]
                p95 = np.percentile(cluster_counts, 95)
                
                # Avoid division by zero
                if p95 > 0:
                    normalized_counts[i, :] = cluster_counts / p95
                else:
                    normalized_counts[i, :] = cluster_counts
                    
            sorted_normalized_counts = normalized_counts[sorted_indices]
    else:
        print("Warning: modulation_index not found in np_results. Cannot create sorted heatmap.")
        sorted_normalized_counts = None
        
    if sorted_normalized_counts is not None:
        # Plot the heatmap
        extent = [time_extent[0], time_extent[1], 0, sorted_normalized_counts.shape[0]]
        vmax_threshold = np.percentile(sorted_normalized_counts, 95)
        
        im1 = ax1.matshow(sorted_normalized_counts, aspect='auto', extent=extent, cmap='binary', 
                        interpolation='none', origin='lower', vmin=0, vmax=vmax_threshold)
        
        # Instead of sleep overlays, color the x-axis during sleep periods
        # Save the original x-axis color
        orig_tick_color = ax1.xaxis.get_ticklabels()[0].get_color() if len(ax1.xaxis.get_ticklabels()) > 0 else 'black'
        
        # Set the ticks to blue where there is sleep
        ax1.xaxis.set_tick_params(colors=orig_tick_color)  # Reset all to original
        
        # Create tick positions based on sleep bout boundaries
        sleep_starts = []
        sleep_ends = []
        for _, bout in results[reference_probe]['sleep_bout_mapping'].iterrows():
            # Remove the in_range check to ensure all bouts are included
            sleep_starts.append(bout['start_bin_time'])
            sleep_ends.append(bout['end_bin_time'])
                
        # Add binary sleep/wake indicator at the top of the plot
        # Calculate appropriate height for the indicator line (just above the top of the plot)
        y_max = sorted_normalized_counts.shape[0]
        y_indicator = y_max * 1.02  # Place slightly above the top
        
        # Draw a thin rectangle for each sleep bout at the top
        for start, end in zip(sleep_starts, sleep_ends):
            # Draw a blue line segment at the top for each sleep bout
            ax1.plot([start, end], [y_indicator, y_indicator], color='blue', linewidth=4)
        
        # Add a thin black line across the entire width to serve as the "axis" for the indicator
        ax1.axhline(y=y_indicator, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add small labels on the right side
        ax1.text(time_extent[1]*1.01, y_indicator, 'Sleep', color='blue', 
                verticalalignment='center', fontsize=8)
        
        ax1.set_title(f'All clusters sorted by sleep-wake modulation (n={sorted_normalized_counts.shape[0]})')
        ax1.set_ylabel('Cluster rank\n(sleep-selective → wake-selective)')
        # Note: removing colorbar for this plot to ensure consistent width
        ax1.set_xlim(time_extent)
    
    # ================ PLOT 3: AVERAGE ACTIVITY ================
    ax2 = fig.add_subplot(gs[2], sharex=ax_behav)
    
    # Get or calculate mean activity
    if 'merged' in np_results and 'mean_activity' in np_results['merged']:
        mean_activity = np_results['merged']['mean_activity']
    else:
        # Need to calculate from original data
        all_counts = []
        for probe in valid_probes:
            # Filter out noise clusters, keep good and mua
            counts, _, _ = filter_clusters_by_quality(
                results, probe, include_qualities=['good', 'mua']
            )
            all_counts.append(counts)
            
        if all_counts:
            merged_counts = np.vstack(all_counts)
            mean_activity = np.mean(merged_counts, axis=0)
        else:
            mean_activity = None
            print("Warning: No valid clusters found to calculate mean activity")
    
    if mean_activity is not None:
        # Calculate reasonable y-limits with padding
        buffer_factor = 0.15
        data_range = np.max(mean_activity) - np.min(mean_activity)
        buffer_amount = data_range * buffer_factor
        y_max = np.max(mean_activity) + buffer_amount
        y_min = np.min(mean_activity) - buffer_amount
        
        ax2.plot(time_bins, mean_activity, color='black', linewidth=1)
        ax2.set_ylim(y_min, y_max)
        
        # Add light colored vertical spans for sleep periods
        for _, bout in sleep_bouts.iterrows():
            ax2.axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')

        ax2.grid(True, alpha=0.2)
        ax2.set_title('Average activity across all filtered clusters')
        ax2.set_ylabel('Mean spike count')
        ax2.set_xlim(time_extent)
    
    # ================ PLOT 4: PC1 over time ================

    ax3 = fig.add_subplot(gs[3], sharex=ax_behav)
    if pca_results is not None:
        pc1_data = pca_results['pca_result'][:, 0]
        time_bins_pca = pca_results['time_bins_used']
        ax3.plot(time_bins_pca, pc1_data, color='black', linewidth=0.7)
        ax3.set_title('PC1 over time (raw)')
        ax3.set_ylabel('PC1 Score')
        for _, bout in sleep_bouts.iterrows():
            ax3.axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')
        ax3.grid(True, alpha=0.2)
        ax3.legend()
        ax3.set_xlim(time_extent)
    else:
        print("Warning: PCA results not available. Cannot plot PC1 over time.")


    # ================ PLOT 5: POWER SPECTRUM ================
    ax4 = fig.add_subplot(gs[4], sharex=ax_behav)
    
    # Get or calculate power spectrum
    if 'merged' in spectrum_results and 'power_spectrum' in spectrum_results['merged']:
        spectrum_data = spectrum_results['merged']
        frequencies = spectrum_data['frequencies']
        Sxx = spectrum_data['power_spectrum']
        
        # Convert to dB if not already
        if not np.any(Sxx < 0):  # Assuming it's not already in dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
        else:
            Sxx_db = Sxx
        
        # Apply frequency filtering
        freq_range = (0, 30)
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies_filtered = frequencies[freq_mask]
        Sxx_db_filtered = Sxx_db[freq_mask, :]
    else:
        print("Warning: power_spectrum not found in spectrum_results. Cannot create spectrogram.")
        Sxx_db_filtered = None
        frequencies_filtered = None
    
    if Sxx_db_filtered is not None and frequencies_filtered is not None:
        # Calculate power limits
        power_5th = np.percentile(Sxx_db_filtered, 5)
        power_95th = np.percentile(Sxx_db_filtered, 95)
        vmin = power_5th - 5
        vmax = power_95th + 5
        
        # Get frequency time bins
        if 'times' in spectrum_data:
            spec_times = spectrum_data['times']
            # Calculate spectrogram extent
            spec_extent = [time_extent[0], time_extent[1], frequencies_filtered[0], frequencies_filtered[-1]]
        else:
            # Retrieve from freq_results
            freq_time_bins = freq_results[reference_probe]['time_bins']
            spec_extent = [freq_time_bins[0], freq_time_bins[-1], frequencies_filtered[0], frequencies_filtered[-1]]
        
        im3 = ax4.matshow(Sxx_db_filtered, aspect='auto', origin='lower', 
                        extent=spec_extent, cmap='viridis',
                        vmin=vmin, vmax=vmax)
        
        # Add sleep bout outlines as vertical lines
        for _, bout in sleep_bouts.iterrows():
            ax4.axvline(x=bout['start_bin_time'], color='white', linestyle='--', alpha=0.7)
            ax4.axvline(x=bout['end_bin_time'], color='white', linestyle='--', alpha=0.7)
        
        # Add horizontal lines for frequency bands
        ax4.axhline(y=1, color='white', linestyle='-', alpha=0.5, label='Delta start (1Hz)')
        ax4.axhline(y=4, color='white', linestyle='-', alpha=0.5, label='Delta end / Theta start (4Hz)')
        ax4.axhline(y=8, color='white', linestyle='-', alpha=0.5, label='Theta end (8Hz)')
        
        # Note: removing colorbar for this plot to ensure consistent width
        ax4.set_title('Power Spectrum')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.legend(loc='upper right', fontsize='small')
        ax4.set_xlim(time_extent)  # Use the same time extent as other plots
    
    # ================ PLOT 6: FREQUENCY BAND POWER ================
    ax5 = fig.add_subplot(gs[5], sharex=ax_behav)

    # Check if we have smoothed data available from the smoothed_results parameter
    band_powers = None
    smoothed_available = False
    filter_type = None
    
    # First check if smoothed_results is provided directly
    if smoothed_results is not None:
        # Try to determine which filter was used in save_sleep_periods_to_csv
        if output_dir:
            sleep_times_csv = os.path.join(output_dir, "sleep_times.csv")
            if os.path.exists(sleep_times_csv):
                try:
                    sleep_df = pd.read_csv(sleep_times_csv)
                    if 'filter' in sleep_df.columns and len(sleep_df) > 0:
                        filter_name = sleep_df['filter'].iloc[0]
                        if 'Savitzky-Golay' in filter_name:
                            filter_type = 'sg'
                            print("Using Savitzky-Golay filter for frequency band visualization")
                            if 'savitzky_golay' in smoothed_results:
                                band_powers = smoothed_results['savitzky_golay']
                                smoothed_available = True
                        elif 'MovingAverage' in filter_name or 'Moving Average' in filter_name:
                            filter_type = 'ma'
                            print("Using Moving Average filter for frequency band visualization")
                            if 'moving_average' in smoothed_results:
                                band_powers = smoothed_results['moving_average']
                                smoothed_available = True
                except Exception as e:
                    print(f"Error determining filter type: {e}")
    
    # Fall back to original if smoothed not available
    if not smoothed_available and 'merged' in spectrum_results and 'band_powers' in spectrum_results['merged']:
        band_powers = spectrum_results['merged']['band_powers']
        print("Using original band powers (smoothed data not available)")
    
    if band_powers is not None:
        # Get or calculate spectrogram times
        if 'times' in spectrum_results.get('merged', {}):
            spec_times = np.linspace(time_extent[0], time_extent[1], len(next(iter(band_powers.values()))))
        else:
            # Retrieve from freq_results
            freq_time_bins = freq_results[reference_probe]['time_bins']
            spec_times = np.linspace(freq_time_bins[0], freq_time_bins[-1], len(next(iter(band_powers.values()))))
        
        # Plot each band
        for band_name, power in band_powers.items():
            ax5.plot(spec_times, power, label=f"{band_name} {'(Smoothed)' if smoothed_available else ''}", linewidth=1.5)
        
        # Add sleep bout highlights
        for _, bout in sleep_bouts.iterrows():
            ax5.axvspan(bout['start_bin_time'], bout['end_bin_time'], 
                        color='lightblue', alpha=0.3, ec='none')
        
        title_suffix = " (Smoothed)" if smoothed_available else ""
        ax5.set_title(f'Power in Frequency Bands Over Time{title_suffix}')
        ax5.set_ylabel('Power (dB)')
        ax5.set_xlabel('Time (s)')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(time_extent)  # Use the same time extent as other plots
    
    # Adjust layout to make better use of space now that colorbars are removed
    plt.subplots_adjust(hspace=0.4, left=0.06, right=0.98, top=0.96, bottom=0.05)
    
    # Save plot if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"combined_sleep_wake_spectrum_analysis_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to: {filepath}")
    
    plt.show()


def power_band_smoothing(spectrum_results, df_sleep, window_size=30, poly_order=3, 
                             threshold_ma=-50, threshold_sg=-50, min_duration_s=10,
                             output_dir=None, save_plots=False):
    """
    Test different smoothing approaches for frequency band power signals with sleep detection 
    and comparison to behavior-based sleep times. Includes a power spectrum visualization.
    
    Parameters:
    -----------
    spectrum_results : dict
        Results from analyze_power_spectrum()
    df_sleep : pandas DataFrame
        DataFrame with behavior-based sleep times, containing 'start_timestamp_s' and 'end_timestamp_s' columns
    window_size : int
        Size of smoothing window (should be odd)
    poly_order : int
        Order of polynomial for Savitzky-Golay filter
    threshold_ma : float
        Power threshold (dB) for moving average filter to classify as sleep
    threshold_sg : float
        Power threshold (dB) for Savitzky-Golay filter to classify as sleep
    min_duration_s : float
        Minimum duration (seconds) above threshold to be classified as sleep
    output_dir : str, optional
        Directory to save plots to (default: None)
    save_plots : bool, optional
        Whether to save plots (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing the smoothed versions of the band powers and detected sleep periods
    """
    
    # Check if band_powers exist in the spectrum results
    if 'merged' not in spectrum_results or 'band_powers' not in spectrum_results['merged']:
        print("Band powers not found in spectrum results")
        return None
    
    # Extract band powers - focusing only on Delta
    band_powers = spectrum_results['merged']['band_powers']
    if not band_powers or 'Delta' not in band_powers:
        print("Delta band power not found in spectrum results")
        return None
    
    # Filter to only include Delta
    delta_band = {'Delta': band_powers['Delta']}
    
    # Get time information if available
    if 'times' in spectrum_results['merged']:
        times = spectrum_results['merged']['times']
    else:
        # Create arbitrary time points if not available
        sample_power = next(iter(band_powers.values()))
        times = np.arange(len(sample_power))
    
    # Calculate time step for determining minimum sequence length
    time_step = (times[-1] - times[0]) / len(times)
    min_samples = int(min_duration_s / time_step)
    print(f"Time step: {time_step:.5f}s, minimum sleep duration: {min_duration_s}s ({min_samples} samples)")
    
    # Ensure window_size is odd for Savitzky-Golay
    if window_size % 2 == 0:
        sg_window = window_size + 1
        print(f"Adjusting Savitzky-Golay window to {sg_window} (must be odd)")
    else:
        sg_window = window_size
    
    # Function to apply moving average with special handling for edges
    def moving_average(x, window_size):
        # Create output array
        smoothed = np.zeros_like(x, dtype=float)
        n = len(x)
        
        # Calculate global mean for edge handling
        global_mean = np.mean(x)
        
        # Process each point
        for i in range(n):
            # Calculate window boundaries
            half_window = window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(n, i + half_window + 1)
            
            # Get actual window values
            window_values = x[window_start:window_end]
            
            # For edges, supplement with global mean
            if i < half_window:
                # Beginning edge case
                missing_values = half_window - i
                # Weighted average: window_values and multiple copies of global_mean
                weighted_sum = np.sum(window_values) + global_mean * missing_values
                weighted_n = len(window_values) + missing_values
                smoothed[i] = weighted_sum / weighted_n
            elif i >= n - half_window:
                # End edge case
                missing_values = half_window - (n - i - 1)
                weighted_sum = np.sum(window_values) + global_mean * missing_values
                weighted_n = len(window_values) + missing_values
                smoothed[i] = weighted_sum / weighted_n
            else:
                # Normal case (full window)
                smoothed[i] = np.mean(window_values)
                
        return smoothed
    
    # Function to detect sleep periods based on threshold and minimum duration
    def detect_sleep_periods(power_values, threshold, min_samples, time_array):
        # Create binary mask where power is above threshold
        above_threshold = power_values > threshold
        
        sleep_periods = []
        in_sleep = False
        start_idx = 0
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_sleep:
                # Start of sleep period - exact frame
                start_idx = i
                in_sleep = True
            elif (not above_threshold[i] or np.isnan(power_values[i])) and in_sleep:
                # End of sleep period - exact frame
                end_idx = i - 1
                
                # Only count as sleep if duration exceeds minimum
                # Use actual index values for precision
                if (end_idx - start_idx + 1) >= min_samples:
                    sleep_periods.append(
                        (time_array[start_idx], time_array[end_idx])
                    )
                in_sleep = False
        
        # Handle case where recording ends during sleep
        if in_sleep:
            end_idx = len(above_threshold) - 1
            if (end_idx - start_idx + 1) >= min_samples:
                sleep_periods.append(
                    (time_array[start_idx], time_array[end_idx])
                )
        
        return sleep_periods, above_threshold
    
    # Function to calculate overlap between detected sleep periods and behavioral sleep
    def calculate_overlap(detected_periods, behavioral_df):
        if len(detected_periods) == 0 or behavioral_df.empty:
            return 0, 0, 0
        
        # Convert detected periods to a binary mask
        time_range = times[-1] - times[0]
        num_samples = len(times)
        time_step = time_range / (num_samples - 1)
        
        detected_mask = np.zeros(num_samples, dtype=bool)
        for start, end in detected_periods:
            start_idx = max(0, int((start - times[0]) / time_step))
            end_idx = min(num_samples - 1, int((end - times[0]) / time_step))
            detected_mask[start_idx:end_idx+1] = True
        
        # Convert behavioral sleep to a binary mask
        behavioral_mask = np.zeros(num_samples, dtype=bool)
        for _, row in behavioral_df.iterrows():
            start = row['start_timestamp_s']
            end = row['end_timestamp_s']
            start_idx = max(0, int((start - times[0]) / time_step))
            end_idx = min(num_samples - 1, int((end - times[0]) / time_step))
            behavioral_mask[start_idx:end_idx+1] = True
        
        # Calculate overlap
        intersection = np.sum(detected_mask & behavioral_mask)
        union = np.sum(detected_mask | behavioral_mask)
        
        # Calculate precision and recall
        if np.sum(detected_mask) > 0:
            precision = intersection / np.sum(detected_mask)
        else:
            precision = 0
            
        if np.sum(behavioral_mask) > 0:
            recall = intersection / np.sum(behavioral_mask)
        else:
            recall = 0
        
        # Calculate IoU (Intersection over Union)
        if union > 0:
            iou = intersection / union
        else:
            iou = 0
            
        return precision, recall, iou
    
    # Apply smoothing to Delta band power
    smoothed_powers_ma = {}  # Moving average
    smoothed_powers_sg = {}  # Savitzky-Golay
    
    for band_name, power in delta_band.items():
        # Apply moving average with edge handling
        smoothed_powers_ma[band_name] = moving_average(power, window_size)
        
        # Apply Savitzky-Golay filter
        # If the signal is too short for the window, reduce window size
        if len(power) < sg_window:
            temp_window = len(power) - 1
            if temp_window % 2 == 0:
                temp_window -= 1
            if temp_window < poly_order + 1:
                # Can't apply SG filter, use moving average instead
                smoothed_powers_sg[band_name] = smoothed_powers_ma[band_name]
                print(f"Signal too short for SG filter, using moving average for {band_name}")
            else:
                smoothed_powers_sg[band_name] = savgol_filter(power, temp_window, poly_order)
                print(f"Reduced SG window to {temp_window} for {band_name} due to short signal")
        else:
            smoothed_powers_sg[band_name] = savgol_filter(power, sg_window, poly_order)
    
    # Get Delta band power
    delta_power = delta_band['Delta']
    delta_ma = smoothed_powers_ma['Delta']
    delta_sg = smoothed_powers_sg['Delta']
    
    # Detect sleep periods based on moving average smoothed power
    sleep_periods_ma, above_threshold_ma = detect_sleep_periods(
        delta_ma, threshold_ma, min_samples, times)
    
    # Calculate total sleep time for MA
    total_sleep_time_ma = sum(end-start for start, end in sleep_periods_ma)
    sleep_percentage_ma = total_sleep_time_ma / (times[-1] - times[0]) * 100
    
    # Detect sleep periods based on Savitzky-Golay smoothed power
    sleep_periods_sg, above_threshold_sg = detect_sleep_periods(
        delta_sg, threshold_sg, min_samples, times)
    
    # Calculate total sleep time for SG
    total_sleep_time_sg = sum(end-start for start, end in sleep_periods_sg)
    sleep_percentage_sg = total_sleep_time_sg / (times[-1] - times[0]) * 100
    
    # Calculate total behavioral sleep time
    behavioral_sleep_time = sum(row['end_timestamp_s'] - row['start_timestamp_s'] for _, row in df_sleep.iterrows())
    behavioral_sleep_percentage = behavioral_sleep_time / (times[-1] - times[0]) * 100
    
    # Calculate overlap between detected and behavioral sleep
    precision_ma, recall_ma, iou_ma = calculate_overlap(sleep_periods_ma, df_sleep)
    precision_sg, recall_sg, iou_sg = calculate_overlap(sleep_periods_sg, df_sleep)
    
    # Print sleep statistics
    print("\nBehavioral Sleep Statistics:")
    print(f"- {len(df_sleep)} sleep bouts")
    print(f"- Total sleep time: {behavioral_sleep_time:.1f}s ({behavioral_sleep_percentage:.1f}% of recording)")
    
    print(f"\nMoving Average Sleep Detection (threshold={threshold_ma}dB, min duration={min_duration_s}s):")
    print(f"- Detected {len(sleep_periods_ma)} sleep bouts")
    print(f"- Total sleep time: {total_sleep_time_ma:.1f}s ({sleep_percentage_ma:.1f}% of recording)")
    print(f"- Comparison to behavioral sleep:")
    print(f"  * Precision: {precision_ma:.2f} (proportion of detected sleep that matches behavior)")
    print(f"  * Recall: {recall_ma:.2f} (proportion of behavioral sleep that was detected)")
    print(f"  * IoU: {iou_ma:.2f} (Intersection over Union)")
    
    print(f"\nSavitzky-Golay Sleep Detection (threshold={threshold_sg}dB, min duration={min_duration_s}s):")
    print(f"- Detected {len(sleep_periods_sg)} sleep bouts")
    print(f"- Total sleep time: {total_sleep_time_sg:.1f}s ({sleep_percentage_sg:.1f}% of recording)")
    print(f"- Comparison to behavioral sleep:")
    print(f"  * Precision: {precision_sg:.2f} (proportion of detected sleep that matches behavior)")
    print(f"  * Recall: {recall_sg:.2f} (proportion of behavioral sleep that was detected)")
    print(f"  * IoU: {iou_sg:.2f} (Intersection over Union)")
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1.5, 1.5, 2], figure=fig)
    
    # Plot 1: Behavioral Sleep
    ax_beh = fig.add_subplot(gs[0])
    
    # Create a behavior mask to plot
    behavior_mask = np.zeros(len(times))
    for _, row in df_sleep.iterrows():
        start = row['start_timestamp_s']
        end = row['end_timestamp_s']
        start_idx = max(0, int((start - times[0]) / time_step))
        end_idx = min(len(times) - 1, int((end - times[0]) / time_step))
        behavior_mask[start_idx:end_idx+1] = 1
    
    # Plot the behavioral sleep mask
    ax_beh.plot(times, behavior_mask, 'k-', linewidth=1.5)
    ax_beh.set_yticks([0, 1])
    ax_beh.set_yticklabels(['Wake', 'Sleep'])
    ax_beh.set_title('Behavioral Sleep Classification')
    ax_beh.grid(True, alpha=0.3)
    
    # Add light colored vertical spans for behavioral sleep periods
    for _, row in df_sleep.iterrows():
        ax_beh.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                      color='lightblue', alpha=0.3, ec='none')
    
    # Plot 2: Original vs Moving Average with sleep detection
    ax1 = fig.add_subplot(gs[1], sharex=ax_beh)
    ax1.plot(times, delta_power, color='gray', alpha=0.7, label='Original', linewidth=1)
    ax1.plot(times, delta_ma, color='blue', label=f'Moving Avg (window={window_size})', linewidth=2)
    ax1.axhline(y=threshold_ma, color='blue', linestyle='--', alpha=0.7, 
                label=f'Threshold ({threshold_ma}dB)')
    
    # Add sleep period backgrounds for moving average
    for start, end in sleep_periods_ma:
        ax1.axvspan(start, end, color='lightblue', alpha=0.3, ec='none')
    
    ax1.set_title(f'Delta Band Power - Moving Average Sleep Detection ({sleep_percentage_ma:.1f}% sleep, IoU={iou_ma:.2f})')
    ax1.set_ylabel('Power (dB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 3: Original vs Savitzky-Golay with sleep detection
    ax2 = fig.add_subplot(gs[2], sharex=ax_beh)
    ax2.plot(times, delta_power, color='gray', alpha=0.7, label='Original', linewidth=1)
    ax2.plot(times, delta_sg, color='red', label=f'Savitzky-Golay (window={sg_window}, order={poly_order})', linewidth=2)
    ax2.axhline(y=threshold_sg, color='red', linestyle='--', alpha=0.7,
                label=f'Threshold ({threshold_sg}dB)')
    
    # Add sleep period backgrounds for Savitzky-Golay
    for start, end in sleep_periods_sg:
        ax2.axvspan(start, end, color='lightblue', alpha=0.3, ec='none')
    
    ax2.set_title(f'Delta Band Power - Savitzky-Golay Sleep Detection ({sleep_percentage_sg:.1f}% sleep, IoU={iou_sg:.2f})')
    ax2.set_ylabel('Power (dB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 4: Power Spectrum
    ax3 = fig.add_subplot(gs[3])
    
    # Get or calculate power spectrum
    if 'merged' in spectrum_results and 'power_spectrum' in spectrum_results['merged']:
        spectrum_data = spectrum_results['merged']
        frequencies = spectrum_data['frequencies']
        Sxx = spectrum_data['power_spectrum']
        
        # Convert to dB if not already
        if not np.any(Sxx < 0):  # Assuming it's not already in dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
        else:
            Sxx_db = Sxx
        
        # Apply frequency filtering
        freq_range = (0, 30)
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        frequencies_filtered = frequencies[freq_mask]
        Sxx_db_filtered = Sxx_db[freq_mask, :]
    else:
        print("Warning: power_spectrum not found in spectrum_results. Cannot create spectrogram.")
        Sxx_db_filtered = None
        frequencies_filtered = None
    
    if Sxx_db_filtered is not None and frequencies_filtered is not None:
        # Calculate power limits
        power_5th = np.percentile(Sxx_db_filtered, 5)
        power_95th = np.percentile(Sxx_db_filtered, 95)
        vmin = power_5th - 5
        vmax = power_95th + 5
        
        # Get frequency time bins
        if 'times' in spectrum_data:
            spec_times = spectrum_data['times']
            # Calculate spectrogram extent
            time_extent = [times[0], times[-1]]
            spec_extent = [time_extent[0], time_extent[1], frequencies_filtered[0], frequencies_filtered[-1]]
        else:
            # Use the same times as for the delta band
            time_extent = [times[0], times[-1]]
            spec_extent = [time_extent[0], time_extent[1], frequencies_filtered[0], frequencies_filtered[-1]]
        
        im3 = ax3.matshow(Sxx_db_filtered, aspect='auto', origin='lower', 
                       extent=spec_extent, cmap='viridis',
                       vmin=vmin, vmax=vmax)
        
        # Add behavioral sleep period vertical lines
        for _, row in df_sleep.iterrows():
            ax3.axvline(x=row['start_timestamp_s'], color='white', linestyle='--', alpha=0.7)
            ax3.axvline(x=row['end_timestamp_s'], color='white', linestyle='--', alpha=0.7)
        
        # Add horizontal lines for frequency bands
        ax3.axhline(y=1, color='white', linestyle='-', alpha=0.5, label='Delta start (1Hz)')
        ax3.axhline(y=4, color='white', linestyle='-', alpha=0.5, label='Delta end / Theta start (4Hz)')
        ax3.axhline(y=8, color='white', linestyle='-', alpha=0.5, label='Theta end (8Hz)')
        
        ax3.set_title('Power Spectrum')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.legend(loc='upper right', fontsize='small')
        ax3.set_xlim(time_extent)  # Use the same time extent as other plots
    else:
        ax3.text(0.5, 0.5, 'Power spectrum data not available', ha='center', va='center')
        ax3.set_title('Power Spectrum (Not Available)')
        ax3.set_ylabel('Frequency (Hz)')
        
    # Adjust layout
    plt.tight_layout()
    
    # Add a common x-axis label
    fig.text(0.5, 0.01, 'Time (s)', ha='center', va='center')
    
    # Save plot if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"delta_power_sleep_detection_w{window_size}_t{threshold_ma}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved sleep detection plot to: {filepath}")
    
    plt.show()
    
    # Create dataframes of sleep periods for both methods
    sleep_df_ma = pd.DataFrame(sleep_periods_ma, columns=['start_time', 'end_time'])
    if not sleep_df_ma.empty:
        sleep_df_ma['duration'] = sleep_df_ma['end_time'] - sleep_df_ma['start_time']
    sleep_df_ma['method'] = 'Moving Average'
    
    sleep_df_sg = pd.DataFrame(sleep_periods_sg, columns=['start_time', 'end_time'])
    if not sleep_df_sg.empty:
        sleep_df_sg['duration'] = sleep_df_sg['end_time'] - sleep_df_sg['start_time']
    sleep_df_sg['method'] = 'Savitzky-Golay'
    
    # Return the smoothed signals and sleep detection results
    return {
        'original': delta_band,
        'moving_average': smoothed_powers_ma,
        'savitzky_golay': smoothed_powers_sg,
        'window_size': window_size,
        'sg_window': sg_window,
        'poly_order': poly_order,
        'sleep_periods_ma': sleep_periods_ma,
        'sleep_periods_sg': sleep_periods_sg,
        'sleep_df_ma': sleep_df_ma,
        'sleep_df_sg': sleep_df_sg,
        'above_threshold_ma': above_threshold_ma,
        'above_threshold_sg': above_threshold_sg,
        'threshold_ma': threshold_ma,
        'threshold_sg': threshold_sg,
        'min_duration_s': min_duration_s,
        'precision_ma': precision_ma,
        'recall_ma': recall_ma,
        'iou_ma': iou_ma,
        'precision_sg': precision_sg,
        'recall_sg': recall_sg,
        'iou_sg': iou_sg
    }


def save_sleep_periods_to_csv(smoothed_results, output_dir, used_filter='SG', frame_rate=60):
    """
    Save the detected sleep periods to a CSV file with accurate frame conversions.
    
    Parameters:
    -----------
    smoothed_results : dict
        Results from power_band_smoothing() containing the detected sleep periods
    output_dir : str
        Directory to save the CSV file
    used_filter : str, optional
        Which filter to use for sleep periods ('SG' for Savitzky-Golay or 'MA' for Moving Average)
        Default is 'SG'
    frame_rate : float, optional
        Camera frame rate in frames per second (default: 60)
    
    Returns:
    --------
    str
        Path to the saved CSV file
    """
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Select the appropriate sleep periods based on the filter type
    if used_filter.upper() == 'SG':
        sleep_periods = smoothed_results['sleep_periods_sg']
        filter_name = "Savitzky-Golay"
    elif used_filter.upper() == 'MA':
        sleep_periods = smoothed_results['sleep_periods_ma']
        filter_name = "MovingAverage"
    else:
        raise ValueError("Invalid filter type. Use 'SG' for Savitzky-Golay or 'MA' for Moving Average")
    
    if not sleep_periods:
        print(f"No sleep periods detected using {filter_name} filter.")
        df = pd.DataFrame(columns=['start_frame', 'end_frame', 'start_time_s', 'end_time_s', 'duration_s', 'filter'])
    else:
        # Create data for the DataFrame with precise frame calculations
        data = []
        for start_time, end_time in sleep_periods:
            # Convert times to frames using the frame rate
            # Adding 1 to ensure we never have frame 0 (frames are 1-indexed)
            start_frame = int(round(start_time * frame_rate)) + 1
            end_frame = int(round(end_time * frame_rate)) + 1
            
            # Calculate duration from timestamps
            duration_s = end_time - start_time
            
            # Add to data list
            data.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_timestamp_s': start_time,
                'end_timestamp_s': end_time,
                'duration_s': duration_s,
                'filter': filter_name
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"sleep_times.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved detected sleep periods to: {filepath}")
    print(f"Total sleep periods: {len(df)}")
    if len(df) > 0:
        total_sleep_time = df['duration_s'].sum()
        print(f"Total sleep time: {total_sleep_time:.1f} seconds")
        print(f"Frames calculated using frame rate of {frame_rate} fps")
        print(f"Frame range: {df['start_frame'].min()}-{df['end_frame'].max()}")
    
    return filepath



def combine_neural_data(results_dict, output_folder, subject, exp_date, exp_num, bin_type='low_res'):
    """
    Combines neural data from multiple probes and saves as numpy array.
    
    Parameters:
    ----------
    results_dict : dict
        Either 'results' (low-res, 100ms bins) or 'freq_results' (high-res, 5ms bins)
    output_folder : str
        Directory to save output file
    subject : str
        Subject ID
    exp_date : str
        Experiment date
    exp_num : str
        Experiment number
    bin_type : str
        'low_res' (100ms) or 'high_res' (5ms) to indicate in filename
        
    Returns:
    --------
    str : Path to saved file
    """
    # Extract and combine good/MUA neurons from both probes
    probe0_matrix = results_dict['probe0']['counts'][results_dict['probe0']['good_mua_cluster_mask'], :]
    probe1_matrix = results_dict['probe1']['counts'][results_dict['probe1']['good_mua_cluster_mask'], :]
    
    # Ensure same number of timepoints
    min_timepoints = min(probe0_matrix.shape[1], probe1_matrix.shape[1])
    probe0_matrix = probe0_matrix[:, :min_timepoints]
    probe1_matrix = probe1_matrix[:, :min_timepoints]
    
    # Stack matrices and convert to float32
    combined_matrix = np.vstack((probe0_matrix, probe1_matrix))
    combined_matrix = combined_matrix.astype(np.float32)
    
    # Save to NPY file
    output_file = os.path.join(output_folder, 
                              f"{subject}_{exp_date}_{exp_num}_combined_neural_{bin_type}.npy")
    np.save(output_file, combined_matrix)
    
    print(f"Saved combined matrix with shape {combined_matrix.shape} to {output_file}")
    return output_file

def analyze_neural_pca(combined_matrix, state_labels, time_bins, neural_sleep_df, 
                     subject, output_folder, n_components=3, components_to_plot=(0,1), 
                     downsample_factor=1, pc_index_to_plot=0):
    """
    Performs PCA on neural data, creates visualizations, and returns results.
    
    Parameters:
    ----------
    combined_matrix : ndarray
        Neural data matrix (neurons x timepoints)
    state_labels : ndarray
        Labels for each timepoint (0=wake, 1=sleep)
    time_bins : ndarray
        Time bins corresponding to data points
    neural_sleep_df : DataFrame
        DataFrame containing sleep bout information
    subject : str
        Subject ID for plot titles
    output_folder : str
        Directory to save output plots
    n_components : int
        Number of PCA components to compute
    components_to_plot : tuple
        Which components to plot (zero-indexed)
    downsample_factor : int
        Factor to downsample data for faster computation (1 = no downsampling)
    pc_index_to_plot : int
        Which PC index to use for the timeseries plot (zero-indexed)
        
    Returns:
    --------
    dict : PCA results and analysis
    """
    # Transpose matrix for PCA (timepoints x neurons)
    X = combined_matrix.T
    
    # Apply downsampling if requested
    if downsample_factor > 1:
        X = X[::downsample_factor]
        state_labels_used = state_labels[::downsample_factor]
        time_bins_used = time_bins[::downsample_factor]
    else:
        state_labels_used = state_labels
        time_bins_used = time_bins
    
    print(f"Shape of data for PCA: {X.shape}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA with requested number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled) # Shape: (n_samples, n_components)
    
    # Create masks for sleep and wake based on state_labels_used
    # Assuming 0 = Wake, 1 = Sleep in state_labels_used
    sleep_mask_orig = state_labels_used == 1
    wake_mask_orig = state_labels_used == 0
    
    # Plot the selected components
    comp1_idx, comp2_idx = components_to_plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot the selected components - CORRECTED COLORS
    ax1.scatter(pca_result[wake_mask_orig, comp1_idx], pca_result[wake_mask_orig, comp2_idx], 
              c='orange', alpha=0.5, s=5, label='Wake') # Wake is orange
    ax1.scatter(pca_result[sleep_mask_orig, comp1_idx], pca_result[sleep_mask_orig, comp2_idx], 
              c='blue', alpha=0.5, s=5, label='Sleep')   # Sleep is blue
    ax1.set_xlabel(f'PC{comp1_idx+1} ({pca.explained_variance_ratio_[comp1_idx]:.1%} variance)')
    ax1.set_ylabel(f'PC{comp2_idx+1} ({pca.explained_variance_ratio_[comp2_idx]:.1%} variance)')
    ax1.set_title(f'{subject}: Neural State Space PC{comp1_idx+1} vs PC{comp2_idx+1}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Plot explained variance
    ax2.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 'o-', color='blue')
    ax2.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'o-', color='red')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50% Explained Variance')
    ax2.grid(True)
    ax2.legend(['Individual', 'Cumulative', '50% Threshold'])
    
    plt.tight_layout()
    original_scatter_filename = os.path.join(output_folder, f"{subject}_pca_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_scatter_states.png")
    plt.savefig(original_scatter_filename, dpi=300)
    plt.show()
    
    # Plot PC (specified by pc_index_to_plot) over time with sleep bouts highlighted
    plt.figure(figsize=(15, 6))
    plt.plot(time_bins_used, pca_result[:, pc_index_to_plot], 'k-', linewidth=0.5)
    
    y_min_ts, y_max_ts = plt.ylim() # Get y-limits for axvspan
    for _, row in neural_sleep_df.iterrows():
        plt.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                   ymin=0, ymax=1, color='blue', alpha=0.2) # ymin/ymax relative to axes

    plt.title(f"{subject}: PC{pc_index_to_plot +1} over Recording Time")
    plt.xlabel('Time (s)')
    plt.ylabel(f'PC{pc_index_to_plot +1} Score')
    plt.tight_layout()
    timeseries_plot_filename = os.path.join(output_folder, f"{subject}_pc{pc_index_to_plot +1}_timeseries.png")
    plt.savefig(timeseries_plot_filename, dpi=300)
    plt.show()
    
    # --- Integration of New Plots ---
    pc1_scores_new_plots = pca_result[:, comp1_idx]
    pc2_scores_new_plots = pca_result[:, comp2_idx]
    explained_var_pc1_new = pca.explained_variance_ratio_[comp1_idx]
    explained_var_pc2_new = pca.explained_variance_ratio_[comp2_idx]

    # --- Prepare bout information (common for new Plot A & B) ---
    all_bouts_list_new = []
    last_time_point_new = time_bins_used[-1] if len(time_bins_used) > 0 else 0
    current_time_new = time_bins_used[0] if len(time_bins_used) > 0 else 0

    df_neural_sleep_sorted_new = neural_sleep_df.sort_values(by='start_timestamp_s').reset_index()

    for _, sleep_bout_new in df_neural_sleep_sorted_new.iterrows():
        if sleep_bout_new['start_timestamp_s'] > current_time_new:
            all_bouts_list_new.append({'start': current_time_new, 'end': sleep_bout_new['start_timestamp_s'], 'type': 'wake'})
        all_bouts_list_new.append({'start': sleep_bout_new['start_timestamp_s'], 'end': sleep_bout_new['end_timestamp_s'], 'type': 'sleep'})
        current_time_new = sleep_bout_new['end_timestamp_s']
    if len(time_bins_used) > 0 and current_time_new < last_time_point_new:
        all_bouts_list_new.append({'start': current_time_new, 'end': last_time_point_new, 'type': 'wake'})
    
    df_all_bouts_new = pd.DataFrame(all_bouts_list_new)
            
    # Create masks for sleep and wake points for density plot (Plot A)
    density_sleep_mask = np.zeros(len(time_bins_used), dtype=bool)
    density_wake_mask = np.zeros(len(time_bins_used), dtype=bool)

    if not df_all_bouts_new.empty and len(time_bins_used) > 0:
        for _, bout_new in df_all_bouts_new.iterrows():
            bout_indices_new = (time_bins_used >= bout_new['start']) & (time_bins_used < bout_new['end'])
            if bout_new['type'] == 'sleep':
                density_sleep_mask[bout_indices_new] = True
            else: # wake
                density_wake_mask[bout_indices_new] = True
            
    pc1_sleep_density_new = pc1_scores_new_plots[density_sleep_mask]
    pc2_sleep_density_new = pc2_scores_new_plots[density_sleep_mask]
    pc1_wake_density_new = pc1_scores_new_plots[density_wake_mask]
    pc2_wake_density_new = pc2_scores_new_plots[density_wake_mask]

    # --- Custom Colormaps for Plot A (Density) ---
    custom_blues_colors_new = [(0, "#08306b"), (0.5, "#4292c6"), (1, "#c6dbef")] 
    custom_blues_cmap_new = LinearSegmentedColormap.from_list("custom_blues_new", custom_blues_colors_new)
    custom_oranges_colors_new = [(0, "#a63603"), (0.5, "#fd8d3c"), (1, "#feedde")]
    custom_oranges_cmap_new = LinearSegmentedColormap.from_list("custom_oranges_new", custom_oranges_colors_new)

    # --- Plot A: PC1 vs PC2 Density Plot (State-Specific Colors, White Background) ---
    plt.figure(figsize=(10, 8))
    if len(pc1_wake_density_new) > 1 and len(pc2_wake_density_new) > 1:
        sns.kdeplot(x=pc1_wake_density_new, y=pc2_wake_density_new, cmap=custom_oranges_cmap_new, fill=True, thresh=0.05, alpha=0.75, n_levels=100, label="Wake Density")
    if len(pc1_sleep_density_new) > 1 and len(pc2_sleep_density_new) > 1:
        sns.kdeplot(x=pc1_sleep_density_new, y=pc2_sleep_density_new, cmap=custom_blues_cmap_new, fill=True, thresh=0.05, alpha=0.75, n_levels=100, label="Sleep Density")
    
    plt.xlabel(f'PC{comp1_idx + 1} ({explained_var_pc1_new:.1%} variance)')
    plt.ylabel(f'PC{comp2_idx + 1} ({explained_var_pc2_new:.1%} variance)')
    plt.title(f'{subject}: PC{comp1_idx + 1} vs PC{comp2_idx + 1} State-Specific Density')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.legend()
    plot_a_filename_new = os.path.join(output_folder, f"{subject}_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_density_states.png")
    plt.savefig(plot_a_filename_new, dpi=300)
    plt.show()

    # --- Plot B: PC1 vs PC2 Scatter with Discrete Phased Coloring ---
    plt.figure(figsize=(12, 10))
    
    num_phases_new = 3 
    phase_labels_new = ['Early', 'Mid', 'Late'] if num_phases_new == 3 else ['Early', 'Mid-Early', 'Mid-Late', 'Late']
    
    sleep_phase_palette_new = ListedColormap(plt.cm.Blues(np.linspace(0.8, 0.3, num_phases_new)))
    wake_phase_palette_new = ListedColormap(plt.cm.Oranges(np.linspace(0.8, 0.3, num_phases_new)))
    
    point_colors_phased_new = ['gray'] * len(time_bins_used) if len(time_bins_used) > 0 else []

    if len(time_bins_used) > 0 and not df_all_bouts_new.empty:
        for i_new, t_new in enumerate(time_bins_used):
            for _, bout_new_phased in df_all_bouts_new.iterrows():
                if bout_new_phased['start'] <= t_new < bout_new_phased['end']:
                    bout_duration_new = bout_new_phased['end'] - bout_new_phased['start']
                    if bout_duration_new > 0:
                        relative_pos_new = (t_new - bout_new_phased['start']) / bout_duration_new
                        phase_index_new = min(int(relative_pos_new * num_phases_new), num_phases_new - 1)
                        
                        if bout_new_phased['type'] == 'sleep':
                            point_colors_phased_new[i_new] = sleep_phase_palette_new(phase_index_new)
                        elif bout_new_phased['type'] == 'wake':
                            point_colors_phased_new[i_new] = wake_phase_palette_new(phase_index_new)
                    break 
    
    if len(pc1_scores_new_plots) > 0 and len(point_colors_phased_new) == len(pc1_scores_new_plots):
        plt.scatter(pc1_scores_new_plots, pc2_scores_new_plots, c=point_colors_phased_new, s=5, alpha=0.6)

    legend_elements_b_new = []
    for phase_idx_new in range(num_phases_new):
        legend_elements_b_new.append(Line2D([0], [0], marker='o', color='w', 
                                         label=f'{phase_labels_new[phase_idx_new]} Sleep',
                                         markerfacecolor=sleep_phase_palette_new(phase_idx_new), markersize=8))
    for phase_idx_new in range(num_phases_new):
        legend_elements_b_new.append(Line2D([0], [0], marker='o', color='w', 
                                         label=f'{phase_labels_new[phase_idx_new]} Wake',
                                         markerfacecolor=wake_phase_palette_new(phase_idx_new), markersize=8))

    plt.xlabel(f'PC{comp1_idx + 1} ({explained_var_pc1_new:.1%} variance)')
    plt.ylabel(f'PC{comp2_idx + 1} ({explained_var_pc2_new:.1%} variance)')
    plt.title(f'{subject}: PC{comp1_idx + 1} vs PC{comp2_idx + 1} - Discrete Phased Coloring')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    if legend_elements_b_new:
        plt.legend(handles=legend_elements_b_new, loc='best', title="Bout Phase:", ncol=2)
    plot_b_filename_new = os.path.join(output_folder, f"{subject}_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_phased_scatter.png")
    plt.savefig(plot_b_filename_new, dpi=300)
    plt.show()

    # --- End of New Plot Integration ---

    # Calculate component thresholds
    variance_thresholds = [0.5, 0.7, 0.9]
    components_needed = {}
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    for threshold_val in variance_thresholds: # Renamed threshold to threshold_val to avoid conflict
        n_needed = np.argmax(cumulative_var >= threshold_val) + 1 if any(cumulative_var >= threshold_val) else n_components
        components_needed[threshold_val] = n_needed
        print(f"Components needed for {threshold_val*100:.0f}% variance: {n_needed}")
    
    # Return results dictionary (pca_result is n_samples x n_components)
    return {
        'pca': pca,
        'pca_result': pca_result, 
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components_needed': components_needed,
        'state_labels_used': state_labels_used,
        'time_bins_used': time_bins_used
    }

def analyze_pc_smoothing(pca_results, df_neural_sleep, subject, output_folder, 
                         window_sizes=[10, 20, 40, 60], comparison_window=10, pc_index_to_plot=0):
    """
    Analyze PC smoothing with different window sizes and create comparison plots
    
    Parameters:
    -----------
    pca_results : dict
        Results from analyze_neural_pca function
    df_neural_sleep : pd.DataFrame
        DataFrame containing sleep bout information with 'start_timestamp_s' and 'end_timestamp_s'
    subject : str
        Subject identifier for plot titles and filenames
    output_folder : str
        Directory to save plots
    window_sizes : list, optional
        List of window sizes to test for smoothing (default: [10, 20, 40, 60])
    comparison_window : int, optional
        Window size to use for the single comparison plot (default: 10)
        
    Returns:
    --------
    dict
        Dictionary containing smoothed PC data for each window size
    """
    
    # Extract PC from pca_results
    if 'pca_result' not in pca_results:
        print("Error: pca_results not found. Please run the PCA analysis first.")
        return None

    pc_data = pca_results['pca_result'][:, pc_index_to_plot]  # First component
    time_bins_pca = pca_results['time_bins_used']  # Use the correct key
    state_labels_pca = pca_results['state_labels_used']  # Use the correct key

    print(f"PC{pc_index_to_plot + 1} data length: {len(pc_data)}")
    print(f"Time bins length: {len(time_bins_pca)}")
    print(f"State labels length: {len(state_labels_pca)}")
    
    # Calculate session average for PC
    session_avg_pc = np.mean(pc_data)

    # Store results
    smoothing_results = {
        f'pc{pc_index_to_plot + 1}_original': pc_data,
        'time_bins': time_bins_pca,
        'state_labels': state_labels_pca,
        'session_average': session_avg_pc,
        'smoothed_data': {}
    }
    
    # Create comparison plot with multiple window sizes
    fig, axes = plt.subplots(len(window_sizes), 1, figsize=(15, 3*len(window_sizes)), sharex=True)
    if len(window_sizes) == 1:
        axes = [axes]
    
    for i, window_size in enumerate(window_sizes):
        # Apply smoothing
        pc_smoothed = moving_average_with_padding(pc_data, window_size, session_avg_pc)
        smoothing_results['smoothed_data'][window_size] = pc_smoothed

        print(f"Window {window_size}: PC{pc_index_to_plot + 1} original length: {len(pc_data)}, smoothed length: {len(pc_smoothed)}")

        # Plot smoothed PC
        axes[i].plot(time_bins_pca, pc_smoothed, 'k-', linewidth=1.5,
                    label=f'PC{pc_index_to_plot + 1} (smoothed, window={window_size})')

        # Add sleep periods as blue background
        for _, row in df_neural_sleep.iterrows():
            start_time = row['start_timestamp_s']
            end_time = row['end_timestamp_s']
            axes[i].axvspan(start_time, end_time, alpha=0.3, color='lightblue', 
                           label='Sleep' if i == 0 and row.name == 0 else "")
        
        # Formatting
        axes[i].set_ylabel(f'PC{pc_index_to_plot + 1} Score')
        axes[i].set_title(f'PC{pc_index_to_plot + 1} over Time - Moving Average (Window: {window_size} bins)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Print some statistics about the smoothing
        original_std = np.std(pc_data)
        smoothed_std = np.std(pc_smoothed)
        print(f"Window {window_size}: Original STD={original_std:.3f}, Smoothed STD={smoothed_std:.3f}, "
              f"Reduction={((original_std-smoothed_std)/original_std*100):.1f}%")
    
    # Finalize comparison plot
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{subject}_pc{pc_index_to_plot + 1}_smoothed_comparison.png"), dpi=300)
    plt.show()
    
    # Create single plot with specified window for comparison
    if comparison_window in smoothing_results['smoothed_data']:
        window_smoothed = smoothing_results['smoothed_data'][comparison_window]
    else:
        # If comparison_window not in the list, compute it
        window_smoothed = moving_average_with_padding(pc_data, comparison_window, session_avg_pc)
        smoothing_results['smoothed_data'][comparison_window] = window_smoothed
    
    plt.figure(figsize=(15, 6))
    
    # Plot both original and smoothed for comparison
    plt.plot(time_bins_pca, pc_data, 'gray', alpha=0.5, linewidth=0.5, label='Original PC')
    plt.plot(time_bins_pca, window_smoothed, 'black', linewidth=2,
             label=f'Smoothed PC (window={comparison_window})')

    # Add sleep periods
    for _, row in df_neural_sleep.iterrows():
        start_time = row['start_timestamp_s']
        end_time = row['end_timestamp_s']
        plt.axvspan(start_time, end_time, alpha=0.3, color='lightblue', 
                   label='Sleep' if row.name == 0 else "")
    
    plt.xlabel('Time (s)')
    plt.ylabel(f'PC{pc_index_to_plot + 1} Score')
    plt.title(f'{subject}: PC{pc_index_to_plot + 1} Neural Trajectory Over Time (Smoothed, Window={comparison_window})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{subject}_pc{pc_index_to_plot + 1}_window{comparison_window}_smoothed.png"), dpi=300)
    plt.show()
    
    return smoothing_results

def plot_pc_spectrogram(pc_data, time_bins_pca, df_neural_sleep, subject_name, output_dir, 
                         nperseg_pc=50, noverlap_ratio=0.5, show_plots=True, save_plots=True, pc_index_to_plot=0):
    """
    Computes and plots the spectrogram of PC data, highlighting sleep periods and delta power.

    Args:
        pc_data (np.array): Array of PC scores.
        time_bins_pca (np.array): Array of time bins corresponding to pc_data.
        df_neural_sleep (pd.DataFrame): DataFrame with 'start_timestamp_s' and 'end_timestamp_s'
                                        for sleep bouts.
        subject_name (str): Subject identifier for titles and filenames.
        output_dir (str): Directory to save the plot.
        nperseg_pc (int): Window length for the spectrogram in samples.
        noverlap_ratio (float): Ratio of overlap to nperseg_pc for the spectrogram.
        show_plots (bool): Whether to display the plot.
        save_plots (bool): Whether to save the plot.
    """
    if pc_data is None or time_bins_pca is None or df_neural_sleep is None:
        print("Error: pc_data, time_bins_pca, or df_neural_sleep is None. Skipping PC spectrogram.")
        return

    if len(pc_data) == 0 or len(time_bins_pca) == 0:
        print("Error: pc_data or time_bins_pca is empty. Skipping PC spectrogram.")
        return

    if len(time_bins_pca) < 2:
        print("Error: time_bins_pca must have at least 2 elements to calculate sampling frequency. Skipping PC spectrogram.")
        return

    # Calculate sampling frequency
    fs = 1 / (time_bins_pca[1] - time_bins_pca[0])
    print(f"PC Spectrogram: Sampling frequency (fs): {fs:.2f} Hz")

    # Define spectrogram parameters
    noverlap_pc = int(nperseg_pc * noverlap_ratio)
    print(f"PC Spectrogram parameters: nperseg={nperseg_pc} ({(nperseg_pc/fs):.1f}s), noverlap={noverlap_pc} ({(noverlap_pc/fs):.1f}s)")

    if len(pc_data) < nperseg_pc:
        print(f"Error: pc_data length ({len(pc_data)}) is less than nperseg ({nperseg_pc}). Skipping PC spectrogram.")
        return

    # Compute spectrogram for PC1
    try:
        frequencies, times_spec, Sxx_pc = signal.spectrogram(
            pc_data,
            fs=fs,
            window='hamming',
            nperseg=nperseg_pc,
            noverlap=noverlap_pc,
            scaling='density', # Power spectral density
            detrend='constant'
        )
    except ValueError as e:
        print(f"Error computing spectrogram: {e}. Skipping PC spectrogram.")
        return

    # Convert power to dB
    Sxx_pc_db = 10 * np.log10(Sxx_pc + 1e-10) # Add epsilon to avoid log(0)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # 1. Plot Spectrogram
    vmin_spec = np.percentile(Sxx_pc_db, 5)
    vmax_spec = np.percentile(Sxx_pc_db, 95)

    times_spec_aligned = times_spec + time_bins_pca[0]

    im = axs[0].pcolormesh(times_spec_aligned, frequencies, Sxx_pc_db, shading='gouraud', cmap='viridis', vmin=vmin_spec, vmax=vmax_spec)
    # fig.colorbar(im, ax=axs[0], label='Power/Frequency (dB/Hz)') # Uncomment if colorbar is desired
    axs[0].set_ylabel('Frequency (Hz)')
    axs[0].set_title(f'{subject_name}: PC{pc_index_to_plot + 1} Spectrogram')
    axs[0].set_ylim(0, fs / 2)

    # Overlay sleep periods on spectrogram
    for _, bout in df_neural_sleep.iterrows():
        start_time = bout['start_timestamp_s']
        end_time = bout['end_timestamp_s']
        axs[0].axvspan(start_time, end_time, alpha=0.2, color='cyan', ec='none')

    # 2. Plot Power in Delta Band (1-4 Hz)
    delta_band_mask = (frequencies >= 1) & (frequencies <= 4)
    if np.any(delta_band_mask):
        delta_power_pc = np.mean(Sxx_pc_db[delta_band_mask, :], axis=0)
        axs[1].plot(times_spec_aligned, delta_power_pc, color='blue', linewidth=1.5, label='Delta Power (1-4 Hz)')
        axs[1].set_ylabel('Avg. Power (dB)')
        
        # Overlay sleep periods on delta power plot
        # Add legend only once for sleep
        sleep_label_added = False
        for _, bout in df_neural_sleep.iterrows():
            start_time = bout['start_timestamp_s']
            end_time = bout['end_timestamp_s']
            current_label = 'Sleep' if not sleep_label_added else ""
            axs[1].axvspan(start_time, end_time, alpha=0.2, color='cyan', ec='none', label=current_label)
            if not sleep_label_added:
                sleep_label_added = True
        
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())
    else:
        axs[1].text(0.5, 0.5, 'Delta band (1-4 Hz) not resolved with current settings.',
                    horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)

    axs[1].set_xlabel('Time (s)')
    if len(time_bins_pca) > 0:
        axs[1].set_xlim(time_bins_pca[0], time_bins_pca[-1])
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        save_path = os.path.join(output_dir, f"{subject_name}_pc{pc_index_to_plot + 1}_spectrogram.png")
        plt.savefig(save_path, dpi=300)
        print(f"PC{pc_index_to_plot + 1} spectrogram saved to {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def analyze_pca_across_bin_sizes(
    spike_recordings,
    cluster_quality_maps,
    neural_sleep_df,
    subject,
    output_folder,
    bin_sizes_ms,
    probes_to_use=['probe0', 'probe1'],
    n_components_pca=50,
    components_to_plot_scatter=(0, 1),
    max_components_for_variance_plot=50,
    cluster_qualities_to_include=['good', 'mua']
):
    """
    Performs PCA for different temporal bin sizes and plots results.

    Parameters:
    -----------
    spike_recordings : dict
        Raw spike data loaded by pinkrigs_tools load_data.
        Example: spike_recordings['probe0'][0]['spikes'] and spike_recordings['probe0'][0]['clusters']
    cluster_quality_maps : dict
        A dictionary mapping probe names to cluster quality arrays.
        Example: {'probe0': quality_array_probe0, 'probe1': quality_array_probe1}
                 where quality_array is from bombcell_sort_units and corresponds to
                 the order of clusters in spike_recordings[probe][0]['clusters']['cluster_id'].
    neural_sleep_df : pd.DataFrame
        DataFrame with sleep bout information ('start_timestamp_s', 'end_timestamp_s').
    subject : str
        Subject identifier.
    output_folder : str
        Directory to save plots.
    bin_sizes_ms : list
        List of bin sizes in milliseconds (e.g., [10, 50, 100]).
    probes_to_use : list, optional
        List of probe names to include (e.g., ['probe0', 'probe1']).
    n_components_pca : int, optional
        Number of PCA components to compute.
    components_to_plot_scatter : tuple, optional
        PC indices for the scatter plot (0-indexed).
    max_components_for_variance_plot : int, optional
        Maximum number of components for the cumulative variance plot.
    cluster_qualities_to_include : list, optional
        List of cluster quality labels to include.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    all_cumulative_variances = {}
    pc1_idx, pc2_idx = components_to_plot_scatter

    for bin_size_ms in bin_sizes_ms:
        bin_size_s = bin_size_ms / 1000.0
        print(f"\nProcessing bin size: {bin_size_ms} ms ({bin_size_s} s)")

        combined_counts_for_bin = []
        time_bins_current_binning = None # Will be set by the first probe

        for probe_idx, probe_name in enumerate(probes_to_use):
            # Check if probe exists and has data using .empty instead of boolean evaluation
            if probe_name not in spike_recordings or spike_recordings[probe_name].empty:
                print(f"Spike data for {probe_name} not found. Skipping.")
                continue
            if probe_name not in cluster_quality_maps:
                print(f"Cluster quality for {probe_name} not found. Skipping.")
                continue

            probe_spikes_data = spike_recordings[probe_name].iloc[0]['spikes']
            probe_clusters_data = spike_recordings[probe_name].iloc[0]['clusters']
            probe_quality_labels = cluster_quality_maps[probe_name]

            # Fixed: Use 'cluster_id' instead of 'ids'
            all_cluster_ids_probe = probe_clusters_data['cluster_id']
            
            # Ensure quality labels array matches the number of cluster IDs
            if len(all_cluster_ids_probe) != len(probe_quality_labels):
                print(f"Warning: Mismatch in cluster ID count ({len(all_cluster_ids_probe)}) and quality labels ({len(probe_quality_labels)}) for {probe_name}. Skipping probe.")
                continue

            # Create mask for selecting clusters of desired quality
            quality_mask = np.isin(probe_quality_labels, cluster_qualities_to_include)
            selected_cluster_ids = all_cluster_ids_probe[quality_mask]

            if len(selected_cluster_ids) == 0:
                print(f"No clusters of desired quality found for {probe_name}. Skipping.")
                continue

            # Filter spikes: only include spikes from selected clusters
            spike_times_all = probe_spikes_data['times']
            # Fixed: Use 'clusters' instead of 'clusters' (this was already correct)
            spike_clusters_all = probe_spikes_data['clusters']
            
            # Mask for spikes belonging to selected clusters
            spike_selection_mask = np.isin(spike_clusters_all, selected_cluster_ids)
            
            filtered_spike_times = spike_times_all[spike_selection_mask]
            filtered_spike_clusters = spike_clusters_all[spike_selection_mask]

            if len(filtered_spike_times) == 0:
                print(f"No spikes found from selected clusters for {probe_name}. Skipping.")
                continue

            # Bin the filtered spikes
            # bincount2D expects x (times), y (clusters)
            # ybin=0 means aggregate by unique values in y (filtered_spike_clusters)
            counts_probe, tb_probe, cids_probe = bincount2D(
                x=filtered_spike_times,
                y=filtered_spike_clusters,
                xbin=bin_size_s,
                ybin=0, # Use unique cluster IDs present in filtered_spike_clusters
                xlim=[np.min(filtered_spike_times), np.max(filtered_spike_times)]
            )
            
            if counts_probe is None or counts_probe.shape[0] == 0 or counts_probe.shape[1] == 0:
                print(f"Binning resulted in empty counts for {probe_name}. Skipping.")
                continue

            print(f"  {probe_name}: Binned counts shape: {counts_probe.shape} (clusters x timebins)")
            combined_counts_for_bin.append(counts_probe)

            if time_bins_current_binning is None: # Set time bins from the first processed probe
                time_bins_current_binning = tb_probe
            elif not np.array_equal(time_bins_current_binning, tb_probe):
                # This case should ideally not happen if recordings are aligned and binning is consistent
                # For simplicity, we'll truncate to the shortest if they differ.
                print(f"  Warning: Time bins differ between probes for bin size {bin_size_ms}ms. Truncating.")
                min_len = min(len(time_bins_current_binning), len(tb_probe))
                time_bins_current_binning = time_bins_current_binning[:min_len]
                # Adjust existing combined_counts if this is not the first probe
                for i in range(len(combined_counts_for_bin)-1):
                    combined_counts_for_bin[i] = combined_counts_for_bin[i][:, :min_len]
                combined_counts_for_bin[-1] = combined_counts_for_bin[-1][:, :min_len]


        if not combined_counts_for_bin or time_bins_current_binning is None:
            print(f"No data to process for bin size {bin_size_ms} ms. Skipping PCA.")
            continue
        
        # Ensure all count matrices have the same number of time bins
        min_timepoints = min(arr.shape[1] for arr in combined_counts_for_bin)
        combined_matrix_list = [arr[:, :min_timepoints] for arr in combined_counts_for_bin]
        time_bins_current_binning = time_bins_current_binning[:min_timepoints]

        combined_matrix = np.vstack(combined_matrix_list).astype(np.float32) # (total_neurons x timepoints)
        print(f"  Combined matrix shape for bin size {bin_size_ms}ms: {combined_matrix.shape}")

        if combined_matrix.shape[0] == 0 or combined_matrix.shape[1] == 0:
            print(f"Combined matrix is empty for bin size {bin_size_ms}ms. Skipping PCA.")
            continue

        # Prepare state labels for current time bins
        state_labels_current_binning = np.zeros(len(time_bins_current_binning), dtype=int) # 0 for wake
        if not neural_sleep_df.empty:
            for _, row in neural_sleep_df.iterrows():
                start_idx = np.searchsorted(time_bins_current_binning, row['start_timestamp_s'], side='left')
                end_idx = np.searchsorted(time_bins_current_binning, row['end_timestamp_s'], side='right')
                state_labels_current_binning[start_idx:end_idx] = 1 # 1 for sleep
        
        # PCA
        X = combined_matrix.T # (timepoints x neurons)
        if X.shape[0] < n_components_pca or X.shape[1] < n_components_pca :
            print(f"  Data shape {X.shape} too small for {n_components_pca} PCA components. Skipping bin size {bin_size_ms}ms.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        current_n_components = min(n_components_pca, X_scaled.shape[0], X_scaled.shape[1])
        if current_n_components < 2: # Need at least 2 components for scatter plot
             print(f"  Not enough features/samples ({current_n_components}) for PCA. Skipping bin size {bin_size_ms}ms.")
             continue

        pca = PCA(n_components=current_n_components)
        pca_result = pca.fit_transform(X_scaled) # (timepoints x components)
        
        # Store cumulative variance
        all_cumulative_variances[bin_size_ms] = np.cumsum(pca.explained_variance_ratio_)

        # Plot PC1 vs PC2 Scatter
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
        wake_mask_scatter = state_labels_current_binning == 0
        sleep_mask_scatter = state_labels_current_binning == 1

        ax_scatter.scatter(pca_result[wake_mask_scatter, pc1_idx], pca_result[wake_mask_scatter, pc2_idx],
                           c='orange', alpha=0.5, s=10, label='Wake')
        ax_scatter.scatter(pca_result[sleep_mask_scatter, pc1_idx], pca_result[sleep_mask_scatter, pc2_idx],
                           c='blue', alpha=0.5, s=10, label='Sleep')
        
        ax_scatter.set_xlabel(f'PC{pc1_idx + 1} ({pca.explained_variance_ratio_[pc1_idx]:.1%} variance)')
        ax_scatter.set_ylabel(f'PC{pc2_idx + 1} ({pca.explained_variance_ratio_[pc2_idx]:.1%} variance)')
        ax_scatter.set_title(f'{subject}: PC{pc1_idx+1} vs PC{pc2_idx+1} (Bin Size: {bin_size_ms} ms)')
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax_scatter.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        
        scatter_filename = os.path.join(output_folder, f"{subject}_pca_scatter_bin_{bin_size_ms}ms.png")
        plt.savefig(scatter_filename, dpi=300)
        plt.close(fig_scatter)
        print(f"  Saved PC scatter plot to {scatter_filename}")

    # Plot Cumulative Explained Variance
    if not all_cumulative_variances:
        print("No PCA results to plot for cumulative variance.")
        return

    fig_variance, ax_variance = plt.subplots(figsize=(12, 8))
    num_components_to_show = min(max_components_for_variance_plot, n_components_pca)

    for bin_size_ms, cum_var in all_cumulative_variances.items():
        components_axis = np.arange(1, min(len(cum_var), num_components_to_show) + 1)
        ax_variance.plot(components_axis, cum_var[:min(len(cum_var), num_components_to_show)], 
                         marker='o', linestyle='-', markersize=4, label=f'{bin_size_ms} ms bin')

    ax_variance.set_xlabel('Number of Principal Components')
    ax_variance.set_ylabel('Cumulative Explained Variance')
    ax_variance.set_title(f'{subject}: PCA Cumulative Explained Variance by Bin Size')
    ax_variance.grid(True, alpha=0.5)
    ax_variance.legend(title="Bin Size")
    ax_variance.set_xticks(np.arange(0, num_components_to_show + 1, 5 if num_components_to_show > 20 else 1))
    ax_variance.set_yticks(np.arange(0, 1.1, 0.1))
    ax_variance.set_ylim(0, 1.05)
    ax_variance.set_xlim(0, num_components_to_show + 1)

    variance_plot_filename = os.path.join(output_folder, f"{subject}_pca_cumulative_variance_by_binsize.png")
    plt.savefig(variance_plot_filename, dpi=300)
    plt.show()
    plt.close(fig_variance)
    print(f"Saved cumulative variance plot to {variance_plot_filename}")

    print("\nFinished PCA analysis across bin sizes.")
