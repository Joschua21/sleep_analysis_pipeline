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



def analyze_sleep_wake_activity(results, output_dir=None, save_plots=False, show_plots=False, num_top_clusters=10):
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
    
    if show_plots or save_plots:
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
    
    if show_plots:
        plt.show()
    
    return modulation_results

def analyze_cluster_state_and_stability(results, output_dir=None, save_plots=False, bin_size_s=120, 
                                       state_threshold=0.9, max_iterations=1000):
    """
    Combined analysis of cluster state distribution and neuronal stability using pooled data from all probes.
    Creates both the state distribution plot and stability scatter plots in a single analysis.
    
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
        Dictionary containing both state distribution and stability metrics for merged probes
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
            normalized_activity_in_bin = np.mean(normalized_counts[j, start_idx:end_idx])
            large_bin_firing_rates[j, i] = normalized_activity_in_bin
    
    # Convert to arrays for easier manipulation
    large_bin_times = np.array(large_bin_times)
    large_bin_states = np.array(large_bin_states)
    
    # === STATE DISTRIBUTION ANALYSIS ===
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
    
    # === STABILITY ANALYSIS ===
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
    
    # Try to find a balanced split for stability analysis
    all_first_half_bins = np.concatenate([first_half_sleep_bins, first_half_wake_bins])
    all_second_half_bins = np.concatenate([second_half_sleep_bins, second_half_wake_bins])
    all_sleep_bins = np.concatenate([first_half_sleep_bins, second_half_sleep_bins])
    all_wake_bins = np.concatenate([first_half_wake_bins, second_half_wake_bins])
    
    # Define constraints
    success = False
    iteration = 0
    bin_assignment = np.zeros(num_large_bins, dtype=int)
    
    start_time = time.time()
    while not success and iteration < max_iterations:
        # Create a random assignment (1 = C1, 0 = C2)
        bin_assignment = np.zeros(num_large_bins, dtype=int)
        
        # Randomly assign bins to C1 (1) or C2 (0)
        all_bins = np.arange(num_large_bins)
        usable_bins = np.where(np.isin(all_bins, np.concatenate([all_first_half_bins, all_second_half_bins])))[0]
        
        if len(usable_bins) > 0:
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
                if first_half_in_c1 + second_half_in_c1 > 0 and first_half_in_c2 + second_half_in_c2 > 0:
                    temporal_balance_c1 = (first_half_in_c1 / (first_half_in_c1 + second_half_in_c1) <= 0.7 and
                                          second_half_in_c1 / (first_half_in_c1 + second_half_in_c1) <= 0.7)
                    
                    temporal_balance_c2 = (first_half_in_c2 / (first_half_in_c2 + second_half_in_c2) <= 0.7 and
                                          second_half_in_c2 / (first_half_in_c2 + second_half_in_c2) <= 0.7)
                    
                    # If both criteria are met, we're successful
                    if temporal_balance_c1 and temporal_balance_c2:
                        success = True
                    elif iteration > max_iterations//2:
                        success = True  # Accept less ideal split after many attempts
        
        iteration += 1
    
    end_time = time.time()
    
    if not success:
        print(f"Could not find a balanced split after {max_iterations} iterations. Using best available split.")
    else:
        print(f"Found balanced split after {iteration} iterations ({end_time - start_time:.2f}s)")
    
    # Calculate firing rates for each category
    c1_mask = bin_assignment == 1
    c2_mask = bin_assignment == 0
    
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
    
    # === COMBINED PLOTTING ===
    # Create figure with 3 subplots: swarm plot + 2 stability plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # === PLOT 1: STATE DISTRIBUTION SWARM PLOT ===
    if not df_plot.empty:
        # Create swarm plot
        sns.swarmplot(data=df_plot, x='state', y='firing_rate', color='gray', alpha=0.7, size=5, ax=axes[0])
        
        # Add box plot over swarm plot
        sns.boxplot(data=df_plot, x='state', y='firing_rate', color='white', fliersize=0, width=0.5, 
                   boxprops={"facecolor": (.9, .9, .9, 0.5), "edgecolor": "black"}, ax=axes[0])
        
        axes[0].set_title(f'State-Dependent Firing Rates')
            
    else:
        print("No data available for state distribution plot.")

    axes[0].set_ylabel('Normalized Firing Rate')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    # === PLOT 2: STABILITY - FIRING RATE CONSISTENCY ===
    valid_mask = ~np.isnan(c1_rates) & ~np.isnan(c2_rates)
    
    if np.sum(valid_mask) > 1:
        # Get max value for axis scaling
        max_val = max(np.nanmax(c1_rates), np.nanmax(c2_rates))
        
        # Create scatter plot
        axes[1].scatter(c1_rates[valid_mask], c2_rates[valid_mask], alpha=0.7)
        
        # Add identity line
        axes[1].plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.7)
        
        # Add regression line
        slope, intercept, r_value, p_value_stability, std_err = stats.linregress(
            c1_rates[valid_mask], c2_rates[valid_mask]
        )
        
        x_vals = np.array([0, max_val*1.1])
        axes[1].plot(x_vals, intercept + slope * x_vals, 'r-', alpha=0.7)


        
        axes[1].set_xlabel('C1 - Average Firing Rate')
        axes[1].set_ylabel('C2 - Average Firing Rate')
        axes[1].set_title('Neuronal Stability')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient data for\nstability analysis',
                   ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Neuronal Stability\nFiring Rate Consistency')
    
    # === PLOT 3: STABILITY - MODULATION CONSISTENCY ===
    valid_mod_mask = (~np.isnan(c1_modulation) & ~np.isnan(c2_modulation))
    
    if np.sum(valid_mod_mask) > 1:
        # Get max absolute value for axis scaling
        max_mod = max(
            np.nanmax(np.abs(c1_modulation)), 
            np.nanmax(np.abs(c2_modulation))
        ) * 1.1
        
        # Create scatter plot
        axes[2].scatter(c1_modulation[valid_mod_mask], c2_modulation[valid_mod_mask], alpha=0.7)
        
        # Add identity line
        axes[2].plot([-max_mod, max_mod], [-max_mod, max_mod], 'k--', alpha=0.7)
        
        # Add regression line
        mod_slope, mod_intercept, mod_r, mod_p, mod_err = stats.linregress(
            c1_modulation[valid_mod_mask], c2_modulation[valid_mod_mask]
        )
        
        x_mod_vals = np.array([-max_mod, max_mod])
        axes[2].plot(x_mod_vals, mod_intercept + mod_slope * x_mod_vals, 'r-', alpha=0.7)

        
        # Add quadrant lines
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[2].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Count points in each quadrant
        q1 = np.sum((c1_modulation > 0) & (c2_modulation > 0) & valid_mod_mask)
        q2 = np.sum((c1_modulation < 0) & (c2_modulation > 0) & valid_mod_mask)
        q3 = np.sum((c1_modulation < 0) & (c2_modulation < 0) & valid_mod_mask)
        q4 = np.sum((c1_modulation > 0) & (c2_modulation < 0) & valid_mod_mask)
        
        axes[2].set_xlabel('C1 - Wake-Sleep Modulation')
        axes[2].set_ylabel('C2 - Wake-Sleep Modulation')
        axes[2].set_title('Sleep-Wake Modulation')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        # Set equal x and y limits
        axes[2].set_xlim(-max_mod, max_mod)
        axes[2].set_ylim(-max_mod, max_mod)
        
    else:
        axes[2].text(0.5, 0.5, 'Insufficient data for\nmodulation analysis',
                   ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Sleep-Wake Modulation\nConsistency')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"combined_state_stability_analysis_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved combined analysis plot to: {filepath}")
    
    plt.show()
    
    # Store and return combined results
    combined_results = {
        'merged': {
            # State distribution results
            'cluster_ids': all_cluster_ids,
            'large_bin_states': large_bin_states,
            'large_bin_firing_rates': large_bin_firing_rates,
            'large_bin_centers': large_bin_times,
            'plot_data': df_plot,
            
            # Stability results
            'c1_rates': c1_rates,
            'c2_rates': c2_rates,
            'c1_sleep_rates': c1_sleep_rates,
            'c1_wake_rates': c1_wake_rates,
            'c2_sleep_rates': c2_sleep_rates,
            'c2_wake_rates': c2_wake_rates,
            'c1_modulation': c1_modulation,
            'c2_modulation': c2_modulation,
            
            # Analysis parameters
            'bin_size_s': bin_size_s,
            'state_threshold': state_threshold,
            'num_large_bins': num_large_bins
        }
    }
    
    return combined_results


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
        'time_bins': time_bins  #
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
            ax_behav.axvspan(bout['start_timestamp_s'], bout['end_timestamp_s'], 
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
        for _, bout in sleep_bouts.iterrows():
            # Remove the in_range check to ensure all bouts are included
            sleep_starts.append(bout['start_timestamp_s'])
            sleep_ends.append(bout['end_bin_time'])
        for _, bout in sleep_bouts.iterrows():
            ax1.axvspan(bout['start_timestamp_s'], bout['end_timestamp_s'], 
            color='lightblue', alpha=0.3, ec='none')
                
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
            ax2.axvspan(bout['start_timestamp_s'], bout['end_timestamp_s'], 
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
            ax3.axvspan(bout['start_timestamp_s'], bout['end_timestamp_s'], 
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
            ax4.axvline(x=bout['start_timestamp_s'], color='white', linestyle='--', alpha=0.7)
            ax4.axvline(x=bout['end_timestamp_s'], color='white', linestyle='--', alpha=0.7)

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
            ax5.axvspan(bout['start_timestamp_s'], bout['end_timestamp_s'], 
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
    
    if 'time_bins' in spectrum_results['merged']:
        # Use the original neural time bins, interpolated to match band power length
        original_time_bins = spectrum_results['merged']['time_bins']
        sample_power = next(iter(band_powers.values()))
        
        # Create times that span the original recording duration
        times = np.linspace(original_time_bins[0], original_time_bins[-1], len(sample_power))
        print(f"Using neural time range: {times[0]:.1f}s to {times[-1]:.1f}s")
        print(f"Original neural recording: {original_time_bins[0]:.1f}s to {original_time_bins[-1]:.1f}s")
    elif 'times' in spectrum_results['merged']:
        # Fallback to spectrogram times
        times = spectrum_results['merged']['times']
        print("Warning: Using spectrogram times (may extend beyond recording)")
    else:
        print("No valid time bins found in spectrum results")
        return None
    
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
        total_behavioral_sleep = 0
        
        for _, row in behavioral_df.iterrows():
            start = row['start_timestamp_s']
            end = row['end_timestamp_s']
            
            # Convert Series to scalar if needed
            if hasattr(start, 'iloc'):
                start = start.iloc[0]
            if hasattr(end, 'iloc'):
                end = end.iloc[0]
                
            total_behavioral_sleep += (end - start)
            
            # ADD THE MISSING PART: Actually populate the behavioral mask
            start_idx = max(0, int((start - times[0]) / time_step))
            end_idx = min(num_samples - 1, int((end - times[0]) / time_step))
            behavioral_mask[start_idx:end_idx+1] = True
        
        behavioral_sleep_time = total_behavioral_sleep
        total_recording_time = times[-1] - times[0]
        behavioral_sleep_percentage = (behavioral_sleep_time / total_recording_time) * 100
        
        print(f"- Total sleep time: {behavioral_sleep_time:.1f}s ({behavioral_sleep_percentage:.1f}% of recording)")
        
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

    
    print(f"\nSavitzky-Golay Sleep Detection (threshold={threshold_sg}dB, min duration={min_duration_s}s):")
    print(f"- Detected {len(sleep_periods_sg)} sleep bouts")
    print(f"- Total sleep time: {total_sleep_time_sg:.1f}s ({sleep_percentage_sg:.1f}% of recording)")
    print(f"- Comparison to behavioral sleep:")

    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(12, 14))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1.5, 1.5, 2], figure=fig)
    
    # Plot 1: Behavioral Sleep
    ax_beh = fig.add_subplot(gs[0])
    
    # Create a behavior mask to plot
    behavior_mask = np.zeros(len(times))
    for _, row in df_sleep.iterrows():
        start = row['start_timestamp_s']
        end = row['end_timestamp_s']
        
        # Convert to scalar if needed
        if isinstance(start, pd.Series):
            start = start.values[0]
        if isinstance(end, pd.Series):
            end = end.values[0]
            
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
        start = row['start_timestamp_s']
        end = row['end_timestamp_s']
        
        # Convert Series to scalar if needed
        if hasattr(start, 'iloc'):
            start = start.iloc[0]
        if hasattr(end, 'iloc'):
            end = end.iloc[0]
            
        ax_beh.axvspan(start, end, color='lightblue', alpha=0.3, ec='none')
    
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
    #ax1.legend(loc='upper right')
    
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
    #ax2.legend(loc='upper right')
    
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
            start = row['start_timestamp_s']
            end = row['end_timestamp_s']
            
            # Convert Series to scalar if needed
            if hasattr(start, 'iloc'):
                start = start.iloc[0]
            if hasattr(end, 'iloc'):
                end = end.iloc[0]
                
            ax3.axvline(x=start, color='white', linestyle='--', alpha=0.7)
            ax3.axvline(x=end, color='white', linestyle='--', alpha=0.7)
        
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
        'iou_sg': iou_sg,
        'sleep_percentage_ma': sleep_percentage_ma,
        'sleep_percentage_sg': sleep_percentage_sg
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
    
    saved_analysis_dir = os.path.join(output_folder, "saved_analysis")
    os.makedirs(saved_analysis_dir, exist_ok=True)

    output_file = os.path.join(saved_analysis_dir, 
                            f"{subject}_{exp_date}_{exp_num}_combined_neural_{bin_type}.npy")
    # Save to NPY file
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        print(f"Skipping creation and using existing file.")
    else:
        # Save to NPY file only if it doesn't exist
        np.save(output_file, combined_matrix)
        print(f"Saved combined matrix with shape {combined_matrix.shape} to {output_file}")

    return output_file
    
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
    
    # === PLOT 1: PC1 vs PC2 Scatter ===
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(pca_result[wake_mask_orig, comp1_idx], pca_result[wake_mask_orig, comp2_idx], 
              c='orange', alpha=0.5, s=5, label='Wake') # Wake is orange
    ax1.scatter(pca_result[sleep_mask_orig, comp1_idx], pca_result[sleep_mask_orig, comp2_idx], 
              c='blue', alpha=0.5, s=5, label='Sleep')   # Sleep is blue
    ax1.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
    ax1.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
    ax1.set_title('PC1 vs PC2', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    scatter_filename = os.path.join(output_folder, f"{subject}_pca_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_scatter_states.png")
    plt.savefig(scatter_filename, dpi=300)
    plt.show()
    
    # === PLOT 2: Explained Variance ===
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 'o-', color='blue')
    ax2.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'o-', color='red')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Component')
    ax2.axhline(y=0.5, color='gray', linestyle='--', label='50% Explained Variance')
    ax2.grid(True)
    ax2.legend(['Individual', 'Cumulative', '50% Threshold'])
    
    variance_filename = os.path.join(output_folder, f"{subject}_pca_explained_variance.png")
    plt.savefig(variance_filename, dpi=300)
    plt.show()
    
    # === PLOT 3: PC over Time ===
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(time_bins_used, pca_result[:, pc_index_to_plot], 'k-', linewidth=0.5)
    
    y_min_ts, y_max_ts = ax3.get_ylim() # Get y-limits for axvspan
    for _, row in neural_sleep_df.iterrows():
        ax3.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                   ymin=0, ymax=1, color='blue', alpha=0.2) # ymin/ymax relative to axes
    ax3.set_title(f"PC{pc_index_to_plot +1} over Time")
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(f'PC{pc_index_to_plot +1} Score')    
    timeseries_filename = os.path.join(output_folder, f"{subject}_pca_pc{pc_index_to_plot+1}_timeseries.png")
    plt.savefig(timeseries_filename, dpi=300)
    plt.show()
    
    # === PLOT 4: Hexbin Plot ===
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    pc1 = pca_result[:, comp1_idx]
    pc2 = pca_result[:, comp2_idx]
    
    # Create hexbin plot
    hexbin = ax4.hexbin(pc1, pc2, gridsize=30, cmap='viridis', alpha=0.8, mincnt=1)
    ax4.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
    ax4.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
    ax4.set_title('PC1 vs PC2 Hexbin Density', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Add colorbar
    plt.colorbar(hexbin, ax=ax4, label='Count')
    
    hexbin_filename = os.path.join(output_folder, f"{subject}_pca_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_hexbin.png")
    plt.savefig(hexbin_filename, dpi=300)
    plt.show()
    
    # === PLOT 5: State-Aware Density Plot ===
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    
    # Create state-specific data
    pc1_sleep = pc1[sleep_mask_orig]
    pc2_sleep = pc2[sleep_mask_orig]
    pc1_wake = pc1[wake_mask_orig]
    pc2_wake = pc2[wake_mask_orig]
    
    # Use KDE (Kernel Density Estimation) for better state visualization
    import seaborn as sns
    
    # Plot wake density first (in orange/red)
    if len(pc1_wake) > 10:  # Need sufficient points for KDE
        sns.kdeplot(x=pc1_wake, y=pc2_wake, ax=ax5, color='red', alpha=0.6, 
                   fill=True, levels=10, label='Wake Density')
    
    # Plot sleep density (in blue) - overlaying wake
    if len(pc1_sleep) > 10:  # Need sufficient points for KDE
        sns.kdeplot(x=pc1_sleep, y=pc2_sleep, ax=ax5, color='blue', alpha=0.6, 
                   fill=True, levels=10, label='Sleep Density')
    
    # Fallback to scatter if not enough points for KDE
    if len(pc1_wake) <= 10:
        ax5.scatter(pc1_wake, pc2_wake, c='red', alpha=0.5, s=10, label='Wake')
    if len(pc1_sleep) <= 10:
        ax5.scatter(pc1_sleep, pc2_sleep, c='blue', alpha=0.5, s=10, label='Sleep')
    
    ax5.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
    ax5.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
    ax5.set_title('PC1 vs PC2 State-Specific Density', fontsize=16, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax5.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    density_filename = os.path.join(output_folder, f"{subject}_pca_pc{comp1_idx+1}_vs_pc{comp2_idx+1}_density.png")
    plt.savefig(density_filename, dpi=300)
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

    # === PLOT 7: PC1 vs PC2 Scatter with Discrete Phased Coloring ===
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


def plot_pca_with_behavioral_features(pca_results, neural_sleep_df, dlc_folder, smoothed_results, 
                                    subject, output_folder, components_to_plot=(0, 1), 
                                    movement_upper_limit=740000, plot_together=True, figsize_individual=(10, 8)):
    """
    Create six PC1 vs PC2 plots with different coloring schemes:
    Row 1: Sleep/Wake, Movement (sorted), Delta power
    Row 2: Movement (random order), Low movement (<0.5), High movement (>=0.5)
    
    Parameters:
    ----------
    pca_results : dict
        Results from analyze_neural_pca
    neural_sleep_df : DataFrame
        DataFrame containing sleep bout information
    dlc_folder : str
        Path to the DLC folder containing behavioral data
    smoothed_results : dict
        Results from power_band_smoothing containing delta power
    subject : str
        Subject ID for plot titles
    output_folder : str
        Directory to save output plots
    components_to_plot : tuple
        Which components to plot (zero-indexed)
    movement_upper_limit : float
        Upper limit for movement data capping
    plot_together : bool
        If True, plot all subplots in one 3x2 figure. If False, plot individually.
    figsize_individual : tuple
        Figure size for individual plots (width, height)
        
    Returns:
    --------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from matplotlib.colors import Normalize
    from scipy.interpolate import interp1d
    
    # Extract PC data
    if 'pca_result' not in pca_results:
        print("Error: PCA results not found")
        return
    
    pc_data = pca_results['pca_result']
    time_bins_pca = pca_results['time_bins_used']
    state_labels_pca = pca_results['state_labels_used']
    
    comp1_idx, comp2_idx = components_to_plot
    pc1_scores = pc_data[:, comp1_idx]
    pc2_scores = pc_data[:, comp2_idx]
    explained_var_pc1 = pca_results['explained_variance_ratio'][comp1_idx]
    explained_var_pc2 = pca_results['explained_variance_ratio'][comp2_idx]
    
    # Load behavioral data (pixel differences)
    pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
    pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                       if f.endswith('.csv') and 'pixel_differences' in f]
    
    behavior_data = None
    if pixel_diff_files:
        pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
        try:
            behavior_data = pd.read_csv(pixel_diff_file)
            print(f"Loaded behavioral data from {pixel_diff_file}")
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            behavior_data = None
    
    # Extract delta power data
    delta_power = None
    delta_times = None
    if smoothed_results and 'savitzky_golay' in smoothed_results and 'Delta' in smoothed_results['savitzky_golay']:
        delta_power = smoothed_results['savitzky_golay']['Delta']
        if 'times' in smoothed_results and len(smoothed_results['times']) == len(delta_power):
            delta_times = smoothed_results['times']
        else:
            delta_times = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
        print(f"Loaded delta power data: {len(delta_power)} time points")
    else:
        print("Warning: No delta power data found in smoothed_results")
    
    # Process movement data once
    movement_values = None
    movement_normalized = None
    if behavior_data is not None:
        time_sec = behavior_data['time_sec'].values
        smoothed_diff = behavior_data['smoothed_difference'].values
        
        valid_mask = ~np.isnan(smoothed_diff)
        time_sec_clean = time_sec[valid_mask]
        smoothed_diff_clean = smoothed_diff[valid_mask]
        
        if len(time_sec_clean) > 0:
            min_time = max(time_bins_pca[0], time_sec_clean[0])
            max_time = min(time_bins_pca[-1], time_sec_clean[-1])
            pca_time_mask = (time_bins_pca >= min_time) & (time_bins_pca <= max_time)
            
            if np.sum(pca_time_mask) > 0:
                interp_func = interp1d(time_sec_clean, smoothed_diff_clean, 
                                     kind='linear', bounds_error=False, fill_value=np.nan)
                movement_values = interp_func(time_bins_pca)
                
                valid_movement = movement_values[~np.isnan(movement_values)]
                if len(valid_movement) > 0:
                    movement_min = np.min(valid_movement)
                    movement_capped = np.clip(movement_values, movement_min, movement_upper_limit)
                    movement_normalized = (movement_capped - movement_min) / (movement_upper_limit - movement_min)
                    
                    print(f"Movement statistics:")
                    print(f"  Raw range: {movement_min:.0f} - {np.max(valid_movement):.0f}")
                    print(f"  Values above threshold ({movement_upper_limit:,.0f}): {np.sum(valid_movement > movement_upper_limit)}")
                    print(f"  Percentage capped: {np.sum(valid_movement > movement_upper_limit)/len(valid_movement)*100:.1f}%")
    
    # Process delta power data once
    delta_values = None
    delta_normalized = None
    if delta_power is not None and delta_times is not None:
        min_time = max(time_bins_pca[0], delta_times[0])
        max_time = min(time_bins_pca[-1], delta_times[-1])
        pca_time_mask = (time_bins_pca >= min_time) & (time_bins_pca <= max_time)
        
        if np.sum(pca_time_mask) > 0:
            interp_func = interp1d(delta_times, delta_power, 
                                 kind='linear', bounds_error=False, fill_value=np.nan)
            delta_values = interp_func(time_bins_pca)
            
            valid_delta = delta_values[~np.isnan(delta_values)]
            if len(valid_delta) > 0:
                delta_min = np.min(valid_delta)
                delta_max = np.max(valid_delta)
                if delta_max > delta_min:
                    delta_normalized = (delta_values - delta_min) / (delta_max - delta_min)
                else:
                    delta_normalized = np.zeros_like(delta_values)
    
    # Helper function to set up each subplot
    def setup_subplot(ax, title):
        ax.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
        ax.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
        ax.set_title(f'{subject}: {title}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Define plot titles and data
    plot_titles = [
        'Sleep/Wake States',
        'Movement Intensity (Sorted)',
        'Delta Power',
        'Movement Intensity (Random)',
        'Low Movement (<0.5)',
        'High Movement (≥0.5)'
    ]
    
    if plot_together:
        # Create 2x3 subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Sleep/Wake state coloring
        sleep_mask = state_labels_pca == 1
        wake_mask = state_labels_pca == 0
        
        axes[0,0].scatter(pc1_scores[wake_mask], pc2_scores[wake_mask], 
                         c='orange', alpha=0.5, s=5, label='Wake')
        axes[0,0].scatter(pc1_scores[sleep_mask], pc2_scores[sleep_mask], 
                         c='blue', alpha=0.5, s=5, label='Sleep')
        axes[0,0].legend()
        setup_subplot(axes[0,0], plot_titles[0])
        
        # Plot 2: Movement intensity (sorted order)
        if movement_normalized is not None:
            valid_points = ~np.isnan(movement_normalized)
            if np.sum(valid_points) > 0:
                valid_indices = np.where(valid_points)[0]
                movement_values_valid = movement_normalized[valid_indices]
                sort_order = np.argsort(movement_values_valid)
                sorted_indices = valid_indices[sort_order]
                
                scatter2 = axes[0,1].scatter(pc1_scores[sorted_indices], pc2_scores[sorted_indices], 
                                           c=movement_normalized[sorted_indices], 
                                           cmap='coolwarm', alpha=0.6, s=5, vmin=0, vmax=1)
                
                invalid_points = np.isnan(movement_normalized)
                if np.sum(invalid_points) > 0:
                    axes[0,1].scatter(pc1_scores[invalid_points], pc2_scores[invalid_points], 
                                    c='lightgray', alpha=0.3, s=5)
                
                cbar2 = plt.colorbar(scatter2, ax=axes[0,1], shrink=0.8)
                cbar2.set_label(f'Movement (Sorted)\nBlue=Low, Red=High')
            else:
                axes[0,1].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
                axes[0,1].text(0.5, 0.5, 'No valid movement data', 
                             ha='center', va='center', transform=axes[0,1].transAxes)
        else:
            axes[0,1].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
            axes[0,1].text(0.5, 0.5, 'Movement data\nnot available', 
                         ha='center', va='center', transform=axes[0,1].transAxes)
        
        setup_subplot(axes[0,1], plot_titles[1])
        
        # Plot 3: Delta power coloring
        if delta_normalized is not None:
            valid_points = ~np.isnan(delta_normalized)
            if np.sum(valid_points) > 0:
                scatter3 = axes[0,2].scatter(pc1_scores[valid_points], pc2_scores[valid_points], 
                                           c=delta_normalized[valid_points], 
                                           cmap='coolwarm_r', alpha=0.7, s=5, vmin=0, vmax=1)
                
                invalid_points = np.isnan(delta_normalized)
                if np.sum(invalid_points) > 0:
                    axes[0,2].scatter(pc1_scores[invalid_points], pc2_scores[invalid_points], 
                                    c='lightgray', alpha=0.3, s=5)
                
                cbar3 = plt.colorbar(scatter3, ax=axes[0,2], shrink=0.8)
                cbar3.set_label('Delta Power\n(Red=Low, Blue=High)')
            else:
                axes[0,2].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
                axes[0,2].text(0.5, 0.5, 'No valid delta power data', 
                             ha='center', va='center', transform=axes[0,2].transAxes)
        else:
            axes[0,2].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
            axes[0,2].text(0.5, 0.5, 'Delta power data\nnot available', 
                         ha='center', va='center', transform=axes[0,2].transAxes)
        
        setup_subplot(axes[0,2], plot_titles[2])
        
        # Plot 4: Movement intensity (random order)
        if movement_normalized is not None:
            valid_points = ~np.isnan(movement_normalized)
            if np.sum(valid_points) > 0:
                valid_indices = np.where(valid_points)[0]
                np.random.shuffle(valid_indices)
                
                scatter4 = axes[1,0].scatter(pc1_scores[valid_indices], pc2_scores[valid_indices], 
                                           c=movement_normalized[valid_indices], 
                                           cmap='coolwarm', alpha=0.6, s=5, vmin=0, vmax=1)
                
                invalid_points = np.isnan(movement_normalized)
                if np.sum(invalid_points) > 0:
                    axes[1,0].scatter(pc1_scores[invalid_points], pc2_scores[invalid_points], 
                                    c='lightgray', alpha=0.3, s=5)
                
                cbar4 = plt.colorbar(scatter4, ax=axes[1,0], shrink=0.8)
                cbar4.set_label(f'Movement (Random)\nBlue=Low, Red=High')
            else:
                axes[1,0].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
                axes[1,0].text(0.5, 0.5, 'No valid movement data', 
                             ha='center', va='center', transform=axes[1,0].transAxes)
        else:
            axes[1,0].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
            axes[1,0].text(0.5, 0.5, 'Movement data\nnot available', 
                         ha='center', va='center', transform=axes[1,0].transAxes)
        
        setup_subplot(axes[1,0], plot_titles[3])
        
        # Plot 5: Low movement (<0.5)
        if movement_normalized is not None:
            valid_points = ~np.isnan(movement_normalized)
            low_movement_mask = valid_points & (movement_normalized < 0.5)
            
            if np.sum(low_movement_mask) > 0:
                scatter5 = axes[1,1].scatter(pc1_scores[low_movement_mask], pc2_scores[low_movement_mask], 
                                           c=movement_normalized[low_movement_mask], 
                                           cmap='coolwarm', alpha=0.7, s=5, vmin=0, vmax=1)
                cbar5 = plt.colorbar(scatter5, ax=axes[1,1], shrink=0.8)
                cbar5.set_label('Movement (<0.5)\nBlue=Low, Red=High')
            else:
                axes[1,1].text(0.5, 0.5, 'No low movement data', 
                             ha='center', va='center', transform=axes[1,1].transAxes)
            
            other_points = ~low_movement_mask
            if np.sum(other_points) > 0:
                axes[1,1].scatter(pc1_scores[other_points], pc2_scores[other_points], 
                                c='lightgray', alpha=0.2, s=3)
        else:
            axes[1,1].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
            axes[1,1].text(0.5, 0.5, 'Movement data\nnot available', 
                         ha='center', va='center', transform=axes[1,1].transAxes)
        
        setup_subplot(axes[1,1], plot_titles[4])
        
        # Plot 6: High movement (>=0.5)
        if movement_normalized is not None:
            valid_points = ~np.isnan(movement_normalized)
            high_movement_mask = valid_points & (movement_normalized >= 0.5)
            
            if np.sum(high_movement_mask) > 0:
                scatter6 = axes[1,2].scatter(pc1_scores[high_movement_mask], pc2_scores[high_movement_mask], 
                                           c=movement_normalized[high_movement_mask], 
                                           cmap='coolwarm', alpha=0.7, s=5, vmin=0, vmax=1)
                cbar6 = plt.colorbar(scatter6, ax=axes[1,2], shrink=0.8)
                cbar6.set_label('Movement (≥0.5)\nBlue=Low, Red=High')
            else:
                axes[1,2].text(0.5, 0.5, 'No high movement data', 
                             ha='center', va='center', transform=axes[1,2].transAxes)
            
            other_points = ~high_movement_mask
            if np.sum(other_points) > 0:
                axes[1,2].scatter(pc1_scores[other_points], pc2_scores[other_points], 
                                c='lightgray', alpha=0.2, s=3)
        else:
            axes[1,2].scatter(pc1_scores, pc2_scores, c='lightgray', alpha=0.5, s=5)
            axes[1,2].text(0.5, 0.5, 'Movement data\nnot available', 
                         ha='center', va='center', transform=axes[1,2].transAxes)
        
        setup_subplot(axes[1,2], plot_titles[5])
        
        plt.tight_layout()
        
        # Save combined plot
        combined_filename = os.path.join(output_folder, f"{subject}_pca_behavioral_features_combined.png")
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        # Helper functions for individual plots
        def plot_movement_sorted(ax, pc1, pc2, movement_norm):
            if movement_norm is not None:
                valid_points = ~np.isnan(movement_norm)
                if np.sum(valid_points) > 0:
                    valid_indices = np.where(valid_points)[0]
                    movement_values_valid = movement_norm[valid_indices]
                    sort_order = np.argsort(movement_values_valid)
                    sorted_indices = valid_indices[sort_order]
                    
                    scatter = ax.scatter(pc1[sorted_indices], pc2[sorted_indices], 
                                       c=movement_norm[sorted_indices], 
                                       cmap='coolwarm', alpha=0.6, s=10, vmin=0, vmax=1)
                    
                    invalid_points = np.isnan(movement_norm)
                    if np.sum(invalid_points) > 0:
                        ax.scatter(pc1[invalid_points], pc2[invalid_points], 
                                 c='lightgray', alpha=0.3, s=10)
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Movement (Sorted)\nBlue=Low, Red=High')
                else:
                    ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                    ax.text(0.5, 0.5, 'No valid movement data', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                ax.text(0.5, 0.5, 'Movement data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        def plot_delta_power(ax, pc1, pc2, delta_norm):
            if delta_norm is not None:
                valid_points = ~np.isnan(delta_norm)
                if np.sum(valid_points) > 0:
                    scatter = ax.scatter(pc1[valid_points], pc2[valid_points], 
                                       c=delta_norm[valid_points], 
                                       cmap='coolwarm_r', alpha=0.7, s=10, vmin=0, vmax=1)
                    
                    invalid_points = np.isnan(delta_norm)
                    if np.sum(invalid_points) > 0:
                        ax.scatter(pc1[invalid_points], pc2[invalid_points], 
                                 c='lightgray', alpha=0.3, s=10)
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Delta Power\n(Red=Low, Blue=High)')
                else:
                    ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                    ax.text(0.5, 0.5, 'No valid delta power data', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                ax.text(0.5, 0.5, 'Delta power data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        def plot_movement_random(ax, pc1, pc2, movement_norm):
            if movement_norm is not None:
                valid_points = ~np.isnan(movement_norm)
                if np.sum(valid_points) > 0:
                    valid_indices = np.where(valid_points)[0]
                    np.random.shuffle(valid_indices)
                    
                    scatter = ax.scatter(pc1[valid_indices], pc2[valid_indices], 
                                       c=movement_norm[valid_indices], 
                                       cmap='coolwarm', alpha=0.6, s=10, vmin=0, vmax=1)
                    
                    invalid_points = np.isnan(movement_norm)
                    if np.sum(invalid_points) > 0:
                        ax.scatter(pc1[invalid_points], pc2[invalid_points], 
                                 c='lightgray', alpha=0.3, s=10)
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Movement (Random)\nBlue=Low, Red=High')
                else:
                    ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                    ax.text(0.5, 0.5, 'No valid movement data', 
                           ha='center', va='center', transform=ax.transAxes)
            else:
                ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                ax.text(0.5, 0.5, 'Movement data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        def plot_movement_threshold(ax, pc1, pc2, movement_norm, operator, threshold):
            if movement_norm is not None:
                valid_points = ~np.isnan(movement_norm)
                if operator == '<':
                    threshold_mask = valid_points & (movement_norm < threshold)
                    label_suffix = f'(<{threshold})'
                else:
                    threshold_mask = valid_points & (movement_norm >= threshold)
                    label_suffix = f'(≥{threshold})'
                
                if np.sum(threshold_mask) > 0:
                    scatter = ax.scatter(pc1[threshold_mask], pc2[threshold_mask], 
                                       c=movement_norm[threshold_mask], 
                                       cmap='coolwarm', alpha=0.7, s=10, vmin=0, vmax=1)
                    plt.colorbar(scatter, ax=ax, shrink=0.8, label=f'Movement {label_suffix}\nBlue=Low, Red=High')
                else:
                    ax.text(0.5, 0.5, f'No {operator.replace("<", "low").replace(">=", "high")} movement data', 
                           ha='center', va='center', transform=ax.transAxes)
                
                other_points = ~threshold_mask
                if np.sum(other_points) > 0:
                    ax.scatter(pc1[other_points], pc2[other_points], 
                             c='lightgray', alpha=0.2, s=5)
            else:
                ax.scatter(pc1, pc2, c='lightgray', alpha=0.5, s=10)
                ax.text(0.5, 0.5, 'Movement data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Create individual plots
        for i, title in enumerate(plot_titles):
            fig, ax = plt.subplots(figsize=figsize_individual)
            
            if i == 0:  # Sleep/Wake plot
                ax.scatter(pc1_scores[state_labels_pca == 0], pc2_scores[state_labels_pca == 0], 
                          c='orange', alpha=0.5, s=10, label='Wake')
                ax.scatter(pc1_scores[state_labels_pca == 1], pc2_scores[state_labels_pca == 1], 
                          c='blue', alpha=0.5, s=10, label='Sleep')
                ax.legend()
            elif i == 1:  # Movement sorted
                plot_movement_sorted(ax, pc1_scores, pc2_scores, movement_normalized)
            elif i == 2:  # Delta power
                plot_delta_power(ax, pc1_scores, pc2_scores, delta_normalized)
            elif i == 3:  # Movement random
                plot_movement_random(ax, pc1_scores, pc2_scores, movement_normalized)
            elif i == 4:  # Low movement
                plot_movement_threshold(ax, pc1_scores, pc2_scores, movement_normalized, '<', 0.5)
            elif i == 5:  # High movement
                plot_movement_threshold(ax, pc1_scores, pc2_scores, movement_normalized, '>=', 0.5)
            
            setup_subplot(ax, title)
            
            plt.tight_layout()
            
            # Save individual plot
            safe_title = title.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('<', 'lt').replace('≥', 'gte')
            individual_filename = os.path.join(output_folder, f"{subject}_pca_{safe_title}_individual.png")
            plt.savefig(individual_filename, dpi=300, bbox_inches='tight')
            plt.show()
    
    print(f"Behavioral feature plots saved to {output_folder}")


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
    Performs PCA for different temporal bin sizes and plots results with enhanced visualization.

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

        # NEW: Create enhanced visualization for this bin size (4 subplots in one figure)
        create_enhanced_pca_visualization(
            pca_result=pca_result,
            state_labels=state_labels_current_binning,
            time_bins=time_bins_current_binning,
            neural_sleep_df=neural_sleep_df,
            bin_size_ms=bin_size_ms,
            subject=subject,
            output_folder=output_folder
        )

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

    return all_cumulative_variances


def create_enhanced_pca_visualization(pca_result, state_labels, time_bins, neural_sleep_df, 
                                    bin_size_ms, subject, output_folder):
    """
    Create enhanced PCA visualization with 4 subplots: scatter, density, wake-only, sleep-only
    """
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'PCA Analysis - {subject} (Bin Size: {bin_size_ms}ms)', fontsize=16, y=0.98)
    
    # Extract PC1 and PC2
    pc1 = pca_result[:, 0]
    pc2 = pca_result[:, 1]
    
    # Define colors
    wake_color = '#FF6B6B'  # Red
    sleep_color = '#4ECDC4'  # Teal
    
    # Create masks
    wake_mask = state_labels == 0
    sleep_mask = state_labels == 1
    
    # Plot 1: Traditional scatter plot (PC1 vs PC2)
    ax1 = axes[0, 0]
    ax1.scatter(pc1[wake_mask], pc2[wake_mask], c=wake_color, alpha=0.6, s=20, label='Wake')
    ax1.scatter(pc1[sleep_mask], pc2[sleep_mask], c=sleep_color, alpha=0.6, s=20, label='Sleep')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PC1 vs PC2 Scatter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Density plot
    ax2 = axes[0, 1]
    
    # Create density plot using hexbin
    hb = ax2.hexbin(pc1, pc2, gridsize=30, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PC1 vs PC2 Density')
    plt.colorbar(hb, ax=ax2, label='Count')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wake only
    ax3 = axes[1, 0]
    if np.sum(wake_mask) > 0:
        ax3.scatter(pc1[wake_mask], pc2[wake_mask], c=wake_color, alpha=0.6, s=20)
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title(f'Wake State Only (n={np.sum(wake_mask)} bins)')
        ax3.grid(True, alpha=0.3)
        
        # Match axis limits to main plot
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_ylim(ax1.get_ylim())
    else:
        ax3.text(0.5, 0.5, 'No wake data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Wake State Only (No Data)')
    
    # Plot 4: Sleep only  
    ax4 = axes[1, 1]
    if np.sum(sleep_mask) > 0:
        ax4.scatter(pc1[sleep_mask], pc2[sleep_mask], c=sleep_color, alpha=0.6, s=20)
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC2')
        ax4.set_title(f'Sleep State Only (n={np.sum(sleep_mask)} bins)')
        ax4.grid(True, alpha=0.3)
        
        # Match axis limits to main plot
        ax4.set_xlim(ax1.get_xlim())
        ax4.set_ylim(ax1.get_ylim())
    else:
        ax4.text(0.5, 0.5, 'No sleep data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Sleep State Only (No Data)')
    
    # Add statistics text
    stats_text = f"""Statistics:
    Total bins: {len(state_labels)}
    Wake bins: {np.sum(wake_mask)} ({np.sum(wake_mask)/len(state_labels)*100:.1f}%)
    Sleep bins: {np.sum(sleep_mask)} ({np.sum(sleep_mask)/len(state_labels)*100:.1f}%)
    Bin size: {bin_size_ms}ms
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_folder, f'{subject}_pca_enhanced_{bin_size_ms}ms.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Enhanced PCA plot saved: {plot_path}")

def correlate_pc_with_movement(pca_results, dlc_folder, pc_index=1, time_range=None, 
                             movement_column='smoothed_difference'):
    """
    Correlate a specific PC with movement data over a defined time range.
    
    Parameters:
    -----------
    pca_results : dict
        Results from analyze_neural_pca function
    dlc_folder : str
        Path to the DLC folder containing behavioral data
    pc_index : int, optional
        Which PC to analyze (0=PC1, 1=PC2, etc.) (default: 1 for PC2)
    time_range : tuple, optional
        (start_time, end_time) in seconds. If None, uses full recording
    movement_column : str, optional
        Column name for movement data (default: 'smoothed_difference')
    
    Returns:
    --------
    dict
        Dictionary containing correlation results and aligned data
    """
    from scipy.interpolate import interp1d
    from scipy.stats import pearsonr, spearmanr
    import os
    import pandas as pd
    import numpy as np
    
    # Extract PC data
    if 'pca_result' not in pca_results:
        print("Error: PCA results not found")
        return None
    
    pc_data = pca_results['pca_result'][:, pc_index]
    time_bins_pca = pca_results['time_bins_used']
    
    print(f"PC{pc_index+1} data: {len(pc_data)} time points from {time_bins_pca[0]:.1f}s to {time_bins_pca[-1]:.1f}s")
    
    # Load behavioral data (pixel differences)
    pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
    pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                       if f.endswith('.csv') and 'pixel_differences' in f]
    
    if not pixel_diff_files:
        print(f"No pixel difference CSV found in {pixel_diff_path}")
        return None
    
    # Load the behavioral data
    pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
    try:
        behavior_data = pd.read_csv(pixel_diff_file)
        print(f"Loaded behavioral data from {pixel_diff_file}")
        
        if movement_column not in behavior_data.columns or 'time_sec' not in behavior_data.columns:
            print(f"Required columns not found. Available: {behavior_data.columns.tolist()}")
            return None
            
    except Exception as e:
        print(f"Error loading behavioral data: {e}")
        return None
    
    # Extract time and movement data
    time_sec = behavior_data['time_sec'].values
    movement_data = behavior_data[movement_column].values
    
    # Remove NaN values from movement data
    valid_mask = ~np.isnan(movement_data)
    time_sec_clean = time_sec[valid_mask]
    movement_data_clean = movement_data[valid_mask]
    
    print(f"Movement data: {len(movement_data_clean)} valid points from {time_sec_clean[0]:.1f}s to {time_sec_clean[-1]:.1f}s")
    
    # Apply time range filter if specified
    if time_range is not None:
        start_time, end_time = time_range
        print(f"Filtering to time range: {start_time}s to {end_time}s")
        
        # Filter PCA data
        pca_time_mask = (time_bins_pca >= start_time) & (time_bins_pca <= end_time)
        time_bins_filtered = time_bins_pca[pca_time_mask]
        pc_data_filtered = pc_data[pca_time_mask]
        
        # Filter movement data
        movement_time_mask = (time_sec_clean >= start_time) & (time_sec_clean <= end_time)
        time_sec_filtered = time_sec_clean[movement_time_mask]
        movement_data_filtered = movement_data_clean[movement_time_mask]
        
        print(f"After filtering - PC data: {len(pc_data_filtered)} points, Movement data: {len(movement_data_filtered)} points")
    else:
        time_bins_filtered = time_bins_pca
        pc_data_filtered = pc_data
        time_sec_filtered = time_sec_clean
        movement_data_filtered = movement_data_clean
    
    # Check if we have overlapping data
    if len(time_bins_filtered) == 0 or len(time_sec_filtered) == 0:
        print("Error: No data in the specified time range")
        return None
    
    # Find common time range
    min_time = max(time_bins_filtered[0], time_sec_filtered[0])
    max_time = min(time_bins_filtered[-1], time_sec_filtered[-1])
    
    if min_time >= max_time:
        print("Error: No overlapping time range between PC and movement data")
        return None
    
    print(f"Common time range: {min_time:.1f}s to {max_time:.1f}s")
    
    # Further filter both datasets to common time range
    pca_common_mask = (time_bins_filtered >= min_time) & (time_bins_filtered <= max_time)
    movement_common_mask = (time_sec_filtered >= min_time) & (time_sec_filtered <= max_time)
    
    time_bins_common = time_bins_filtered[pca_common_mask]
    pc_data_common = pc_data_filtered[pca_common_mask]
    time_sec_common = time_sec_filtered[movement_common_mask]
    movement_data_common = movement_data_filtered[movement_common_mask]
    
    # Interpolate movement data to match PCA time points
    if len(time_sec_common) < 2:
        print("Error: Insufficient movement data points for interpolation")
        return None
    
    interp_func = interp1d(time_sec_common, movement_data_common, 
                          kind='linear', bounds_error=False, fill_value=np.nan)
    movement_interpolated = interp_func(time_bins_common)
    
    # Remove any remaining NaN values
    valid_interp_mask = ~np.isnan(movement_interpolated)
    final_pc_data = pc_data_common[valid_interp_mask]
    final_movement_data = movement_interpolated[valid_interp_mask]
    final_time_bins = time_bins_common[valid_interp_mask]
    
    print(f"Final aligned data: {len(final_pc_data)} points")
    
    if len(final_pc_data) < 10:
        print("Error: Too few data points for reliable correlation")
        return None
    
    # Calculate correlations
    pearson_r, pearson_p = pearsonr(final_pc_data, final_movement_data)
    spearman_r, spearman_p = spearmanr(final_pc_data, final_movement_data)
    
    # Print results
    print(f"\n=== PC{pc_index+1} vs Movement Correlation Results ===")
    print(f"Time range analyzed: {final_time_bins[0]:.1f}s to {final_time_bins[-1]:.1f}s")
    print(f"Data points: {len(final_pc_data)}")
    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
    
    return {
        'pc_index': pc_index,
        'time_range': time_range,
        'final_time_bins': final_time_bins,
        'pc_data': final_pc_data,
        'movement_data': final_movement_data,
        'pearson_correlation': pearson_r,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_r,
        'spearman_p_value': spearman_p,
        'n_points': len(final_pc_data)
    }

def analyze_pc_regions_over_time(pca_results, dlc_folder, smoothed_results, neural_sleep_df, 
                               subject, output_folder, regions_to_analyze, 
                               movement_column='smoothed_difference', components_to_plot=(0, 1),
                               use_sg_filter=True, movement_upper_limit=800000, behavior='movement'):  
    """
    Analyze specific regions of PC space over time with behavioral and spectral context.
    Creates separate plot sets for each region with movement-intensity coloring.
    
    Parameters:
    -----------
    pca_results : dict
        Results from analyze_neural_pca function
    dlc_folder : str
        Path to the DLC folder containing behavioral data
    smoothed_results : dict
        Results from power_band_smoothing containing delta power
    neural_sleep_df : DataFrame
        DataFrame with sleep bout information
    subject : str
        Subject identifier
    output_folder : str
        Directory to save plots
    regions_to_analyze : list of dict
        List of regions to analyze, each dict should have:
        {'name': 'Region1', 'x_range': [x_min, x_max], 'y_range': [y_min, y_max]}
        Note: 'color' field is no longer needed as colors come from movement intensity
    movement_column : str
        Column name for movement data
    components_to_plot : tuple
        Which PC components to use (default: (0, 1) for PC1 vs PC2)
    use_sg_filter : bool
        Whether to use Savitzky-Golay (True) or moving average (False) filtered delta power
    behavior : str
        Type of behavioral data to plot: 'movement' or 'delta'
        
    Returns:
    --------
    dict : Analysis results including region assignments and timing
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    from scipy.interpolate import interp1d
    
    # Extract PC data
    if 'pca_result' not in pca_results:
        print("Error: PCA results not found")
        return None
    
    pc_data = pca_results['pca_result']
    time_bins_pca = pca_results['time_bins_used']
    
    comp1_idx, comp2_idx = components_to_plot
    pc1 = pc_data[:, comp1_idx]
    pc2 = pc_data[:, comp2_idx]
    
    print(f"PC{comp1_idx+1} vs PC{comp2_idx+1} analysis: {len(pc1)} time points")
    print(f"Behavior type: {behavior}")
    
    # Load behavioral data based on behavior parameter
    if behavior == 'movement':
        # Load movement data
        pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
        pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                           if f.endswith('.csv') and 'pixel_differences' in f]
        
        if not pixel_diff_files:
            print(f"No pixel difference CSV found in {pixel_diff_path}")
            return None
        
        # Load movement data
        pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
        try:
            behavior_data = pd.read_csv(pixel_diff_file)
            behavior_time = behavior_data['time_sec'].values
            behavior_values = behavior_data[movement_column].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(behavior_values)
            behavior_time = behavior_time[valid_mask]
            behavior_values = behavior_values[valid_mask]
            
            behavior_label = 'Movement (pixels)'
            behavior_title = 'Movement Over Time'
            
            print(f"Movement data loaded: {len(behavior_values)} points")
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None
            
        if behavior_values is not None:
            # Apply movement capping (same as plot_pca_with_behavioral_features)
            movement_min = np.nanmin(behavior_values)
            movement_max = np.nanmax(behavior_values)
            
            print(f"Original movement range: {movement_min:.0f} to {movement_max:.0f}")
            print(f"Applying upper limit cap at: {movement_upper_limit}")
            
            # Cap the movement values
            behavior_values_capped = np.clip(behavior_values, movement_min, movement_upper_limit)
            
            # Update the behavior_values for all subsequent processing
            behavior_values = behavior_values_capped
            
            capped_max = np.nanmax(behavior_values)
            print(f"Capped movement range: {movement_min:.0f} to {capped_max:.0f}")
            
    elif behavior == 'delta':
        # Get delta power data from smoothed_results
        band_powers = None
        filter_type = None
        
        # Determine which filter was used
        if output_folder:
            sleep_times_csv = os.path.join(output_folder, "sleep_times.csv")
            if os.path.exists(sleep_times_csv):
                try:
                    sleep_df = pd.read_csv(sleep_times_csv)
                    if 'filter' in sleep_df.columns and len(sleep_df) > 0:
                        filter_name = sleep_df['filter'].iloc[0]
                        if 'Savitzky-Golay' in filter_name:
                            filter_type = 'SG'
                            if 'savitzky_golay' in smoothed_results:
                                band_powers = smoothed_results['savitzky_golay']
                        elif 'MovingAverage' in filter_name or 'Moving Average' in filter_name:
                            filter_type = 'MA'
                            if 'moving_average' in smoothed_results:
                                band_powers = smoothed_results['moving_average']
                except Exception as e:
                    print(f"Error determining filter type: {e}")
        
        # Fallback to user-specified filter if CSV method failed
        if band_powers is None:
            if use_sg_filter:
                if 'savitzky_golay' in smoothed_results:
                    band_powers = smoothed_results['savitzky_golay']
                    filter_type = "SG"
                else:
                    print("Error: Savitzky-Golay filtered data not found in smoothed_results")
                    return None
            else:
                if 'moving_average' in smoothed_results:
                    band_powers = smoothed_results['moving_average']
                    filter_type = "MA"
                else:
                    print("Error: Moving average filtered data not found in smoothed_results")
                    return None
        
        # Extract Delta band data
        if 'Delta' not in band_powers:
            print(f"Error: Delta band not found in {filter_type} filtered data")
            return None
        
        delta_power = band_powers['Delta']
        behavior_time = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
        behavior_values = delta_power
        
        behavior_label = f'Delta Power (dB, {filter_type})'
        behavior_title = f'Delta Power Over Time ({filter_type} filtered)'
        
        print(f"Delta power data loaded: {len(behavior_values)} points")
    
    else:
        print(f"Error: Unknown behavior type '{behavior}'. Use 'movement' or 'delta'.")
        return None    # Get behavior values at PC time points for coloring
    if len(behavior_time) > 1:
        interp_func = interp1d(behavior_time, behavior_values, 
                            kind='linear', bounds_error=False, fill_value=np.nan)
        behavior_at_pc_times = interp_func(time_bins_pca)        
        # Remove NaN values for coloring
        valid_behavior_mask = ~np.isnan(behavior_at_pc_times)
        
        # Use values for normalization (5th to 95th percentile of data)
        behavior_norm = Normalize(vmin=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 5), 
                                vmax=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 95))
    else:
        print("Error: Insufficient behavior data for interpolation")
        return None
    
    # Get delta power data from smoothed_results
    band_powers = None
    smoothed_available = False
    filter_type = None
    
    # First check if smoothed_results is provided directly
    if smoothed_results is not None:
        # Try to determine which filter was used in save_sleep_periods_to_csv
        if output_folder:
            sleep_times_csv = os.path.join(output_folder, "sleep_times.csv")
            if os.path.exists(sleep_times_csv):
                try:
                    sleep_df = pd.read_csv(sleep_times_csv)
                    if 'filter' in sleep_df.columns and len(sleep_df) > 0:
                        filter_name = sleep_df['filter'].iloc[0]
                        if 'Savitzky-Golay' in filter_name:
                            filter_type = 'SG'
                            if 'savitzky_golay' in smoothed_results:
                                band_powers = smoothed_results['savitzky_golay']
                                smoothed_available = True
                        elif 'MovingAverage' in filter_name or 'Moving Average' in filter_name:
                            filter_type = 'MA'
                            if 'moving_average' in smoothed_results:
                                band_powers = smoothed_results['moving_average']
                                smoothed_available = True
                except Exception as e:
                    print(f"Error determining filter type: {e}")
        
        # Fallback to user-specified filter if CSV method failed
        if not smoothed_available:
            if use_sg_filter:
                if 'savitzky_golay' in smoothed_results:
                    band_powers = smoothed_results['savitzky_golay']
                    filter_type = "SG"
                    smoothed_available = True
                else:
                    print("Error: Savitzky-Golay filtered data not found in smoothed_results")
            else:
                if 'moving_average' in smoothed_results:
                    band_powers = smoothed_results['moving_average']
                    filter_type = "MA"
                    smoothed_available = True
                else:
                    print("Error: Moving average filtered data not found in smoothed_results")
    
    # Extract Delta band data from the band_powers dictionary
    delta_power = None
    delta_time = None
    
    if band_powers is not None and 'Delta' in band_powers:
        delta_power = band_powers['Delta']
        delta_time = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
        print(f"Delta power data ({filter_type}): {len(delta_power)} points")
    else:
        print("Error: Delta band not found in smoothed results")
        return None
    
    # Assign each PC point to regions and collect results
    all_region_results = {}
    
    for region_idx, region in enumerate(regions_to_analyze):        # Find points in this region
        x_mask = (pc1 >= region['x_range'][0]) & (pc1 <= region['x_range'][1])
        y_mask = (pc2 >= region['y_range'][0]) & (pc2 <= region['y_range'][1])
        region_mask = x_mask & y_mask & valid_behavior_mask
        
        if np.sum(region_mask) == 0:
            print(f"Warning: No valid points found in region '{region['name']}'")
            continue
        
        # Get data for this region
        region_times = time_bins_pca[region_mask]
        region_behavior_values = behavior_at_pc_times[region_mask]
        region_pc1 = pc1[region_mask]  
        region_pc2 = pc2[region_mask]        
        print(f"Region '{region['name']}': {len(region_times)} points")
        
        # Create color map based on behavior intensity (like in the PCA plot)
        # Use the same colormap as plot_pca_with_behavioral_features
        norm = Normalize(vmin=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 5), 
                        vmax=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 95))
        cmap = cm.get_cmap('coolwarm')  # Blue for low behavior, red for high behavior
        region_colors = cmap(norm(region_behavior_values))        
        # Create the plot for this region
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'{subject}: {region["name"]} - PC Region Analysis Over Time', fontsize=16)
        
        # Top plot: Behavior over time based on behavior parameter
        ax1 = axes[0]
        ax1.plot(behavior_time, behavior_values, 'k-', linewidth=0.5, alpha=0.3, label=f'All {behavior.title()}')
        
        # Add sleep periods as background
        for _, row in neural_sleep_df.iterrows():
            ax1.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                       color='blue', alpha=0.2, label='Sleep' if _ == 0 else "")
        
        # Get behavior values at region times (interpolate)
        if len(behavior_time) > 1:
            behavior_at_region_times = interp_func(region_times)
            valid_interp = ~np.isnan(behavior_at_region_times)
            
            # Plot region points with behavior-intensity colors
            scatter = ax1.scatter(region_times[valid_interp], behavior_at_region_times[valid_interp], 
                                c=region_behavior_values[valid_interp], cmap='coolwarm', 
                                s=20, alpha=0.8, norm=norm, zorder=5, 
                                label=f"{region['name']} (n={np.sum(valid_interp)})")
        
        ax1.set_ylabel(behavior_label)
        ax1.set_title(behavior_title + f' - {region["name"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)        
        # Middle plot: Delta power over time
        ax2 = axes[1]
        
        # Get delta power data from smoothed_results for delta plot
        delta_band_powers = None
        delta_filter_type = None
        
        # First check if smoothed_results is provided directly
        if smoothed_results is not None:
            # Try to determine which filter was used in save_sleep_periods_to_csv
            if output_folder:
                sleep_times_csv = os.path.join(output_folder, "sleep_times.csv")
                if os.path.exists(sleep_times_csv):
                    try:
                        sleep_df = pd.read_csv(sleep_times_csv)
                        if 'filter' in sleep_df.columns and len(sleep_df) > 0:
                            filter_name = sleep_df['filter'].iloc[0]
                            if 'Savitzky-Golay' in filter_name:
                                delta_filter_type = 'SG'
                                if 'savitzky_golay' in smoothed_results:
                                    delta_band_powers = smoothed_results['savitzky_golay']
                            elif 'MovingAverage' in filter_name or 'Moving Average' in filter_name:
                                delta_filter_type = 'MA'
                                if 'moving_average' in smoothed_results:
                                    delta_band_powers = smoothed_results['moving_average']
                    except Exception as e:
                        print(f"Error determining filter type: {e}")
            
            # Fallback to user-specified filter if CSV method failed
            if delta_band_powers is None:
                if use_sg_filter:
                    if 'savitzky_golay' in smoothed_results:
                        delta_band_powers = smoothed_results['savitzky_golay']
                        delta_filter_type = "SG"
                    else:
                        print("Error: Savitzky-Golay filtered data not found in smoothed_results")
                else:
                    if 'moving_average' in smoothed_results:
                        delta_band_powers = smoothed_results['moving_average']
                        delta_filter_type = "MA"
                    else:
                        print("Error: Moving average filtered data not found in smoothed_results")
        
        # Extract Delta band data from the delta_band_powers dictionary
        if delta_band_powers is not None and 'Delta' in delta_band_powers:
            delta_power = delta_band_powers['Delta']
            delta_time = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
            
            ax2.plot(delta_time, delta_power, 'purple', linewidth=0.8, alpha=0.3, label=f'All Delta Power ({delta_filter_type})')
            
            # Add sleep periods
            for _, row in neural_sleep_df.iterrows():
                ax2.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                           color='blue', alpha=0.2, label='Sleep' if _ == 0 else "")
            
            # Get delta power values at region times (interpolate)
            if len(delta_time) > 1:
                interp_func_delta = interp1d(delta_time, delta_power, 
                                           kind='linear', bounds_error=False, fill_value=np.nan)
                delta_at_region_times = interp_func_delta(region_times)
                valid_interp_delta = ~np.isnan(delta_at_region_times)
                
                # Plot region points with behavior-intensity colors
                scatter2 = ax2.scatter(region_times[valid_interp_delta], delta_at_region_times[valid_interp_delta], 
                                     c=region_behavior_values[valid_interp_delta], cmap='coolwarm', 
                                     s=20, alpha=0.8, norm=norm, zorder=5,
                                     label=f"{region['name']} (n={np.sum(valid_interp_delta)})")
            
            ax2.set_ylabel('Delta Power (dB)')
            ax2.set_title(f'Delta Power Over Time ({delta_filter_type} filtered) - {region["name"]}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Delta power data not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Delta Power Over Time - {region["name"]}')
            ax2.set_ylabel('Delta Power (dB)')
            ax2.grid(True, alpha=0.3)
        
        # Bottom plot: PC space showing this region
        ax3 = axes[2]
        
        # Plot all points in light gray
        ax3.scatter(pc1, pc2, c='lightgray', s=5, alpha=0.2, label='All Data')        
        # Plot this region's points with behavior-intensity colors
        scatter3 = ax3.scatter(region_pc1, region_pc2, 
                             c=region_behavior_values, cmap='coolwarm', 
                             s=25, alpha=0.8, norm=norm, 
                             label=f"{region['name']} (n={len(region_pc1)})")
        
        # Draw region boundary
        x_min, x_max = region['x_range']
        y_min, y_max = region['y_range']
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                           fill=False, edgecolor='black', linewidth=2, linestyle='--')
        ax3.add_patch(rect)
        
        ax3.set_xlabel(f'PC{comp1_idx+1}')
        ax3.set_ylabel(f'PC{comp2_idx+1}')
        ax3.set_title(f'PC{comp1_idx+1} vs PC{comp2_idx+1} - {region["name"]}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax3.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        
        # Add colorbar for behavior intensity
        plt.colorbar(scatter3, ax=ax3, label=behavior_label)
        
        plt.tight_layout()        
        # Save the plot for this region
        safe_region_name = region['name'].replace(' ', '_').replace('/', '_')
        if behavior == 'delta':
            plot_path = os.path.join(output_folder, f'{subject}_pc_region_{safe_region_name}_analysis_delta.png')
        else:
            plot_path = os.path.join(output_folder, f'{subject}_pc_region_{safe_region_name}_analysis_movement.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"PC region analysis plot saved: {plot_path}")
        
        # Store results for this region
        all_region_results[region['name']] = {
            'region_mask': region_mask,
            'time_points': region_times,
            'behavior_values': region_behavior_values,
            'pc1_values': region_pc1,
            'pc2_values': region_pc2,
            'n_points': len(region_times)
        }
        
        # Print summary statistics for this region
        print(f"\n=== {region['name']} Analysis Summary ===")
          # Check sleep/wake distribution
        sleep_count = 0
        wake_count = 0
        
        for time_point in region_times:
            is_sleep = False
            for _, row in neural_sleep_df.iterrows():
                if row['start_timestamp_s'] <= time_point <= row['end_timestamp_s']:
                    is_sleep = True
                    break
            
            if is_sleep:
                sleep_count += 1
            else:
                wake_count += 1
        
        print(f"  Total points: {len(region_times)}")
        print(f"  Sleep points: {sleep_count} ({sleep_count/len(region_times)*100:.1f}%)")
        print(f"  Wake points: {wake_count} ({wake_count/len(region_times)*100:.1f}%)")
        print(f"  Time range: {region_times.min():.1f}s - {region_times.max():.1f}s")
        print(f"  {behavior.title()} range: {region_behavior_values.min():.1f} - {region_behavior_values.max():.1f}")
    
    return {
        'region_results': all_region_results,
        'pc1': pc1,
        'pc2': pc2,
        'time_bins': time_bins_pca,
        'behavior_at_pc_times': behavior_at_pc_times,
        'regions_analyzed': regions_to_analyze,
        'behavior_type': behavior
    }

def analyze_time_regions_in_pc_space(pca_results, dlc_folder, smoothed_results, neural_sleep_df, 
                                   subject, output_folder, time_regions_to_analyze, 
                                   movement_column='smoothed_difference', components_to_plot=(0, 1),
                                   use_sg_filter=True, behavior='movement'):
    """
    Analyze specific time regions by highlighting them in PC space with behavioral context.
    Creates three-panel plots: behavior over time, PC space with movement coloring, and PC space with time gradient coloring.
    
    Parameters:
    -----------
    pca_results : dict
        Results from analyze_neural_pca function
    dlc_folder : str
        Path to the DLC folder containing behavioral data
    smoothed_results : dict
        Results from power_band_smoothing containing delta power
    neural_sleep_df : DataFrame
        DataFrame with sleep bout information
    subject : str
        Subject identifier
    output_folder : str
        Directory to save plots
    time_regions_to_analyze : list of dict
        List of time regions to analyze, each dict should have:
        {'name': 'Region1', 'start_time': start_s, 'end_time': end_s}
    movement_column : str
        Column name for movement data
    components_to_plot : tuple
        Which PC components to use (default: (0, 1) for PC1 vs PC2)
    use_sg_filter : bool
        Whether to use Savitzky-Golay (True) or moving average (False) filtered delta power
    behavior : str
        Type of behavioral data to plot: 'movement' or 'delta'
        
    Returns:
    --------
    dict : Analysis results including time region assignments and data
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    from scipy.interpolate import interp1d
    
    # Extract PC data
    if 'pca_result' not in pca_results:
        print("Error: PCA results not found")
        return None
    
    pc_data = pca_results['pca_result']
    time_bins_pca = pca_results['time_bins_used']
    
    comp1_idx, comp2_idx = components_to_plot
    pc1 = pc_data[:, comp1_idx]
    pc2 = pc_data[:, comp2_idx]
    
    print(f"PC{comp1_idx+1} vs PC{comp2_idx+1} analysis: {len(pc1)} time points")
    print(f"Time range: {time_bins_pca[0]:.1f}s to {time_bins_pca[-1]:.1f}s")
    print(f"Behavior type: {behavior}")
    
    # Load behavioral data based on behavior parameter
    if behavior == 'movement':
        # Load movement data
        pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
        pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                           if f.endswith('.csv') and 'pixel_differences' in f]
        
        if not pixel_diff_files:
            print(f"No pixel difference CSV found in {pixel_diff_path}")
            return None
        
        pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
        try:
            behavior_data = pd.read_csv(pixel_diff_file)
            behavior_time = behavior_data['time_sec'].values
            behavior_values = behavior_data[movement_column].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(behavior_values)
            behavior_time = behavior_time[valid_mask]
            behavior_values = behavior_values[valid_mask]
            
            behavior_label = 'Movement (pixels)'
            behavior_title = 'Movement Over Time'
            color_map = 'coolwarm'  # Blue = low movement, Red = high movement
            reverse_colormap = False
            
            print(f"Movement data loaded: {len(behavior_values)} points")
            
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None
            
    elif behavior == 'delta':
        # Get delta power data from smoothed_results
        band_powers = None
        filter_type = None
        
        # Determine which filter was used
        if output_folder:
            sleep_times_csv = os.path.join(output_folder, "sleep_times.csv")
            if os.path.exists(sleep_times_csv):
                try:
                    sleep_df = pd.read_csv(sleep_times_csv)
                    if 'filter' in sleep_df.columns and len(sleep_df) > 0:
                        filter_name = sleep_df['filter'].iloc[0]
                        if 'Savitzky-Golay' in filter_name:
                            filter_type = 'SG'
                            if 'savitzky_golay' in smoothed_results:
                                band_powers = smoothed_results['savitzky_golay']
                        elif 'MovingAverage' in filter_name or 'Moving Average' in filter_name:
                            filter_type = 'MA'
                            if 'moving_average' in smoothed_results:
                                band_powers = smoothed_results['moving_average']
                except Exception as e:
                    print(f"Error determining filter type: {e}")
        
        # Fallback to user-specified filter if CSV method failed
        if band_powers is None:
            if use_sg_filter:
                if 'savitzky_golay' in smoothed_results:
                    band_powers = smoothed_results['savitzky_golay']
                    filter_type = "SG"
                else:
                    print("Error: Savitzky-Golay filtered data not found in smoothed_results")
                    return None
            else:
                if 'moving_average' in smoothed_results:
                    band_powers = smoothed_results['moving_average']
                    filter_type = "MA"
                else:
                    print("Error: Moving average filtered data not found in smoothed_results")
                    return None
        
        # Extract Delta band data
        if 'Delta' not in band_powers:
            print(f"Error: Delta band not found in {filter_type} filtered data")
            return None
        
        delta_power = band_powers['Delta']
        behavior_time = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
        behavior_values = delta_power
        
        behavior_label = f'Delta Power (dB, {filter_type})'
        behavior_title = f'Delta Power Over Time ({filter_type} filtered)'
        color_map = 'coolwarm_r'  # Reverse: Blue = high delta (sleep), Red = low delta (wake)
        reverse_colormap = True
        
        print(f"Delta power data loaded: {len(behavior_values)} points")
    
    else:
        print(f"Error: Unknown behavior type '{behavior}'. Use 'movement' or 'delta'.")
        return None
    
    # Get behavior values at PC time points for coloring
    if len(behavior_time) > 1:
        interp_func = interp1d(behavior_time, behavior_values, 
                             kind='linear', bounds_error=False, fill_value=np.nan)
        behavior_at_pc_times = interp_func(time_bins_pca)
        
        # Remove NaN values for coloring
        valid_behavior_mask = ~np.isnan(behavior_at_pc_times)
        print(f"Valid behavior data at PC times: {np.sum(valid_behavior_mask)}/{len(behavior_at_pc_times)}")
    else:
        print("Error: Insufficient behavior data for interpolation")
        return None
    
    # Define different color gradients for time coloring
    time_colormaps = ['inferno', 'plasma', 'viridis', 'magma', 'cividis', 'turbo']
    
    # Store results for all time regions
    all_time_region_results = {}
    
    for region_idx, time_region in enumerate(time_regions_to_analyze):
        start_time = time_region['start_time']
        end_time = time_region['end_time']
        region_name = time_region['name']
        
        print(f"\nAnalyzing time region '{region_name}': {start_time}s to {end_time}s")
        
        # Find PC time points within this time region
        time_mask = (time_bins_pca >= start_time) & (time_bins_pca <= end_time) & valid_behavior_mask
        
        if np.sum(time_mask) == 0:
            print(f"Warning: No valid points found in time region '{region_name}'")
            continue
        
        # Get data for this time region
        region_times = time_bins_pca[time_mask]
        region_behavior_values = behavior_at_pc_times[time_mask]
        region_pc1 = pc1[time_mask]
        region_pc2 = pc2[time_mask]
        
        print(f"Time region '{region_name}': {len(region_times)} points")
        
        # Create color map based on behavior intensity
        behavior_norm = Normalize(vmin=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 5), 
                                vmax=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 95))
        
        # Create time-based color map (from start to end of time region)
        time_norm = Normalize(vmin=region_times.min(), vmax=region_times.max())
        time_cmap = cm.get_cmap(time_colormaps[region_idx % len(time_colormaps)])
        
        # Create the 3-panel plot for this time region
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
        
        # Set overall title for the figure (bigger and bold)
        fig.suptitle(region_name, fontsize=20, fontweight='bold', y=0.98)
        
        # Top plot (spans both columns): Behavior over time with highlighted region
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(behavior_time, behavior_values, 'k-', linewidth=0.8, alpha=0.7)
        
        # Add sleep periods as background
        for _, row in neural_sleep_df.iterrows():
            ax1.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                       color='blue', alpha=0.2)
        
        # Highlight the selected time region
        ax1.axvspan(start_time, end_time, color='yellow', alpha=0.3, zorder=1)
        
        # Get behavior values at region times (interpolate)
        if len(behavior_time) > 1:
            behavior_at_region_times = interp_func(region_times)
            valid_interp = ~np.isnan(behavior_at_region_times)
            
            # Plot region points with behavior-intensity colors
            if np.sum(valid_interp) > 0:
                scatter1 = ax1.scatter(region_times[valid_interp], behavior_at_region_times[valid_interp], 
                                     c=region_behavior_values[valid_interp], cmap=color_map, 
                                     s=30, alpha=0.9, norm=behavior_norm, zorder=5, 
                                     edgecolors='black', linewidths=0.5)
        
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax1.set_ylabel(behavior_label, fontsize=14)
        # Remove subplot title since we have the main title now
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(behavior_time[0], behavior_time[-1])
        
        # Bottom left: PC space with behavior-intensity coloring
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Plot all points in light gray first
        ax2.scatter(pc1, pc2, c='lightgray', s=10, alpha=0.3)
        
        # Plot the time region's points with behavior-intensity colors
        if len(region_pc1) > 0:
            scatter2 = ax2.scatter(region_pc1, region_pc2, 
                                 c=region_behavior_values, cmap=color_map, 
                                 s=25, alpha=0.8, norm=behavior_norm, 
                                 edgecolors='black', linewidths=0.3)
            
            # Add colorbar for behavior intensity
            cbar2 = plt.colorbar(scatter2, ax=ax2, label=behavior_label, shrink=0.8)
        
        ax2.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
        ax2.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
        if behavior == 'movement':
            ax2.set_title('PC Space (Movement)', fontsize=14, fontweight='bold')
        else:
            ax2.set_title('PC Space (Delta Power)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax2.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        
        # Bottom right: PC space with time gradient coloring
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Plot all points in light gray first
        ax3.scatter(pc1, pc2, c='lightgray', s=10, alpha=0.3)
        
        # Plot the time region's points with time gradient colors
        if len(region_pc1) > 0:
            scatter3 = ax3.scatter(region_pc1, region_pc2, 
                                 c=region_times, cmap=time_cmap, 
                                 s=25, alpha=0.8, norm=time_norm,
                                 edgecolors='black', linewidths=0.3)
            
            # Add colorbar for time gradient
            cbar3 = plt.colorbar(scatter3, ax=ax3, label='Time (s)', shrink=0.8)
            
            # Add start/end markers
            if len(region_pc1) > 1:
                start_idx = np.argmin(region_times)
                end_idx = np.argmax(region_times)
                
                ax3.plot(region_pc1[start_idx], region_pc2[start_idx], 'go', 
                        markersize=8, zorder=11)
                ax3.plot(region_pc1[end_idx], region_pc2[end_idx], 'ro', 
                        markersize=8, zorder=11)
        
        ax3.set_xlabel(f'PC{comp1_idx+1}', fontsize=14)
        ax3.set_ylabel(f'PC{comp2_idx+1}', fontsize=14)
        ax3.set_title('PC Space (Chronologically)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax3.axvline(0, color='gray', linestyle='--', linewidth=0.7)
        
        plt.tight_layout()
        
        # Save the plot for this time region
        safe_region_name = region_name.replace(' ', '_').replace('/', '_')
        plot_path = os.path.join(output_folder, f'{subject}_{behavior}_{start_time}s-{end_time}s_PC{comp1_idx+1}_{comp2_idx+1}_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Time region analysis plot saved: {plot_path}")
        
        # Store results for this time region
        all_time_region_results[region_name] = {
            'start_time': start_time,
            'end_time': end_time,
            'time_mask': time_mask,
            'time_points': region_times,
            'behavior_values': region_behavior_values,
            'pc1_values': region_pc1,
            'pc2_values': region_pc2,
            'n_points': len(region_times),
            'time_colormap': time_colormaps[region_idx % len(time_colormaps)],
            'behavior_type': behavior
        }
        
        # Print summary statistics for this time region
        print(f"\n=== {region_name} Analysis Summary ===")
        
        # Check sleep/wake distribution
        sleep_count = 0
        wake_count = 0
        
        for time_point in region_times:
            is_sleep = False
            for _, row in neural_sleep_df.iterrows():
                if row['start_timestamp_s'] <= time_point <= row['end_timestamp_s']:
                    is_sleep = True
                    break
            
            if is_sleep:
                sleep_count += 1
            else:
                wake_count += 1
        
        print(f"  Time window: {start_time}s - {end_time}s ({end_time - start_time}s duration)")
        print(f"  Total points: {len(region_times)}")
        print(f"  Sleep points: {sleep_count} ({sleep_count/len(region_times)*100:.1f}%)")
        print(f"  Wake points: {wake_count} ({wake_count/len(region_times)*100:.1f}%)")
        
        if behavior == 'movement':
            print(f"  Movement range: {region_behavior_values.min():.1f} - {region_behavior_values.max():.1f} pixels")
        else:
            print(f"  Delta power range: {region_behavior_values.min():.2f} - {region_behavior_values.max():.2f} dB")
            
        print(f"  PC{comp1_idx+1} range: {region_pc1.min():.2f} - {region_pc1.max():.2f}")
        print(f"  PC{comp2_idx+1} range: {region_pc2.min():.2f} - {region_pc2.max():.2f}")
    
    return {
        'time_region_results': all_time_region_results,
        'pc1': pc1,
        'pc2': pc2,
        'time_bins': time_bins_pca,
        'behavior_at_pc_times': behavior_at_pc_times,
        'time_regions_analyzed': time_regions_to_analyze,
        'behavior_type': behavior
    }



def analyze_averaged_time_windows_in_pc_space(pca_results, dlc_folder, smoothed_results, neural_sleep_df, 
                                            subject, output_folder, time_windows_to_analyze, 
                                            movement_column='smoothed_difference', components_to_plot=(0, 1),
                                            use_sg_filter=True, behavior='delta'):
    """
    Analyze multiple time windows of the same length by averaging corresponding time points across windows.
    Creates plots showing the averaged trajectory in PC space.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    from scipy.interpolate import interp1d
    
    # Extract PC data
    if 'pca_result' not in pca_results:
        print("Error: PCA results not found")
        return None
    
    pc_data = pca_results['pca_result']
    time_bins_pca = pca_results['time_bins_used']
    
    comp1_idx, comp2_idx = components_to_plot
    pc1 = pc_data[:, comp1_idx]
    pc2 = pc_data[:, comp2_idx]
    
    print(f"PC{comp1_idx+1} vs PC{comp2_idx+1} analysis: {len(pc1)} time points")
    print(f"Time range: {time_bins_pca[0]:.1f}s to {time_bins_pca[-1]:.1f}s")
    print(f"Behavior type: {behavior}")
    
    # Check that all windows are the same length with floating-point error tolerance
    window_durations = [w['end_time'] - w['start_time'] for w in time_windows_to_analyze]
    
    # Round durations to 6 decimal places to handle floating-point errors
    rounded_durations = [round(duration, 6) for duration in window_durations]
    
    print(f"Original durations: {window_durations}")
    print(f"Rounded durations: {rounded_durations}")
    
    if len(set(rounded_durations)) > 1:
        print(f"Error: All time windows must be the same length. Found durations: {window_durations}")
        return None
    
    window_duration = rounded_durations[0]  # Use the rounded duration
    print(f"Analyzing {len(time_windows_to_analyze)} windows of {window_duration}s each")
    
    # EXACT COPY FROM WORKING plot_pca_with_behavioral_features function
    # Load behavioral data (pixel differences)
    pixel_diff_path = os.path.join(dlc_folder, "pixel_difference")
    pixel_diff_files = [f for f in os.listdir(pixel_diff_path) 
                       if f.endswith('.csv') and 'pixel_differences' in f]
    
    movement_data = None
    if pixel_diff_files:
        pixel_diff_file = os.path.join(pixel_diff_path, pixel_diff_files[0])
        try:
            movement_data = pd.read_csv(pixel_diff_file)
            print(f"Loaded movement data from {pixel_diff_file}")
        except Exception as e:
            print(f"Error loading movement data: {e}")
    
    # Extract delta power data (same as working function)
    delta_power = None
    if smoothed_results and 'savitzky_golay' in smoothed_results and 'Delta' in smoothed_results['savitzky_golay']:
        delta_power = smoothed_results['savitzky_golay']['Delta']
        print(f"Loaded delta power data: {len(delta_power)} time points")
    else:
        print("Warning: No delta power data found in smoothed_results")
    
    # Set up data based on behavior type (EXACT COPY)
    if behavior == 'movement':
        if movement_data is not None:
            behavior_time = movement_data['time_sec'].values
            behavior_values = movement_data[movement_column].values
            # Remove NaN values
            valid_mask = ~np.isnan(behavior_values)
            behavior_time = behavior_time[valid_mask]
            behavior_values = behavior_values[valid_mask]
            behavior_label = 'Movement (pixels)'
            behavior_title = 'Movement Over Time'
            color_map = 'coolwarm'
        else:
            print("Error: No movement data available")
            return None
    elif behavior == 'delta':
        if delta_power is not None:
            # CREATE TIME ARRAY THAT MATCHES DELTA POWER LENGTH - EXACT COPY
            behavior_time = np.linspace(time_bins_pca[0], time_bins_pca[-1], len(delta_power))
            behavior_values = delta_power
            behavior_label = 'Delta Power (dB, SG)'
            behavior_title = 'Delta Power Over Time'
            color_map = 'coolwarm_r'
        else:
            print("Error: No delta power data available")
            return None
    else:
        print(f"Error: Unknown behavior type '{behavior}'")
        return None
    
    # Get behavior values at PC time points for coloring
    if len(behavior_time) > 1:
        interp_func = interp1d(behavior_time, behavior_values, 
                             kind='linear', bounds_error=False, fill_value=np.nan)
        behavior_at_pc_times = interp_func(time_bins_pca)
        
        # Remove NaN values for coloring
        valid_behavior_mask = ~np.isnan(behavior_at_pc_times)
        print(f"Valid behavior data at PC times: {np.sum(valid_behavior_mask)}/{len(behavior_at_pc_times)}")
    else:
        print("Error: Insufficient behavior data for interpolation")
        return None
    
    # Extract data for each time window and calculate averages
    all_window_pc1 = []
    all_window_pc2 = []
    all_window_behavior = []
    all_window_times = []
    
    for window_idx, time_window in enumerate(time_windows_to_analyze):
        start_time = time_window['start_time']
        end_time = time_window['end_time']
        window_name = time_window['name']
        
        # Find PC time points within this time window
        time_mask = (time_bins_pca >= start_time) & (time_bins_pca <= end_time) & valid_behavior_mask
        
        if np.sum(time_mask) == 0:
            print(f"Warning: No valid points found in window '{window_name}'")
            continue
        
        # Get data for this window
        window_times = time_bins_pca[time_mask]
        window_behavior_values = behavior_at_pc_times[time_mask]
        window_pc1 = pc1[time_mask]
        window_pc2 = pc2[time_mask]
        
        # Convert to relative time within window (0 to window_duration)
        relative_times = window_times - start_time
        
        all_window_pc1.append(window_pc1)
        all_window_pc2.append(window_pc2)
        all_window_behavior.append(window_behavior_values)
        all_window_times.append(relative_times)
        
        print(f"Window '{window_name}': {len(window_times)} points")
    
    if len(all_window_pc1) == 0:
        print("Error: No valid windows found")
        return None
    
    max_points = max(len(w) for w in all_window_pc1)
    print(f"Using {max_points} points for full trajectory (max across windows)")

    # Initialize arrays for averaging - use the maximum number of points
    averaged_pc1 = np.full(max_points, np.nan)
    averaged_pc2 = np.full(max_points, np.nan)
    averaged_behavior = np.full(max_points, np.nan)
    point_counts = np.zeros(max_points)  # Track how many windows contribute to each point

    # Create time array for the full expected duration
    averaged_times = np.linspace(0, window_duration, max_points)

    # Accumulate values from all windows
    for window_idx, (w_pc1, w_pc2, w_behavior, w_times) in enumerate(zip(
        all_window_pc1, all_window_pc2, all_window_behavior, all_window_times)):
        
        window_length = len(w_pc1)
        
        # For each point in this window, add to the running sum
        for point_idx in range(window_length):
            if point_idx < max_points:  # Safety check
                if np.isnan(averaged_pc1[point_idx]):
                    # First contribution to this point
                    averaged_pc1[point_idx] = w_pc1[point_idx]
                    averaged_pc2[point_idx] = w_pc2[point_idx]
                    averaged_behavior[point_idx] = w_behavior[point_idx]
                else:
                    # Add to existing sum
                    averaged_pc1[point_idx] += w_pc1[point_idx]
                    averaged_pc2[point_idx] += w_pc2[point_idx]
                    averaged_behavior[point_idx] += w_behavior[point_idx]
                
                point_counts[point_idx] += 1

    # Calculate averages by dividing by the number of contributing windows
    for point_idx in range(max_points):
        if point_counts[point_idx] > 0:
            averaged_pc1[point_idx] /= point_counts[point_idx]
            averaged_pc2[point_idx] /= point_counts[point_idx]
            averaged_behavior[point_idx] /= point_counts[point_idx]

    # Remove any remaining NaN values (shouldn't happen, but safety check)
    valid_avg_mask = ~(np.isnan(averaged_pc1) | np.isnan(averaged_pc2) | np.isnan(averaged_behavior))

    if np.sum(valid_avg_mask) < max_points:
        print(f"Warning: Only {np.sum(valid_avg_mask)}/{max_points} points have valid averages")
        averaged_pc1 = averaged_pc1[valid_avg_mask]
        averaged_pc2 = averaged_pc2[valid_avg_mask]
        averaged_behavior = averaged_behavior[valid_avg_mask]
        averaged_times = averaged_times[valid_avg_mask]

    print(f"Final averaged trajectory: {len(averaged_pc1)} points")
    
    # Create color map based on behavior intensity
    behavior_norm = Normalize(vmin=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 5), 
                            vmax=np.nanpercentile(behavior_at_pc_times[valid_behavior_mask], 95))
    
    # Create time-based color map
    time_norm = Normalize(vmin=0, vmax=window_duration)
    
    # Create the 4-panel plot
    fig = plt.figure(figsize=(24, 16))
    gs = plt.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2, 1])
    
    # Top plot (spans both columns): Full behavior over time with highlighted windows
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(behavior_time, behavior_values, 'k-', linewidth=0.8, alpha=0.7, label=behavior_title)
    
    # Add sleep periods as background
    for idx, row in neural_sleep_df.iterrows():
        ax1.axvspan(row['start_timestamp_s'], row['end_timestamp_s'], 
                   color='blue', alpha=0.2, label='Sleep' if idx == 0 else "")
    
    # Highlight all selected time windows (NO WINDOW LABELS IN LEGEND)
    window_colors = plt.cm.Set3(np.linspace(0, 1, len(time_windows_to_analyze)))
    for window_idx, time_window in enumerate(time_windows_to_analyze):
        start_time = time_window['start_time']
        end_time = time_window['end_time']
        
        ax1.axvspan(start_time, end_time, color=window_colors[window_idx], alpha=0.3, zorder=1)
        
        # Get behavior values for this window and plot points COLORED BY BEHAVIOR VALUES
        window_mask = (behavior_time >= start_time) & (behavior_time <= end_time)
        if np.sum(window_mask) > 0:
            window_behavior_vals = behavior_values[window_mask]
            window_time_vals = behavior_time[window_mask]
            # Color points by behavior values, not window colors
            ax1.scatter(window_time_vals, window_behavior_vals, 
                       c=window_behavior_vals, cmap=color_map, s=20, alpha=0.8, 
                       norm=behavior_norm, zorder=5)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(behavior_label)
    ax1.set_title(f'{behavior_title} - {len(time_windows_to_analyze)} Windows Highlighted ({window_duration}s each)')
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(behavior_time[0], behavior_time[-1])
    
    # Middle plot (spans both columns): Averaged behavior over relative time
    ax2 = fig.add_subplot(gs[1, :])
    
    # Plot individual windows in light colors (only up to their available length)
    for window_idx, (window_behavior, window_times) in enumerate(zip(all_window_behavior, all_window_times)):
        ax2.plot(window_times, window_behavior, 
                color='lightgray', alpha=0.5, linewidth=1)
    
    # Plot averaged behavior
    ax2.plot(averaged_times, averaged_behavior, 'k-', linewidth=3, alpha=0.8, label='Averaged')
    
    # Plot points with behavior-intensity colors
    scatter2 = ax2.scatter(averaged_times, averaged_behavior, 
                         c=averaged_behavior, cmap=color_map, 
                         s=40, alpha=0.9, norm=behavior_norm, zorder=5, 
                         edgecolors='black', linewidths=0.5)
    
    ax2.set_xlabel('Relative Time (s)')
    ax2.set_ylabel(behavior_label)
    ax2.set_title(f'Averaged {behavior_title} - {len(time_windows_to_analyze)} Windows')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, window_duration)
    
    # Bottom left: PC space with behavior-intensity coloring
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Plot all PC points in light gray first
    ax3.scatter(pc1, pc2, c='lightgray', s=8, alpha=0.2, label='All Other Times')
    
    # Plot ONLY the averaged trajectory points with behavior-intensity colors (no connections)
    scatter3 = ax3.scatter(averaged_pc1, averaged_pc2, 
                         c=averaged_behavior, cmap=color_map, 
                         s=60, alpha=0.9, norm=behavior_norm, 
                         edgecolors='black', linewidths=0.8,
                         label=f"Averaged (n={len(time_windows_to_analyze)} windows)")
    
    # Add start/end markers with SAME SIZE and COLORED BY BEHAVIOR with GREEN/RED BORDERS
    ax3.scatter(averaged_pc1[0], averaged_pc2[0], 
               c=averaged_behavior[0], cmap=color_map, s=60, alpha=0.9, norm=behavior_norm,
               edgecolors='green', linewidths=2, label='Start', zorder=11)
    ax3.scatter(averaged_pc1[-1], averaged_pc2[-1], 
               c=averaged_behavior[-1], cmap=color_map, s=60, alpha=0.9, norm=behavior_norm,
               edgecolors='red', linewidths=2, label='End', zorder=11)
    
    # Add colorbar for behavior intensity
    cbar3 = plt.colorbar(scatter3, ax=ax3, label=behavior_label, shrink=0.8)
    
    ax3.set_xlabel(f'PC{comp1_idx+1}')
    ax3.set_ylabel(f'PC{comp2_idx+1}')
    if behavior == 'movement':
        ax3.set_title(f'PC Space - Movement Coloring (Averaged)')
    else:
        ax3.set_title(f'PC Space - Delta Power Coloring (Averaged)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Bottom right: PC space with time gradient coloring
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Plot all PC points in light gray first
    ax4.scatter(pc1, pc2, c='lightgray', s=8, alpha=0.2, label='All Other Times')
    
    # Plot ONLY the averaged trajectory points with time gradient colors (no connections)
    scatter4 = ax4.scatter(averaged_pc1, averaged_pc2, 
                         c=averaged_times, cmap='viridis', 
                         s=60, alpha=0.9, norm=time_norm,
                         edgecolors='black', linewidths=0.8,
                         label=f"Averaged (n={len(time_windows_to_analyze)} windows)")
    
    # Add start/end markers with SAME SIZE and TIME-COLORED with GREEN/RED BORDERS
    ax4.scatter(averaged_pc1[0], averaged_pc2[0], 
               c=averaged_times[0], cmap='viridis', s=60, alpha=0.9, norm=time_norm,
               edgecolors='green', linewidths=2, label='Start', zorder=11)
    ax4.scatter(averaged_pc1[-1], averaged_pc2[-1], 
               c=averaged_times[-1], cmap='viridis', s=60, alpha=0.9, norm=time_norm,
               edgecolors='red', linewidths=2, label='End', zorder=11)
    
    # Add colorbar for time gradient
    cbar4 = plt.colorbar(scatter4, ax=ax4, label='Relative Time (s)', shrink=0.8)
    
    ax4.set_xlabel(f'PC{comp1_idx+1}')
    ax4.set_ylabel(f'PC{comp2_idx+1}')
    ax4.set_title(f'PC Space - Time Gradient (Averaged)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    
    # Overall title
    behavior_type_title = "Movement" if behavior == 'movement' else "Delta Power"
    fig.suptitle(f'{subject}: Averaged Time Windows Analysis ({behavior_type_title}) - {window_duration}s windows', 
                fontsize=18, y=0.96)
    
    plt.tight_layout()
    
    # Save the plot
    safe_name = f"averaged_{len(time_windows_to_analyze)}windows_{window_duration}s_{behavior}"
    plot_path = os.path.join(output_folder, f'{subject}_{safe_name}_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Averaged time windows analysis plot saved: {plot_path}")
    
    # Calculate trajectory statistics
    distances = np.sqrt(np.diff(averaged_pc1)**2 + np.diff(averaged_pc2)**2)
    total_distance = np.sum(distances)
    net_displacement = np.sqrt((averaged_pc1[-1] - averaged_pc1[0])**2 + 
                             (averaged_pc2[-1] - averaged_pc2[0])**2)
    tortuosity = total_distance / net_displacement if net_displacement > 0 else np.inf
    
    print(f"\n=== Averaged Trajectory Analysis Summary ===")
    print(f"  Number of windows: {len(time_windows_to_analyze)}")
    print(f"  Window duration: {window_duration}s")  
    print(f"  Total path length: {total_distance:.2f}")
    print(f"  Net displacement: {net_displacement:.2f}")
    print(f"  Tortuosity: {tortuosity:.2f}")
    print(f"  Behavior range: {averaged_behavior.min():.2f} - {averaged_behavior.max():.2f}")
    
    return {
        'averaged_pc1': averaged_pc1,
        'averaged_pc2': averaged_pc2,
        'averaged_behavior': averaged_behavior,
        'averaged_times': averaged_times,
        'individual_windows': {
            'pc1': all_window_pc1,
            'pc2': all_window_pc2,
            'behavior': all_window_behavior,
            'times': all_window_times
        },
        'window_duration': window_duration,
        'num_windows': len(time_windows_to_analyze),
        'behavior_type': behavior,
        'trajectory_stats': {
            'total_distance': total_distance,
            'net_displacement': net_displacement,
            'tortuosity': tortuosity
        }
    }