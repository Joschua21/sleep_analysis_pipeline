def batch_analyze_delta_spectrograms(subject_ids, output_base_dir, exp_defs=['postactive', 'spontaneous'],
                                   window_size=71, nperseg=200, noverlap=150, freq_range=(0, 50),
                                   bin_size=0.005, save_plots=True, show_plots=False):
    """
    Batch process multiple sessions for one or more subjects to generate delta power spectrograms.
    Each session is processed completely independently with its own bombcell classification.
    
    Parameters:
    -----------
    subject_ids : str or list
        Subject identifier(s) (e.g., 'AV043' or ['AV043', 'AV049', 'AV050'])
    output_base_dir : str
        Base directory where plots will be saved
    exp_defs : list
        List of experiment definitions to include
    window_size : int
        Window size for smoothing (default: 71)
    nperseg : int
        Length of each segment for spectrogram (default: 200)
    noverlap : int
        Number of points to overlap between segments (default: 150)
    freq_range : tuple
        Frequency range to analyze in Hz (default: (0, 50))
    bin_size : float
        Spike binning size in seconds (default: 0.005 for 5ms)
    save_plots : bool
        Whether to save plots (default: True)
    show_plots : bool
        Whether to display plots (default: False)
        
    Returns:
    --------
    dict
        Summary of processed sessions for all subjects
    """
    from pinkrigs_tools.dataset.query import queryCSV, load_data
    from pinkrigs_tools.utils.spk_utils import bombcell_sort_units
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import signal
    from datetime import datetime
    import gc  # For garbage collection
    from tqdm import tqdm  # Add this import
    from neuropixel_utils import process_spike_data, bin_spikes
    
    # Handle single subject or list of subjects
    if isinstance(subject_ids, str):
        subject_ids = [subject_ids]
    
    print(f"Starting batch analysis for {len(subject_ids)} subject(s): {subject_ids}")
    print(f"Looking for experiment types: {exp_defs}")
    
    # Create main output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = os.path.join(output_base_dir, f"batch_delta_spectrograms")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Initialize overall tracking
    all_processed_sessions = []
    all_failed_sessions = []
    subject_summaries = []
    
    # Main progress bar for subjects
    subject_pbar = tqdm(subject_ids, desc="Processing subjects", unit="subject")
    
    for subject_idx, subject_id in enumerate(subject_pbar):
        subject_pbar.set_description(f"Processing subject {subject_id}")
        
        print(f"\n{'='*80}")
        print(f"PROCESSING SUBJECT {subject_idx + 1}/{len(subject_ids)}: {subject_id}")
        print(f"{'='*80}")
        
        # Create subject-specific output directory
        subject_output_dir = os.path.join(main_output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        try:
            # Query sessions using the simplified approach
            print(f"Querying sessions for {subject_id}...")
            
            try:
                exp_data = queryCSV(
                    subject=subject_id,
                    expDate='all',
                    expDef=exp_defs,  # Pass the list directly
                    checkSpikes='1'   # Only sessions with spike data
                )
                
                if exp_data.empty:
                    print(f"  No sessions found for {subject_id}!")
                    subject_summaries.append({
                        'subject_id': subject_id,
                        'total_found': 0,
                        'processed': 0,
                        'failed': 0,
                        'status': 'no_sessions'
                    })
                    continue
                
                print(f"  Found {len(exp_data)} sessions matching criteria")
                # Add exp_type column based on expDef
                exp_data['exp_type'] = exp_data['expDef']
                
            except Exception as e:
                print(f"  Error querying sessions: {e}")
                subject_summaries.append({
                    'subject_id': subject_id,
                    'total_found': 0,
                    'processed': 0,
                    'failed': 0,
                    'status': f'query_error: {str(e)}'
                })
                continue
            
            # Filter for valid sessions with both probes
            valid_sessions = []
            print(f"  Validating session folders for {subject_id}...")
            
            for _, row in tqdm(exp_data.iterrows(), 
                             total=len(exp_data), 
                             desc=f"Checking {subject_id} session validity", 
                             leave=False):
                # Check if required folder paths exist
                exp_folder_exists = pd.notna(row.get('expFolder')) and os.path.exists(row.get('expFolder', ''))
                ephys_probe0_exists = pd.notna(row.get('ephysPathProbe0')) and os.path.exists(row.get('ephysPathProbe0', ''))
                ephys_probe1_exists = pd.notna(row.get('ephysPathProbe1')) and os.path.exists(row.get('ephysPathProbe1', ''))
                
                # Require BOTH probes for this analysis
                if exp_folder_exists and ephys_probe0_exists and ephys_probe1_exists:
                    session_info = {
                        'subject': row['subject'],
                        'expDate': row['expDate'],
                        'expNum': row.get('expNum', 'N/A'),
                        'exp_type': row['expDef'],  # Use the actual expDef value
                        'expFolder': row.get('expFolder'),
                        'ephysPathProbe0': row.get('ephysPathProbe0'),
                        'ephysPathProbe1': row.get('ephysPathProbe1')
                    }
                    valid_sessions.append(session_info)
            
            print(f"  Valid sessions with both probes for {subject_id}: {len(valid_sessions)}")
            
            if not valid_sessions:
                print(f"  No sessions found with both probe0 and probe1 for {subject_id}!")
                subject_summaries.append({
                    'subject_id': subject_id,
                    'total_found': len(exp_data),
                    'processed': 0,
                    'failed': 0,
                    'status': 'no_valid_sessions'
                })
                continue
            
            # Process each session for this subject
            subject_processed = []
            subject_failed = []
            
            # Create session progress bar for this subject
            session_pbar = tqdm(valid_sessions, 
                              desc=f"Processing {subject_id} sessions", 
                              unit="session",
                              leave=False)
            
            for session in session_pbar:
                session_id = f"{session['expDate']}_{session['expNum']}_{session['exp_type']}"
                
                # Update progress bar description with current session
                session_pbar.set_description(f"Processing {subject_id}: {session_id}")
                
                try:
                    print(f"\n    === PROCESSING SESSION INDEPENDENTLY ===")
                    print(f"    Session: {session_id}")
                    
                    # Prepare experiment kwargs FOR THIS SPECIFIC SESSION
                    exp_kwargs = {
                        'subject': [session['subject']],
                        'expDate': session['expDate'],
                        'expNum': session['expNum'],
                    }
                    
                    # STEP 1: Load process_spike_data FOR THIS SESSION
                    print(f"    Step 1: Loading spike data for {session_id}...")
                    freq_results = process_spike_data(
                        exp_kwargs=exp_kwargs,
                        bin_size=bin_size,
                        show_plots=False,
                    )
                    
                    # STEP 2: Load spike recordings FOR THIS SESSION and apply bombcell FOR THIS SESSION
                    print(f"    Step 2: Applying bombcell classification for {session_id}...")
                    ephys_dict = {'spikes':'all','clusters':'all'}
                    data_name_dict = {'probe0':ephys_dict,'probe1':ephys_dict} 
                    spike_recordings = load_data(data_name_dict=data_name_dict, **exp_kwargs)
                    
                    # Apply bombcell classification independently for each probe in THIS SESSION
                    total_good_clusters = 0
                    combined_counts = []
                    time_bins = None
                    
                    for probe in ['probe0', 'probe1']:
                        if probe in freq_results and probe in spike_recordings and len(spike_recordings[probe]) > 0:
                            print(f"      Processing {probe} for {session_id}...")
                            
                            # Get cluster data for THIS SESSION and THIS PROBE
                            cluster_data = spike_recordings[probe][0]['clusters']
                            
                            # Apply bombcell classification for THIS SESSION and THIS PROBE
                            cluster_quality = bombcell_sort_units(cluster_data)
                            
                            # Store session-specific results
                            freq_results[probe]['cluster_quality'] = cluster_quality
                            freq_results[probe]['good_mua_cluster_mask'] = np.array([q in ['good', 'mua'] for q in cluster_quality])
                            
                            print(f"        {probe}: Found {len(cluster_quality)} total clusters")
                            
                            # Print bombcell summary for this session/probe
                            unique_qualities, counts = np.unique(cluster_quality, return_counts=True)
                            for quality, count in zip(unique_qualities, counts):
                                print(f"        {probe} - {quality}: {count} clusters")
                            
                            # Filter to good+mua clusters for THIS SESSION
                            good_mua_mask = freq_results[probe]['good_mua_cluster_mask']
                            
                            # Handle dimension mismatch if needed (session-specific)
                            counts_shape = freq_results[probe]['counts'].shape[0]
                            if len(good_mua_mask) != counts_shape:
                                print(f"        {probe}: Fixing dimension mismatch ({len(good_mua_mask)} -> {counts_shape})")
                                good_mua_mask = good_mua_mask[:counts_shape]
                                freq_results[probe]['good_mua_cluster_mask'] = good_mua_mask
                            
                            # Get filtered counts for this session/probe
                            filtered_counts = freq_results[probe]['counts'][good_mua_mask]
                            good_clusters = np.sum(good_mua_mask)
                            total_good_clusters += good_clusters
                            
                            print(f"        {probe}: Using {good_clusters} good+mua clusters")
                            
                            combined_counts.append(filtered_counts)
                            if time_bins is None:
                                time_bins = freq_results[probe]['time_bins']
                    
                    # Clear spike recordings to save memory
                    del spike_recordings
                    gc.collect()
                    
                    if not combined_counts:
                        raise ValueError(f"No valid spike data found for {session_id}")
                    
                    # STEP 3: Combine data from both probes for THIS SESSION
                    print(f"    Step 3: Combining probe data for {session_id}...")
                    
                    # Ensure same time length for this session
                    if len(combined_counts) > 1:
                        min_length = min(c.shape[1] for c in combined_counts)
                        combined_counts = [c[:, :min_length] for c in combined_counts]
                        time_bins = time_bins[:min_length]
                    
                    # Stack vertically for this session
                    merged_counts = np.vstack(combined_counts)
                    
                    print(f"    Combined for {session_id}: {merged_counts.shape[0]} neurons, {merged_counts.shape[1]} time bins")
                    print(f"    Time range: {time_bins[0]:.1f}s to {time_bins[-1]:.1f}s")
                    
                    # STEP 4: Power frequency analysis for THIS SESSION
                    print(f"    Step 4: Computing power spectrum for {session_id}...")
                    
                    # Calculate mean activity across all neurons for this session
                    mean_activity = np.mean(merged_counts, axis=0)
                    
                    # Compute spectrogram for this session
                    sampling_rate = 1 / bin_size
                    frequencies, times, Sxx = signal.spectrogram(
                        mean_activity,
                        fs=sampling_rate,
                        window='hamming',
                        nperseg=nperseg,
                        noverlap=noverlap,
                        scaling='density',
                        detrend='constant'
                    )
                    
                    # Convert to dB scale and filter frequency range
                    Sxx_db = 10 * np.log10(Sxx + 1e-10)
                    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
                    frequencies_filtered = frequencies[freq_mask]
                    Sxx_db_filtered = Sxx_db[freq_mask, :]
                    
                    # Calculate delta band power (1-4 Hz)
                    delta_mask = (frequencies >= 1) & (frequencies <= 4)
                    delta_power = np.mean(Sxx_db[delta_mask, :], axis=0)
                    
                    # Apply smoothing to delta power (simple moving average)
                    if window_size > 1:
                        # Pad with edge values for smoothing
                        pad_size = window_size // 2
                        padded_delta = np.pad(delta_power, pad_size, mode='edge')
                        kernel = np.ones(window_size) / window_size
                        smoothed_delta = np.convolve(padded_delta, kernel, mode='valid')
                    else:
                        smoothed_delta = delta_power
                    
                    # Create time array for delta power
                    spec_times = np.linspace(time_bins[0], time_bins[-1], len(delta_power))
                    
                    # STEP 5: Create plots for THIS SESSION
                    print(f"    Step 5: Creating plots for {session_id}...")
                    
                    # Create the combined plot
                    fig = plt.figure(figsize=(14, 10))
                    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], figure=fig)
                    
                    # 1. Spectrogram
                    ax1 = fig.add_subplot(gs[0])
                    
                    # Calculate color limits
                    power_5th = np.percentile(Sxx_db_filtered, 5)
                    power_95th = np.percentile(Sxx_db_filtered, 95)
                    vmin = power_5th - 5
                    vmax = power_95th + 5
                    
                    extent = [time_bins[0], time_bins[-1], 
                             frequencies_filtered[0], frequencies_filtered[-1]]
                    
                    im1 = ax1.matshow(Sxx_db_filtered, aspect='auto', origin='lower',
                                    extent=extent, cmap='viridis', vmin=vmin, vmax=vmax)
                    
                    # Add frequency band lines
                    ax1.axhline(y=1, color='white', linestyle='-', alpha=0.7, linewidth=1.5, label='Delta (1-4 Hz)')
                    ax1.axhline(y=4, color='white', linestyle='-', alpha=0.7, linewidth=1.5)
                    
                    ax1.set_ylabel('Frequency (Hz)')
                    ax1.set_title(f'{subject_id} - {session["expDate"]} #{session["expNum"]} ({session["exp_type"]})\n'
                                 f'Population Power Spectrum ({total_good_clusters} neurons)')
                    ax1.legend(loc='upper right')
                                        
                    # 2. Raw delta power
                    ax2 = fig.add_subplot(gs[1], sharex=ax1)
                    ax2.plot(spec_times, delta_power, 'b-', linewidth=1, alpha=0.7, label='Raw Delta Power')
                    ax2.set_ylabel('Delta Power (dB)')
                    ax2.set_title('Delta Band Power (1-4 Hz)')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    # 3. Smoothed delta power
                    ax3 = fig.add_subplot(gs[2], sharex=ax1)
                    ax3.plot(spec_times, smoothed_delta, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
                    ax3.set_xlabel('Time (s)')
                    ax3.set_ylabel('Smoothed Delta (dB)')
                    ax3.set_title('Smoothed Delta Power')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()
                    
                    # Set common x-axis limits
                    ax3.set_xlim(time_bins[0], time_bins[-1])
                    
                    plt.tight_layout()
                    
                    # Save the plot in subject-specific directory
                    if save_plots:
                        # Clean up exp_type for filename (remove special characters)
                        clean_exp_type = session['exp_type'].replace('_', '').replace('-', '')
                        plot_filename = f"{subject_id}_{session['expDate']}_{session['expNum']}_{clean_exp_type}_delta_spectrogram.png"
                        plot_path = os.path.join(subject_output_dir, plot_filename)
                        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
                        print(f"    Saved: {plot_filename}")
                    
                    if show_plots:
                        plt.show()
                    else:
                        plt.close(fig)
                    
                    # Clean up memory after THIS SESSION
                    del freq_results, merged_counts, mean_activity, Sxx, Sxx_db, delta_power, smoothed_delta
                    gc.collect()
                    
                    # Record successful processing for THIS SESSION
                    session_summary = {
                        'session_id': session_id,
                        'subject': session['subject'],
                        'expDate': session['expDate'],
                        'expNum': session['expNum'],
                        'exp_type': session['exp_type'],
                        'n_neurons': total_good_clusters,
                        'duration_s': time_bins[-1] - time_bins[0],
                        'plot_saved': save_plots,
                        'status': 'success'
                    }
                    subject_processed.append(session_summary)
                    all_processed_sessions.append(session_summary)
                    
                    print(f"    ✓ Successfully processed {session_id}")
                    
                    # Update session progress bar postfix with success info
                    session_pbar.set_postfix({
                        'Success': len(subject_processed),
                        'Failed': len(subject_failed),
                        'Neurons': total_good_clusters
                    })
                    
                except Exception as e:
                    print(f"    ✗ Failed to process {session_id}: {e}")
                    
                    failed_session = {
                        'session_id': session_id,
                        'subject': session['subject'],
                        'expDate': session['expDate'],
                        'expNum': session['expNum'],
                        'exp_type': session['exp_type'],
                        'error': str(e),
                        'status': 'failed'
                    }
                    subject_failed.append(failed_session)
                    all_failed_sessions.append(failed_session)
                    
                    # Update session progress bar postfix with failure info
                    session_pbar.set_postfix({
                        'Success': len(subject_processed),
                        'Failed': len(subject_failed),
                        'Error': str(e)[:20] + '...' if len(str(e)) > 20 else str(e)
                    })
                    
                    # Clean up any partial data
                    gc.collect()
                    continue
            
            # Close the session progress bar
            session_pbar.close()
            
            # Save subject-specific summary
            if subject_processed or subject_failed:
                subject_sessions_summary = subject_processed + subject_failed
                subject_summary_df = pd.DataFrame(subject_sessions_summary)
                subject_summary_path = os.path.join(subject_output_dir, f"{subject_id}_processing_summary.csv")
                subject_summary_df.to_csv(subject_summary_path, index=False)
            
            # Record subject summary
            subject_summary = {
                'subject_id': subject_id,
                'total_found': len(valid_sessions),
                'processed': len(subject_processed),
                'failed': len(subject_failed),
                'status': 'completed'
            }
            subject_summaries.append(subject_summary)
            
            print(f"  Subject {subject_id} complete: {len(subject_processed)} processed, {len(subject_failed)} failed")
            
        except Exception as e:
            print(f"  ERROR processing subject {subject_id}: {e}")
            subject_summaries.append({
                'subject_id': subject_id,
                'total_found': 0,
                'processed': 0,
                'failed': 0,
                'status': f'error: {str(e)}'
            })
            continue
        
        # Update main subject progress bar
        total_processed = sum(s['processed'] for s in subject_summaries)
        total_failed = sum(s['failed'] for s in subject_summaries)
        subject_pbar.set_postfix({
            'Total Success': total_processed,
            'Total Failed': total_failed
        })
    
    # Close the main subject progress bar
    subject_pbar.close()
    
    # Create overall summary report
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE - ALL SUBJECTS")
    print(f"{'='*80}")
    print(f"Subjects processed: {len(subject_ids)}")
    print(f"Output directory: {main_output_dir}")
    print(f"Total sessions processed: {len(all_processed_sessions)}")
    print(f"Total sessions failed: {len(all_failed_sessions)}")
    
    # Subject-by-subject summary
    print(f"\nSUBJECT SUMMARY:")
    for summary in subject_summaries:
        status_symbol = "✓" if summary['status'] == 'completed' else "✗"
        print(f"  {status_symbol} {summary['subject_id']}: "
              f"{summary['processed']} processed, {summary['failed']} failed "
              f"(out of {summary['total_found']} valid sessions)")
    
    if all_processed_sessions:
        print(f"\nAll successfully processed sessions:")
        for session in all_processed_sessions:
            print(f"  ✓ {session['subject']} - {session['session_id']} - "
                  f"{session['n_neurons']} neurons, {session['duration_s']:.1f}s")
    
    if all_failed_sessions:
        print(f"\nAll failed sessions:")
        for session in all_failed_sessions:
            print(f"  ✗ {session['subject']} - {session['session_id']} - {session['error']}")
    
    # Save overall summary
    overall_summary_data = {
        'processed_sessions': all_processed_sessions,
        'failed_sessions': all_failed_sessions,
        'subject_summaries': subject_summaries,
        'total_subjects': len(subject_ids),
        'total_sessions_found': sum(s['total_found'] for s in subject_summaries),
        'parameters': {
            'subject_ids': subject_ids,
            'exp_defs': exp_defs,
            'window_size': window_size,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'freq_range': freq_range,
            'bin_size': bin_size
        }
    }
    
    # Save overall summary as CSV
    if all_processed_sessions or all_failed_sessions:
        all_sessions_summary = all_processed_sessions + all_failed_sessions
        overall_summary_df = pd.DataFrame(all_sessions_summary)
        overall_summary_path = os.path.join(main_output_dir, "overall_processing_summary.csv")
        overall_summary_df.to_csv(overall_summary_path, index=False)
        
        # Also save subject summaries
        subject_summary_df = pd.DataFrame(subject_summaries)
        subject_summary_path = os.path.join(main_output_dir, "subject_summaries.csv")
        subject_summary_df.to_csv(subject_summary_path, index=False)
        
        print(f"\nOverall summary saved: {overall_summary_path}")
        print(f"Subject summaries saved: {subject_summary_path}")
    
    return overall_summary_data