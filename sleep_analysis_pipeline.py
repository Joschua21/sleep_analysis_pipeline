import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless operation
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import linregress
import logging
import cv2
import time
from datetime import datetime

try:
    from plotting_utils import *
    from debugging_functions import *
except ImportError:
    print("ImportError: could not import plotting_utils or debugging_functions. Ensure they are in the same directory as this script.")

frame_rate = 60
likelihood_threshold = 0.95

def setup_directories(input_dir):
    """Set up the input and output directories for the analysis pipeline."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "sleep_pipeline_output")
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def find_files(input_dir):
    """Find the CSV file and video file in the input directory."""
    # Find the only CSV file in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    if len(csv_files) > 1:
        print(f"Warning: Multiple CSV files found in {input_dir}. Using {csv_files[0]}")
    csv_file = os.path.join(input_dir, csv_files[0])
    
    # Find the only video file in the directory
    video_extensions = ['.mp4']
    video_files = [f for f in os.listdir(input_dir) 
                  if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"Warning: No .mp4 files found in {input_dir}")
        video_file = None
    elif len(video_files) > 1:
        print(f"Warning: Multiple .mp4 files found in {input_dir}. Using {video_files[0]}")
        video_file = os.path.join(input_dir, video_files[0])
    else:
        video_file = os.path.join(input_dir, video_files[0])
    
    return csv_file, video_file

def load_dlc_data(file_path):
    """Load DLC CSV file and return DataFrame."""
    try:
        df_dlc = pd.read_csv(file_path, header=[1, 2], index_col=0)
        return df_dlc
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def get_default_bodyparts():
    """Return the default list of bodyparts for analysis."""
    DEFAULT_BODYPARTS = [
        'neck', 'mid_back', 'mouse_center', 'mid_backend', 'left_midside',
        'right_midside', 'right_hip', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip',
        'left_shoulder', 'right_shoulder', 'head_midpoint'
    ]
    return DEFAULT_BODYPARTS

def select_available_bodyparts(df_dlc, default_bodyparts=None):
    """
    Filter the default bodyparts list to only include those available in the data.
    
    Args:
        df_dlc: DataFrame containing DLC tracking data
        default_bodyparts: List of default bodyparts to check (if None, uses get_default_bodyparts())
        
    Returns:
        list: Final list of bodyparts that exist in the data
    """
    if default_bodyparts is None:
        default_bodyparts = get_default_bodyparts()
    
    all_bodyparts = df_dlc.columns.get_level_values(0).unique().tolist()
    
    final_bodyparts_list = []
    for bp in default_bodyparts:
        if bp in all_bodyparts:
            final_bodyparts_list.append(bp)
    
    if not final_bodyparts_list:
        raise ValueError("None of the default bodyparts were found in the data. Check your DLC output format.")
        
    return final_bodyparts_list

def interpolate_gaps_conditionally(series: pd.Series, max_gap_length: int) -> pd.Series:
    """
    Interpolates NaN values in a Series using linear interpolation, but only
    if the consecutive NaN gap is shorter than or equal to max_gap_length.
    Returns:
        pd.Series: Series with short NaN gaps interpolated.
    """
    s_out = series.copy()
    is_na = s_out.isna()

    if not is_na.any(): 
        return s_out

    # Identify groups of consecutive NaNs
    na_group_ids = is_na.ne(is_na.shift()).cumsum()
    for group_id, na_block_series in s_out[is_na].groupby(na_group_ids[is_na]):
        if not na_block_series.empty and len(na_block_series) <= max_gap_length:
            temp_interpolated_series = s_out.interpolate(method='linear', limit_direction='both')
            s_out.loc[na_block_series.index] = temp_interpolated_series.loc[na_block_series.index]
            
    return s_out

def process_bodypart_coordinates(df_dlc, final_bodyparts_list, likelihood_threshold=0.95):
    """
    Calculate displacement per bodypart with likelihood check and interpolation.
    Returns:
        Updated DataFrame with median coordinates and displacement metrics
    """
    MAX_CONSECUTIVE_NAN_INTERPOLATE = 60  # Max frames that can be NaN and still interpolated
    
    # Prepare columns for median values
    df_dlc[('analysis', 'median_x')] = np.nan
    df_dlc[('analysis', 'median_y')] = np.nan

    # Store filtered and interpolated coordinates temporarily to calculate medians
    filtered_x_coords = pd.DataFrame(index=df_dlc.index)
    filtered_y_coords = pd.DataFrame(index=df_dlc.index)
    
    print(f"Applying likelihood filter and interpolation to selected bodyparts...")
    for bp in final_bodyparts_list:
        if (bp, 'x') not in df_dlc.columns or \
           (bp, 'y') not in df_dlc.columns or \
           (bp, 'likelihood') not in df_dlc.columns:
            print(f"Warning: Data for bodypart {bp} (x, y, or likelihood) not found. Skipping.")
            filtered_x_coords[bp] = np.nan  # Fill with NaNs to maintain DataFrame structure
            filtered_y_coords[bp] = np.nan
            continue

        x = df_dlc[(bp, 'x')].copy()
        y = df_dlc[(bp, 'y')].copy()
        likelihood = df_dlc[(bp, 'likelihood')]
        
        # Apply likelihood threshold: set x, y to NaN where likelihood is low
        mask = likelihood < likelihood_threshold
        x[mask] = np.nan
        y[mask] = np.nan
        
        # Apply conditional interpolation
        x_processed = interpolate_gaps_conditionally(x, MAX_CONSECUTIVE_NAN_INTERPOLATE)
        y_processed = interpolate_gaps_conditionally(y, MAX_CONSECUTIVE_NAN_INTERPOLATE)
        
        filtered_x_coords[bp] = x_processed
        filtered_y_coords[bp] = y_processed

    # Calculate median across the filtered and interpolated coordinates for each frame
    df_dlc[('analysis', 'median_x')] = filtered_x_coords.median(axis=1, skipna=True)
    df_dlc[('analysis', 'median_y')] = filtered_y_coords.median(axis=1, skipna=True)

    # Calculate displacement of the median point
    median_x_coords = df_dlc[('analysis', 'median_x')]
    median_y_coords = df_dlc[('analysis', 'median_y')]

    # Calculate difference between consecutive frames for the median point
    delta_median_x = median_x_coords.diff()
    delta_median_y = median_y_coords.diff()

    # Calculate Euclidean distance for the median point's displacement
    displacement_median_pixels = np.sqrt(delta_median_x**2 + delta_median_y**2)

    # Store the displacement of the median point
    df_dlc[('analysis', 'displacement_median_pixels')] = displacement_median_pixels

    print("Displacement calculations complete.")
    
    return df_dlc, filtered_x_coords, filtered_y_coords

def calculate_speed(df_dlc, frame_rate, output_dir, file_name, final_bodyparts_list, save_plots=True):
    """
    Calculate animal speed from displacement data and save results.
    Returns:
        Updated DataFrame with speed calculations
    """
    smoothing_window_seconds = 15/60  
    time_per_frame = 1.0 / frame_rate  # seconds per frame
    speed_pixels_per_second = df_dlc[('analysis', 'displacement_median_pixels')] * frame_rate

    if ('analysis', 'average_displacement_pixels') in df_dlc.columns:
        df_dlc = df_dlc.drop(columns=[('analysis', 'average_displacement_pixels')])

    df_dlc[('analysis', 'speed_pixels_per_second')] = speed_pixels_per_second

    df_dlc = df_dlc.sort_index(axis=1)

    print("\nSpeed calculation complete.")
    # --- Save the DataFrame and generate plots ---
    base_name_without_ext = os.path.splitext(file_name)[0]
    base_output_name = f"{base_name_without_ext}_speed_analysis"
    output_filename_csv = os.path.join(output_dir, base_output_name + '.csv')

    # Save the DataFrame
    try:
        df_dlc.to_csv(output_filename_csv)
        print(f"DataFrame with speed data saved to: {output_filename_csv}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

    if save_plots:
        try:
            # Create empty DataFrame since we're only plotting median displacements
            empty_df_displacements = pd.DataFrame(index=df_dlc.index)
            
            plot_speed(
                df_dlc=df_dlc,
                df_displacements=empty_df_displacements,
                final_bodyparts_list=final_bodyparts_list,
                frame_rate=frame_rate,
                output_dir=output_dir,
                base_filename=base_output_name,
                plot_individual=False,  # Only plotting median-based speed
                save_plot=True,
                smoothing_window_seconds=smoothing_window_seconds
            )
            print(f"Speed plots saved to: {output_dir}")
        except Exception as e:
            print(f"Error generating speed plots: {e}")
    
    return df_dlc

def calculate_body_posture_metric(df_dlc, filtered_x_coords, filtered_y_coords, final_bodyparts_list, output_dir, file_name, save_plots=True):
    """
    Calculate body posture metric by measuring the average distance of each bodypart to the median point.
    Returns:
        Updated DataFrame with body posture metrics
    """
    print("Calculating body posture metric (average distance to median)...")
    
    # Check if required data is available
    if ('analysis', 'median_x') not in df_dlc.columns or ('analysis', 'median_y') not in df_dlc.columns:
        print("Error: Median coordinates not found in DataFrame. Run process_bodypart_coordinates first.")
        return df_dlc
    
    # --- Step 1: Calculate distance of each processed bodypart to the median point ---
    median_x_series = df_dlc[('analysis', 'median_x')]
    median_y_series = df_dlc[('analysis', 'median_y')]
    
    df_distances_to_median = pd.DataFrame(index=df_dlc.index)

    # Calculate distance for each bodypart
    for bp in final_bodyparts_list:
        if bp in filtered_x_coords.columns and bp in filtered_y_coords.columns:
            x_bp = filtered_x_coords[bp]
            y_bp = filtered_y_coords[bp]
            
            # Distance calculation: sqrt((x_bp - median_x)^2 + (y_bp - median_y)^2)
            dist_bp = np.sqrt((x_bp - median_x_series)**2 + (y_bp - median_y_series)**2)
            df_distances_to_median[bp] = dist_bp

    # --- Step 2: Calculate sum of distances and count of valid bodyparts per frame ---
    sum_of_distances = df_distances_to_median.sum(axis=1, skipna=True)
    count_of_valid_bodyparts = df_distances_to_median.notna().sum(axis=1)

    # --- Step 3: Calculate average distance to median (handle division by zero) ---
    metric_col_name = ('analysis', 'avg_dist_to_median')
    df_dlc[metric_col_name] = np.nan  # Initialize column
    
    # Calculate average only where count_of_valid_bodyparts > 0
    df_dlc[metric_col_name] = np.where(
        count_of_valid_bodyparts > 0, 
        sum_of_distances / count_of_valid_bodyparts, 
        np.nan
    )
    
    print(f"Body posture metric calculated.")
    
    # --- Step 4: Plot the new metric ---
    if save_plots:
        try:
            # Create base output name
            base_name_without_ext = os.path.splitext(file_name)[0]
            base_output_name = f"{base_name_without_ext}_posture_analysis"
            
            # Fixed parameter
            smoothing_window_seconds = 15/60
            
            plot_body_posture_metric(
                df_dlc=df_dlc,
                metric_column_tuple=metric_col_name,
                frame_rate=frame_rate,
                output_dir_path=output_dir,
                base_output_name=base_output_name,
                save_plot=save_plots,
                display_plot=False,  # No display in pipeline mode
                smoothing_window_seconds_metric=smoothing_window_seconds,
                plot_with_speed=True,
                smoothing_window_seconds_speed=smoothing_window_seconds
            )
            print(f"Body posture metric plots saved to: {output_dir}")
        except Exception as e:
            print(f"Error generating body posture plots: {e}")
    
    return df_dlc

def calculate_body_movement_derivative(df_dlc, frame_rate, output_dir, file_name, save_plots=True):
    """
    Calculate and plot the absolute derivative of the body posture metric.        
    Returns:
        Updated DataFrame with body movement derivative metrics
    """
    metric_col_tuple = ('analysis', 'avg_dist_to_median')
    
    # Check if required data is available
    if metric_col_tuple not in df_dlc.columns:
        print("Error: Body posture metric not found in DataFrame. Run calculate_body_posture_metric first.")
        return df_dlc
    
    # Create time axis in seconds
    time_axis = df_dlc.index / frame_rate
    
    # --- Extract Data ---
    body_posture_metric = df_dlc[metric_col_tuple]

    # --- Calculate Derivative ---
    # 1. Calculate frame-to-frame difference of the metric
    metric_derivative_frames = body_posture_metric.diff()

    # 2. Convert from metric_units/frame to metric_units/second
    # The metric 'avg_dist_to_median' is in pixels.
    # So the derivative will be in pixels/second.
    metric_derivative_per_s = metric_derivative_frames * frame_rate

    # 3. Calculate the absolute value of the derivative (rate of change)
    absolute_metric_derivative_per_s = np.abs(metric_derivative_per_s)

    # Add columns for both raw and absolute derivatives
    df_dlc[('analysis', 'posture_metric_derivative')] = metric_derivative_per_s
    df_dlc[('analysis', 'posture_metric_abs_derivative')] = absolute_metric_derivative_per_s
    
    # --- Plotting the Absolute Derivative Only ---
    if save_plots:
        try:
            plt.figure(figsize=(15, 5))
            plt.plot(time_axis, absolute_metric_derivative_per_s, label='Absolute Rate of Change', 
                     color='purple', linewidth=1.0)
            plt.title('Absolute Rate of Change of Body Posture Metric (Avg. Dist. to Median)')
            plt.xlabel("Time (s)")
            plt.ylabel("Absolute Rate of Change (pixels/second)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Create base output name
            base_name_without_ext = os.path.splitext(file_name)[0]
            base_output_name = f"{base_name_without_ext}_posture_analysis"
            
            plot_filename_body_metric_derivative = os.path.join(output_dir, 
                                                               f"{base_output_name}_body_posture_metric_abs_derivative.png")
            plt.savefig(plot_filename_body_metric_derivative, dpi=150)
            plt.close()  # Close the plot to free memory
            
            print(f"Saved absolute body posture metric derivative plot: {plot_filename_body_metric_derivative}")
        except Exception as e:
            print(f"Error generating body posture derivative plot: {e}")

    print("Body movement derivative calculation complete.")
    
    return df_dlc

def prepare_body_axis_coordinates(df_dlc, filtered_x_coords, filtered_y_coords, final_bodyparts_list):
    """
    Prepare body part coordinates for PCA-based body axis calculation.
    Creates datasets of core axial parts and midpoints of paired parts for determining 
    the principal axis of the animal's body.
    Returns:
        dict: Dictionary containing coordinate sets for body axis calculation
    """    
    # --- Define Body Part Sets for Axis Calculation ---
    core_axial_parts = ['neck', 'mid_back', 'mouse_center', 'mid_backend']
    paired_parts_for_midpoints = {
        'midsides_midpoint': ('left_midside', 'right_midside'),
        'ears_midpoint': ('left_ear', 'right_ear'),
        'eartips_midpoint': ('left_ear_tip', 'right_ear_tip'),
        'shoulders_midpoint': ('left_shoulder', 'right_shoulder')
    }

    # Start with the core axial parts
    axial_plus_midpoints_x_list = []
    axial_plus_midpoints_y_list = []

    for bp in core_axial_parts:
        if bp in filtered_x_coords.columns and bp in filtered_y_coords.columns:
            axial_plus_midpoints_x_list.append(filtered_x_coords[bp])
            axial_plus_midpoints_y_list.append(filtered_y_coords[bp])
        else:
            print(f"  Warning: Core axial part '{bp}' not found. Skipping.")

    # Add midpoints of paired parts
    for midpoint_name, (bp_left, bp_right) in paired_parts_for_midpoints.items():
        if bp_left in filtered_x_coords.columns and bp_right in filtered_x_coords.columns and \
           bp_left in filtered_y_coords.columns and bp_right in filtered_y_coords.columns:
            
            mid_x = (filtered_x_coords[bp_left] + filtered_x_coords[bp_right]) / 2
            mid_y = (filtered_y_coords[bp_left] + filtered_y_coords[bp_right]) / 2
            
            # Name the series for easier concatenation
            mid_x.name = midpoint_name 
            mid_y.name = midpoint_name
            
            axial_plus_midpoints_x_list.append(mid_x)
            axial_plus_midpoints_y_list.append(mid_y)
        else:
            missing_for_midpoint = []
            if bp_left not in filtered_x_coords.columns or bp_left not in filtered_y_coords.columns:
                missing_for_midpoint.append(bp_left)
            if bp_right not in filtered_x_coords.columns or bp_right not in filtered_y_coords.columns:
                missing_for_midpoint.append(bp_right)
            print(f"  Warning: One or both parts for '{midpoint_name}' ({', '.join(missing_for_midpoint)}) not found. Skipping.")

    # Create dataframes from the lists
    if not axial_plus_midpoints_x_list:
        print("Error: No bodyparts could be prepared for PCA axis calculation. Cannot proceed.")
        axial_plus_midpoints_x_df = pd.DataFrame(index=df_dlc.index)  # Empty DF
        axial_plus_midpoints_y_df = pd.DataFrame(index=df_dlc.index)  # Empty DF
    else:
        axial_plus_midpoints_x_df = pd.concat(axial_plus_midpoints_x_list, axis=1)
        axial_plus_midpoints_y_df = pd.concat(axial_plus_midpoints_y_list, axis=1)

    # Create empty dataframes for the axial_only approach to maintain compatibility
    axial_points_x_df = pd.DataFrame(index=df_dlc.index)
    axial_points_y_df = pd.DataFrame(index=df_dlc.index)

    # Store the coordinate sets in a dictionary for easier access
    bodypart_coordinate_sets = {
        "axial_only": {"x": axial_points_x_df, "y": axial_points_y_df},
        "axial_plus_midpoints": {"x": axial_plus_midpoints_x_df, "y": axial_plus_midpoints_y_df}
    }

    print("Body axis coordinate preparation complete.")
    
    return bodypart_coordinate_sets

def calculate_body_axis_pca(df_dlc, bodypart_coordinate_sets):
    """
    Calculate principal component analysis (PCA) on bodypart coordinates to determine body axis.
    For each frame, calculates a PCA-based primary axis using available bodypart coordinates.
    This represents the main orientation axis of the animal's body.  
    Returns:
        tuple: (df_dlc with PCA columns added, DataFrame with raw PCA results)
    """   
    # Check if valid coordinate sets are available
    if bodypart_coordinate_sets["axial_plus_midpoints"]["x"].empty:
        print("Error: No valid coordinate sets available for PCA calculation.")
        return df_dlc, pd.DataFrame(index=df_dlc.index)
        
    results_midpoints_pca = []
    
    # Define a minimum number of points required to fit a line
    MIN_POINTS_FOR_FIT = 2
    
    # Iterate over each frame index present in the original df_dlc
    for frame_idx in df_dlc.index:
        # Initialize parameters to NaN
        pca_dx, pca_dy, pca_mean_x, pca_mean_y, pca_explained_variance = np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Get the "axial_plus_midpoints" data for the current frame
        current_x_df = bodypart_coordinate_sets["axial_plus_midpoints"]["x"]
        current_y_df = bodypart_coordinate_sets["axial_plus_midpoints"]["y"]
    
        if frame_idx in current_x_df.index and frame_idx in current_y_df.index:
            x_coords_frame = current_x_df.loc[frame_idx]
            y_coords_frame = current_y_df.loc[frame_idx]
    
            # Combine into a DataFrame and drop NaNs
            points_df = pd.DataFrame({'x': x_coords_frame, 'y': y_coords_frame}).dropna()
    
            if len(points_df) >= MIN_POINTS_FOR_FIT:
                # --- PCA ---
                try:
                    pca = PCA(n_components=1)
                    pca.fit(points_df[['x', 'y']].values)
                    pca_dx, pca_dy = pca.components_[0]
                    pca_mean_x, pca_mean_y = pca.mean_
                    pca_explained_variance = pca.explained_variance_[0] if pca.explained_variance_ is not None else np.nan
                except Exception as e:
                    print(f"PCA calculation error at frame {frame_idx}: {e}")
    
        # Store results
        results_midpoints_pca.append({
            'frame': frame_idx, 'pca_dx': pca_dx, 'pca_dy': pca_dy, 
            'pca_mean_x': pca_mean_x, 'pca_mean_y': pca_mean_y,
            'pca_explained_variance': pca_explained_variance
        })
    
    # Convert list of dictionaries to DataFrame
    df_midpoints_pca_raw = pd.DataFrame(results_midpoints_pca).set_index('frame')
    
    # Add PCA results to main DataFrame
    df_dlc[('analysis', 'pca_dx')] = df_midpoints_pca_raw['pca_dx']
    df_dlc[('analysis', 'pca_dy')] = df_midpoints_pca_raw['pca_dy']
    df_dlc[('analysis', 'pca_mean_x')] = df_midpoints_pca_raw['pca_mean_x']
    df_dlc[('analysis', 'pca_mean_y')] = df_midpoints_pca_raw['pca_mean_y']
    df_dlc[('analysis', 'pca_explained_variance')] = df_midpoints_pca_raw['pca_explained_variance']
    
    # Calculate success rate
    total_frames = len(df_dlc)
    successful_frames = df_midpoints_pca_raw['pca_dx'].notna().sum()
    
    print(f"Body axis PCA calculation complete: successful in {successful_frames}/{total_frames} frames "
          f"({successful_frames/total_frames*100:.1f}%)")
    
    # Report NaN counts
    nan_counts = df_midpoints_pca_raw.isna().sum()
    print(f"NaN counts in PCA results:\n{nan_counts}")
    
    return df_dlc, df_midpoints_pca_raw

def calculate_and_plot_body_axis_angles(df_dlc, df_midpoints_pca_raw, filtered_x_coords, filtered_y_coords, 
                                        frame_rate, output_dir, file_name, save_plots=True):
    """
    Calculate the angles of body axis vs. positive Y-axis and create a polar plot
    showing the changes in orientation over time.        
    Returns:
        Updated DataFrame with body axis angle data
    """   
    # --- Orientation Setup ---
    front_bp_name = 'neck'
    back_bp_name = 'mid_backend'
    can_orient_globally = True  # Renamed for clarity

    # Check if orientation bodyparts are available
    if not all(bp in filtered_x_coords.columns for bp in [front_bp_name, back_bp_name]) or \
       not all(bp in filtered_y_coords.columns for bp in [front_bp_name, back_bp_name]):
        print(f"Warning: Orientation bodyparts ('{front_bp_name}', '{back_bp_name}') not fully available.")
        print("Angles will be calculated based on raw PCA vectors without anatomical orientation.")
        can_orient_globally = False
        # Create empty series if orientation parts are missing, so .get() doesn't fail later
        front_bp_x_series = pd.Series(dtype=float, index=df_dlc.index)
        front_bp_y_series = pd.Series(dtype=float, index=df_dlc.index)
        back_bp_x_series = pd.Series(dtype=float, index=df_dlc.index)
        back_bp_y_series = pd.Series(dtype=float, index=df_dlc.index)
    else:
        front_bp_x_series = filtered_x_coords[front_bp_name]
        front_bp_y_series = filtered_y_coords[front_bp_name]
        back_bp_x_series = filtered_x_coords[back_bp_name]
        back_bp_y_series = filtered_y_coords[back_bp_name]
    # --- Calculate Percentage of Frames Where Anatomical Orientation Was Applied ---
    frames_oriented_successfully = 0
    total_frames_for_orientation_check = len(df_dlc.index)
    percentage_oriented = 0.0

    if can_orient_globally and total_frames_for_orientation_check > 0:
        for frame_idx in df_dlc.index:
            f_x = front_bp_x_series.get(frame_idx, np.nan)
            f_y = front_bp_y_series.get(frame_idx, np.nan)
            b_x = back_bp_x_series.get(frame_idx, np.nan)
            b_y = back_bp_y_series.get(frame_idx, np.nan)
            
            if not (pd.isna(f_x) or pd.isna(f_y) or pd.isna(b_x) or pd.isna(b_y) or (f_x == b_x and f_y == b_y)):
                frames_oriented_successfully += 1
        
        if total_frames_for_orientation_check > 0:
            percentage_oriented = (frames_oriented_successfully / total_frames_for_orientation_check) * 100

    # --- Helper Function for Angle Calculation (PCA) ---
    def calculate_pca_angle_vs_y(df_pca_raw, method_name):
        angles = []
        for frame_idx, row in df_pca_raw.iterrows():
            dx, dy = row['pca_dx'], row['pca_dy']
            if pd.isna(dx) or pd.isna(dy):
                angles.append(np.nan)
                continue

            oriented_dx, oriented_dy = dx, dy
            if can_orient_globally:  # Use the global flag
                f_x = front_bp_x_series.get(frame_idx, np.nan)
                f_y = front_bp_y_series.get(frame_idx, np.nan)
                b_x = back_bp_x_series.get(frame_idx, np.nan)
                b_y = back_bp_y_series.get(frame_idx, np.nan)

                if not (pd.isna(f_x) or pd.isna(f_y) or pd.isna(b_x) or pd.isna(b_y) or (f_x == b_x and f_y == b_y)):
                    anat_dx, anat_dy = f_x - b_x, f_y - b_y
                    if (dx * anat_dx + dy * anat_dy) < 0: 
                        oriented_dx, oriented_dy = -dx, -dy
            
            angles.append(np.degrees(np.arctan2(oriented_dx, oriented_dy)))
        
        df_pca_raw[f'angle_y_deg_{method_name}'] = angles
        print(f"Calculated angles for PCA ({method_name}).")

    # --- Calculate Angles only for midpoints_pca ---
    calculate_pca_angle_vs_y(df_midpoints_pca_raw, 'midpoints_pca')
    
    # Add the calculated angles to the main DataFrame
    df_dlc[('analysis', 'body_axis_angle')] = df_midpoints_pca_raw['angle_y_deg_midpoints_pca']

    if save_plots:
        try:
            # --- Prepare Time Axis and Angular Data for Plotting ---
            time_axis = df_dlc.index / frame_rate
            angle_data = df_midpoints_pca_raw['angle_y_deg_midpoints_pca'].copy()

            # --- Convert to Polar Coordinates ---
            # Convert angles from [-180, 180] to radians [0, 2pi]
            # First, map angles from [-180, 180] to [0, 360]
            angles_0_360 = (angle_data + 180) % 360
            # Then convert to radians
            angles_radians = np.radians(angles_0_360)

            # --- Create time-based radius and colors ---
            # Create a normalized time vector (0 to 1) for color mapping
            norm_time = (time_axis - time_axis[0]) / (time_axis[-1] - time_axis[0]) if len(time_axis) > 1 else [0]
            # Radius grows with time (0 to 1.0) - linear progression
            radius = norm_time  # This makes radius start at 0 for the first frame

            # Create the polar plot with time-colored lines
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='polar')

            # Use viridis colormap but reversed (yellow->green->blue->purple)
            cmap = plt.cm.viridis_r  # Using reversed viridis (yellow at start, purple at end)

            # Plot the data with thin lines
            # Skip NaN values and connect only consecutive valid points
            valid_mask = ~np.isnan(angles_radians)
            segments = []
            colors = []

            for i in range(len(angles_radians) - 1):
                if valid_mask[i] and valid_mask[i + 1]:
                    segments.append([(angles_radians[i], radius[i]), (angles_radians[i + 1], radius[i + 1])])
                    colors.append(cmap(norm_time[i]))  # Color corresponds to time

            # Use line collection for efficient plotting
            from matplotlib.collections import LineCollection
            if segments:  # Only create line collection if there are segments to plot
                line_segments = LineCollection(segments, colors=colors, linewidths=0.5, alpha=0.7)
                ax.add_collection(line_segments)

            # Mark the starting point with a dot at exactly the center
            if valid_mask[0]:
                start_angle = angles_radians[0]
                # Force the start point to be at the exact center (radius=0)
                ax.scatter(start_angle, 0, s=30, color='yellow', edgecolor='black', zorder=10, clip_on=False)
            
            # Set up the plot appearance
            ax.set_theta_zero_location('N')  # 0 degrees at the top
            ax.set_theta_direction(-1)  # clockwise
            ax.set_rorigin(0)
            ax.set_rmin(0)
            ax.set_rmax(1.0)
            ax.grid(True, alpha=0.3)

            # Remove radius ticks
            ax.set_rticks([])  # No radius ticks

            # Add angle labels
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])

            # Add colorbar to show time mapping (reversed color scale)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(time_axis.min(), time_axis.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Time (seconds)')

            # Add title and adjust layout
            plt.title('Body Axis Angle Over Time (Polar Representation)', fontsize=14)
            plt.tight_layout()

            # Save plot
            base_name_without_ext = os.path.splitext(file_name)[0]
            base_output_name = f"{base_name_without_ext}_orientation_analysis"
            plot_filename = os.path.join(output_dir, f"{base_output_name}_body_axis_angle_polar.png")
            plt.savefig(plot_filename, dpi=200)
            plt.close()
            
            print(f"Saved polar angle plot: {plot_filename}")
        
        except Exception as e:
            print(f"Error generating body axis angle plot: {e}")
    
    # --- Orientation Summary ---
    print("\n--- Orientation Summary ---")
    if not can_orient_globally:
        print(f"Anatomical orientation was not attempted because required bodyparts ('{front_bp_name}', '{back_bp_name}') were missing.")
        print("Percentage of frames where anatomical orientation was applied: 0.00%")
    else:
        print(f"Anatomical orientation was attempted using '{front_bp_name}' (front) and '{back_bp_name}' (back).")
        print(f"Percentage of frames with anatomical orientation: {percentage_oriented:.2f}%")
    print("Body axis angle calculation and plotting complete.")
    
    return df_dlc

def plot_smoothed_body_axis_angles(df_dlc, df_midpoints_pca_raw, frame_rate, output_dir, file_name, save_plots=True):
    """
    Apply rolling window smoothing to body axis angles and create a polar plot 
    showing smoothed orientation changes over time.
    Returns:
        DataFrame with smoothed angles added
    """
    print("\nApplying rolling window smoothing to body axis angles...")
    
    # Check if angles were calculated
    if 'angle_y_deg_midpoints_pca' not in df_midpoints_pca_raw.columns:
        print("Error: Angle data not found. Please run calculate_and_plot_body_axis_angles first.")
        return df_dlc
    
    # --- Data and Parameters ---
    angle_data_series = df_midpoints_pca_raw['angle_y_deg_midpoints_pca'].copy()
    time_axis = df_dlc.index / frame_rate

    # Rolling window parameters - fixed at 15 frames
    smoothing_window_frames = 15

    # --- Apply Rolling Window Smoothing ---
    if angle_data_series.empty:
        print("Skipping smoothing: No angle data available.")
        smoothed_angle = pd.Series(np.nan, index=angle_data_series.index)
    elif len(angle_data_series) < smoothing_window_frames:
        print(f"Skipping smoothing: Data length ({len(angle_data_series)}) is less than window size ({smoothing_window_frames}).")
        smoothed_angle = pd.Series(np.nan, index=angle_data_series.index)
    else:
        # Apply rolling window average with proper handling of NaNs and circular data
        angles_rad = np.deg2rad(angle_data_series)
        
        # Create a DataFrame to store complex representations
        complex_df = pd.DataFrame(index=angle_data_series.index)
        complex_df['real'] = np.cos(angles_rad)
        complex_df['imag'] = np.sin(angles_rad)
        
        # Apply rolling mean to real and imaginary components separately
        # min_periods=1 ensures we get values even at the edges
        rolled_real = complex_df['real'].rolling(window=smoothing_window_frames, center=True, min_periods=1).mean()
        rolled_imag = complex_df['imag'].rolling(window=smoothing_window_frames, center=True, min_periods=1).mean()
        
        # Convert back to angles, handling NaNs appropriately
        # Where both real and imaginary parts are NaN, the result should be NaN
        mask = ~(rolled_real.isna() | rolled_imag.isna())
        result = pd.Series(np.nan, index=angle_data_series.index)
        result[mask] = np.rad2deg(np.arctan2(rolled_imag[mask], rolled_real[mask]))
        
        smoothed_angle = result
        print(f"Applied rolling window smoothing with window size: {smoothing_window_frames} frames to angular data.")
        print(f"NaN count in raw angles: {angle_data_series.isna().sum()}")
        print(f"NaN count in smoothed angles: {smoothed_angle.isna().sum()}")
    
    # Add smoothed angle to the main DataFrame
    df_dlc[('analysis', 'body_axis_angle_smoothed')] = smoothed_angle
    df_midpoints_pca_raw['angle_y_deg_midpoints_pca_smoothed'] = smoothed_angle
    
    if save_plots:
        try:
            # Convert to 0-360 format for plotting
            smoothed_angles_0_360 = (smoothed_angle + 180) % 360
            # Then convert to radians
            smoothed_angles_radians = np.radians(smoothed_angles_0_360)
            
            # --- Create time-based radius and colors ---
            # Create a normalized time vector (0 to 1) for color mapping
            norm_time = (time_axis - time_axis.min()) / (time_axis.max() - time_axis.min()) if len(time_axis) > 1 else [0]
            
            # Radius grows with time (0 to 1.0) - linear progression
            radius = norm_time  # This makes radius start at 0 for the first frame
            
            # Create the polar plot with time-colored lines
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='polar')
            
            # Use viridis colormap but reversed (yellow->green->blue->purple)
            cmap = plt.cm.viridis_r  # Using reversed viridis (yellow at start, purple at end)
            
            # Plot the data with thin lines
            # Skip NaN values and connect only consecutive valid points
            valid_mask = ~np.isnan(smoothed_angles_radians)
            segments = []
            colors = []
            
            for i in range(len(smoothed_angles_radians) - 1):
                if valid_mask[i] and valid_mask[i + 1]:
                    segments.append([(smoothed_angles_radians[i], radius[i]), (smoothed_angles_radians[i + 1], radius[i + 1])])
                    colors.append(cmap(norm_time[i]))  # Color corresponds to time
            
            # Use line collection for efficient plotting
            from matplotlib.collections import LineCollection
            if segments:  # Only create line collection if there are segments to plot
                line_segments = LineCollection(segments, colors=colors, linewidths=0.5, alpha=0.7)
                ax.add_collection(line_segments)
            
            # Mark the starting point with a dot at exactly the center
            if valid_mask[0]:
                start_angle = smoothed_angles_radians[0]
                # Force the start point to be at the exact center (radius=0)
                ax.scatter(start_angle, 0, s=30, color='yellow', edgecolor='black', zorder=10, clip_on=False)
            
            # Set up the plot appearance
            ax.set_theta_zero_location('N')  # 0 degrees at the top
            ax.set_theta_direction(-1)  # clockwise
            ax.set_rorigin(0)  # Set the origin of the radial axis at 0
            ax.set_rmin(0)    # Set minimum radius to 0
            ax.set_rmax(1.0)
            ax.grid(True, alpha=0.3)
            
            # Remove radius ticks
            ax.set_rticks([])  # No radius ticks
            
            # Add angle labels
            ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])
            
            # Add colorbar to show time mapping (reversed color scale)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(time_axis.min(), time_axis.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
            cbar.set_label('Time (seconds)')
            
            # Add title and adjust layout
            plt.title('Smoothed Body Axis Angle Over Time (Polar Representation)', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            base_name_without_ext = os.path.splitext(file_name)[0]
            base_output_name = f"{base_name_without_ext}_orientation_analysis"
            plot_filename = os.path.join(output_dir, f"{base_output_name}_body_axis_angle_smoothed_polar.png")
            plt.savefig(plot_filename, dpi=200)
            plt.close()
            
            print(f"Saved smoothed polar angle plot: {plot_filename}")
            
        except Exception as e:
            print(f"Error generating smoothed body axis angle plot: {e}")
    
    print("Smoothed body axis angle plotting complete.")
    
    return df_dlc

def calculate_angular_velocity(df_dlc, df_midpoints_pca_raw, frame_rate, output_dir, file_name, save_plots=True):
    """
    Calculate the derivative (angular velocity) of smoothed body axis angles.
    Creates plots for both signed angular velocity and absolute angular velocity.
    Returns:
        Updated DataFrame with angular velocity metrics added
    """
    print("\nCalculating and plotting derivative of smoothed body axis angle...")
    
    # Check if smoothed angle data is available
    if 'angle_y_deg_midpoints_pca_smoothed' not in df_midpoints_pca_raw.columns:
        print("Error: Smoothed angle data not found. Please run plot_smoothed_body_axis_angles first.")
        return df_dlc
    
    # Get smoothed angle data
    smoothed_angle = df_midpoints_pca_raw['angle_y_deg_midpoints_pca_smoothed']
    time_axis = df_dlc.index / frame_rate
    
    # Check if data is all NaN
    if smoothed_angle.isna().all():
        print("Warning: Smoothed angle data is all NaN. Cannot calculate derivative.")
        return df_dlc
    
    # Get the smoothing window size for reference in outputs
    smoothing_window_frames = 15  # This is the fixed window size used in plot_smoothed_body_axis_angles
    
    # --- Calculate Derivative (Angular Velocity) ---
    # 1. Calculate frame-to-frame difference of the smoothed angle
    angle_derivative_raw = smoothed_angle.diff()

    # 2. Correct for circularity (angles are in degrees, -180 to 180)
    # The change from 170 deg to -170 deg should be -20 deg, not -340 deg.
    # (difference + 180) % 360 - 180 maps differences to the range [-180, 180]
    angle_derivative_corrected = (angle_derivative_raw + 180) % 360 - 180

    # 3. Convert from degrees/frame to degrees/second
    angular_velocity_deg_per_s = angle_derivative_corrected * frame_rate

    # 4. Calculate the absolute angular velocity (magnitude only)
    absolute_angular_velocity = np.abs(angular_velocity_deg_per_s)
    
    # Store results in the main DataFrame and PCA results DataFrame
    df_dlc[('analysis', 'angular_velocity_deg_per_s')] = angular_velocity_deg_per_s
    df_dlc[('analysis', 'absolute_angular_velocity')] = absolute_angular_velocity
    df_midpoints_pca_raw['angular_velocity_deg_per_s'] = angular_velocity_deg_per_s
    df_midpoints_pca_raw['absolute_angular_velocity'] = absolute_angular_velocity

    # Print summary statistics
    print(f"Calculated angular velocity from smoothed angle (window={smoothing_window_frames} frames).")
    
    if not angular_velocity_deg_per_s.isna().all():
        print(f"Angular velocity range: {angular_velocity_deg_per_s.min():.2f} to {angular_velocity_deg_per_s.max():.2f} deg/s (excluding NaNs)")
        print(f"Absolute angular velocity range: {absolute_angular_velocity.min():.2f} to {absolute_angular_velocity.max():.2f} deg/s (excluding NaNs)")
    else:
        print("Could not calculate angular velocity range - all values are NaN.")
    
    if save_plots:
        try:
            # --- Create time-based colors using the same colormap as the polar plot ---
            # Create a normalized time vector (0 to 1) for color mapping
            norm_time = (time_axis - time_axis.min()) / (time_axis.max() - time_axis.min()) if len(time_axis) > 1 else [0]

            # Use viridis colormap but reversed (yellow->green->blue->purple) for consistency
            cmap = plt.cm.viridis_r

            # --- Plot both angular velocity and its absolute value with time-colored lines ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

            # Create a color array for the entire time series
            colors = [cmap(t) for t in norm_time]

            # Plot the angular velocity (with direction)
            from matplotlib.collections import LineCollection
            
            valid_mask = ~np.isnan(angular_velocity_deg_per_s)
            segments = []
            seg_colors = []
            
            for i in range(len(time_axis) - 1):
                if valid_mask.iloc[i] and valid_mask.iloc[i+1]:
                    segments.append([(time_axis[i], angular_velocity_deg_per_s.iloc[i]), 
                                    (time_axis[i+1], angular_velocity_deg_per_s.iloc[i+1])])
                    seg_colors.append(colors[i])
            
            if segments:
                lc1 = LineCollection(segments, colors=seg_colors, linewidth=1.5)
                ax1.add_collection(lc1)
                
                if not angular_velocity_deg_per_s.isna().all():
                    ax1.set_ylim(angular_velocity_deg_per_s.min()*1.05, angular_velocity_deg_per_s.max()*1.05)
                
            ax1.set_xlim(time_axis.min(), time_axis.max())
            ax1.set_title('Angular Velocity of Body Axis')
            ax1.set_ylabel("Angular Velocity (degrees/second)")
            ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Angular Velocity')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper right')

            # Plot the absolute angular velocity (magnitude only)
            segments_abs = []
            seg_abs_colors = []
            
            valid_mask_abs = ~np.isnan(absolute_angular_velocity)
            
            for i in range(len(time_axis) - 1):
                if valid_mask_abs.iloc[i] and valid_mask_abs.iloc[i+1]:
                    segments_abs.append([(time_axis[i], absolute_angular_velocity.iloc[i]), 
                                         (time_axis[i+1], absolute_angular_velocity.iloc[i+1])])
                    seg_abs_colors.append(colors[i])
            
            if segments_abs:
                lc2 = LineCollection(segments_abs, colors=seg_abs_colors, linewidth=1.5)
                ax2.add_collection(lc2)
                
                if not absolute_angular_velocity.isna().all():
                    ax2.set_ylim(0, absolute_angular_velocity.max()*1.05)
                
            ax2.set_xlim(time_axis.min(), time_axis.max())
            ax2.set_title('Absolute Angular Velocity (Turning Rate)')
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Absolute Angular Velocity (degrees/second)")
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(['Absolute Angular Velocity'], loc='upper right')

            fig.subplots_adjust(right=0.85)

            # Add colorbar to show time mapping
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(time_axis.min(), time_axis.max()))
            sm.set_array([])
            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Time (seconds)')

            fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave room on the right

            # Save the plot
            base_name_without_ext = os.path.splitext(file_name)[0]
            base_output_name = f"{base_name_without_ext}_orientation_analysis"
            plot_filename = os.path.join(output_dir, f"{base_output_name}_angular_velocity.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close()
            
            print(f"Saved angular velocity plot: {plot_filename}")
            
        except Exception as e:
            print(f"Error generating angular velocity plot: {e}")
    
    print("Angular velocity calculation and plotting complete.")
    
    return df_dlc

def identify_sleep_bouts_and_plot(df_dlc, df_midpoints_pca_raw, frame_rate, output_dir, file_name, save_plots=True):
    """
    Identify sleep bouts using three different metrics (speed, body posture change, angular velocity)
    and create comparative visualizations.
    Returns:
        tuple: (df_dlc with sleep metrics added, dict of sleep bout DataFrames)
    """
    print("\nIdentifying and visualizing sleep bouts using multiple metrics...")
    
    # Define thresholds for sleep detection
    sleep_speed_threshold_pixels_per_second = 60.0
    posture_change_threshold_pps = 60.0
    angular_velocity_threshold_deg_per_s = 50.0
    
    # Minimum durations for sleep bouts
    min_sleep_duration_seconds = 10
    
    # Create time axis in seconds
    time_axis = df_dlc.index / frame_rate
    
    # Base output name for plot files
    base_name_without_ext = os.path.splitext(file_name)[0]
    base_output_name = f"{base_name_without_ext}_sleep_analysis"
    
    # Dictionary to store results
    sleep_bouts_dict = {}
    
    # --- 1. SPEED-BASED SLEEP DETECTION ---
    print("\n--- Identifying Sleep Bouts Based on Speed ---")
    
    # Make sure we have the speed data
    if ('analysis', 'speed_pixels_per_second') not in df_dlc.columns:
        print("Error: Speed data not found. Run calculate_speed first.")
        return df_dlc, {}
    
    # Apply smoothing to speed data
    smoothing_window_seconds = 15/60
    smoothing_window_frames = int(smoothing_window_seconds * frame_rate)
    if smoothing_window_frames < 1: 
        smoothing_window_frames = 1
        
    smoothed_speed = df_dlc[('analysis', 'speed_pixels_per_second')].rolling(
        window=smoothing_window_frames, min_periods=1, center=True
    ).mean()
    
    # Store smoothed speed in the DataFrame
    df_dlc[('analysis', 'speed_smoothed')] = smoothed_speed
    
    # 1. Identify periods of low speed (potential sleep)
    is_low_activity = smoothed_speed < sleep_speed_threshold_pixels_per_second

    # 2. Identify contiguous blocks of low activity
    activity_groups = is_low_activity.ne(is_low_activity.shift()).cumsum()
    low_activity_periods = is_low_activity[is_low_activity]  # Filter for only True periods

    # 3. Process these blocks into sleep bouts
    speed_sleep_bouts = []
    if not low_activity_periods.empty:
        for group_id, group_data in low_activity_periods.groupby(activity_groups[is_low_activity]):
            start_frame = group_data.index[0]
            end_frame = group_data.index[-1]
            
            # Calculate duration in frames and seconds
            duration_frames = (end_frame - start_frame) + 1  # Inclusive
            duration_seconds = duration_frames / frame_rate
            
            # Only include if it meets minimum duration requirement
            if duration_seconds >= min_sleep_duration_seconds:
                start_time_seconds = start_frame / frame_rate
                end_time_seconds = (end_frame + 1) / frame_rate  # End time is exclusive for slicing
                
                speed_sleep_bouts.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_s': start_time_seconds,
                    'end_time_s': end_time_seconds,
                    'duration_s': duration_seconds,
                    'avg_speed_in_bout': smoothed_speed.loc[start_frame:end_frame].mean(),
                    'max_speed_in_bout': smoothed_speed.loc[start_frame:end_frame].max()
                })

    # Convert to DataFrame
    df_speed_sleep_bouts = pd.DataFrame(speed_sleep_bouts)
    sleep_bouts_dict['speed'] = df_speed_sleep_bouts
    
    # Report results
    if not df_speed_sleep_bouts.empty:
        print(f"Identified {len(df_speed_sleep_bouts)} sleep bout(s) based on speed:")
        print(df_speed_sleep_bouts[['start_time_s', 'end_time_s', 'duration_s', 'avg_speed_in_bout']].head().to_string())
        if len(df_speed_sleep_bouts) > 5:
            print(f"...and {len(df_speed_sleep_bouts) - 5} more bouts")
    else:
        print("No sleep bouts identified based on speed.")
    
    # --- 2. BODY POSTURE CHANGE-BASED SLEEP DETECTION ---
    print("\n--- Identifying Sleep Bouts Based on Body Posture Change ---")
    
    # Make sure we have the body posture metric derivative
    if ('analysis', 'posture_metric_abs_derivative') not in df_dlc.columns:
        print("Body posture metric derivative not found. Run calculate_body_movement_derivative first.")
        posture_sleep_bouts_available = False
        df_posture_sleep_bouts = pd.DataFrame()
    else:
        posture_sleep_bouts_available = True
        posture_change_rate = df_dlc[('analysis', 'posture_metric_abs_derivative')]
        
        # 1. Identify periods of low posture change (potential sleep)
        is_low_posture_change = posture_change_rate < posture_change_threshold_pps
        
        # 2. Identify contiguous blocks of low posture change
        posture_change_groups = is_low_posture_change.ne(is_low_posture_change.shift()).cumsum()
        low_posture_change_periods = is_low_posture_change[is_low_posture_change]  # Filter for only True periods
        
        # 3. Process these blocks into sleep bouts
        posture_sleep_bouts = []
        if not low_posture_change_periods.empty:
            for group_id, group_data in low_posture_change_periods.groupby(posture_change_groups[is_low_posture_change]):
                start_frame = group_data.index[0]
                end_frame = group_data.index[-1]
                
                # Calculate duration in frames and seconds
                duration_frames = (end_frame - start_frame) + 1  # Inclusive
                duration_seconds = duration_frames / frame_rate
                
                # Only include if it meets minimum duration requirement
                if duration_seconds >= min_sleep_duration_seconds:
                    start_time_seconds = start_frame / frame_rate
                    end_time_seconds = (end_frame + 1) / frame_rate  # End time is exclusive
                    
                    posture_sleep_bouts.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_time_s': start_time_seconds,
                        'end_time_s': end_time_seconds,
                        'duration_s': duration_seconds,
                        'avg_posture_change_in_bout': posture_change_rate.loc[start_frame:end_frame].mean(),
                        'max_posture_change_in_bout': posture_change_rate.loc[start_frame:end_frame].max()
                    })
        
        # Convert to DataFrame
        df_posture_sleep_bouts = pd.DataFrame(posture_sleep_bouts)
        sleep_bouts_dict['posture'] = df_posture_sleep_bouts
        
        # Report results
        if not df_posture_sleep_bouts.empty:
            print(f"Identified {len(df_posture_sleep_bouts)} sleep bout(s) based on posture change:")
            print(df_posture_sleep_bouts[['start_time_s', 'end_time_s', 'duration_s', 'avg_posture_change_in_bout']].head().to_string())
            if len(df_posture_sleep_bouts) > 5:
                print(f"...and {len(df_posture_sleep_bouts) - 5} more bouts")
        else:
            print("No sleep bouts identified based on posture change.")
    
    # --- 3. ANGULAR VELOCITY-BASED SLEEP DETECTION ---
    print("\n--- Identifying Sleep Bouts Based on Angular Velocity ---")
    
    # Make sure we have the angular velocity data
    if ('analysis', 'absolute_angular_velocity') not in df_dlc.columns:
        print("Angular velocity data not found. Run calculate_angular_velocity first.")
        angular_sleep_bouts_available = False
        df_angular_sleep_bouts = pd.DataFrame()
    else:
        angular_sleep_bouts_available = True
        angular_velocity_data = df_dlc[('analysis', 'absolute_angular_velocity')]
        
        # 1. Identify periods of low angular velocity (potential sleep)
        is_low_angular_velocity = angular_velocity_data < angular_velocity_threshold_deg_per_s
        
        # 2. Identify contiguous blocks of low angular velocity
        angular_velocity_groups = is_low_angular_velocity.ne(is_low_angular_velocity.shift()).cumsum()
        low_angular_velocity_periods = is_low_angular_velocity[is_low_angular_velocity]  # Filter for only True periods
        
        # 3. Process these blocks into sleep bouts
        angular_sleep_bouts = []
        if not low_angular_velocity_periods.empty:
            for group_id, group_data in low_angular_velocity_periods.groupby(angular_velocity_groups[is_low_angular_velocity]):
                start_frame = group_data.index[0]
                end_frame = group_data.index[-1]
                
                # Calculate duration in frames and seconds
                duration_frames = (end_frame - start_frame) + 1  # Inclusive
                duration_seconds = duration_frames / frame_rate
                
                # Only include if it meets minimum duration requirement
                if duration_seconds >= min_sleep_duration_seconds:
                    start_time_seconds = start_frame / frame_rate
                    end_time_seconds = (end_frame + 1) / frame_rate  # End time is exclusive
                    
                    angular_sleep_bouts.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_time_s': start_time_seconds,
                        'end_time_s': end_time_seconds,
                        'duration_s': duration_seconds,
                        'avg_angular_velocity_in_bout': angular_velocity_data.loc[start_frame:end_frame].mean(),
                        'max_angular_velocity_in_bout': angular_velocity_data.loc[start_frame:end_frame].max()
                    })
        
        # Convert to DataFrame
        df_angular_sleep_bouts = pd.DataFrame(angular_sleep_bouts)
        sleep_bouts_dict['angular'] = df_angular_sleep_bouts
        
        # Report results
        if not df_angular_sleep_bouts.empty:
            print(f"Identified {len(df_angular_sleep_bouts)} sleep bout(s) based on angular velocity:")
            print(df_angular_sleep_bouts[['start_time_s', 'end_time_s', 'duration_s', 'avg_angular_velocity_in_bout']].head().to_string())
            if len(df_angular_sleep_bouts) > 5:
                print(f"...and {len(df_angular_sleep_bouts) - 5} more bouts")
        else:
            print("No sleep bouts identified based on angular velocity.")
    
    # --- GENERATE PLOTS ---
    if save_plots:
        # --- 1. Individual metric plots ---
        # Check if we need to create individual plots
        has_speed_bouts = not df_speed_sleep_bouts.empty
        has_posture_bouts = posture_sleep_bouts_available and not df_posture_sleep_bouts.empty
        has_angular_bouts = angular_sleep_bouts_available and not df_angular_sleep_bouts.empty
        
        # 1. Speed plot
        if has_speed_bouts:
            try:
                plt.figure(figsize=(15, 5))
                plt.plot(time_axis, smoothed_speed, label='Smoothed Speed', color='grey', alpha=0.7, lw=1)
                
                # Highlight sleep bouts
                for _, bout in df_speed_sleep_bouts.iterrows():
                    plt.axvspan(
                        bout['start_time_s'], 
                        bout['end_time_s'], 
                        color='palegreen', 
                        alpha=0.4, 
                        label='Sleep Bout' if _ == 0 else ""  # Label only once
                    )
                
                plt.axhline(
                    sleep_speed_threshold_pixels_per_second, 
                    color='r', 
                    linestyle='--', 
                    lw=1, 
                    label=f'Sleep Threshold ({sleep_speed_threshold_pixels_per_second} px/s)'
                )
                
                plt.xlabel("Time (seconds)")
                plt.ylabel("Speed (pixels/second)")
                plt.title("Animal Speed with Identified Sleep Bouts")
                
                # Improve legend uniqueness
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                
                # Save plot
                plot_filename = os.path.join(output_dir, f"{base_output_name}_speed_sleep_bouts.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                
                print(f"Speed-based sleep bouts plot saved to: {plot_filename}")
            except Exception as e:
                print(f"Error generating speed-based sleep plot: {e}")
        
        # 2. Posture change plot
        if has_posture_bouts:
            try:
                plt.figure(figsize=(15, 5))
                plt.plot(time_axis, posture_change_rate, label='Posture Change Rate', color='purple', alpha=0.7, lw=1)
                
                # Highlight sleep bouts
                for _, bout in df_posture_sleep_bouts.iterrows():
                    plt.axvspan(
                        bout['start_time_s'], 
                        bout['end_time_s'], 
                        color='palegreen', 
                        alpha=0.4, 
                        label='Low Posture Change Bout' if _ == 0 else ""
                    )
                
                plt.axhline(
                    posture_change_threshold_pps, 
                    color='r', 
                    linestyle='--', 
                    lw=1, 
                    label=f'Posture Change Threshold ({posture_change_threshold_pps} px/s)'
                )
                
                plt.xlabel("Time (seconds)")
                plt.ylabel("Posture Change Rate (pixels/second)")
                plt.title("Body Posture Change Rate with Identified Low-Change Bouts")
                
                # Improve legend uniqueness
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                
                # Save plot
                plot_filename = os.path.join(output_dir, f"{base_output_name}_posture_sleep_bouts.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                
                print(f"Posture-based sleep bouts plot saved to: {plot_filename}")
            except Exception as e:
                print(f"Error generating posture-based sleep plot: {e}")
        
        # 3. Angular velocity plot
        if has_angular_bouts:
            try:
                plt.figure(figsize=(15, 5))
                plt.plot(time_axis, angular_velocity_data, label='Absolute Angular Velocity', color='blue', alpha=0.7, lw=1)
                
                # Highlight sleep bouts
                for _, bout in df_angular_sleep_bouts.iterrows():
                    plt.axvspan(
                        bout['start_time_s'], 
                        bout['end_time_s'], 
                        color='palegreen', 
                        alpha=0.4, 
                        label='Low Angular Velocity Bout' if _ == 0 else ""
                    )
                
                plt.axhline(
                    angular_velocity_threshold_deg_per_s, 
                    color='r', 
                    linestyle='--', 
                    lw=1, 
                    label=f'Angular Velocity Threshold ({angular_velocity_threshold_deg_per_s} deg/s)'
                )
                
                plt.xlabel("Time (seconds)")
                plt.ylabel("Absolute Angular Velocity (degrees/second)")
                plt.title("Angular Velocity with Identified Low-Velocity Bouts")
                
                # Improve legend uniqueness
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                
                # Save plot
                plot_filename = os.path.join(output_dir, f"{base_output_name}_angular_velocity_sleep_bouts.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                
                print(f"Angular velocity-based sleep bouts plot saved to: {plot_filename}")
            except Exception as e:
                print(f"Error generating angular velocity-based sleep plot: {e}")
        
        # --- 2. Combined comparative visualization ---
        try:
            # Count how many plots we'll need
            active_methods = sum([
                True,  # Speed is always available
                posture_sleep_bouts_available,
                angular_sleep_bouts_available
            ])
            
            fig, axes = plt.subplots(active_methods, 1, figsize=(15, 5 * active_methods), sharex=True)
            
            # If only one plot, make axes iterable
            if active_methods == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # 1. Speed plot
            axes[plot_idx].plot(time_axis, smoothed_speed, label='Smoothed Speed', color='grey', alpha=0.7, lw=1)
            
            if has_speed_bouts:
                for _, bout in df_speed_sleep_bouts.iterrows():
                    axes[plot_idx].axvspan(
                        bout['start_time_s'], 
                        bout['end_time_s'], 
                        color='palegreen', 
                        alpha=0.4, 
                        label='Speed-Based Sleep' if _ == 0 else ""
                    )
            
            axes[plot_idx].axhline(
                sleep_speed_threshold_pixels_per_second, 
                color='r', 
                linestyle='--', 
                lw=1, 
                label=f'Speed Threshold ({sleep_speed_threshold_pixels_per_second} px/s)'
            )
            
            axes[plot_idx].set_ylabel("Speed (pixels/second)")
            axes[plot_idx].set_title("Speed-Based Sleep Detection")
            axes[plot_idx].grid(True, linestyle=':', alpha=0.6)
            
            # Improve legend uniqueness
            handles, labels = axes[plot_idx].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[plot_idx].legend(by_label.values(), by_label.keys(), loc='upper right')
            
            plot_idx += 1
            
            # 2. Posture change plot
            if posture_sleep_bouts_available:
                axes[plot_idx].plot(time_axis, posture_change_rate, label='Posture Change Rate', color='purple', alpha=0.7, lw=1)
                
                if has_posture_bouts:
                    for _, bout in df_posture_sleep_bouts.iterrows():
                        axes[plot_idx].axvspan(
                            bout['start_time_s'], 
                            bout['end_time_s'], 
                            color='lightblue', 
                            alpha=0.4, 
                            label='Posture-Based Sleep' if _ == 0 else ""
                        )
                
                axes[plot_idx].axhline(
                    posture_change_threshold_pps, 
                    color='r', 
                    linestyle='--', 
                    lw=1, 
                    label=f'Posture Change Threshold ({posture_change_threshold_pps} px/s)'
                )
                
                axes[plot_idx].set_ylabel("Posture Change Rate (pixels/second)")
                axes[plot_idx].set_title("Posture Change-Based Sleep Detection")
                axes[plot_idx].grid(True, linestyle=':', alpha=0.6)
                
                # Improve legend uniqueness
                handles, labels = axes[plot_idx].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[plot_idx].legend(by_label.values(), by_label.keys(), loc='upper right')
                
                plot_idx += 1
            
            # 3. Angular velocity plot
            if angular_sleep_bouts_available:
                axes[plot_idx].plot(time_axis, angular_velocity_data, label='Angular Velocity', color='blue', alpha=0.7, lw=1)
                
                if has_angular_bouts:
                    for _, bout in df_angular_sleep_bouts.iterrows():
                        axes[plot_idx].axvspan(
                            bout['start_time_s'], 
                            bout['end_time_s'], 
                            color='salmon', 
                            alpha=0.4, 
                            label='Angular-Based Sleep' if _ == 0 else ""
                        )
                
                axes[plot_idx].axhline(
                    angular_velocity_threshold_deg_per_s, 
                    color='r', 
                    linestyle='--', 
                    lw=1, 
                    label=f'Angular Velocity Threshold ({angular_velocity_threshold_deg_per_s} deg/s)'
                )
                
                axes[plot_idx].set_xlabel("Time (seconds)")
                axes[plot_idx].set_ylabel("Angular Velocity (degrees/second)")
                axes[plot_idx].set_title("Angular Velocity-Based Sleep Detection")
                axes[plot_idx].grid(True, linestyle=':', alpha=0.6)
                
                # Improve legend uniqueness
                handles, labels = axes[plot_idx].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[plot_idx].legend(by_label.values(), by_label.keys(), loc='upper right')
            
            plt.tight_layout()
            
            # Save comparative plot
            comp_plot_filename = os.path.join(output_dir, f"{base_output_name}_all_sleep_methods_comparison.png")
            plt.savefig(comp_plot_filename, dpi=300)
            plt.close()
            
            print(f"Comparative sleep bouts plot saved to: {comp_plot_filename}")
            
        except Exception as e:
            print(f"Error generating comparative sleep plots: {e}")
    
    # --- CALCULATE OVERLAPS BETWEEN METHODS ---    
    # Count how many methods detected sleep
    methods_with_bouts = sum([
        has_speed_bouts, 
        has_posture_bouts,
        has_angular_bouts
    ])
    
    if methods_with_bouts >= 2:
        # Create time-indexed series for each bout type
        time_resolution = 1 / frame_rate
        
        # Find max time
        max_time_list = []
        if has_speed_bouts:
            max_time_list.append(df_speed_sleep_bouts['end_time_s'].max())
        if has_posture_bouts:
            max_time_list.append(df_posture_sleep_bouts['end_time_s'].max())
        if has_angular_bouts:
            max_time_list.append(df_angular_sleep_bouts['end_time_s'].max())
        
        max_time = max(max_time_list)
        time_points = np.arange(0, max_time, time_resolution)
        
        # Initialize series
        speed_bout_series = pd.Series(0, index=time_points) if has_speed_bouts else None
        posture_bout_series = pd.Series(0, index=time_points) if has_posture_bouts else None
        angular_bout_series = pd.Series(0, index=time_points) if has_angular_bouts else None
        
        # Fill in the series
        if has_speed_bouts:
            for _, bout in df_speed_sleep_bouts.iterrows():
                bout_time_points = speed_bout_series[(speed_bout_series.index >= bout['start_time_s']) & 
                                                    (speed_bout_series.index < bout['end_time_s'])].index
                speed_bout_series.loc[bout_time_points] = 1
        
        if has_posture_bouts:
            for _, bout in df_posture_sleep_bouts.iterrows():
                bout_time_points = posture_bout_series[(posture_bout_series.index >= bout['start_time_s']) & 
                                                      (posture_bout_series.index < bout['end_time_s'])].index
                posture_bout_series.loc[bout_time_points] = 1
        
        if has_angular_bouts:
            for _, bout in df_angular_sleep_bouts.iterrows():
                bout_time_points = angular_bout_series[(angular_bout_series.index >= bout['start_time_s']) & 
                                                      (angular_bout_series.index < bout['end_time_s'])].index
                angular_bout_series.loc[bout_time_points] = 1
        
        # Calculate pairwise overlaps
        overlap_results = {}
        
        # Function to calculate overlap between two methods
        def calculate_overlap(series1, series2, name1, name2):
            if series1 is None or series2 is None:
                return None
            
            both = (series1 == 1) & (series2 == 1)
            either = (series1 == 1) | (series2 == 1)
            
            if either.sum() > 0:
                overlap_percent = (both.sum() / either.sum()) * 100
                total_either_time = either.sum() * time_resolution
                total_both_time = both.sum() * time_resolution
                
                print(f"Overlap between {name1} and {name2}: {overlap_percent:.2f}%")
                print(f"  Total time with either: {total_either_time:.2f} seconds")
                print(f"  Time with both: {total_both_time:.2f} seconds")
                
                return {
                    'overlap_percent': overlap_percent,
                    'total_either_time': total_either_time,
                    'total_both_time': total_both_time
                }
            else:
                print(f"No overlap data available for {name1} vs {name2}.")
                return None
        
        # Speed vs. Posture
        if has_speed_bouts and has_posture_bouts:
            overlap_results['speed_vs_posture'] = calculate_overlap(
                speed_bout_series, posture_bout_series, "speed", "posture"
            )
        
        # Speed vs. Angular
        if has_speed_bouts and has_angular_bouts:
            overlap_results['speed_vs_angular'] = calculate_overlap(
                speed_bout_series, angular_bout_series, "speed", "angular"
            )
        
        # Posture vs. Angular
        if has_posture_bouts and has_angular_bouts:
            overlap_results['posture_vs_angular'] = calculate_overlap(
                posture_bout_series, angular_bout_series, "posture", "angular"
            )
        
        # All three methods
        if has_speed_bouts and has_posture_bouts and has_angular_bouts:
            all_three = (speed_bout_series == 1) & (posture_bout_series == 1) & (angular_bout_series == 1)
            any_method = (speed_bout_series == 1) | (posture_bout_series == 1) | (angular_bout_series == 1)
            
            if any_method.sum() > 0:
                all_three_percent = (all_three.sum() / any_method.sum()) * 100
                total_any_time = any_method.sum() * time_resolution
                total_all_three_time = all_three.sum() * time_resolution
                
                print(f"\nOverlap across ALL THREE methods: {all_three_percent:.2f}%")
                print(f"  Total time with any method: {total_any_time:.2f} seconds")
                print(f"  Time with all three methods: {total_all_three_time:.2f} seconds")
                
                overlap_results['all_three'] = {
                    'overlap_percent': all_three_percent,
                    'total_any_time': total_any_time,
                    'total_all_three_time': total_all_three_time
                }
        
        # Store overlap results in df_dlc for reference
        if 'speed_vs_posture' in overlap_results and overlap_results['speed_vs_posture']:
            df_dlc[('analysis', 'sleep_overlap_speed_posture')] = overlap_results['speed_vs_posture']['overlap_percent']
        
        if 'speed_vs_angular' in overlap_results and overlap_results['speed_vs_angular']:
            df_dlc[('analysis', 'sleep_overlap_speed_angular')] = overlap_results['speed_vs_angular']['overlap_percent']
        
        if 'posture_vs_angular' in overlap_results and overlap_results['posture_vs_angular']:
            df_dlc[('analysis', 'sleep_overlap_posture_angular')] = overlap_results['posture_vs_angular']['overlap_percent']
        
        if 'all_three' in overlap_results:
            df_dlc[('analysis', 'sleep_overlap_all_three')] = overlap_results['all_three']['overlap_percent']
        
        # Add overlap data to the return dictionary
        sleep_bouts_dict['overlap'] = overlap_results
    else:
        print(f"Only {methods_with_bouts} method(s) detected sleep bouts. Need at least 2 for overlap analysis.")
    
    print("\nSleep bout analysis complete.")
    return df_dlc, sleep_bouts_dict

def generate_sleep_analysis_video(df_dlc, df_midpoints_pca_raw, sleep_bouts_dict, 
                                  smoothed_speed, body_movement_derivative, angular_velocity,
                                  smoothed_angle, frame_rate, video_file, output_dir, file_name, 
                                  bodypart_coordinate_sets, thresholds=None):
    """
    Generate a comprehensive sleep analysis video with synchronized plots of multiple metrics.
    Returns:
        str: Path to the created video file, or None if video creation failed
    """
    print("\nGenerating comprehensive sleep analysis video...")

    # Check if video file exists
    if not os.path.isfile(video_file):
        print(f"Error: Video file not found at {video_file}")
        return None
    
    # Prepare output path
    base_name_without_ext = os.path.splitext(file_name)[0]
    video_output_filename = f"{base_name_without_ext}_sleep_analysis.mp4"
    output_video_path = os.path.join(output_dir, video_output_filename)
    
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = {
            'speed': 60.0,      # pixels per second
            'posture': 60.0,    # pixels per second
            'angular': 50.0     # degrees per second
        }
    
    # Get sleep bouts from the dictionary
    df_speed_sleep_bouts = sleep_bouts_dict.get('speed', pd.DataFrame())
    df_posture_sleep_bouts = sleep_bouts_dict.get('posture', pd.DataFrame()) 
    df_angular_sleep_bouts = sleep_bouts_dict.get('angular', pd.DataFrame())
    
    # Check for available sleep bout data
    has_speed_bouts = not df_speed_sleep_bouts.empty
    has_posture_bouts = not df_posture_sleep_bouts.empty
    has_angular_bouts = not df_angular_sleep_bouts.empty
    
    print("Available sleep bout data:")
    print(f"- Speed-based: {'Yes' if has_speed_bouts else 'No'}")
    print(f"- Posture-based: {'Yes' if has_posture_bouts else 'No'}")
    print(f"- Angular velocity-based: {'Yes' if has_angular_bouts else 'No'}")
    
    # Generate the video
    try:
        # Import here to allow the function to be used without cv2 if not generating video
        from plotting_utils import create_comprehensive_sleep_analysis_video
        
        create_comprehensive_sleep_analysis_video(
            video_path=video_file,
            speed_data=smoothed_speed,
            body_movement_derivative=body_movement_derivative,
            angular_velocity_data=angular_velocity,
            smoothed_angle_data=smoothed_angle,
            frame_rate=frame_rate,
            output_video_path=output_video_path,
            df_speed_sleep_bouts=df_speed_sleep_bouts,
            df_midpoints_pca_raw=df_midpoints_pca_raw,
            df_posture_sleep_bouts=df_posture_sleep_bouts,
            df_angular_sleep_bouts=df_angular_sleep_bouts,
            median_coords=df_dlc[[('analysis', 'median_x'), ('analysis', 'median_y')]],
            bodypart_coordinate_sets=bodypart_coordinate_sets,
            plot_width_seconds=5.0,
            plot_height_pixels=150,
            median_point_radius=5,
            median_point_color=(255, 0, 0),  # Red in BGR
            arrow_color=(130, 0, 130),       # Purple in BGR
            arrow_length=40,
            arrow_size=10,
            arrow_smoothing_frames=15,
            speed_threshold=thresholds['speed'],
            posture_threshold=thresholds['posture'],
            angular_threshold=thresholds['angular'],
        )
        
        print(f"Comprehensive sleep analysis video created at: {output_video_path}")
        return output_video_path
        
    except Exception as e:
        print(f"Error creating comprehensive sleep analysis video: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def main(input_dir, params=None):
    """
    Run the complete sleep analysis pipeline on a single directory.
    
    Args:
        input_dir: Path to the directory containing the DLC CSV file and video
        params: Optional dictionary of analysis parameters
    
    Returns:
        bool: True if analysis completed successfully, False otherwise
    """
    print(f"\n{'='*80}\nProcessing directory: {input_dir}\n{'='*80}")
    
    try:
        # Set default parameters
        default_params = {
            'frame_rate': 60,
            'likelihood_threshold': 0.95,
            'sleep_speed_threshold': 60.0,
            'posture_change_threshold': 60.0,
            'angular_velocity_threshold': 50.0,
            'min_sleep_duration_seconds': 10,
            'save_plots': True
        }
        
        # Override with user-provided parameters if any
        if params:
            default_params.update(params)
            
        # Use the updated parameters
        params = default_params
        print("Running with parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
            
        # 1. Setup output directory
        output_dir = setup_directories(input_dir)
        print(f'Output directory: {output_dir}')
        
        # 2. Find input files
        csv_file, video_file = find_files(input_dir)
        if csv_file is None:
            print("Error: No CSV file found. Skipping directory.")
            return False
            
        print(f"Found CSV file: {csv_file}")
        if video_file:
            print(f"Found video file: {video_file}")
        else:
            print("No video file found")
        
        # 3. Load DLC data
        df_dlc = load_dlc_data(csv_file)
        print(f"DLC data loaded: {df_dlc.shape[0]} rows, {len(df_dlc.columns)} columns")
        
        # 4. Get default bodyparts list
        default_bodyparts = get_default_bodyparts()
        
        # 5. Use select_available_bodyparts to find which bodyparts are available
        final_bodyparts_list = select_available_bodyparts(df_dlc, default_bodyparts)
        print(f"Final bodyparts list for analysis: {final_bodyparts_list}")
        
        # 6. Process coordinates
        df_dlc, filtered_x_coords, filtered_y_coords = process_bodypart_coordinates(
            df_dlc, final_bodyparts_list, params['likelihood_threshold']
        )
        print("Coordinate processing complete")
        
        # 7. Calculate speed
        file_name = os.path.basename(csv_file)
        df_dlc = calculate_speed(
            df_dlc, params['frame_rate'], output_dir, file_name, 
            final_bodyparts_list, save_plots=params['save_plots']
        )
        print("Speed calculation complete")
        
        # 8. Calculate body posture metric
        df_dlc = calculate_body_posture_metric(
            df_dlc, filtered_x_coords, filtered_y_coords, 
            final_bodyparts_list, output_dir, file_name, 
            save_plots=params['save_plots']
        )
        print("Body posture analysis complete")
        
        # 9. Calculate body movement derivative
        df_dlc = calculate_body_movement_derivative(
            df_dlc, params['frame_rate'], output_dir, file_name, 
            save_plots=params['save_plots']
        )
        print("Body movement derivative analysis complete")
        
        # 10. Prepare coordinates for body axis calculation
        bodypart_coordinate_sets = prepare_body_axis_coordinates(
            df_dlc, filtered_x_coords, filtered_y_coords, final_bodyparts_list
        )
        print("Body axis coordinate preparation complete")
        
        # 11. Calculate body axis PCA
        df_dlc, df_midpoints_pca_raw = calculate_body_axis_pca(df_dlc, bodypart_coordinate_sets)
        print("Body axis PCA calculation complete")
        
        # 12. Calculate the body axis angles and create polar plot
        df_dlc = calculate_and_plot_body_axis_angles(
            df_dlc, df_midpoints_pca_raw, filtered_x_coords, filtered_y_coords,
            params['frame_rate'], output_dir, file_name, save_plots=params['save_plots']
        )
        print("Body axis angle analysis complete")
        
        # 13. Calculate and plot smoothed body axis angles
        df_dlc = plot_smoothed_body_axis_angles(
            df_dlc, df_midpoints_pca_raw, params['frame_rate'], 
            output_dir, file_name, save_plots=params['save_plots']
        )
        print("Smoothed body axis angle analysis complete")
        
        # 14. Calculate angular velocity from smoothed body axis angles
        df_dlc = calculate_angular_velocity(
            df_dlc, df_midpoints_pca_raw, params['frame_rate'], 
            output_dir, file_name, save_plots=params['save_plots']
        )
        print("Angular velocity analysis complete")
        
        # 15. Identify sleep bouts across all metrics
        df_dlc, sleep_bouts_dict = identify_sleep_bouts_and_plot(
            df_dlc, df_midpoints_pca_raw, params['frame_rate'], 
            output_dir, file_name, save_plots=params['save_plots']
        )
        print("Sleep analysis complete")
        
        # 16. Generate comprehensive video if available
        if video_file:
            video_path = generate_sleep_analysis_video(
                df_dlc=df_dlc,
                df_midpoints_pca_raw=df_midpoints_pca_raw,
                sleep_bouts_dict=sleep_bouts_dict,
                smoothed_speed=df_dlc[('analysis', 'speed_smoothed')],
                body_movement_derivative=df_dlc[('analysis', 'posture_metric_abs_derivative')],
                angular_velocity=df_dlc[('analysis', 'absolute_angular_velocity')],
                smoothed_angle=df_midpoints_pca_raw['angle_y_deg_midpoints_pca_smoothed'],
                frame_rate=params['frame_rate'],
                video_file=video_file,
                output_dir=output_dir,
                file_name=file_name,
                bodypart_coordinate_sets=bodypart_coordinate_sets,
                thresholds={
                    'speed': params['sleep_speed_threshold'],
                    'posture': params['posture_change_threshold'],
                    'angular': params['angular_velocity_threshold']
                }
            )
            
            if video_path:
                print("Video generation complete.")
            else:
                print("Video generation failed but continuing with analysis.")
        else:
            print("No video file found for analysis. Skipping video generation.")
        
        # 17. Save the final results
        final_output_csv = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_final_results.csv")
        df_dlc.to_csv(final_output_csv)
        print(f"Final results saved to: {final_output_csv}")
        
        print(f"\n{'='*80}\nAnalysis of {input_dir} completed successfully\n{'='*80}")
        return True
        
    except Exception as e:
        print(f"\nERROR: Analysis failed for {input_dir}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def parse_config_file(config_file):
    """
    Parse a JSON configuration file containing parameters for each directory.
    
    Args:
        config_file: Path to the JSON configuration file
        
    Returns:
        dict: Dictionary mapping directory paths to parameter dictionaries
    """
    import json
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate the config format
        if not isinstance(config, dict):
            print(f"Error: Config file should contain a JSON object/dictionary")
            return {}
            
        # Convert all paths to absolute paths if they're relative
        config_with_abs_paths = {}
        for dir_path, params in config.items():
            abs_path = os.path.abspath(dir_path)
            config_with_abs_paths[abs_path] = params
            
        return config_with_abs_paths
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file: {e}")
        return {}
    except Exception as e:
        print(f"Error reading config file: {e}")
        return {}


def batch_process(input_config, pause_seconds=2):
    """
    Process multiple directories with individual parameters based on a config file.
    
    Args:
        input_config: Either a list of directories (using default parameters) or
                     a path to a JSON config file with directory-specific parameters
        pause_seconds: Number of seconds to pause between directories
    
    Returns:
        tuple: (list of successful directories, list of failed directories)
    """
    import time
    
    successful = []
    failed = []
    directory_params = {}
    
    # Check if input_config is a config file path or a list of directories
    if isinstance(input_config, str) and os.path.isfile(input_config):
        # It's a file, check if it's a JSON config file or a directory list
        if input_config.lower().endswith('.json'):
            # Parse the JSON config file
            directory_params = parse_config_file(input_config)
            if not directory_params:
                print("Error parsing config file or empty config. Aborting.")
                return [], []
        else:
            # Assume it's a simple list of directories
            with open(input_config, 'r') as f:
                directories = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            # Use default parameters for all directories
            directory_params = {os.path.abspath(d): None for d in directories}
    elif isinstance(input_config, list):
        # It's already a list of directories
        directory_params = {os.path.abspath(d): None for d in input_config}
    else:
        print(f"Error: Invalid input_config format. Expected a file path or list of directories.")
        return [], []
    
    total_dirs = len(directory_params)
    print(f"\nBatch processing {total_dirs} directories...")
    
    for i, (directory, params) in enumerate(directory_params.items()):
        print(f"\nProcessing directory {i+1} of {total_dirs}: {directory}")
        try:
            # Ensure the directory exists
            if not os.path.isdir(directory):
                print(f"Directory does not exist: {directory}")
                failed.append(directory)
                continue
                
            success = main(directory, params)
            if success:
                successful.append(directory)
            else:
                failed.append(directory)
                
            # Pause between directories to let any file operations complete
            if i < total_dirs - 1 and pause_seconds > 0:
                print(f"Pausing for {pause_seconds} seconds before next directory...")
                time.sleep(pause_seconds)
                
        except Exception as e:
            print(f"Unexpected error processing directory {directory}: {str(e)}")
            failed.append(directory)
    
    # Print summary
    print("\n" + "="*80)
    print(f"BATCH PROCESSING COMPLETE: {len(successful)}/{total_dirs} directories processed successfully")
    
    if failed:
        print(f"\nFailed directories ({len(failed)}):")
        for dir_path in failed:
            print(f"  - {dir_path}")
    
    return successful, failed


if __name__ == "__main__":
    import argparse
    import time
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run sleep analysis pipeline on DLC data')
    
    # Required arguments
    parser.add_argument('input', help='Input directory, text file with directory list, or JSON config file')
    
    # Optional arguments
    parser.add_argument('--frame-rate', type=int, default=60, help='Video frame rate (default: 60)')
    parser.add_argument('--likelihood-threshold', type=float, default=0.95, help='DLC likelihood threshold (default: 0.95)')
    parser.add_argument('--speed-threshold', type=float, default=60.0, help='Sleep speed threshold (default: 60.0 px/s)')
    parser.add_argument('--posture-threshold', type=float, default=60.0, help='Posture change threshold (default: 60.0 px/s)')
    parser.add_argument('--angular-threshold', type=float, default=50.0, help='Angular velocity threshold (default: 50.0 deg/s)')
    parser.add_argument('--min-sleep', type=float, default=10.0, help='Minimum sleep bout duration (default: 10.0 s)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    parser.add_argument('--pause', type=int, default=2, help='Seconds to pause between batch directories (default: 2)')
    
    args = parser.parse_args()
    
    # Prepare default parameters (used for single directories or when config doesn't specify)
    params = {
        'frame_rate': args.frame_rate,
        'likelihood_threshold': args.likelihood_threshold,
        'sleep_speed_threshold': args.speed_threshold,
        'posture_change_threshold': args.posture_threshold,
        'angular_velocity_threshold': args.angular_threshold,
        'min_sleep_duration_seconds': args.min_sleep,
        'save_plots': not args.no_plots
    }
    
    start_time = time.time()
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        # Single directory mode
        success = main(args.input, params)
        if success:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed.")
            sys.exit(1)
    elif os.path.isfile(args.input):
        # Batch mode - could be a list of directories or a JSON config
        successful, failed = batch_process(args.input, args.pause)
        
        if failed:
            print("\nSome directories failed. Check the output for details.")
            sys.exit(1)
        else:
            print("\nAll directories processed successfully!")
    else:
        print(f"Error: Input '{args.input}' is neither a directory nor a valid file.")
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal processing time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")