import os
import pandas as pd
import numpy as np
from typing import Any

def get_sensor_mapping() -> dict[str, tuple[str, str]]:
    """
    Get the sensor mapping for merging sensor data.
    Maps output filename to (sensor_type, sensor_key) tuple.
    """
    return {
        'chest_acc': ('chest', 'ACC'),
        'chest_ecg': ('chest', 'ECG'),
        'chest_emg': ('chest', 'EMG'),
        'chest_eda': ('chest', 'EDA'),
        'chest_temp': ('chest', 'Temp'),
        'chest_resp': ('chest', 'Resp'),
        'wrist_acc': ('wrist', 'ACC'),
        'wrist_bvp': ('wrist', 'BVP'),
        'wrist_eda': ('wrist', 'EDA'),
        'wrist_temp': ('wrist', 'TEMP')
    }

def add_subject_column(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """Add subject column to a DataFrame."""
    df_copy = df.copy()
    df_copy['subject_id'] = subject_id
    return df_copy

def merge_event_timings(all_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Merge event timings from all subjects into a single DataFrame.
    
    Args:
        all_results: List of dicts
        
    Returns:
        DataFrame with merged event timings including subject column
    """
    merged_event_timings: list[pd.DataFrame] = []
    
    for result in all_results:
        df_with_subject = add_subject_column(result['event_timings'], result['subject_id'])
        merged_event_timings.append(df_with_subject)
    
    return pd.concat(merged_event_timings, ignore_index=True)

def merge_questionnaire_responses(all_results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Merge questionnaire responses from all subjects into a single DataFrame.
    
    Args:
        all_results: List of dicts
        
    Returns:
        DataFrame with merged questionnaire responses including subject column
    """
    merged_questionnaire: list[pd.DataFrame] = []
    
    for result in all_results:
        df_with_subject = add_subject_column(result['questionnaire_responses'], result['subject_id'])
        merged_questionnaire.append(df_with_subject)
    
    return pd.concat(merged_questionnaire, ignore_index=True)

def merge_sensor_data(all_results: list[dict[str, Any]], 
                     sensor_map: dict[str, tuple[str, str]] | None = None) -> dict[str, pd.DataFrame]:
    """
    Merge sensor data from all subjects for each sensor type.
    
    Args:
        all_results: List of dicts
        sensor_map: Optional dict mapping filenames to (sensor_type, sensor_key) tuples.
                   If None, uses default mapping from get_sensor_mapping()
        
    Returns:
        Dict mapping sensor filename to merged DataFrame
    """
    if sensor_map is None:
        sensor_map = get_sensor_mapping()
    
    merged_sensors: dict[str, pd.DataFrame] = {}
    
    for fname, (sensor_type, sensor_key) in sensor_map.items():
        parts: list[pd.DataFrame] = []
        
        for result in all_results:
            # Get the appropriate dataframes dict (chest or wrist)
            dataframes = result['chest'] if sensor_type == 'chest' else result['wrist']
            
            # Check if this sensor exists for this subject
            if sensor_key in dataframes:
                df_with_subject = add_subject_column(dataframes[sensor_key], result['subject_id'])
                parts.append(df_with_subject)
        
        # Only add to merged dict if we have data
        if parts:
            merged_sensors[fname] = pd.concat(parts, ignore_index=True)
    
    return merged_sensors

def create_merged_dataset(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Create a complete merged dataset from all subject results.
    
    Args:
        all_results: List of dicts
        
    Returns:
        Dict containing:
            - 'event_timings': Merged event timings DataFrame
            - 'questionnaire_responses': Merged questionnaire DataFrame
            - 'sensors': Dict of merged sensor DataFrames
    """
    return {
        'event_timings': merge_event_timings(all_results),
        'questionnaire_responses': merge_questionnaire_responses(all_results),
        'sensors': merge_sensor_data(all_results)
    }

def save_merged_dataset(merged_data: dict[str, Any], output_dir: str) -> None:
    """
    Save merged dataset to CSV files.
    
    Args:
        merged_data: Dict from create_merged_dataset()
        output_dir: Directory to save merged files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save event timings
    event_path = os.path.join(output_dir, 'merged_event_timings.csv')
    merged_data['event_timings'].to_csv(event_path, index=False)
    print(f"Saved merged_event_timings.csv ({len(merged_data['event_timings'])} rows)")
    
    # Save questionnaire responses
    quest_path = os.path.join(output_dir, 'merged_questionnaire_responses.csv')
    merged_data['questionnaire_responses'].to_csv(quest_path, index=False)
    print(f"Saved merged_questionnaire_responses.csv ({len(merged_data['questionnaire_responses'])} rows)")
    
    # Save sensor data
    for fname, df in merged_data['sensors'].items():
        sensor_path = os.path.join(output_dir, f'merged_{fname}.csv')
        df.to_csv(sensor_path, index=False)
        print(f"Saved merged_{fname}.csv ({len(df)} rows)")

def upsample_to_64hz(subject_id: str, df_target_sub: pd.DataFrame, low_freq_signal: dict[str, Any]) -> pd.DataFrame:
    """
    Aligns low-freq EDA (4Hz) to high-freq BVP (64Hz) for a single subject.
    """

    # Sampling Rates (Fixed by Empatica E4 hardware)
    FS_TARGET = 64.0  # Hz
    FS_EDA_TEMP = 4.0   # Hz
    FS_ACC = 32.0  # Hz

    # 1. Generate Relative Time Indices (Seconds since start)
    # -----------------------------------------------------
    # BVP Time: 0, 0.015625, 0.03125...
    n_bvp = len(df_target_sub)
    time_target = np.arange(n_bvp) / FS_TARGET
    target_series = pd.Series(df_target_sub['BVP'].values, index=time_target)

    aligned_df = pd.DataFrame({
        'subject_id': subject_id,  # Scalar - will be broadcast
        'time_sec': target_series.index,  # Array - defines the number of rows
        'bvp': target_series.values  # Array - target signal values
    })
    
    for sensor_key in low_freq_signal:
        # EDA Time: 0, 0.25, 0.50...
        n_eda = len(low_freq_signal[sensor_key])

        if sensor_key == 'ACC':
            time_acc = np.arange(n_eda) / FS_ACC

            # Handle multi-axis data (e.g., ACC has x, y, z)
            for axis in low_freq_signal[sensor_key].columns:
                axis_key = f"{sensor_key}_{axis}"
                low_series_axis = pd.Series(low_freq_signal[sensor_key][axis].values, index=time_acc)
                low_upsampled_axis = low_series_axis.reindex(target_series.index).interpolate(method='linear').ffill().bfill()
                aligned_df[axis_key.lower()] = low_upsampled_axis.values
            continue
        elif sensor_key == 'BVP':
            continue  # Skip BVP since it's the target signal and already included

        # Non-ACC signals are single-channel
        signal_data = low_freq_signal[sensor_key]
        if isinstance(signal_data, pd.DataFrame):
            signal_data = signal_data.iloc[:, 0]
        
        time_eda_temp = np.arange(n_eda) / FS_EDA_TEMP

        low_series = pd.Series(np.asarray(signal_data).reshape(-1), index=time_eda_temp)

        low_upsampled = low_series.reindex(target_series.index).interpolate(method='linear').ffill().bfill()
        aligned_df[sensor_key.lower()] = low_upsampled.values
    
    return aligned_df

def downsample_to_64hz(subject_id: str, df_target_sub: pd.DataFrame, high_freq_signal: dict[str, Any]) -> pd.DataFrame:

    # Sampling Rates
    FS_TARGET = 64.0     # Hz (target frequency - wrist BVP)
    FS_HIGH = 700.0  # Hz (source frequency - e.g., chest ECG at 700Hz)

    # 1. Generate Relative Time Indices (Seconds since start)
    # -----------------------------------------------------
    # Target Time: 0, 0.015625, 0.03125... (64Hz)
    n_target = len(df_target_sub)
    time_target = np.arange(n_target) / FS_TARGET
    target_series = pd.Series(df_target_sub['BVP'].values, index=time_target)

    aligned_df = pd.DataFrame({
        'subject_id': subject_id,  # Scalar - will be broadcast
        'time_sec': target_series.index  # Array - defines the number of rows
    })

    for sensor_key in high_freq_signal:

        # High-Freq Time: 0, 0.00142857, 0.00285714... (700Hz)
        n_high = len(high_freq_signal[sensor_key])
        time_high = np.arange(n_high) / FS_HIGH

        if sensor_key == 'ACC':
            # Handle multi-axis data (e.g., ACC has x, y, z)
            for axis in high_freq_signal[sensor_key].columns:
                axis_key = f"{sensor_key}_{axis}"
                high_series_axis = pd.Series(high_freq_signal[sensor_key][axis].values, index=time_high)
                high_downsampled_axis = high_series_axis.reindex(target_series.index).interpolate(method='linear').ffill().bfill()
                aligned_df[axis_key.lower()] = high_downsampled_axis.values
            continue

        # Non-ACC signals are single-channel
        signal_data = high_freq_signal[sensor_key]
        if isinstance(signal_data, pd.DataFrame):
            signal_data = signal_data.iloc[:, 0]

        high_series = pd.Series(np.asarray(signal_data).reshape(-1), index=time_high)
        high_downsampled = high_series.reindex(target_series.index).interpolate(method='linear').ffill().bfill()
        aligned_df[sensor_key.lower()] = high_downsampled.values
    
    return aligned_df


