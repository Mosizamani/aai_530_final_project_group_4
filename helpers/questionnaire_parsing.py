import pandas as pd
import numpy as np

def clean_split_list(s: str) -> list[str]:
    """
    Clean and split a semicolon-separated string from WESAD questionnaire format.
    
    Removes hash symbols (#) and splits on semicolons, filtering out empty strings.
    This is used to parse the structured questionnaire CSV format where entries
    are prefixed with '#' and separated by ';'.
    
    Args:
        s: Semicolon-separated string from questionnaire (e.g., '# Order; Event1; Event2;')
        
    Returns:
        List of cleaned string parts with whitespace trimmed and empty entries removed
    """
    parts = s.replace('#', '').split(';')
    return [p.strip() for p in parts if p.strip()]

def parse_event_order(order_string: str) -> list[str]:
    """
    Extract event names from the questionnaire order string.
    
    The order string contains the sequence of experimental events (e.g., baseline,
    stress tasks, relaxation). The first element is typically a label ('Order'),
    so we skip it and return the actual event names.
    
    Args:
        order_string: Questionnaire row containing event order
                     (e.g., '# Order; Baseline; TSST; Meditation;')
        
    Returns:
        List of event names in the order they occurred during the experiment
    """
    return clean_split_list(order_string)[1:]

def parse_event_times(time_string: str, event_names: list[str]) -> dict[str, float | None]:
    """
    Parse start or end times from questionnaire and map to event names.
    
    Times are stored in seconds from the start of recording. The first element
    in the time string is a label ('Start time' or 'End time'), so we skip it
    and pair the remaining times with their corresponding event names.
    
    Args:
        time_string: Questionnaire row containing timestamps
                    (e.g., '# Start time; 0.0; 120.5; 480.2;')
        event_names: List of event names to associate with the timestamps
        
    Returns:
        Dictionary mapping event names to their timestamps (in seconds).
        Returns None for events with missing or invalid timestamps.
    """
    time_parts = clean_split_list(time_string)
    times: dict[str, float | None] = {}
    for i, event in enumerate(event_names):
        if (i + 1) < len(time_parts):
            try:
                times[event] = float(time_parts[i + 1])
            except (ValueError, IndexError):
                times[event] = None
        else:
            times[event] = None
    return times

def parse_event_timings(df_quest: pd.DataFrame) -> pd.DataFrame:
    """
    Parse event timings from WESAD questionnaire DataFrame.
    
    The WESAD questionnaire CSV has a structured format where:
    - Row 0: Event order/names
    - Row 1: Start times (seconds from recording start)
    - Row 2: End times (seconds from recording start)
    
    This function extracts these three rows and creates a structured DataFrame
    with event names, start/end times, and calculated durations.
    
    Args:
        df_quest: DataFrame loaded from subject's questionnaire CSV file
        
    Returns:
        DataFrame with columns: Event, Start_Time, End_Time, Duration (all in seconds)
    """
    order_str = str(df_quest.iloc[0, 0])
    start_str = str(df_quest.iloc[1, 0])
    end_str = str(df_quest.iloc[2, 0])
    
    event_names = parse_event_order(order_str)
    start_times = parse_event_times(start_str, event_names)
    end_times = parse_event_times(end_str, event_names)
    
    df_event_timings = pd.DataFrame({
        'Event': event_names,
        'Start_Time': [start_times.get(e) for e in event_names],
        'End_Time': [end_times.get(e) for e in event_names]
    })
    df_event_timings['Duration'] = df_event_timings['End_Time'] - df_event_timings['Start_Time']
    
    return df_event_timings

def parse_scale_data(df_quest: pd.DataFrame, scale_name: str, start_row: int = 4) -> pd.DataFrame:
    """
    Parse data for a specific psychological questionnaire scale.
    
    WESAD includes several validated psychological questionnaires:
    - PANAS: Positive and Negative Affect Schedule (mood assessment)
    - STAI: State-Trait Anxiety Inventory (anxiety levels)
    - DIM: Dimensional questionnaire (emotional dimensions)
    - SSSQ: Short Stress State Questionnaire (stress perception)
    
    Each scale has multiple items/questions, with responses recorded at
    different time points during the experiment (pre/post conditions).
    
    Args:
        df_quest: DataFrame loaded from subject's questionnaire CSV file
        scale_name: Name of the psychological scale to parse (PANAS, STAI, DIM, or SSSQ)
        start_row: Row index to start searching for scale data (default: 4)
        
    Returns:
        DataFrame where each row is a time point and columns are scale items/questions
    """
    scale_data: list[list[float]] = []
    prefix = f'# {scale_name};'
    
    for i in range(start_row, len(df_quest)):
        row_string = str(df_quest.iloc[i, 0])
        if row_string.startswith(f'# {scale_name}'):
            parts = row_string.replace(prefix, '').split(';')
            try:
                scale_data.append([float(p.strip()) for p in parts if p.strip()])
            except ValueError:
                continue
    
    return pd.DataFrame(scale_data)

def combine_questionnaire_scales(df_panas: pd.DataFrame, df_stai: pd.DataFrame, 
                                df_dim: pd.DataFrame, df_sssq: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all questionnaire scales into a single DataFrame.
    
    Merges PANAS, STAI, DIM, and SSSQ responses into one unified DataFrame.
    PANAS, STAI, and DIM typically have multiple time point measurements,
    while SSSQ often has only one measurement. This function handles the
    differing lengths by extending SSSQ with NaN values to match the others.
    
    Args:
        df_panas: DataFrame with PANAS (Positive/Negative Affect) responses
        df_stai: DataFrame with STAI (State-Trait Anxiety) responses
        df_dim: DataFrame with DIM (Dimensional emotion) responses
        df_sssq: DataFrame with SSSQ (Short Stress State) responses
        
    Returns:
        DataFrame containing all scales with prefixed column names
        (e.g., PANAS_0, STAI_0, etc.)
    """
    # Add column names
    df_panas.columns = [f'PANAS_{i}' for i in range(len(df_panas.columns))]
    df_stai.columns = [f'STAI_{i}' for i in range(len(df_stai.columns))]
    df_dim.columns = [f'DIM_{i}' for i in range(len(df_dim.columns))]
    df_sssq.columns = [f'SSSQ_{i}' for i in range(len(df_sssq.columns))]
    
    # Combine PANAS, STAI, DIM
    df_combined = pd.concat([df_panas, df_stai, df_dim], axis=1)
    
    # Extend SSSQ to match combined length (SSSQ typically has only 1 row)
    df_sssq_ext = pd.DataFrame(np.nan, index=range(len(df_combined)), columns=df_sssq.columns)
    if not df_sssq.empty:
        df_sssq_ext.iloc[0] = df_sssq.iloc[0]
    
    return pd.concat([df_combined, df_sssq_ext], axis=1)

def parse_questionnaire_responses(df_quest: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all psychological questionnaire responses from WESAD questionnaire file.
    
    This is the main function for extracting questionnaire data. It parses all four
    psychological assessment scales (PANAS, STAI, DIM, SSSQ) and combines them into
    a single DataFrame for analysis. These questionnaires capture subjective
    psychological states before, during, and after stress exposure.
    
    Args:
        df_quest: DataFrame loaded from subject's _quest.csv file
        
    Returns:
        Combined DataFrame with all questionnaire responses, where each row
        represents a measurement time point and columns are individual scale items
    """
    df_panas = parse_scale_data(df_quest, 'PANAS')
    df_stai = parse_scale_data(df_quest, 'STAI')
    df_dim = parse_scale_data(df_quest, 'DIM')
    df_sssq = parse_scale_data(df_quest, 'SSSQ')
    
    return combine_questionnaire_scales(df_panas, df_stai, df_dim, df_sssq)
