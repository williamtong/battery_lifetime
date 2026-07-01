import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.fft import fft, fftfreq


def group_values(df, groupby_col, how, column_name):
    '''
    Aggregates raw per-measurement battery data into per-cycle summary statistics.
    For each cycle, computes the mean, min, or max of the specified column.
    Column-specific post-processing is applied where needed (see inline comments).

    Input:
    df (pandas DataFrame):  Raw battery measurement data with a 'Cycle_Index' column.
    groupby_col (str):      Column to aggregate. One of: 'Charge_Capacity', 'DateTime',
                            'Internal_Resistance', 'Temperature'.
    how (str):              Aggregation method: 'mean', 'min', or 'max'.
    column_name (str):      Name to assign to the output column.

    Output:
    df_group (pandas DataFrame): Per-cycle aggregated values, with index labels of the
                                 form '{cycle_number}_{groupby_col}'.
    '''
    df_group = df.groupby("Cycle_Index")
    if how == 'mean':
        df_groupby = df_group.mean()[groupby_col]
    elif how == 'min':
        df_groupby = df_group.min()[groupby_col]
    elif how == 'max':
        df_groupby = df_group.max()[groupby_col]
    else:
        raise ValueError(f"'how' parameter '{how}' not recognized. Must be 'mean', 'min', or 'max'.")

    if groupby_col == 'Charge_Capacity':
        # Normalize by the first cycle's value so all batteries start at 1.0,
        # making relative capacity degradation comparable across batteries.
        df_groupby = df_groupby/df_groupby.iloc[0]

    if groupby_col == 'DateTime':
        # Convert absolute timestamps to per-cycle durations in minutes.
        df_groupby = (df_groupby - df_groupby.iloc[0])
        df_groupby = df_groupby.diff()/60  # units = minutes
        # Drop the first two rows: row 0 is NaN after diff(), row 1 is unreliable.
        df_groupby = df_groupby.iloc[2:]

    if groupby_col == 'Internal_Resistance':
        # Drop the first two rows which are unreliable during initial conditioning.
        df_groupby = df_groupby.iloc[2:]

    if groupby_col == 'Temperature':
        # Drop the first two rows which are unreliable during initial conditioning.
        df_groupby = df_groupby.iloc[2:]

    df_group = pd.DataFrame(data=df_groupby.values,
                            columns=[column_name],
                            index=[str(int(idx)) + '_' + groupby_col for idx in df_groupby.index])
    return df_group


def replace_DateTime_CycleTime(dataframes):
    '''
    Renames index labels containing "DateTime" to "CycleTime" across a list of DataFrames.
    Called after group_values() to give the aggregated cycle duration feature a more
    descriptive name.

    Input:
    dataframes (list of pandas DataFrames): DataFrames whose index labels may contain "DateTime".

    Output:
    data_frames_renamed (list of pandas DataFrames): Same DataFrames with updated index labels.
    '''
    data_frames_renamed = []
    for dataframe in dataframes:
        indices = dataframe.index
        new_indices = [indx.replace('DateTime', 'CycleTime') for indx in indices]
        dataframe.index = new_indices
        data_frames_renamed.append(dataframe)
    return data_frames_renamed


def strong_low_pass_filter(df, filter_value=0.0):
    '''
    Applies a custom spike-removal filter to a time series DataFrame.
    If a data point differs from the previous one by more than filter_value
    (as a fractional change), it is replaced by the previous value.
    This suppresses measurement spikes without smoothing genuine trends.

    Input:
    df (pandas DataFrame):  Time series data, shape (n_cycles, n_batteries).
    filter_value (float):   Maximum allowed fractional change between consecutive
                            values. E.g. 0.05 allows up to 5% change per step.
                            Default 0.0 replaces any change at all (effectively
                            forward-fills the entire series).

    Output:
    df (pandas DataFrame):  Filtered DataFrame (modified in place and returned).
    '''
    print(f"Processing {'_'.join(df.index[0].split('_')[1:])}")
    columns = range(df.shape[1])
    rows = range(1, df.shape[0])  # Skip row 0 — no previous value to compare against.
    for column in columns:
        for row in rows:
            if np.abs(1 - (df.iloc[row, column]/df.iloc[row-1, column])) > filter_value:
                # Replace spike with the previous value (forward fill).
                df.iloc[row, column] = df.iloc[row-1, column]
    return df


def plot_Current(df_file):
    '''
    Plots the charge/discharge current waveform for cycles 2 through 6.
    Used for visual inspection of the charging protocol shape.

    Input:
    df_file (pandas DataFrame): Raw battery data with 'Cycle_Index', 'DateTime',
                                and 'Current' columns.
    '''
    df_file = df_file[(df_file['Cycle_Index'] >= 2) & (df_file['Cycle_Index'] < 7)]
    df_file["Time_elapsed"] = (df_file["DateTime"] - df_file["DateTime"].min())/3600
    plt.figure(figsize=(15, 5))
    plt.plot(df_file["Time_elapsed"], df_file["Current"])
    plt.xlabel("Time (hour)")
    plt.xticks(np.arange(0, 5))
    plt.ylabel("Current (A)")
    plt.grid()
    plt.show()


def create_Peak_Areas_df(PSD, dt, filename, harmonics=4, peakhalfwidth=5, feature_name='CC'):
    '''
    Extracts frequency-domain features from a Power Spectral Density (PSD) series.
    Specifically:
      1. Finds the fundamental frequency (first harmonic) of the charging waveform.
      2. Computes the integrated peak area around each of the first n harmonics.
      3. Returns these as a one-row DataFrame indexed by battery filename.

    The peak half-width is set adaptively as 1/5 of the fundamental frequency index,
    so that wider peaks (lower fundamental frequency) get wider integration windows.

    These PSD features were found to be the most predictive of battery lifetime in
    this dataset, consistent with the finding that simpler (single-step) charging
    protocols — which lack higher harmonics — tend to produce longer-lived batteries.

    Input:
    PSD (numpy array):      Power Spectral Density of the charging current or voltage.
    dt (float):             Mean time interval between measurements (seconds).
    filename (str):         Battery data file path; used to label the output row.
    harmonics (int):        Number of harmonics to extract peak areas for.
    peakhalfwidth (int):    Initial half-width for peak integration (overridden adaptively).
    feature_name (str):     Prefix for output column names ('CC' for current, 'CV' for voltage).

    Output:
    df_peak_areas (pandas DataFrame): One-row DataFrame with columns:
                                      'fundfreq{feature_name}' (fundamental frequency) and
                                      '{i:02d}_harmonic_{feature_name}Area' for each harmonic.
    '''
    # Skip the first 5 frequency bins (DC and near-DC components) to find the true fundamental.
    maxfreq_index = PSD[5:].argmax() + 5
    print(maxfreq_index)
    columns, data, harmonic_areas = [], [], []

    # Compute peak area for each harmonic.
    # The integration window width is set adaptively to 1/5 of the fundamental index.
    for harmonic in range(1, harmonics+1):
        peakhalfwidth = int(np.round(maxfreq_index/5))
        harmonic_areas.append(np.sum(PSD[harmonic*maxfreq_index - peakhalfwidth:
                                         harmonic*maxfreq_index + peakhalfwidth]))
    print(f'harmonic_areas: {harmonic_areas}')

    Freqs = fftfreq(n=PSD.shape[0], d=dt)

    data.append(Freqs[maxfreq_index])
    data.extend(harmonic_areas)

    columns_area = [str(i).zfill(2) + '_harmonic_' + feature_name + 'Area' for i in range(1, harmonics+1)]
    columns.append('fundfreq' + feature_name)
    columns.extend(columns_area)

    df_peak_areas = pd.DataFrame(index=columns,
                                 data=data,
                                 columns=[filename[:-4]]).T
    return df_peak_areas

