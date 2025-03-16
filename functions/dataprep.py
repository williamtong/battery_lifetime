import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.fft import fft, fftfreq


def group_values(df, groupby_col, how, column_name):
    '''This function finds the max, min, mean of each cycle of a battery charge, depending on the input 
    parameter 'how'.
    '''
    df_group = df.groupby("Cycle_Index")
    if how == 'mean':
        df_groupby = df_group.mean()[groupby_col]
    elif how == 'min':
        df_groupby = df_group.min()[groupby_col]
    elif how == 'max':
        df_groupby = df_group.max()[groupby_col]
    else:
        print('group by how not recognized')
        return None

    if groupby_col == 'Charge_Capacity':
        # Normalizes the data by dividing the series by the first data point.
        df_groupby = df_groupby/df_groupby.iloc[0]
    if groupby_col == 'DateTime':
        # The interval for each cycle. 
        df_groupby = (df_groupby - df_groupby.iloc[0])
        # Convert the data to minutes.
        df_groupby = df_groupby.diff()/60 # units = minutes
        # Remove first and second data point, which are useless.
        df_groupby = df_groupby.iloc[2:]

    if groupby_col == 'Internal_Resistance':
        # Remove first and second data point, which are useless.
        df_groupby = df_groupby.iloc[2:]
    if groupby_col == 'Temperature':
        # Remove first and second data point, which are useless.
        df_groupby = df_groupby.iloc[2:]
    df_group = pd.DataFrame(data = df_groupby.values,
                            columns = [column_name],
                            index = [str(int(idx)) + '_' + groupby_col  for idx in df_groupby.index]
                           )
    return df_group


def replace_DateTime_CycleTime(dataframes):
    '''This simple function replaces "DateTime" with "CycleTime".'''
    data_frames_renamed = []
    for dataframe in dataframes:
        indices = dataframe.index
        new_indices = [indx.replace('DateTime', 'CycleTime') for indx in indices]
        dataframe.index = new_indices
        data_frames_renamed.append(dataframe)
    return data_frames_renamed   


def strong_low_pass_filter(df, filter_value = 0.0):
    '''This function applies a custom low pass filter to a time series.  
    If the value of a data point is different from the previous data point by more
    than filter_values in percent, then replace the data point with the previous one.'''
    print(f'Processing {'_'.join(df.index[0].split('_')[1:])}')
    columns = range(df.shape[1])
    rows = range(1, df.shape[0]) # skip row 0
    for column in columns:
        for row in rows:
            if np.abs(1 - (df.iloc[row, column]/df.iloc[row-1, column])) > filter_value:
                df.iloc[row, column] = df.iloc[row-1, column]
    return df
 

def plot_Current(df_file):
    df_file = df_file[(df_file['Cycle_Index'] >= 2) & (df_file['Cycle_Index'] < 7)]
    df_file["Time_elapsed"] = (df_file["DateTime"] - df_file["DateTime"].min())/3600
    plt.figure(figsize = (15, 5))
    plt.plot(df_file["Time_elapsed"], df_file["Current"])
    plt.xlabel("Time (hour)")
    plt.xticks(np.arange(0,5))
    plt.ylabel("Current (A)")
    plt.grid()
    plt.show()


def create_Peak_Areas_df(PSD, dt, filename, harmonics = 4, peakhalfwidth = 5, feature_name = 'CC'):
    '''This function does the following to Power Spectral Density series
    1.  Find the main frequency of the signal.
    2.  The peak areas of the n harmonics (parameter = harmonics)
    3.  Output them in a pandas dataframe.

    INPUT
    PSD:        The Power Spectral Density of a time series.
    dt:         The mean time interval of the time series.
    harmonics:  How many harmonics to include.
    filename:   The battery data file path containing the time series.
    peakhalfwidth:  How far from the max to calculate the peak area.

    OUTPUT
    df_peak_areas: A pandas dataframe containing the max frequency and the areas of the harmonics.

    '''
    maxfreq_index = PSD[5:].argmax()
    columns,  data, harmonic_areas = [], [], []

    # Area calculation
    for harmonic in range(1, harmonics+1):
        harmonic_areas.append(np.sum(PSD[harmonic*maxfreq_index-peakhalfwidth: harmonic*maxfreq_index+peakhalfwidth]))
    
    Freqs = fftfreq(n=PSD.shape[0], d=dt)
    
    data.append(Freqs[maxfreq_index])
    data.extend(harmonic_areas)

    columns_area = [str(i).zfill(2) + '_harmonic_' + feature_name + 'Area' for i in range(harmonics)]
    columns.append('maxfreq' + feature_name)
    columns.extend(columns_area)
    df_peak_areas = pd.DataFrame(index = columns,
                                 data = data,
                                 columns = [filename[:-4]]
                                 ).T
    
    return df_peak_areas
