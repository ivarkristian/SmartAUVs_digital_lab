import pandas as pd
from pandas.api.indexers import BaseIndexer
import numpy as np
# filters = {
#     'angle': [30, 60], (inclusive interval)
#     'angle_open': [None, 60], (open-ended interval)
#     'speed': 10, (exact value)
#     'pattern': {'A', 'B'} (set of allowed exact values)
#}

class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed, step):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            max_left = min(i, self.window_size)
            max_right = min(num_values - i, self.window_size)
            if self.equal_left_right:
                max_left = min(max_left, max_right)
                max_right = max_left

            start[i] = i - max_left
            end[i] = max(i + max_right, 1) # to avoid start[i] = end[i] = 0
        
        return start, end

def filter_df(df, filters):
    df_filtered = df.copy()
    for key, value in filters.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            # Value is an interval: [lower_bound, upper_bound]
            lower_bound, upper_bound = value
            if lower_bound is not None and upper_bound is not None:
                df_filtered = df_filtered[(df_filtered[key] >= lower_bound) & (df_filtered[key] <= upper_bound)]
            elif lower_bound is None:
                df_filtered = df_filtered[df_filtered[key] <= upper_bound]
            elif upper_bound is None:
                df_filtered = df_filtered[df_filtered[key] >= lower_bound]
        elif isinstance(value, set):
            # Value is a set of allowed values
            df_filtered = df_filtered[df_filtered[key].isin(value)]
        else:
            # Value is a constant
            df_filtered = df_filtered[df_filtered[key] == value]

    return df_filtered

def find_nearest(series_a, series_b):
    # Convert series_b to numpy array for faster indexing
    b_values = series_b.values
    
    # Search for insert positions
    # 'side=left' returns the position to insert value in sorted order
    idxs = series_b.searchsorted(series_a, side='left')
    
    nearest_values = []
    for i, val in enumerate(series_a):
        pos = idxs[i]
        
        # Candidates can be element at pos-1 and pos, if they exist
        candidates = []
        if pos > 0:
            candidates.append(b_values[pos-1])
        if pos < len(b_values):
            candidates.append(b_values[pos])
        
        # Find candidate with the smallest absolute difference
        closest = min(candidates, key=lambda x: abs(x - val))
        nearest_values.append(closest)
    
    return pd.Series(nearest_values, index=series_a.index)