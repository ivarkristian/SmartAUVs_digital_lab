import pandas as pd

# filters = {
#     'angle': [30, 60], (inclusive interval)
#     'angle_open': [None, 60], (open-ended interval)
#     'speed': 10, (exact value)
#     'pattern': {'A', 'B'} (set of allowed exact values)
#}

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
