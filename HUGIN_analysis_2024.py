# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots even if not directly referenced)
import plotly.express as px
import plotly.graph_objs as go
import importlib

# %%
#importlib.reload()

# %%
# read csv into df
root_dir = "../HUGIN_chemical_data_horten_2024"
df = pd.read_csv(root_dir + '/merged_missions.csv')

# %%
def plot_3d_measurement(
    df,
    x_col='Lat_m',
    y_col='Lon_m',
    z_col='Depth_m',
    time_col='timestamp',
    measurement_col='ADAMModule-EHpH_pH',
    start_time=None,
    end_time=None,
    connect=False,
    cmap='viridis'
):
    """
    Creates a 3D scatter plot of the specified measurement, color-coded by its value,
    filtered by time range. Optionally connects points with a line in chronological order.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing x, y, z coordinates in meters, a timestamp column,
        and one or more measurement columns.
    x_col : str
        Name of the column for x coordinates.
    y_col : str
        Name of the column for y coordinates.
    z_col : str
        Name of the column for z coordinates.
    time_col : str
        Name of the time column (datetime format).
    measurement_col : str
        The measurement column to plot/color by.
    start_time : datetime-like or None
        Start of the time range filter. If None, no lower bound is applied.
    end_time : datetime-like or None
        End of the time range filter. If None, no upper bound is applied.
    connect : bool
        If True, the points are connected in chronological order by a line.
    cmap : str
        The matplotlib colormap to use for coloring data points by measurement.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib Figure object.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D Axes object containing the plot.
    """

    # 1. Filter DataFrame by time range
    df_plot = df.copy()
    if start_time is not None:
        df_plot = df_plot[df_plot[time_col] >= start_time]
    if end_time is not None:
        df_plot = df_plot[df_plot[time_col] <= end_time]

    # Ensure data is sorted by time if we plan to connect the dots
    df_plot = df_plot.sort_values(by=time_col)

    # 2. Extract coordinate and measurement arrays
    x_vals = df_plot[x_col].values
    y_vals = df_plot[y_col].values
    z_vals = df_plot[z_col].values
    meas_vals = df_plot[measurement_col].values

    # 3. Create figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Create scatter plot, color by measurement
    sc = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=meas_vals,
        cmap=cmap,
        s=20,
        alpha=0.8,
        marker='o'
    )

    # Optionally connect the points in chronological order
    if connect and len(x_vals) > 1:
        ax.plot(
            x_vals,
            y_vals,
            z_vals,
            color='black',
            linewidth=1,
            alpha=0.5,
            label='Connection Line'
        )

    # 5. Add colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(measurement_col)

    # 6. Set labels and title
    ax.set_xlabel(f"{x_col} [m]")
    ax.set_ylabel(f"{y_col} [m]")
    ax.set_zlabel(f"{z_col} [m]")
    ax.set_title(f"{start_time} - {end_time} ({measurement_col})")

    # Add legend if connected
    if connect:
        ax.legend()

    # Adjust layout
    fig.tight_layout()

    return fig, ax

def plot_3d_measurement_plotly(
    df,
    x_col='Lat_m',
    y_col='Lon_m',
    z_col='Depth_m',
    time_col='timestamp',
    measurement_col='ADAMModule-EHpH_pH',
    start_time=None,
    end_time=None,
    connect=False,
    color_scale='viridis',
    marker_size = 4
):
    """
    Creates an interactive 3D scatter plot of a measurement using Plotly,
    filtered by time range, color-coded by the measurement column, and
    optionally connects points in chronological order with a line.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing x, y, z coordinates, a timestamp column,
        and one or more measurement columns.
    x_col : str
        Name of the column for x coordinates.
    y_col : str
        Name of the column for y coordinates.
    z_col : str
        Name of the column for z coordinates.
    time_col : str
        Name of the time column (in datetime or comparable type).
    measurement_col : str
        The measurement column to plot/color by.
    start_time : datetime-like or None
        Start of the time range filter. If None, no lower bound is applied.
    end_time : datetime-like or None
        End of the time range filter. If None, no upper bound is applied.
    connect : bool
        If True, the points are connected in chronological order by a 3D line.
    color_scale : str
        Plotly colormap for coloring data points by measurement.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object for the 3D scatter plot.
    """

    # 1. Filter DataFrame by time range
    df_plot = df.copy()
    if start_time is not None:
        df_plot = df_plot[df_plot[time_col] >= start_time]
    if end_time is not None:
        df_plot = df_plot[df_plot[time_col] <= end_time]

    # 2. Sort by time for correct chronological ordering
    df_plot = df_plot.sort_values(by=time_col)

    # 3. Create the basic 3D scatter plot
    fig = px.scatter_3d(
        df_plot,
        x=x_col,
        y=y_col,
        z=z_col,
        color=measurement_col,
        color_continuous_scale=color_scale,
        title=f"{start_time} - {end_time} ({measurement_col})",
        size=None
    )

    fig.update_traces(marker=dict(size=marker_size))

    # 4. Optionally connect the points with a 3D line
    if connect and len(df_plot) > 1:
        # Extract x, y, z arrays
        x_vals = df_plot[x_col].values
        y_vals = df_plot[y_col].values
        z_vals = df_plot[z_col].values

        # Add a line trace
        line_trace = go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode='lines',
            line=dict(color='black', width=2),
            name='Connection Line'
        )
        fig.add_trace(line_trace)

    # 5. Adjust the layout and axes labels
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x_col} [m]",
            yaxis_title=f"{y_col} [m]",
            zaxis_title=f"{z_col} [m]",
        )
    )

    return fig

def plot_3d_measurement_animated(
    df,
    x_col='Lat_m',
    y_col='Lon_m',
    z_col='Depth_m',
    time_col='timestamp',
    measurement_col='ADAMModule-EHpH_pH',
    freq='15min',         # Pandas frequency string for 15 minutes
    color_scale='cividis_r',
    marker_size=1,
    static_axes=True
):
    """
    Creates a single Plotly 3D scatter figure with a time slider.
    The data is binned by 'freq' (default 15 minutes), and each bin
    is shown as a separate animation frame in the same window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns x, y, z coordinates, a time column, and a measurement column.
    x_col : str
        Name of the column for x coordinates (in meters).
    y_col : str
        Name of the column for y coordinates (in meters).
    z_col : str
        Name of the column for z coordinates (in meters).
    time_col : str
        Column name for timestamps (in datetime).
    measurement_col : str
        Column name for the measurement to color by.
    freq : str
        Pandas offset alias for time bin size (default '15T' = 15 minutes).
    color_scale : str
        Plotly color scale name (e.g., 'Viridis', 'Plasma', 'Cividis', etc.).
    marker_size : int
        Size of the markers in the 3D plot.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The Plotly figure with an interactive time slider (animation).
    """

    # 1) Ensure time_col is in datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 1. Compute global min and max for x, y, z to fix axis ranges
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    z_min, z_max = df[z_col].min(), df[z_col].max()

    # 2) Compute global min/max for the measurement
    meas_min = df[measurement_col].min()
    meas_max = df[measurement_col].max()

    # 2) Create a new column 'time_bin' by flooring each timestamp to the freq
    df = df.copy()
    df['time_bin'] = df[time_col].dt.floor(freq)

    # 3) Plotly Express 3D scatter with animation_frame
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=measurement_col,
        color_continuous_scale=color_scale,
        #range_color=[meas_min, meas_max],
        animation_frame='time_bin',
        title=f"{measurement_col.split('_')[1]} in {freq} bins",
    )

    # 4) Update the marker size
    fig.update_traces(
        marker=dict(size=marker_size),
        selector=dict(mode='markers')
    )

    # 5. Fix the axis ranges to remain static
    if static_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[x_min, x_max], title=f"{x_col} (m)"),
                yaxis=dict(range=[y_min, y_max], title=f"{y_col} (m)"),
                zaxis=dict(range=[z_min, z_max], title=f"{z_col} (m)"),
            )
        )

    # 5) Adjust the 3D axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x_col} (m)",
            yaxis_title=f"{y_col} (m)",
            zaxis_title=f"{z_col} (m)",
        )
    )
    
    fig.update_coloraxes(colorbar_title=measurement_col.split('_')[1])
    
    return fig

# %%
fig = plot_3d_measurement_animated(df, marker_size=4, static_axes=False)
fig.show(renderer="browser")
#fig.show()