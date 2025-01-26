import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
import pandas as pd
import numpy as np

def dms_to_dd(dms_str):
    """Convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees."""
    try:
        dms_str = dms_str.replace("°", "").replace("'", "").replace('"', "").strip()
        parts = dms_str.split()
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2]) if len(parts) > 2 else 0
        dd = degrees + minutes / 60 + seconds / 3600
        return dd
    except Exception as e:
        print(f"Error converting DMS '{dms_str}': {e}")
        return np.nan

def plot_3d_vector_field(vel_df, sum_df):
    """
    Plots a 3D vector field of water flow speed with location data.

    Parameters:
        vel_df (pd.DataFrame): Velocity data DataFrame
        sum_df (pd.DataFrame): Location summary data DataFrame
    """
    # Merge velocity and location dataframes
    merged_df = pd.merge(vel_df, sum_df, on='Sample #', how='inner', suffixes=('', '_double'))
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_double')]
    
    # Convert latitude and longitude to decimal degrees
    merged_df['Latitude (deg)'] = merged_df['Latitude (deg)'].apply(dms_to_dd)
    merged_df['Longitude (deg)'] = merged_df['Longitude (deg)'].apply(dms_to_dd)
    
    # Convert to geodataframe for coordinate transformation
    gdf = gpd.GeoDataFrame(
        merged_df,
        geometry=gpd.points_from_xy(merged_df["Longitude (deg)"], merged_df["Latitude (deg)"]),
        crs="EPSG:4326"
    )
    gdf = gdf.to_crs("EPSG:32632")  # Convert to UTM Zone 32
    
    # Initialize lists to store data for plotting
    x, y, z = [], [], []  # Position coordinates
    u, v, w = [], [], []  # Velocity components

    # Loop through each cell column to extract data
    for i in range(1, 64):  # Assuming there are 63 cells (Cell1 to Cell63)
        location_col = f'Cell{i} Location (m)'
        ve_col = f'Cell{i} Ve (m/s)'
        vn_col = f'Cell{i} Vn (m/s)'
        vu_col = f'Cell{i} Vu (m/s)'

        if location_col in gdf.columns and ve_col in gdf.columns:
            # Filter out rows where any relevant value is NaN
            valid_rows = gdf[[location_col, 'Depth (m)', ve_col, vn_col, vu_col, 'geometry']].dropna()

            # Append data to the lists
            x.extend(valid_rows['geometry'].x)
            y.extend(valid_rows['geometry'].y)
            z.extend(valid_rows[location_col])
            u.extend(valid_rows[ve_col])
            v.extend(valid_rows[vn_col])
            w.extend(valid_rows[vu_col])

    # Convert lists to numpy arrays for plotting
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)

    # Calculate flow speed for coloring
    speed = np.sqrt(u**2 + v**2 + w**2)
    norm_speed = (speed - speed.min()) / (speed.max() - speed.min())

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize vectors for better visualization
    u_norm = u / speed
    v_norm = v / speed
    w_norm = w / speed

    # Plot the vector field
    quiver = ax.quiver(
        x, y, z, u_norm, v_norm, w_norm,
        length=0.1,
        normalize=True,
        color=plt.cm.viridis(norm_speed)
    )

    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap="viridis")
    mappable.set_array(speed)
    plt.colorbar(mappable, ax=ax, label="Flow Speed (m/s)")

    # Set labels and title
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title('3D Vector Field of Water Flow Speeds')

    plt.tight_layout()
    plt.savefig("vector_field_plot.png")
    plt.show()

# Example usage:
data_vel_df = pd.read_csv("data/20250122130314.vel")
data_sum_df = pd.read_csv("data/20250122130314.sum",encoding='ISO-8859-1')

plot_3d_vector_field(data_vel_df.iloc[:1000], data_sum_df.iloc[:1000])
