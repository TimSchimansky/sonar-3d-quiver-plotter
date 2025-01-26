


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Replace 'your_dataframe.csv' with the path to your dataset or load your DataFrame
# df = pd.read_csv('your_dataframe.csv')
# Assuming df is already loaded as a DataFrame

def plot_3d_vector_field(df):
    """
    Plots a 3D vector field of water flow speed for all measured depths.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the water flow data.
    """
    # Initialize lists to store data for plotting
    x, y, z = [], [], []  # Position coordinates (depth and cell locations)
    u, v, w = [], [], []  # Velocity components (Ve, Vn, Vu)

    # Loop through each cell column to extract data
    for i in range(1, 64):  # Assuming there are 63 cells (Cell1 to Cell63)
        location_col = f'Cell{i} Location (m)'
        ve_col = f'Cell{i} Ve (m/s)'
        vn_col = f'Cell{i} Vn (m/s)'
        vu_col = f'Cell{i} Vu (m/s)'

        if location_col in df.columns and ve_col in df.columns:
            # Filter out rows where any relevant value is NaN
            valid_rows = df[[location_col, 'Depth (m)', ve_col, vn_col, vu_col]].dropna()

            # Append data to the lists
            x.extend(valid_rows[location_col])
            y.extend(valid_rows['Depth (m)'])
            z.extend([0] * len(valid_rows))  # Assume all measurements occur at the same horizontal plane
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

    # Create a 3D quiver plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize vectors for better visualization
    magnitude = np.sqrt(u**2 + v**2 + w**2)
    u /= magnitude
    v /= magnitude
    w /= magnitude

    # Plot the vector field
    ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True, color='b', alpha=0.7)

    # Set axis labels
    ax.set_xlabel('Location (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_zlabel('Horizontal Plane (m)')

    # Set plot title
    ax.set_title('3D Vector Field of Water Flow Speeds')

    plt.savefig("test.png")

# Example usage:
data_df = pd.read_csv("20250122130314.vel")

plot_3d_vector_field(data_df)
