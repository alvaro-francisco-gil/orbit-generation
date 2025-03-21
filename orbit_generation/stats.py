"""Scripts to generate statistics out of orbit data"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_orbit_statistics.ipynb.

# %% auto 0
__all__ = ['calculate_overall_spatial_statistics', 'calculate_per_orbit_spatial_statistics', 'plot_time_increments',
           'plot_orbit_data_lengths', 'plot_histograms_position', 'plot_histograms_comparison']

# %% ../nbs/04_orbit_statistics.ipynb 2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# %% ../nbs/04_orbit_statistics.ipynb 7
def calculate_overall_spatial_statistics(orbits: np.ndarray) -> np.ndarray:
    """
    Calculate the overall min, mean, max, and percentile statistics for each scalar 
    (position and velocity in X, Y, Z) across all time instants and orbits.

    Parameters:
    - orbits (np.ndarray): A numpy array of shape (number_of_orbits, 6 or 7, number_of_time_instants) containing orbit data.

    Returns:
    - np.ndarray: A NumPy array containing statistics for each scalar.
    """
    stats = []  # List to store statistics for each scalar.
    scalar_names = ['posx', 'posy', 'posz', 'velx', 'vely', 'velz']  # List of scalar names.
    
    # Check if the first dimension is for time
    if orbits.shape[1] == 7:
        scalar_names.insert(0, 'time')  # Add time as the first scalar if present.

    # Calculate statistics for each scalar
    for scalar_index, scalar_name in enumerate(scalar_names):
        if 'time' in scalar_names and scalar_name == 'time':
            continue  # Skip time in the calculations for statistics.

        # Flatten data across orbits and time points
        scalar_data = orbits[:, scalar_index, :].flatten()

        # Calculate statistics and append to the list
        scalar_stats = [
            np.min(scalar_data),
            np.mean(scalar_data),
            np.max(scalar_data),
            np.percentile(scalar_data, 25),
            np.median(scalar_data),
            np.percentile(scalar_data, 75)
        ]
        stats.append(scalar_stats)

    return np.array(stats)

# %% ../nbs/04_orbit_statistics.ipynb 9
def calculate_per_orbit_spatial_statistics(orbits: np.ndarray) -> np.ndarray:
    """
    Calculate per-orbit min, mean, max, and percentile statistics for each scalar 
    (position and velocity in X, Y, Z) across all time instants.

    Parameters:
    - orbits (np.ndarray): A numpy array of shape (number_of_orbits, 6 or 7, number_of_time_instants) containing orbit data.

    Returns:
    - np.ndarray: A NumPy array of shape (number_of_orbits, number_of_scalars, number_of_stats)
                  containing statistics for each scalar per orbit.
    """
    stats = []  # List to store statistics for each orbit.
    scalar_names = ['posx', 'posy', 'posz', 'velx', 'vely', 'velz']  # List of scalar names.
    num_stats = 6  # min, mean, max, 25th percentile, median, 75th percentile

    # Check if the second dimension includes time
    if orbits.shape[1] == 7:
        scalar_names.insert(0, 'time')  # Add 'time' as the first scalar if present.

    num_orbits = orbits.shape[0]
    num_scalars = len(scalar_names) - 1 if 'time' in scalar_names else len(scalar_names)
    num_stats = 6  # min, mean, max, 25th percentile, median, 75th percentile

    # Iterate over each orbit
    for orbit_index in range(num_orbits):
        orbit_stats = []  # List to store stats for all scalars in the current orbit

        for scalar_index, scalar_name in enumerate(scalar_names):
            if 'time' in scalar_names and scalar_name == 'time':
                continue  # Skip 'time' scalar if present

            # Extract scalar data for the current orbit
            scalar_data = orbits[orbit_index, scalar_index, :]

            # Calculate statistics
            scalar_stats = [
                np.min(scalar_data),
                np.mean(scalar_data),
                np.max(scalar_data),
                np.percentile(scalar_data, 25),
                np.median(scalar_data),
                np.percentile(scalar_data, 75)
            ]
            orbit_stats.append(scalar_stats)

        stats.append(orbit_stats)

    return np.array(stats)

# %% ../nbs/04_orbit_statistics.ipynb 11
def plot_time_increments(orbit_dataset: np.ndarray,  # The 3D numpy array representing the orbits
                         orbits_to_plot: List[int] = None,  # Optional list of integers referring to the orbits to plot
                         show_legend: bool = True  # Boolean to control the display of the legend
                        ) -> None:
    """
    Plots the time as a function to visualize how it increments for each orbit.

    Parameters:
    orbit_dataset (np.ndarray): A 3D numpy array where the first dimension is the number of orbits,
                                the second dimension contains 7 scalars (time, posx, posy, posz, velx, vely, velz),
                                and the third dimension is the time steps.
    orbits_to_plot (list[int], optional): List of integers referring to the orbits to plot. If None, plots all orbits.
    show_legend (bool, optional): Whether to display the legend. Default is True.
    """
    num_orbits = orbit_dataset.shape[0]

    # If orbits_to_plot is not provided, plot all orbits
    if orbits_to_plot is None:
        orbits_to_plot = list(range(num_orbits))

    plt.figure(figsize=(10, 6))

    for i in orbits_to_plot:
        time_steps = orbit_dataset[i, 0]  # Extract the time steps for the current orbit
        plt.plot(time_steps, label=f'Orbit {i}')

    plt.xlabel('Time Step Index')
    plt.ylabel('Time')
    plt.title('Time Increments for Orbits')
    
    if show_legend:
        plt.legend()
    
    plt.grid(True)
    plt.show()

# %% ../nbs/04_orbit_statistics.ipynb 13
def plot_orbit_data_lengths(orbit_data, key_range=(1, 36072), dimension=0, bins=30, color='blue', plot=True, title='Histogram of Orbits Time Steps'):
    lengths = []
    
    # Iterate over each dataset name within the provided range
    start, end = key_range  # Unpack the tuple for range
    for key in range(start, end):
        if key in orbit_data:
            try:
                # Append the length of the specified dimension of the dataset
                lengths.append(len(orbit_data[key][dimension]))
            except IndexError:
                # Handle the case where the dimension is not available
                print(f"Warning: Dimension {dimension} is not available in dataset {key}.")
                continue
    
    if plot:
        # Plot the histogram of these lengths if plot is True
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=bins, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Frequency')
        plt.show()
    else:
        # Return lengths data for further analysis
        return lengths

# %% ../nbs/04_orbit_statistics.ipynb 14
def plot_histograms_position(data: np.ndarray,                  # The orbit data array of shape (num_orbits, num_scalars, num_time_points).
                             save_path: str = None,             # Optional path to save the plot image.
                             last_time_elements: bool = True    # Whether to plot only the last elements of the time vectors.
                            ) -> None:
    """
    Plots histograms for the scalar values (position and velocity in X, Y, Z, and optionally time) across all orbits
    and time points. Handles arrays with 6 or 7 scalar dimensions, with the 7th being 'time'.

    Parameters:
    - data (np.ndarray): The orbit data array.
    - save_path (str, optional): If provided, the plot will be saved to this file path.
    - last_time_elements (bool): If True, plot only the last elements of the time vectors for the time histogram.
    """
    # Check the number of scalars and adjust scalar names accordingly
    num_scalars = data.shape[1]
    if num_scalars == 7:
        scalar_names = ['time', 'posX', 'posY', 'posZ', 'velX', 'velY', 'velZ']
    elif num_scalars == 6:
        scalar_names = ['posX', 'posY', 'posZ', 'velX', 'velY', 'velZ']
    else:
        raise ValueError("Data arrays must have either 6 or 7 scalar dimensions.")

    # Setting up the subplot grid, dynamically adjusting if we have 7 scalars
    rows, cols = (3, 3) if num_scalars == 7 else (2, 3)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # Adjust height based on rows
    fig.suptitle('Histograms of Position, Velocity Components, and Time (if present) Across All Orbits')
    
    for i in range(num_scalars):
        if i == 0 and num_scalars == 7 and last_time_elements:
            # Plot only the last elements of the time vectors
            scalar_values = data[:, i, -1]  # Last elements of the time vectors
        else:
            # Flatten combines all orbits and time points for each scalar
            scalar_values = data[:, i, :].flatten()
        
        row, col = divmod(i, cols)  # Determine subplot position
        axs[row, col].hist(scalar_values, bins=50, alpha=0.75)  # You can adjust the number of bins
        axs[row, col].set_title(f'{scalar_names[i]}')
        axs[row, col].set_ylabel('Frequency')
        axs[row, col].set_xlabel('Value')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path)  # Save the figure to the specified path

    # Display the figure regardless of saving
    plt.show()

# %% ../nbs/04_orbit_statistics.ipynb 16
def plot_histograms_comparison(data1: np.ndarray,  # First orbit data array of shape (num_orbits, num_scalars, num_time_points).
                               data2: np.ndarray,  # Second orbit data array of shape (num_orbits, num_scalars, num_time_points).
                               label1: str = "Dataset 1",  # Label for the first dataset.
                               label2: str = "Dataset 2",  # Label for the second dataset.
                               save_path: str = None,  # Optional path to save the plot image.
                               normalize: bool = False  # Normalize histograms to show relative frequencies.
                               ) -> None:
    """
    Plots histograms for scalar values (position, velocity in X, Y, Z, and optionally time) from two datasets on 
    the same chart with different colors. Supports both 6 and 7 scalar dimensions, with the 7th being 'time'.
    Optionally saves the plot to a specified file path and can normalize histograms for relative comparison.
    """
    # Check the number of scalars and adjust scalar names accordingly
    num_scalars = data1.shape[1]
    scalar_names = ['posX', 'posY', 'posZ', 'velX', 'velY', 'velZ']
    if num_scalars == 7:
        scalar_names.insert(0, 'time')

    if num_scalars not in [6, 7]:
        raise ValueError("Data arrays must have either 6 or 7 scalar dimensions.")

    # Setting up the subplot grid
    rows, cols = (3, 3) if num_scalars == 7 else (2, 3)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle('Comparative Histograms of Position, Velocity Components, and Time (if present)')

    # Plot histograms
    for i in range(num_scalars):
        scalar_values1 = data1[:, i, :].flatten()
        scalar_values2 = data2[:, i, :].flatten()
        row, col = divmod(i, 3)
        
        density = normalize  # Use the same variable for clarity in the hist function call
        axs[row, col].hist(scalar_values1, bins=50, alpha=0.75, color='blue', label=label1, density=density)
        axs[row, col].hist(scalar_values2, bins=50, alpha=0.75, color='green', label=label2, density=density)
        
        axs[row, col].set_title(scalar_names[i])
        axs[row, col].set_ylabel('Density' if normalize else 'Frequency')
        axs[row, col].set_xlabel('Value')
        axs[row, col].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Saving or showing the plot
    if save_path:
        plt.savefig(save_path)
    plt.show()
