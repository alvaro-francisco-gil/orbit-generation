# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/08_experiment.ipynb.

# %% auto 0
__all__ = ['setup_new_experiment', 'convert_notebook']

# %% ../nbs/08_experiment.ipynb 2
import os
import csv
import pandas as pd
import subprocess
import shutil
from typing import Dict, Any, List, Optional, Tuple

from .constants import ORBIT_CLASS_DF

# %% ../nbs/08_experiment.ipynb 3
def setup_new_experiment(params: Dict[str, Any],              # Dictionary of parameters for the new experiment.
                         experiments_folder: str,             # Path to the folder containing all experiments.
                         csv_file: Optional[str] = None       # Optional path to the CSV file tracking experiment parameters.
                        ) -> str:                             # The path to the newly created experiment folder.
    """
    Sets up a new experiment by creating a new folder and updating the CSV file with experiment parameters.
    """
    # Ensure the experiments folder exists
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder)

    # Default CSV file to 'experiments.csv' in the experiments_folder if not provided
    if csv_file is None:
        csv_file = os.path.join(experiments_folder, 'experiments.csv')

    existing_experiment_folder = None
    existing_experiment_ids = set()

    # Check if the parameters already exist in the CSV file
    if os.path.isfile(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_experiment_ids.add(int(row['id']))
                if all(row[key] == str(value) for key, value in params.items()):
                    candidate_folder = os.path.join(experiments_folder, f"experiment_{row['id']}")
                    if os.path.exists(candidate_folder):
                        print(f'Parameters already exist for experiment: {candidate_folder}')
                        return candidate_folder
                    else:
                        existing_experiment_folder = candidate_folder

    # Determine the next experiment number, avoiding existing IDs in the CSV
    existing_experiment_folders = [d for d in os.listdir(experiments_folder) if os.path.isdir(os.path.join(experiments_folder, d))]
    existing_experiment_numbers = {int(folder.split('_')[-1]) for folder in existing_experiment_folders if folder.startswith('experiment')}
    next_experiment_number = 1
    while next_experiment_number in existing_experiment_ids or next_experiment_number in existing_experiment_numbers:
        next_experiment_number += 1

    # Create a new folder for the next experiment
    if existing_experiment_folder and not os.path.exists(existing_experiment_folder):
        new_experiment_folder = existing_experiment_folder
    else:
        new_experiment_folder = os.path.join(experiments_folder, f'experiment_{next_experiment_number}')
    os.makedirs(new_experiment_folder, exist_ok=True)

    # Update the CSV file with the new experiment's parameters
    csv_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the CSV does not exist
        if not csv_exists:
            header = ['id'] + list(params.keys())
            writer.writerow(header)
        # Write the experiment parameters
        row = [next_experiment_number] + list(params.values())
        writer.writerow(row)

    print(f'New experiment setup complete: {new_experiment_folder}')
    print(f'Parameters saved to {csv_file}.')

    return new_experiment_folder

# %% ../nbs/08_experiment.ipynb 4
def convert_notebook(notebook_path: str,                # The path to the notebook to convert.
                     output_folder: str,                # The folder to save the converted file.
                     output_filename: str,              # The name of the output file.
                     format: str = 'html'               # The format to convert the notebook to ('html' or 'pdf').
                    ) -> None:                          # This function does not return a value.
    """
    Convert the specified Jupyter notebook to HTML or PDF.

    :param notebook_path: The path to the notebook to convert.
    :param output_folder: The folder to save the converted file.
    :param output_filename: The name of the output file.
    :param format: The format to convert the notebook to ('html' or 'pdf').
    """
    if format == 'pdf' and shutil.which('pandoc') is None:
        raise RuntimeError("Pandoc is required for PDF conversion but was not found. Please install Pandoc: https://pandoc.org/installing.html")

    # Create the full path for the output file
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{output_filename}.{format}")

    # Convert the notebook using nbconvert
    command = f"jupyter nbconvert --to {format} \"{notebook_path}\" --output \"{output_path}\""
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Notebook converted to {format.upper()} and saved at {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting the notebook to {format.upper()}:")
        print(e.stderr)
        raise