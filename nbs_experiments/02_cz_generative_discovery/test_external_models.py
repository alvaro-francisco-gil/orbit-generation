from orbit_generation.external_models import ExternalModelWrapper
from orbit_generation.visualize import visualize_static_orbits
import numpy as np
import os

# Path to the config YAML and data
config_path = 'periodic_orbits_prop100_cr3bp_nasa.yaml'
data_path = '/orbit-generation/data/periodic_orbit_labels_cr3bp_nasa.npy'

# Create the wrapper
wrapper = ExternalModelWrapper()

# Run the diffusion_ts model
results = wrapper.run_diffusion_model(
    input_data=data_path,
    model_config=config_path
)

print('Diffusion-TS results:', results)

# --- Load generated orbits ---
# You may need to adjust the path and filename depending on your config and wrapper
# Try to find the output file in the results directory or as specified in the config
output_dir = os.path.join(os.path.dirname(__file__), '../../orbit_generation/results')
possible_files = [
    os.path.join(output_dir, 'ddpm_fake_orbits.npy'),
    os.path.join(output_dir, 'diffusion_results.npy'),
    os.path.join(output_dir, 'diffusion_results.json'),
]
generated_orbits = None
for f in possible_files:
    if os.path.exists(f) and f.endswith('.npy'):
        generated_orbits = np.load(f)
        break
    elif os.path.exists(f) and f.endswith('.json'):
        # If the output is a JSON, try to extract the path to the .npy file
        with open(f, 'r') as jf:
            import json
            data = json.load(jf)
            if 'output_file' in data and os.path.exists(data['output_file']):
                generated_orbits = np.load(data['output_file'])
                break

if generated_orbits is None:
    raise FileNotFoundError('Could not find generated orbits output file.')

# --- Sample 10 orbits and visualize ---
num_to_plot = 10
if generated_orbits.shape[0] > num_to_plot:
    indices = np.random.choice(generated_orbits.shape[0], num_to_plot, replace=False)
else:
    indices = np.arange(generated_orbits.shape[0])

visualize_static_orbits(generated_orbits, orbit_indices=indices.tolist())
