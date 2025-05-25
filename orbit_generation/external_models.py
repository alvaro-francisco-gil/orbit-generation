import subprocess
import json
from pathlib import Path
import os
import tempfile

class ModelExecutor:
    """Handles execution of models from different external libraries in their respective conda environments."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.env_mapping = {
            'diffusion_ts': 'diffusion_ts',
            'image_time': 'imagentime'
        }
    
    def _get_env_activate_cmd(self, env_name):
        """Get the conda activate command for a specific environment."""
        return f"conda activate {env_name}"
    
    def execute_model(self, library_name, script_path, params=None, output_file=None):
        """
        Execute a model from an external library in its specific conda environment.
        
        Args:
            library_name (str): Name of the external library (e.g., 'diffusion_ts')
            script_path (str): Path to the script to execute, relative to the library root
            params (dict): Parameters to pass to the model
            output_file (str): Path to save the output (optional)
        
        Returns:
            dict: Results from the model execution
        """
        if library_name not in self.env_mapping:
            raise ValueError(f"Unknown library: {library_name}")
        
        # Create a temporary file for parameters if provided
        params_file = None
        if params:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(params, f)
                params_file = f.name
        
        # Construct the full script path
        full_script_path = self.root_dir / 'external_libs' / library_name / script_path
        
        # Prepare the command
        cmd = [
            "bash", "-c",
            f"source {os.path.expanduser('~')}/miniconda3/etc/profile.d/conda.sh && "
            f"{self._get_env_activate_cmd(self.env_mapping[library_name])} && "
            f"python {full_script_path}"
        ]
        
        if params_file:
            cmd[-1] += f" --params {params_file}"
        
        if output_file:
            cmd[-1] += f" --output {output_file}"
        
        try:
            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            if output_file and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    output = json.load(f)
            else:
                output = {"stdout": result.stdout, "stderr": result.stderr}
            
            return output
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Model execution failed: {e.stderr}")
        
        finally:
            # Clean up temporary files
            if params_file and os.path.exists(params_file):
                os.remove(params_file)


class ExternalModelWrapper:
    """Wrapper class to handle external model execution and results processing."""
    
    def __init__(self):
        self.executor = ModelExecutor()
        self.results_dir = Path(__file__).parent / 'results'
        self.results_dir.mkdir(exist_ok=True)
    
    def run_diffusion_model(self, input_data, model_config):
        """Run a model from the diffusion_ts library."""
        params = {
            "input_data": str(input_data),
            "model_config": model_config
        }
        
        output_file = self.results_dir / 'diffusion_results.json'
        
        try:
            results = self.executor.execute_model(
                library_name="diffusion_ts",
                script_path="Models/train.py",  # Adjust path as needed
                params=params,
                output_file=str(output_file)
            )
            return results
        except Exception as e:
            print(f"Error running diffusion model: {e}")
            raise
    
    def run_image_time_model(self, image_path, processing_config):
        """Run a model from the image_time library."""
        params = {
            "image_path": str(image_path),
            "processing_config": processing_config
        }
        
        output_file = self.results_dir / 'image_time_results.json'
        
        try:
            results = self.executor.execute_model(
                library_name="image_time",
                script_path="process.py",  # Adjust path as needed
                params=params,
                output_file=str(output_file)
            )
            return results
        except Exception as e:
            print(f"Error running image_time model: {e}")
            raise


# Example usage:
"""
# Create wrapper
wrapper = ExternalModelWrapper()

# Run diffusion model
diffusion_config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    # ... other config parameters
}
diffusion_results = wrapper.run_diffusion_model(
    input_data="path/to/data",
    model_config=diffusion_config
)

# Run image time model
image_config = {
    "resolution": (256, 256),
    "processing_steps": 10,
    # ... other config parameters
}
image_results = wrapper.run_image_time_model(
    image_path="path/to/image",
    processing_config=image_config
)
""" 