"""Generate ring attractor data with external inputs injected into observations.

This script:
1. Generates clean ring attractor data using analyze_ring_attractor
2. Creates diverse external input patterns
3. Injects these inputs into the observations to make the task harder for xFADS
4. Saves the dataset with observations, true latents, inputs, and manifold
"""

import torch
import numpy as np
import sys
import os
from typing import Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from analyze_ring_attractor import main as generate_ring_data


def generate_external_inputs(n_trials: int, n_time_bins: int, dt: float = 0.05) -> np.ndarray:
    """Generate simple directional input pushes.
    
    Args:
        n_trials: Number of trials
        n_time_bins: Number of time steps per trial
        dt: Time step size
        
    Returns:
        Input array of shape [n_trials, n_time_bins, 2]
    """
    inputs = np.zeros((n_trials, n_time_bins, 2))
    
    for trial in range(n_trials):
        input_type = trial % 3  # 3 simple input patterns
        
        if input_type == 0:  # X direction push
            inputs[trial, :, 0] = 0.03
            inputs[trial, :, 1] = 0.0
        elif input_type == 1:  # Y direction push  
            inputs[trial, :, 0] = 0.0
            inputs[trial, :, 1] = 0.03
        # input_type == 2: No input (control condition)
        
    return inputs


def inject_inputs_into_observations(
    observations: np.ndarray, 
    latents: np.ndarray, 
    inputs: np.ndarray,
    injection_scale: float = 0.5
) -> np.ndarray:
    """Inject external inputs into observations through latent space.
    
    This simulates the effect of external inputs affecting the neural population
    in a way that correlates with the latent space structure.
    
    Args:
        observations: Original observations [n_trials, n_time_bins, n_neurons]
        latents: True latent trajectories [n_trials, n_time_bins, 2]
        inputs: External inputs [n_trials, n_time_bins, 2]
        injection_scale: How strongly inputs affect observations
        
    Returns:
        Modified observations with injected inputs
    """
    n_trials, n_time_bins, n_neurons = observations.shape
    n_latents = latents.shape[2]
    
    # Create an input-to-observation mapping matrix
    # This simulates how external inputs would affect neural activity
    np.random.seed(42)  # For reproducibility
    input_to_obs_matrix = np.random.randn(n_neurons, 2) * 0.2
    
    # Make some neurons more sensitive to specific input directions
    for i in range(n_neurons):
        if i % 3 == 0:  # X-input sensitive neurons
            input_to_obs_matrix[i, 0] *= 1.5
            input_to_obs_matrix[i, 1] *= 0.7
        elif i % 3 == 1:  # Y-input sensitive neurons
            input_to_obs_matrix[i, 0] *= 0.7
            input_to_obs_matrix[i, 1] *= 1.5
        # i % 3 == 2: Balanced sensitivity
    
    # Create modified observations
    modified_observations = observations.copy()
    
    for trial in range(n_trials):
        for t in range(n_time_bins):
            # Compute input effect on observations
            input_effect = input_to_obs_matrix @ inputs[trial, t]
            
            # Scale the effect
            input_effect *= injection_scale
            
            # Add to observations
            modified_observations[trial, t] += input_effect
    
    return modified_observations




def main() -> None:
    """Generate ring attractor dataset with external inputs."""
    print("="*70)
    print("GENERATING RING ATTRACTOR DATA WITH EXTERNAL INPUTS")
    print("="*70)
    
    # Generate base ring attractor data
    print("Generating base ring attractor data...")
    data = generate_ring_data(return_data=True)
    if data is None:
        print('❌ Error: No data returned from generate_ring_data.')
        return
    
    # Extract base data
    original_observations = data['gaussian_observations']  # [n_trials, n_time_bins, n_neurons]
    trajectories = data['trajectories']                   # [n_trials, n_time_bins, 2]
    inv_man = data['inv_man']                            # [n_points, 2]
    vector_field = data['vector_field']                  # True vector field data
    
    n_trials, n_time_bins, n_neurons = original_observations.shape
    
    print(f"Base data shapes:")
    print(f"  Observations: {original_observations.shape}")
    print(f"  Trajectories: {trajectories.shape}")
    print(f"  Invariant manifold: {inv_man.shape}")
    
    # Generate external inputs
    print("\nGenerating external input patterns...")
    dt = 5.0 / n_time_bins  # Assuming 5 second total time
    inputs = generate_external_inputs(n_trials, n_time_bins, dt)
    
    print(f"Input patterns:")
    print(f"  Shape: {inputs.shape}")
    print(f"  Range: [{inputs.min():.4f}, {inputs.max():.4f}]")
    print(f"  Mean magnitude: {np.mean(np.linalg.norm(inputs, axis=2)):.4f}")
    
    # Count input types
    input_counts = {}
    for trial in range(n_trials):
        input_type = trial % 3
        type_names = {0: 'X-push', 1: 'Y-push', 2: 'Control'}
        type_name = type_names[input_type]
        input_counts[type_name] = input_counts.get(type_name, 0) + 1
    
    print(f"  Input distribution: {input_counts}")
    
    # Inject inputs into observations
    print("\nInjecting inputs into observations...")
    modified_observations = inject_inputs_into_observations(
        original_observations, 
        trajectories, 
        inputs,
        injection_scale=0.3  # Reduced injection strength for simpler inputs
    )
    
    # Calculate the effect of input injection
    observation_change = modified_observations - original_observations
    print(f"Input injection effects:")
    print(f"  Mean absolute change: {np.mean(np.abs(observation_change)):.4f}")
    print(f"  Max change: {np.max(np.abs(observation_change)):.4f}")
    print(f"  Relative change: {np.mean(np.abs(observation_change)) / np.mean(np.abs(original_observations)) * 100:.2f}%")
    
    # Convert to tensors
    print("\nConverting to tensors and saving...")
    save_dict = {
        'y_obs': torch.tensor(modified_observations, dtype=torch.float32),
        'z_true': torch.tensor(trajectories, dtype=torch.float32),
        'u_inputs': torch.tensor(inputs, dtype=torch.float32),
        'inv_man': torch.tensor(inv_man, dtype=torch.float32),
        'y_obs_clean': torch.tensor(original_observations, dtype=torch.float32),  # Keep clean version for comparison
        'true_vector_field': {
            'X': torch.tensor(vector_field['X'], dtype=torch.float32),
            'Y': torch.tensor(vector_field['Y'], dtype=torch.float32),
            'U': torch.tensor(vector_field['U'], dtype=torch.float32),
            'V': torch.tensor(vector_field['V'], dtype=torch.float32)
        }
    }
    
    # Save dataset
    torch.save(save_dict, 'gauss_ring_data.pt')
    
    print(f"\n✅ Successfully saved 'gauss_ring_data.pt' with keys: {list(save_dict.keys())}")
    print(f"\nDataset summary:")
    for key, tensor in save_dict.items():
        if hasattr(tensor, 'shape'):
            print(f"  {key}: {tensor.shape} [{tensor.min():.3f}, {tensor.max():.3f}]")
        elif isinstance(tensor, dict):
            print(f"  {key}: dict with keys {list(tensor.keys())}")
    
    print(f"\n🎯 Dataset ready for xFADS training!")
    print(f"   - xFADS will see only y_obs (with injected inputs)")
    print(f"   - u_inputs available for analysis but not used in training")
    print(f"   - true_vector_field available for evaluation comparisons")
    print(f"   - Task: Learn ring dynamics despite external perturbations")


if __name__ == '__main__':
    main() 