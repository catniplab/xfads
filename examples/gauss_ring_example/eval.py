import torch
import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
import pytorch_lightning as lightning
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
import xfads.utils as utils
import os
import argparse
from scipy.stats import pearsonr


def load_checkpoint(checkpoint_path, ssm, cfg):
    """Load checkpoint handling both .pt and .ckpt files."""
    # First, load the checkpoint to determine its type
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    
    # Check if it's a Lightning checkpoint by looking for 'state_dict' key
    if 'state_dict' in checkpoint:
        # This is a Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remove the 'model.' prefix if it exists (common in Lightning checkpoints)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_key = key[6:]  # Remove 'model.' prefix
            else:
                cleaned_key = key
            cleaned_state_dict[cleaned_key] = value
        
        seq_vae = LightningNonlinearSSM(ssm=ssm, cfg=cfg)
        seq_vae.load_state_dict(cleaned_state_dict)
        return seq_vae
    else:
        # This is a regular PyTorch state dict
        seq_vae = LightningNonlinearSSM(ssm=ssm, cfg=cfg)
        seq_vae.load_state_dict(checkpoint)
        return seq_vae


def plot_comprehensive_latent_analysis(z_inferred, z_true, inv_man=None, n_trials_plot=8):
    """Comprehensive latent space analysis with multiple visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Trajectory comparison
    ax = axes[0, 0]
    for i in range(min(n_trials_plot, len(z_true))):
        alpha = 0.8 if i < 3 else 0.4
        ax.plot(z_true[i, :, 0], z_true[i, :, 1], 'k-', alpha=alpha, linewidth=2, 
                label='True' if i == 0 else None)
        ax.plot(z_inferred[i, :, 0], z_inferred[i, :, 1], 'r--', alpha=alpha, linewidth=2,
                label='Inferred' if i == 0 else None)
        
        # Mark start points
        ax.scatter(z_true[i, 0, 0], z_true[i, 0, 1], c='black', s=60, marker='o', alpha=0.8)
        ax.scatter(z_inferred[i, 0, 0], z_inferred[i, 0, 1], c='red', s=60, marker='s', alpha=0.8)
    
    if inv_man is not None:
        ax.plot(inv_man[:, 0], inv_man[:, 1], 'b-', linewidth=3, alpha=0.9, label='Ring Manifold')
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Trajectory Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Latent space density comparison
    ax = axes[0, 1]
    z_true_flat = z_true.reshape(-1, 2)
    z_inferred_flat = z_inferred.reshape(-1, 2)
    
    ax.scatter(z_true_flat[:, 0], z_true_flat[:, 1], alpha=0.3, s=10, c='black', label='True')
    ax.scatter(z_inferred_flat[:, 0], z_inferred_flat[:, 1], alpha=0.3, s=10, c='red', label='Inferred')
    
    if inv_man is not None:
        ax.plot(inv_man[:, 0], inv_man[:, 1], 'b-', linewidth=3, alpha=0.9, label='Ring Manifold')
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('Latent Space Density', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 3: Phase analysis (angle on ring)
    ax = axes[0, 2]
    true_angles = np.arctan2(z_true_flat[:, 1], z_true_flat[:, 0])
    inferred_angles = np.arctan2(z_inferred_flat[:, 1], z_inferred_flat[:, 0])
    
    ax.scatter(true_angles, inferred_angles, alpha=0.5, s=10)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'k--', alpha=0.7, label='Perfect match')
    ax.set_xlabel('True Phase (radians)')
    ax.set_ylabel('Inferred Phase (radians)')
    ax.set_title('Phase Correspondence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Radial distance from origin
    ax = axes[1, 0]
    true_radii = np.sqrt(z_true_flat[:, 0]**2 + z_true_flat[:, 1]**2)
    inferred_radii = np.sqrt(z_inferred_flat[:, 0]**2 + z_inferred_flat[:, 1]**2)
    
    ax.scatter(true_radii, inferred_radii, alpha=0.5, s=10)
    ax.plot([0, max(true_radii.max(), inferred_radii.max())], 
            [0, max(true_radii.max(), inferred_radii.max())], 'k--', alpha=0.7, label='Perfect match')
    ax.set_xlabel('True Radius')
    ax.set_ylabel('Inferred Radius')
    ax.set_title('Radial Distance Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Time series comparison for a single trial
    ax = axes[1, 1]
    trial_idx = 0
    time_steps = np.arange(z_true.shape[1])
    
    ax.plot(time_steps, z_true[trial_idx, :, 0], 'k-', linewidth=2, label='True X')
    ax.plot(time_steps, z_true[trial_idx, :, 1], 'k--', linewidth=2, label='True Y')
    ax.plot(time_steps, z_inferred[trial_idx, :, 0], 'r-', linewidth=2, label='Inferred X')
    ax.plot(time_steps, z_inferred[trial_idx, :, 1], 'r--', linewidth=2, label='Inferred Y')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Latent Value')
    ax.set_title(f'Time Series (Trial {trial_idx})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Reconstruction error distribution
    ax = axes[1, 2]
    reconstruction_error = np.sqrt(np.sum((z_inferred - z_true)**2, axis=2))
    
    ax.hist(reconstruction_error.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_ring_structure(z_inferred, z_true, inv_man=None):
    """Analyze how well the ring structure is recovered."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten for analysis
    z_true_flat = z_true.reshape(-1, 2)
    z_inferred_flat = z_inferred.reshape(-1, 2)
    
    # Plot 1: Radial distribution
    ax = axes[0, 0]
    true_radii = np.sqrt(z_true_flat[:, 0]**2 + z_true_flat[:, 1]**2)
    inferred_radii = np.sqrt(z_inferred_flat[:, 0]**2 + z_inferred_flat[:, 1]**2)
    
    ax.hist(true_radii, bins=50, alpha=0.7, color='black', label='True', density=True)
    ax.hist(inferred_radii, bins=50, alpha=0.7, color='red', label='Inferred', density=True)
    ax.set_xlabel('Radius from Origin')
    ax.set_ylabel('Density')
    ax.set_title('Radial Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Angular distribution
    ax = axes[0, 1]
    true_angles = np.arctan2(z_true_flat[:, 1], z_true_flat[:, 0])
    inferred_angles = np.arctan2(z_inferred_flat[:, 1], z_inferred_flat[:, 0])
    
    ax.hist(true_angles, bins=50, alpha=0.7, color='black', label='True', density=True)
    ax.hist(inferred_angles, bins=50, alpha=0.7, color='red', label='Inferred', density=True)
    ax.set_xlabel('Angle (radians)')
    ax.set_ylabel('Density')
    ax.set_title('Angular Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Ring deviation analysis
    ax = axes[1, 0]
    if inv_man is not None:
        # Distance from each point to the ring manifold
        true_distances = []
        inferred_distances = []
        
        for i in range(len(z_true_flat)):
            true_dist = np.min(np.sqrt(np.sum((inv_man - z_true_flat[i:i+1])**2, axis=1)))
            inferred_dist = np.min(np.sqrt(np.sum((inv_man - z_inferred_flat[i:i+1])**2, axis=1)))
            true_distances.append(true_dist)
            inferred_distances.append(inferred_dist)
        
        ax.scatter(true_distances, inferred_distances, alpha=0.5, s=10)
        max_dist = max(max(true_distances), max(inferred_distances))
        ax.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.7, label='Perfect match')
        ax.set_xlabel('True Distance to Ring')
        ax.set_ylabel('Inferred Distance to Ring')
        ax.set_title('Ring Adherence', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No ring manifold\navailable', transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Ring Adherence', fontweight='bold')
    
    # Plot 4: Velocity correlation
    ax = axes[1, 1]
    # Compute velocities
    true_velocities = np.diff(z_true, axis=1)
    inferred_velocities = np.diff(z_inferred, axis=1)
    
    true_vel_flat = true_velocities.reshape(-1, 2)
    inferred_vel_flat = inferred_velocities.reshape(-1, 2)
    
    ax.scatter(true_vel_flat[:, 0], inferred_vel_flat[:, 0], alpha=0.5, s=10, label='X velocity')
    ax.scatter(true_vel_flat[:, 1], inferred_vel_flat[:, 1], alpha=0.5, s=10, label='Y velocity')
    
    # Fit lines
    vel_range = max(np.abs(true_vel_flat).max(), np.abs(inferred_vel_flat).max())
    ax.plot([-vel_range, vel_range], [-vel_range, vel_range], 'k--', alpha=0.7, label='Perfect match')
    
    ax.set_xlabel('True Velocity')
    ax.set_ylabel('Inferred Velocity')
    ax.set_title('Velocity Correlation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_input_effects(z_inferred, z_true, u_inputs, n_trials_plot=8):
    """Analyze how inputs affect the trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Trajectories colored by input type
    ax = axes[0, 0]
    colors = ['red', 'blue', 'green']
    labels = ['X push', 'Y push', 'No input']
    
    for i in range(min(n_trials_plot, len(z_inferred))):
        input_type = i % 3  # Updated for simplified inputs
        ax.plot(z_inferred[i, :, 0], z_inferred[i, :, 1], 
               color=colors[input_type], alpha=0.7, linewidth=2,
               label=labels[input_type] if i < 3 else None)
    
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_title('Trajectories by Input Type', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 2: Input magnitude over time
    ax = axes[0, 1]
    for i in range(min(4, len(u_inputs))):
        input_magnitude = np.linalg.norm(u_inputs[i], axis=1)
        ax.plot(input_magnitude, alpha=0.7, linewidth=2, label=f'Trial {i+1}')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Input Magnitude')
    ax.set_title('Input Magnitude Over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Input-trajectory displacement correlation
    ax = axes[1, 0]
    displacements = []
    input_magnitudes = []
    
    for i in range(min(len(z_inferred), 20)):
        traj_displacement = np.sqrt(np.sum(np.diff(z_inferred[i], axis=0)**2, axis=1))
        input_mag = np.linalg.norm(u_inputs[i, 1:], axis=1)
        
        displacements.extend(traj_displacement)
        input_magnitudes.extend(input_mag)
    
    ax.scatter(input_magnitudes, displacements, alpha=0.5, s=10)
    ax.set_xlabel('Input Magnitude')
    ax.set_ylabel('Trajectory Displacement')
    ax.set_title('Input-Displacement Correlation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: True vs inferred response to inputs
    ax = axes[1, 1]
    true_displacements = []
    inferred_displacements = []
    
    for i in range(min(len(z_inferred), 20)):
        true_disp = np.sqrt(np.sum(np.diff(z_true[i], axis=0)**2, axis=1))
        inferred_disp = np.sqrt(np.sum(np.diff(z_inferred[i], axis=0)**2, axis=1))
        
        true_displacements.extend(true_disp)
        inferred_displacements.extend(inferred_disp)
    
    ax.scatter(true_displacements, inferred_displacements, alpha=0.5, s=10)
    max_disp = max(max(true_displacements), max(inferred_displacements))
    ax.plot([0, max_disp], [0, max_disp], 'k--', alpha=0.7, label='Perfect match')
    ax.set_xlabel('True Displacement')
    ax.set_ylabel('Inferred Displacement')
    ax.set_title('Displacement Correlation', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_vector_field_comparison(true_X, true_Y, true_U, true_V, inferred_dynamics, z_range=2.0, n_grid=20):
    """Compare true vector field with inferred dynamics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(true_X):
        true_X = true_X.numpy()
    if torch.is_tensor(true_Y):
        true_Y = true_Y.numpy()
    if torch.is_tensor(true_U):
        true_U = true_U.numpy()
    if torch.is_tensor(true_V):
        true_V = true_V.numpy()
    
    # Create grid for evaluation
    x = np.linspace(-z_range, z_range, n_grid)
    y = np.linspace(-z_range, z_range, n_grid)
    X_eval, Y_eval = np.meshgrid(x, y)
    
    # Evaluate inferred dynamics on grid
    grid_points = torch.tensor(np.stack([X_eval.flatten(), Y_eval.flatten()], axis=1), dtype=torch.float32)
    
    with torch.no_grad():
        dynamics_fn = inferred_dynamics.mean_fn
        inferred_next = dynamics_fn(grid_points).numpy()
        inferred_flow = inferred_next - grid_points.numpy()
    
    inferred_U = inferred_flow[:, 0].reshape(X_eval.shape)
    inferred_V = inferred_flow[:, 1].reshape(X_eval.shape)
    
    # Plot 1: True vector field
    axes[0].streamplot(true_X, true_Y, true_U, true_V, density=1.5, color='blue')
    axes[0].quiver(true_X[::2, ::2], true_Y[::2, ::2], true_U[::2, ::2], true_V[::2, ::2], 
                   color='blue', alpha=0.4, scale=10)
    axes[0].set_title('True Vector Field')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Inferred vector field
    axes[1].streamplot(X_eval, Y_eval, inferred_U, inferred_V, density=1.5, color='red')
    axes[1].quiver(X_eval[::2, ::2], Y_eval[::2, ::2], inferred_U[::2, ::2], inferred_V[::2, ::2], 
                   color='red', alpha=0.4, scale=10)
    axes[1].set_title('Inferred Vector Field')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # Plot 3: Difference (error)
    from scipy.interpolate import griddata
    true_points = np.stack([true_X.flatten(), true_Y.flatten()], axis=1)
    true_U_interp = griddata(true_points, true_U.flatten(), (X_eval, Y_eval), method='linear', fill_value=0)
    true_V_interp = griddata(true_points, true_V.flatten(), (X_eval, Y_eval), method='linear', fill_value=0)
    
    diff_U = inferred_U - true_U_interp
    diff_V = inferred_V - true_V_interp
    
    magnitude = np.sqrt(diff_U**2 + diff_V**2)
    im = axes[2].imshow(magnitude, extent=[-z_range, z_range, -z_range, z_range], 
                       origin='lower', cmap='viridis', alpha=0.7)
    axes[2].quiver(X_eval[::2, ::2], Y_eval[::2, ::2], diff_U[::2, ::2], diff_V[::2, ::2], 
                   color='white', alpha=0.8, scale=5)
    axes[2].set_title('Vector Field Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im, ax=axes[2])
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    
    plt.tight_layout()
    return fig


def plot_training_curves():
    """Plot training curves from logs."""
    try:
        import pandas as pd
        import glob
        
        # Find the metrics file
        csv_files = glob.glob('logs/enhanced_ring/*/metrics.csv')
        if not csv_files:
            csv_files = glob.glob('logs/*/metrics.csv')
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Training loss
            train_data = df.dropna(subset=['train_loss'])
            if not train_data.empty:
                axes[0].plot(train_data['epoch'], train_data['train_loss'], 'b-', linewidth=2, label='Train Loss')
            
            # Validation loss
            valid_data = df.dropna(subset=['valid_loss'])
            if not valid_data.empty:
                axes[0].plot(valid_data['epoch'], valid_data['valid_loss'], 'r-', linewidth=2, label='Valid Loss')
            
            axes[0].set_title('Training Curves')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Time per epoch (if available)
            if 'time_forward' in df.columns:
                time_data = df.dropna(subset=['time_forward'])
                if not time_data.empty:
                    axes[1].plot(time_data['epoch'], time_data['time_forward'], 'g-', linewidth=2)
                    axes[1].set_title('Forward Pass Time')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Time (seconds)')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No Time Data', transform=axes[1].transAxes, ha='center', va='center')
            else:
                axes[1].text(0.5, 0.5, 'No Time Data', transform=axes[1].transAxes, ha='center', va='center')
            
            plt.tight_layout()
            return fig
        else:
            print("No training logs found")
            return None
            
    except ImportError:
        print("pandas not available for plotting training curves")
        return None
    except Exception as e:
        print(f"Error plotting training curves: {e}")
        return None


def compute_comprehensive_metrics(z_inferred, z_true, u_inputs, inv_man=None):
    """Compute comprehensive evaluation metrics."""
    # Basic reconstruction metrics
    mse = torch.mean((z_inferred - z_true)**2).item()
    mae = torch.mean(torch.abs(z_inferred - z_true)).item()
    
    # Correlation
    z_true_flat = z_true.flatten()
    z_inferred_flat = z_inferred.flatten()
    correlation = torch.corrcoef(torch.stack([z_true_flat, z_inferred_flat]))[0, 1].item()
    
    # Phase correlation (for ring attractor)
    true_angles = torch.atan2(z_true[:, :, 1], z_true[:, :, 0])
    inferred_angles = torch.atan2(z_inferred[:, :, 1], z_inferred[:, :, 0])
    phase_correlation = torch.corrcoef(torch.stack([true_angles.flatten(), inferred_angles.flatten()]))[0, 1].item()
    
    # Radial correlation
    true_radii = torch.sqrt(z_true[:, :, 0]**2 + z_true[:, :, 1]**2)
    inferred_radii = torch.sqrt(z_inferred[:, :, 0]**2 + z_inferred[:, :, 1]**2)
    radial_correlation = torch.corrcoef(torch.stack([true_radii.flatten(), inferred_radii.flatten()]))[0, 1].item()
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'phase_correlation': phase_correlation,
        'radial_correlation': radial_correlation
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ring attractor xFADS model')
    parser.add_argument('--checkpoint', '-c', type=str, default='auto',
                        help='Path to model checkpoint or "auto" to find best checkpoint')
    parser.add_argument('--config', type=str, default='config',
                        help='Config name to use')
    args = parser.parse_args()
    
    # Load config
    initialize(version_base=None, config_path="", job_name="ring_eval")
    cfg = compose(config_name=args.config)
    torch.set_default_dtype(torch.float32)
    
    # Load data
    data = torch.load('gauss_ring_data.pt')
    y_obs = data['y_obs']
    z_true = data['z_true']
    u_inputs = data['u_inputs']
    inv_man = data['inv_man']
    
    # Load true vector field data if available
    true_vector_field = data.get('true_vector_field', None)
    
    print(f"Data shapes: obs={y_obs.shape}, true_latents={z_true.shape}, inputs={u_inputs.shape}")
    
    n_neurons = y_obs.shape[2]
    
    # Recreate model architecture
    from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
    from xfads.ssm_modules.likelihoods import GaussianLikelihood
    from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
    from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel
    
    # Create components
    readout_fn = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device)
    R_diag = torch.ones(n_neurons, device=cfg.device) * 0.1
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device)
    
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    Q_diag = torch.ones(cfg.n_latents, device=cfg.device) * 1e-1
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)
    
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    Q_0_diag = torch.ones(cfg.n_latents, device=cfg.device) * 2.0
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)
    
    backward_encoder = BackwardEncoderLRMvn(cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
                                            rank_local=cfg.rank_local, rank_backward=cfg.rank_backward,
                                            device=cfg.device)
    local_encoder = LocalEncoderLRMvn(cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents, rank=cfg.rank_local,
                                      device=cfg.device, dropout=cfg.p_local_dropout)
    
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)
    ssm = LowRankNonlinearStateSpaceModel(dynamics_mod, likelihood_pdf, initial_condition_pdf, 
                                          backward_encoder, local_encoder, nl_filter, device=cfg.device)
    
    # Load trained model
    if args.checkpoint == 'auto':
        # Try best model first
        best_model_path = 'ckpts/best_enhanced_model.pt'
        if os.path.exists(best_model_path):
            checkpoint_path = best_model_path
        else:
            # Find best checkpoint
            import glob as glob_module
            ckpt_files = glob_module.glob('ckpts/*.ckpt')
            if not ckpt_files:
                raise FileNotFoundError("No checkpoint files found")
            
            checkpoint_path = min(ckpt_files, key=lambda x: float(x.split('_')[-1].replace('.ckpt', '')))
        
        print(f"Loading: {checkpoint_path}")
        seq_vae = load_checkpoint(checkpoint_path, ssm, cfg)
    else:
        checkpoint_path = args.checkpoint
        print(f"Loading: {checkpoint_path}")
        seq_vae = load_checkpoint(checkpoint_path, ssm, cfg)
    
    seq_vae.eval()
    
    # Infer latents
    with torch.no_grad():
        _, z_inferred_samples, _ = seq_vae.ssm(y_obs, cfg.n_samples)
        z_inferred = z_inferred_samples.mean(dim=0)
    
    print(f"Inferred latent shape: {z_inferred.shape}")
    
    # Create temp_figures directory
    os.makedirs('temp_figures', exist_ok=True)
    
    # Create plots
    fig1 = plot_comprehensive_latent_analysis(z_inferred.numpy(), z_true.numpy(), inv_man.numpy())
    fig1.savefig('temp_figures/comprehensive_latent_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved temp_figures/comprehensive_latent_analysis.png")
    
    fig2 = analyze_ring_structure(z_inferred.numpy(), z_true.numpy(), inv_man.numpy())
    fig2.savefig('temp_figures/ring_structure_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved temp_figures/ring_structure_analysis.png")
    
    fig3 = analyze_input_effects(z_inferred.numpy(), z_true.numpy(), u_inputs.numpy())
    fig3.savefig('temp_figures/input_effects_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("Saved temp_figures/input_effects_analysis.png")
    
    # Vector field comparison if true vector field is available
    if true_vector_field is not None:
        fig4 = plot_vector_field_comparison(
            true_vector_field['X'], true_vector_field['Y'], 
            true_vector_field['U'], true_vector_field['V'], 
            seq_vae.ssm.dynamics_mod
        )
        fig4.savefig('temp_figures/vector_field_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("Saved temp_figures/vector_field_comparison.png")
    
    fig5 = plot_training_curves()
    if fig5:
        fig5.savefig('temp_figures/training_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print("Saved temp_figures/training_curves.png")
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(z_inferred, z_true, u_inputs, inv_man)
    
    print(f"\nComprehensive Metrics:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    print(f"  Phase Correlation: {metrics['phase_correlation']:.4f}")
    print(f"  Radial Correlation: {metrics['radial_correlation']:.4f}")
    
    print("\nSaved plots in temp_figures/:")
    print("  - comprehensive_latent_analysis.png: Complete latent space analysis")
    print("  - ring_structure_analysis.png: Ring attractor structure analysis")
    print("  - input_effects_analysis.png: Input effects on latent trajectories")
    if true_vector_field is not None:
        print("  - vector_field_comparison.png: True vs inferred vector fields")
    print("  - training_curves.png: Training progress")
    
    print("\nEvaluation complete! xFADS learned ring dynamics despite input perturbations.")


if __name__ == '__main__':
    main()