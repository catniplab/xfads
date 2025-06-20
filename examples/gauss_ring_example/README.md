# Gaussian Ring Attractor Example (XFADS)

This example demonstrates how to use XFADS to recover the latent structure of a 2D ring attractor from noisy Gaussian observations.

## Files
- `config.yaml`: Experiment configuration.
- `generate_data.py`: Generates synthetic ring attractor data and noisy Gaussian observations.
- `inference_lightning.py`: Trains XFADS on the generated data.
- `eval.py`: Evaluates and visualizes the inferred latents vs. the true ring.

## Usage

1. **Generate Data**

   ```bash
   python generate_data.py
   ```
   This will create `gauss_ring_data.pt` in this folder.

2. **Train the Model**

   ```bash
   python inference_lightning.py
   ```
   This will train the model and save the best checkpoint in `ckpts/best_model_path.pt`.

3. **Evaluate and Visualize**

   ```bash
   python eval.py
   ```
   This will plot the inferred latent trajectories against the true ring and invariant manifold.

## Notes
- The data is generated using a perturbed ring attractor system (see `analyze_ring_attractor.py`).
- The model is configured for a 2D latent space to match the ring structure.
- You can adjust noise and other parameters in `config.yaml` or by modifying the data generation script. 