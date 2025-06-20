import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from analyze_ring_attractor import main as generate_ring_data

def main():
    data = generate_ring_data(return_data=True)
    if data is None:
        print('Error: No data returned from generate_ring_data.')
        return
    y_obs = torch.tensor(data['gaussian_observations'], dtype=torch.float32)
    z_true = torch.tensor(data['trajectories'], dtype=torch.float32)
    inv_man = torch.tensor(data['inv_man'], dtype=torch.float32)
    save_dict = {'y_obs': y_obs, 'z_true': z_true, 'inv_man': inv_man}
    torch.save(save_dict, 'gauss_ring_data.pt')
    print('Saved gauss_ring_data.pt with keys:', list(save_dict.keys()))

if __name__ == '__main__':
    main() 