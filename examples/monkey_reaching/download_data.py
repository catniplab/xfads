import h5py
import torch
import requests

from pathlib import Path


def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    url = 'https://github.com/arsedler9/lfads-torch/blob/main/datasets/mc_maze-20ms-val.h5'
    r = requests.get(url, allow_redirects=True)
    open('data/mc_maze_20ms.h5', 'wb').write(r.content)

    bin_sz = 20
    save_root_path = 'data/'
    data_path = 'data/mc_maze_20ms.h5'

    with h5py.File(data_path, "r") as h5file:
        data_dict = {k: v[()] for k, v in h5file.items()}

    train_data, valid_data, test_data = {}, {}, {}
    seq_len = data_dict['train_encod_data'].shape[1]
    n_valid_trials = data_dict['valid_recon_data'].shape[0]

    train_data['y_obs'] = torch.Tensor(data_dict['train_recon_data'])
    train_data['velocity'] = torch.Tensor(data_dict['train_behavior'])
    train_data['n_neurons_enc'] = data_dict['train_encod_data'].shape[-1]
    train_data['n_neurons_obs'] = data_dict['train_recon_data'].shape[-1]
    train_data['n_time_bins_enc'] = seq_len

    valid_data['y_obs'] = torch.Tensor(data_dict['valid_recon_data'])[:n_valid_trials//2]
    valid_data['velocity'] = torch.Tensor(data_dict['valid_behavior'])[:n_valid_trials//2]
    valid_data['n_neurons_enc'] = data_dict['valid_encod_data'].shape[-1]
    train_data['n_neurons_obs'] = data_dict['valid_recon_data'].shape[-1]
    valid_data['n_time_bins_enc'] = seq_len

    test_data['y_obs'] = torch.Tensor(data_dict['valid_recon_data'])[n_valid_trials//2:]
    test_data['velocity'] = torch.Tensor(data_dict['valid_behavior'])[n_valid_trials//2:]
    test_data['n_neurons_enc'] = data_dict['valid_encod_data'].shape[-1]
    test_data['n_neurons_obs'] = data_dict['valid_recon_data'].shape[-1]
    test_data['n_time_bins_enc'] = seq_len

    torch.save(train_data, save_root_path + f'data_train_{bin_sz}ms.pt')
    torch.save(valid_data, save_root_path + f'data_valid_{bin_sz}ms.pt')
    torch.save(test_data, save_root_path + f'data_test_{bin_sz}ms.pt')
    print(f'train shape: {train_data["y_obs"].shape}')
    print(f'valid shape: {valid_data["y_obs"].shape}')


if __name__ == '__main__':
    main()
