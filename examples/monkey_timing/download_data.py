import yaml
import torch

from itertools import product
from nlb_tools.nwb_interface import NWBDataset


def get_int_to_verbose_map():
    options = [('S', 'L'), ('E', 'H'), ('1', '2', '3', '4', '5')]
    combinations = list(product(*options))
    mapping = {i: combination for i, combination in enumerate(combinations)}

    return mapping


def get_verbose_to_int_map():
    options = [('S', 'L'), ('E', 'H'), ('1', '2', '3', '4', '5')]
    combinations = list(product(*options))
    reverse_mapping = {combination: i for i, combination in enumerate(combinations)}

    return reverse_mapping


def main():
    datapath = 'data/000130/sub-Haydn/'
    dataset = NWBDataset(datapath)
    save_root_path = 'data/'

    # Extract neural data and lagged hand velocity
    binsize = 10
    n_neurons = 54
    dataset.resample(binsize)

    start = -1300
    end = 1300
    trial_length = (end - start) // binsize

    verbose_to_int_map = get_verbose_to_int_map()
    int_to_verbose_map = get_int_to_verbose_map()

    # Extract neural data
    trial_info = dataset.trial_info  # .dropna()
    trial_info['color'] = None
    trial_info['position_id'] = None
    trial_data = dataset.make_trial_data(align_field='set_time', align_range=(start, end))
    n_trials = trial_data.shape[0] // trial_length

    y = []
    tp = []
    ts = []
    task_id = []

    print('done')
    print(trial_data.columns)
    count = 0
    for trial_id, trial in trial_data.groupby('trial_id'):
        trial_id_trial_info = trial_info[trial_info['trial_id'] == trial_id]
        is_outlier_t = trial_id_trial_info['is_outlier'].iloc[0]
        tp_t = torch.tensor(trial_id_trial_info['tp'].iloc[0])
        ts_t = torch.tensor(trial_id_trial_info['ts'].iloc[0])
        is_short_t = trial_id_trial_info['is_short'].iloc[0]
        is_eye_t = trial_id_trial_info['is_eye'].iloc[0]

        if is_outlier_t or tp_t < 0:
            continue

        if is_short_t:
            task_str_1 = 'S'

            if ts_t == 480:
                task_str_3 = '1'
            elif ts_t == 560:
                task_str_3 = '2'
            elif ts_t == 640:
                task_str_3 = '3'
            elif ts_t == 720:
                task_str_3 = '4'
            elif ts_t == 800:
                task_str_3 = '5'
        else:
            task_str_1 = 'L'

            if ts_t == 800:
                task_str_3 = '1'
            elif ts_t == 900:
                task_str_3 = '2'
            elif ts_t == 1000:
                task_str_3 = '3'
            elif ts_t == 1100:
                task_str_3 = '4'
            elif ts_t == 1200:
                task_str_3 = '5'

        if is_eye_t:
            task_str_2 = 'E'
        else:
            task_str_2 = 'H'

        y_heldin_t = torch.tensor(trial.spikes.values)
        y_heldout_t = torch.tensor(trial.heldout_spikes.values)
        y_t = torch.concat([y_heldin_t, y_heldout_t], dim=-1)
        y.append(y_t.reshape(1, trial_length, n_neurons))

        task_id_key = (task_str_1, task_str_2, task_str_3)
        task_id_int = verbose_to_int_map[task_id_key]
        task_id.append(torch.tensor(task_id_int).unsqueeze(-1))

        tp.append(torch.tensor(tp_t).unsqueeze(-1))
        ts.append(torch.tensor(ts_t).unsqueeze(-1))

        if is_outlier_t:
            count += 1

    y = torch.concat(y, dim=0)
    task_id = torch.concat(task_id, dim=0)

    subset_ex = 10
    subset_ex_loc = torch.where(task_id == subset_ex)[0]

    y_subset = y[subset_ex_loc]
    y_psth = y_subset.mean(dim=0)

    ts = torch.stack(ts, dim=0)
    tp = torch.stack(tp, dim=0)

    with open('data/old_data/int_condition_map.yaml', 'w') as outfile:
        yaml.dump(int_to_verbose_map, outfile, default_flow_style=False)

    train_data, valid_data, test_data = {}, {}, {}
    untrained_trials = 300
    seq_len = trial_length

    train_data['y_obs'] = y[:-untrained_trials]
    train_data['task_id'] = task_id[:-untrained_trials]
    train_data['ts'] = ts[:-untrained_trials]
    train_data['tp'] = tp[:-untrained_trials]
    train_data['n_neurons_enc'] = n_neurons
    train_data['n_neurons_obs'] = n_neurons
    train_data['n_time_bins_enc'] = seq_len

    valid_data['y_obs'] = y[-untrained_trials:-untrained_trials // 2]
    valid_data['task_id'] = task_id[-untrained_trials:-untrained_trials // 2]
    valid_data['ts'] = ts[-untrained_trials:-untrained_trials // 2]
    valid_data['tp'] = tp[-untrained_trials:-untrained_trials // 2]
    valid_data['n_neurons_enc'] = n_neurons
    valid_data['n_neurons_obs'] = n_neurons
    valid_data['n_time_bins_enc'] = seq_len

    test_data['y_obs'] = y[-untrained_trials // 2:]
    test_data['task_id'] = task_id[-untrained_trials // 2:]
    test_data['ts'] = ts[-untrained_trials // 2:]
    test_data['tp'] = tp[-untrained_trials // 2:]
    test_data['n_neurons_enc'] = n_neurons
    test_data['n_neurons_obs'] = n_neurons
    test_data['n_time_bins_enc'] = seq_len

    torch.save(train_data, save_root_path + f'data_train_{binsize}ms.pt')
    torch.save(valid_data, save_root_path + f'data_valid_{binsize}ms.pt')
    torch.save(test_data, save_root_path + f'data_test_{binsize}ms.pt')


if __name__ == '__main__':
    main()

# Index(['trial_id', 'start_time', 'end_time', 'go_time', 'split', 'fix_on_time',
#        'fix_time', 'target_on_time', 'ready_time', 'set_time',
#        'target_acq_time', 'reward_time', 'bad_time', 'is_short', 'is_eye',
#        'theta', 'ts', 'tp', 'fix_time_dur', 'target_time_dur', 'iti',
#        'reward_dur', 'is_outlier'],
#       dtype='object')