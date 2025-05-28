import torch
import numpy as np

from nlb_tools.nwb_interface import NWBDataset


def tuple_mapping(tuples_list):
    count = 0
    tuple_dict = {}
    id_type_dict = {}

    proper_tuples_list = []

    for tup in tuples_list:
        proper_tuple = (int(tup[0]), int(tup[1]))
        proper_tuples_list.append(proper_tuple)

        if proper_tuple not in tuple_dict:
            tuple_dict[proper_tuple] = count
            id_type_dict[count] = proper_tuple
            count += 1

    mapped_list = [tuple_dict[tup] for tup in proper_tuples_list]
    return mapped_list, id_type_dict


def get_expanded_dataset(dataset, binsize, start, end):
    lag = 100

    go_cue = []
    spikes = []
    velocity = []
    trial_type = []
    trial_version = []
    unique_rt = dataset.trial_info.set_index(["rt"]).index.unique().tolist()

    for dx, rt in enumerate(unique_rt):
        if dx == 0:
            continue
        if rt // binsize >= 49:
            continue

        mask = np.all(dataset.trial_info[["rt"]] == rt, axis=1)

        trial_data = dataset.make_trial_data(
            align_field="move_onset_time",
            align_range=(start, end),
            ignored_trials=(~mask),
        )
        lagged_trial_data = dataset.make_trial_data(
            align_field="move_onset_time",
            align_range=(start + lag, end + lag),
            ignored_trials=(~mask),
        )

        velocity_trial = lagged_trial_data.hand_vel.to_numpy()

        trial_length = int(end - start) // binsize
        n_trials = dataset.trial_info[mask].shape[0]

        spikes_per_trial = torch.tensor(
            trial_data.spikes.values.reshape(n_trials, trial_length, -1),
            dtype=torch.float32,
        )
        heldout_spikes_per_trial = torch.tensor(
            trial_data.heldout_spikes.values.reshape(n_trials, trial_length, -1),
            dtype=torch.float32,
        )

        all_spikes_per_trial = np.concatenate(
            [spikes_per_trial, heldout_spikes_per_trial], axis=-1
        )
        velocity_per_trial = velocity_trial.reshape(n_trials, trial_length, -1)
        velocity_per_trial = np.nan_to_num(velocity_per_trial, nan=0.0)

        go_cue.append(
            torch.tensor(
                -start // binsize - dataset.trial_info[mask].rt.values // binsize
            )
        )
        trial_version.append(
            torch.tensor(dataset.trial_info[mask].trial_version.values)
        )
        trial_type.append(torch.tensor(dataset.trial_info[mask].trial_type.values))
        spikes.append(torch.tensor(all_spikes_per_trial))
        velocity.append(torch.tensor(velocity_per_trial))

    go_cue = torch.cat(go_cue)
    trial_version = torch.cat(trial_version)
    trial_type = torch.cat(trial_type)
    velocity = torch.cat(velocity, dim=0)
    spikes = torch.cat(spikes, dim=0)

    trial_ids, id_type_dict = tuple_mapping(
        torch.cat([trial_version.unsqueeze(-1), trial_type.unsqueeze(-1)], dim=-1)
    )
    go_signal = torch.zeros((spikes.shape[0], spikes.shape[1], 1))
    trial_ids = torch.tensor(trial_ids)

    mv_onset = torch.zeros((spikes.shape[0], spikes.shape[1], 1))
    mv_onset[:, -start // binsize] = 1

    for i in range(go_signal.shape[0]):
        go_signal[i, go_cue.type(torch.int)[i]] = 1

    trial_dx = torch.randperm(spikes.shape[0])
    return (
        spikes[trial_dx],
        velocity[trial_dx],
        go_signal[trial_dx],
        mv_onset,
        trial_ids[trial_dx],
        id_type_dict,
    )


def get_expanded_dataset_target_aligned(
    dataset, binsize, start, end, lag=100, mv_signal_lag=10
):
    trial_length = int(end - start) // binsize
    mask = np.all(dataset.trial_info[["trial_id"]] >= 574, axis=1)
    n_trials = mask[mask].shape[0]

    rt = torch.tensor(dataset.trial_info["rt"][mask].values)
    delay = torch.tensor(dataset.trial_info["delay"][mask].values)
    trial_data = dataset.make_trial_data(
        align_field="target_on_time", align_range=(start, end), ignored_trials=~mask
    )
    lagged_trial_data = dataset.make_trial_data(
        align_field="target_on_time",
        align_range=(start + lag, end + lag),
        ignored_trials=~mask,
    )

    velocity = lagged_trial_data.hand_vel.to_numpy()
    velocity = velocity.reshape(n_trials, trial_length, -1)
    velocity = torch.tensor(np.nan_to_num(velocity, nan=0.0))

    heldin_spikes = torch.tensor(
        trial_data.spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    heldout_spikes = torch.tensor(
        trial_data.heldout_spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    spikes = torch.cat([heldin_spikes, heldout_spikes], dim=-1)

    trial_type = torch.tensor(dataset.trial_info[mask].trial_type.values)
    trial_version = torch.tensor(dataset.trial_info[mask].trial_version.values)
    trial_ids, id_type_dict = tuple_mapping(
        torch.cat([trial_version.unsqueeze(-1), trial_type.unsqueeze(-1)], dim=-1)
    )
    trial_ids = torch.tensor(trial_ids)

    go_signal = torch.zeros((n_trials, trial_length, 1))
    mv_onset_signal = torch.zeros((n_trials, trial_length, 1))

    for trial_dx, (rt_trial, delay_trial) in enumerate(zip(rt, delay)):
        rt_bins = (rt_trial // 20).type(torch.int)
        delay_bins = (delay_trial // 20).type(torch.int)

        go_signal[trial_dx, delay_bins - (start // binsize)] = 1.0

        if delay_bins + rt_bins < end // binsize:
            mv_onset_signal[
                trial_dx, delay_bins + rt_bins - (start // 20) - mv_signal_lag
            ] = 1.0

    trial_dx = torch.randperm(spikes.shape[0])
    return (
        spikes[trial_dx],
        velocity[trial_dx],
        go_signal[trial_dx],
        mv_onset_signal[trial_dx],
        trial_ids[trial_dx],
        id_type_dict,
        rt[trial_dx],
        delay[trial_dx],
    )


def get_expanded_dataset_go_aligned(
    dataset, binsize, start, end, lag=100, mv_signal_lag=10
):
    trial_length = int(end - start) // binsize
    mask = np.all(dataset.trial_info[["trial_id"]] >= 574, axis=1)
    n_trials = mask[mask].shape[0]

    rt = torch.tensor(dataset.trial_info["rt"][mask].values)
    delay = torch.tensor(dataset.trial_info["delay"][mask].values)
    trial_data = dataset.make_trial_data(
        align_field="go_cue_time", align_range=(start, end), ignored_trials=~mask
    )
    lagged_trial_data = dataset.make_trial_data(
        align_field="go_cue_time",
        align_range=(start + lag, end + lag),
        ignored_trials=~mask,
    )

    velocity = lagged_trial_data.hand_vel.to_numpy()
    velocity = velocity.reshape(n_trials, trial_length, -1)
    velocity = torch.tensor(np.nan_to_num(velocity, nan=0.0))

    heldin_spikes = torch.tensor(
        trial_data.spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    heldout_spikes = torch.tensor(
        trial_data.heldout_spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    spikes = torch.cat([heldin_spikes, heldout_spikes], dim=-1)

    trial_type = torch.tensor(dataset.trial_info[mask].trial_type.values)
    trial_version = torch.tensor(dataset.trial_info[mask].trial_version.values)
    trial_ids, id_type_dict = tuple_mapping(
        torch.cat([trial_version.unsqueeze(-1), trial_type.unsqueeze(-1)], dim=-1)
    )
    trial_ids = torch.tensor(trial_ids)

    go_signal = torch.zeros((n_trials, trial_length, 1))
    mv_onset_signal = torch.zeros((n_trials, trial_length, 1))

    for trial_dx, (rt_trial, delay_trial) in enumerate(zip(rt, delay)):
        rt_bins = (rt_trial // binsize).type(torch.int)
        # delay_bins = (delay_trial // binsize).type(torch.int)

        go_signal[trial_dx, -start // binsize] = 1.0

        if rt_bins < end // binsize:
            mv_onset_signal[trial_dx, -(start // binsize) + rt_bins - mv_signal_lag] = (
                1.0
            )

    trial_dx = torch.randperm(spikes.shape[0])
    return (
        spikes[trial_dx],
        velocity[trial_dx],
        go_signal[trial_dx],
        mv_onset_signal[trial_dx],
        trial_ids[trial_dx],
        id_type_dict,
        rt[trial_dx],
        delay[trial_dx],
    )


def get_expanded_dataset_move_onset_aligned(
    dataset, binsize, start, end, lag=100, mv_signal_lag=10
):
    trial_length = int(end - start) // binsize
    mask = np.all(dataset.trial_info[["trial_id"]] >= 574, axis=1)
    n_trials = mask[mask].shape[0]

    rt = torch.tensor(dataset.trial_info["rt"][mask].values)
    delay = torch.tensor(dataset.trial_info["delay"][mask].values)
    trial_type = torch.tensor(dataset.trial_info["trial_type"][mask].values)

    trial_data = dataset.make_trial_data(
        align_field="move_onset_time", align_range=(start, end), ignored_trials=~mask
    )
    lagged_trial_data = dataset.make_trial_data(
        align_field="move_onset_time",
        align_range=(start + lag, end + lag),
        ignored_trials=~mask,
    )

    velocity = lagged_trial_data.hand_vel.to_numpy()
    velocity = velocity.reshape(n_trials, trial_length, -1)
    velocity = torch.tensor(np.nan_to_num(velocity, nan=0.0))

    heldin_spikes = torch.tensor(
        trial_data.spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    heldout_spikes = torch.tensor(
        trial_data.heldout_spikes.values.reshape(n_trials, trial_length, -1),
        dtype=torch.float32,
    )
    spikes = torch.cat([heldin_spikes, heldout_spikes], dim=-1)

    trial_type = torch.tensor(dataset.trial_info[mask].trial_type.values)
    trial_version = torch.tensor(dataset.trial_info[mask].trial_version.values)
    trial_ids, id_type_dict = tuple_mapping(
        torch.cat([trial_version.unsqueeze(-1), trial_type.unsqueeze(-1)], dim=-1)
    )
    trial_ids = torch.tensor(trial_ids)

    go_signal = torch.zeros((n_trials, trial_length, 1))
    mv_onset_signal = torch.zeros((n_trials, trial_length, 1))

    for trial_dx, (rt_trial, delay_trial) in enumerate(zip(rt, delay, trial_type)):
        rt_bins = (rt_trial // 20).type(torch.int)

        if -(start // binsize) - rt_bins >= 0:
            go_signal[trial_dx, -(start // binsize) - rt_bins] = 1.0

        if -(start // binsize) - mv_signal_lag > 0:
            mv_onset_signal[trial_dx, -(start // binsize) - mv_signal_lag] = 1.0

    trial_dx = torch.randperm(spikes.shape[0])
    return (
        spikes[trial_dx],
        velocity[trial_dx],
        go_signal[trial_dx],
        mv_onset_signal[trial_dx],
        trial_ids[trial_dx],
        id_type_dict,
        rt[trial_dx],
        delay[trial_dx],
    )


def main():
    # lag = 100
    # end = 500  # + 460 for movement
    # start = -1000  # - 240 for movement
    n_test_trials = 500

    binsize = 20
    save_path = "../data/"
    torch.set_default_dtype(torch.float32)

    align_at = "target_on_time"
    # align_at = 'go_cue_time'
    datapath = "../data/000128/sub-Jenkins/"
    dataset = NWBDataset(datapath)
    dataset.resample(binsize)

    # for align_at in ['move_onset_time', 'target_on_time', 'go_cue_time']:
    for align_at in ["move_onset_time", "target_on_mo_align"]:
        if align_at == "target_on_time":
            (
                spikes,
                velocity,
                go_signal,
                mv_onset,
                trial_ids,
                id_type_dict,
                rt,
                delay,
            ) = get_expanded_dataset_target_aligned(dataset, binsize, -200, 1200)
        elif align_at == "go_cue_time":
            (
                spikes,
                velocity,
                go_signal,
                mv_onset,
                trial_ids,
                id_type_dict,
                rt,
                delay,
            ) = get_expanded_dataset_go_aligned(dataset, binsize, -200, 1100)
        elif align_at == "target_on_mo_align":
            (
                spikes,
                velocity,
                go_signal,
                mv_onset,
                trial_ids,
                id_type_dict,
                rt,
                delay,
            ) = get_expanded_dataset_move_onset_aligned(dataset, binsize, -800, 500)
        else:
            (
                spikes,
                velocity,
                go_signal,
                mv_onset,
                trial_ids,
                id_type_dict,
                rt,
                delay,
            ) = get_expanded_dataset_move_onset_aligned(dataset, binsize, -240, 460)

        def train_test_split(x):
            return (
                x[n_test_trials:],
                x[: n_test_trials // 2],
                x[n_test_trials // 2 : n_test_trials],
            )

        train_data = {}
        valid_data = {}
        test_data = {}

        train_data["y_obs"], test_data["y_obs"], valid_data["y_obs"] = train_test_split(
            spikes
        )
        train_data["y_enc"], test_data["y_enc"], valid_data["y_enc"] = train_test_split(
            spikes
        )
        train_data["y_hld"], test_data["y_hld"], valid_data["y_hld"] = train_test_split(
            spikes
        )
        train_data["velocity"], test_data["velocity"], valid_data["velocity"] = (
            train_test_split(velocity)
        )
        train_data["go_input"], test_data["go_input"], valid_data["go_input"] = (
            train_test_split(go_signal)
        )
        train_data["mv_input"], test_data["mv_input"], valid_data["mv_input"] = (
            train_test_split(mv_onset)
        )
        train_data["trial_id"], test_data["trial_id"], valid_data["trial_id"] = (
            train_test_split(trial_ids)
        )
        train_data["delay"], test_data["delay"], valid_data["delay"] = train_test_split(
            delay
        )
        train_data["rt"], test_data["rt"], valid_data["rt"] = train_test_split(rt)

        test_data["position"] = torch.cumsum(test_data["velocity"], dim=1)
        train_data["position"] = torch.cumsum(train_data["velocity"], dim=1)
        valid_data["position"] = torch.cumsum(valid_data["velocity"], dim=1)

        test_data["input"] = torch.cat(
            [test_data["mv_input"], test_data["go_input"]], dim=-1
        )
        train_data["input"] = torch.cat(
            [train_data["mv_input"], train_data["go_input"]], dim=-1
        )
        valid_data["input"] = torch.cat(
            [valid_data["mv_input"], valid_data["go_input"]], dim=-1
        )

        test_data["n_time_bins_enc"] = spikes.shape[1]
        train_data["n_time_bins_enc"] = spikes.shape[1]
        valid_data["n_time_bins_enc"] = spikes.shape[1]

        torch.save(
            train_data,
            save_path + f"align_at_{align_at}/data_train_{binsize}ms_{align_at}.pt",
        )
        torch.save(
            test_data,
            save_path + f"align_at_{align_at}/data_test_{binsize}ms_{align_at}.pt",
        )
        torch.save(
            valid_data,
            save_path + f"align_at_{align_at}/data_valid_{binsize}ms_{align_at}.pt",
        )
        torch.save(id_type_dict, save_path + "id_type_map.pt")


if __name__ == "__main__":
    main()
