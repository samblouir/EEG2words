from shared_functions import init, load_et_data, load_eeg_data, print_dict

if __name__ == '__main__':
    # Ensure expected folders are created
    init()

    # Loads et data
    et_dict = load_et_data()

    channel_indices = [3, 4, 13]
    eeg_dict = load_eeg_data(channel_indices)

    print_dict(eeg_dict)
    # print_dict(et_dict)

