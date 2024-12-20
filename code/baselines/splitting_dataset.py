# the idea is to use the two out of n and 4 out of n data purely for training and only use the 1 out of n for testing.

import pandas as pd
import numpy as np
import torch as t

# this file has some functions to preprocess the data and split it into training, validation and test sets
# the code needs to be cleaned up a bit and the functions need to be documented

def create_dataset(base_path):
    whole_df = pd.read_csv(base_path + 'single-choice-data.csv')
    whole_single_df = pd.read_csv(base_path + 'originally-single-choice-data.csv')
    item_reference_id_map = create_id_map(whole_df)
    multi_df = create_multi_df(whole_df, whole_single_df)
    single_dfs = separate_single_dfs(whole_single_df)
    size_map = {2: 4, 4: 3, 8: 2, 12: 1, 16: 0}
    return multi_df, single_dfs, size_map, item_reference_id_map

def create_multi_df(whole_df, whole_single_df):
    whole_data = split_df(whole_df)
    single_data = split_df(whole_single_df)
    matching_indices = identify_matching_indices(whole_data, single_data)
    multi_df = whole_df.drop(matching_indices)
    assert len(multi_df) == len(whole_df) - len(whole_single_df)
    return multi_df

def split_df(df):
    data = []
    for i, row in df.iterrows():
        reference = row['reference']
        selected = row['selected']
        rest = set([row[col] for col in df.columns if col not in ['reference', 'selected'] and not pd.isna(row[col])])
        data.append((reference, selected, rest))
    return data

def create_id_map(df):
    # Flatten all unique entries from the DataFrame
    unique_entries = pd.unique(df.values.ravel())
    # move NaNs to the end
    unique_entries = np.append(unique_entries[~pd.isna(unique_entries)], unique_entries[pd.isna(unique_entries)])
    # Map each unique entry to a unique ID starting from 0
    item_to_id_map = {entry: idx for idx, entry in enumerate(unique_entries)}

    item_reference_id_map = dict()
    for item in unique_entries:
        for reference in unique_entries:
            if pd.isna(reference):
                item_reference_id_map[(item, reference)] = item_to_id_map[item]
            if item_to_id_map[item] <= item_to_id_map[reference]:
                item_reference_id_map[(item, reference)] = item_to_id_map[item]
            else:
                item_reference_id_map[(item, reference)] = item_to_id_map[item] - 1
    return item_reference_id_map

def separate_single_dfs(whole_single_df):
    single_dfs = dict()
    for i in [16, 12, 8, 4, 2]:
        # drop rows that have less than i+1 non-nan values
        df = whole_single_df.dropna(thresh=i+1)
        single_dfs[i] = df
        # remove the rows from the original df
        whole_single_df = whole_single_df.drop(df.index)
    return single_dfs

def identify_matching_indices(whole_data, single_data):
    matching_indices = []
    for i, (ref1, sel1, rest1) in enumerate(single_data):
        found = False
        for j, (ref2, sel2, rest2) in enumerate(whole_data):
            if ref1 == ref2 and sel1 == sel2 and rest1 == rest2:
                found = True
                matching_indices.append(j)
                break
        # sanity check measure
        if not found:
            print(i, ref1, sel1, rest1)
            print("not found")
            break
    return matching_indices

def transform_df_to_list(df, item_reference_id_map, reference):
    data = []
    for i, row in df.iterrows():
        winning_item = item_reference_id_map[(row['selected'], reference)]
        losing_items = list([item_reference_id_map[(item, reference)] for item in row.drop(['selected', 'reference']) if not pd.isna(item)])
        data.append([winning_item, losing_items])
    return data

def transform_data_to_tensor(data, max_choice_set_size, num_items):
    split_data = [[], [], []]
    # data = [data_dict['context_ids'], data_dict['choice_set_lengths'], data_dict['slot_chosen']]
    for row in data:
        choice_set = [row[0]] + row[1]
        choice_set_size = len(choice_set)
        split_data[1].append(choice_set_size)
        chosen_item = 0
        split_data[2].append(chosen_item)
        while len(choice_set) < max_choice_set_size:
            choice_set.append(num_items)
        split_data[0].append(choice_set)
    tensor_data = list(map(t.tensor, split_data))
    return tensor_data

def index_data_by_reference(multi_df, single_dfs, item_reference_id_map):
    datasets = {}
    for reference in multi_df['reference'].unique():
        dataset = {}
        dataset['multi'] = transform_df_to_list(multi_df[multi_df['reference'] == reference], item_reference_id_map, reference)
        for size, df in single_dfs.items():
            dataset['single_' + str(size)] = transform_df_to_list(df[df['reference'] == reference], item_reference_id_map, reference)
        datasets[reference] = dataset
    return datasets

def compute_split_sizes(dataset, k):
    sizes = [2, 4, 8, 12, 16]
    split_size = dict()
    for size in sizes:
        split_size[size] = int(len(dataset['single_' + str(size)]) / k)
    return sizes, split_size

def train_val_test_kfold_split(dataset, i, k):
    sizes, split_size = compute_split_sizes(dataset, k)
    assert i < k
    training_data = dataset['multi'].copy()
    validation_data = dict()
    test_data = dict()
    for size in sizes:
        single_data = dataset['single_' + str(size)].copy()
        split = split_size[size]
        test_data[size] = single_data[i*split:(i+1)*split]
        validation_data[size] = single_data[:i*split] + single_data[(i+1)*split:]
        training_data += validation_data[size].copy()
    return training_data, validation_data, test_data

def train_val_test_split(dataset, test_fraction=0.1):
    sizes = [2, 4, 8, 12, 16]
    split_size = dict()
    for size in sizes:
        split_size[size] = max(int(len(dataset['single_' + str(size)]) * test_fraction), 1)
    training_data = dataset['multi'].copy()
    validation_data = dict()
    test_data = dict()
    for size in sizes:
        single_data = dataset['single_' + str(size)].copy()
        np.random.shuffle(single_data)
        test_data[size] = single_data[:split_size[size]]
        validation_data[size] = single_data[split_size[size]:]
        training_data += validation_data[size].copy()
    return training_data, validation_data, test_data

if __name__ == '__main__':
    base_path = 'data/similarity/'
    create_dataset(base_path)
