import csv
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_anger_anx_group_r1():
    df_anx = pd.read_csv('../minimap_data/minimap_hmm.csv')
    df_anx = df_anx.replace(r'^s*$', float('NaN'), regex = True)
    df_anx = df_anx.replace(r' ', float('NaN'), regex = True)
    df_anx = df_anx[df_anx['anger_premeasure'].notna()]
    df_anx = df_anx[df_anx['anxiety_premeasure'].notna()]
    uids_list = df_anx['uid'].unique()

    mean_anger = np.mean([float(elem) for elem in df_anx['anger_premeasure'].to_numpy()])
    mean_anx = np.mean([float(elem) for elem in df_anx['anxiety_premeasure'].to_numpy()])

    uid_to_anger_anx = {}
    uid_to_group = {}
    for index, row in df_anx.iterrows():
        uid = row['uid']
        anger = float(row['anger_premeasure'])
        anx = float(row['anxiety_premeasure'])
        if anger >= mean_anger:
            if anx >= mean_anx:
                group = 1
            else:
                group = 3
        else:
            if anx >= mean_anx:
                group = 2
            else:
                group = 4

        uid_to_anger_anx[uid] = {'anger': anger, 'anxiety': anx, 'group': group}
        uid_to_group[uid] = group

    return uid_to_group

def create_anger_group_r1():
    df_anx = pd.read_csv('../minimap_data/minimap_hmm.csv')
    df_anx = df_anx.replace(r'^s*$', float('NaN'), regex=True)
    df_anx = df_anx.replace(r' ', float('NaN'), regex=True)
    df_anx = df_anx[df_anx['anger_premeasure'].notna()]
    df_anx = df_anx[df_anx['anxiety_premeasure'].notna()]
    uids_list = df_anx['uid'].unique()

    mean_anger = np.mean([float(elem) for elem in df_anx['anger_premeasure'].to_numpy()])
    # mean_anx = np.mean([float(elem) for elem in df_anx['anxiety_premeasure'].to_numpy()])

    uid_to_group = {}
    for index, row in df_anx.iterrows():
        uid = row['uid']
        anger = float(row['anger_premeasure'])
        # anx = float(row['anxiety_premeasure'])
        if anger >= mean_anger:
            group = 1
        else:
            group = 2

        uid_to_group[uid] = group

    return uid_to_group

def create_anx_group_r1():
    df_anx = pd.read_csv('../minimap_data/minimap_hmm.csv')
    df_anx = df_anx.replace(r'^s*$', float('NaN'), regex=True)
    df_anx = df_anx.replace(r' ', float('NaN'), regex=True)
    df_anx = df_anx[df_anx['anger_premeasure'].notna()]
    df_anx = df_anx[df_anx['anxiety_premeasure'].notna()]
    uids_list = df_anx['uid'].unique()

    mean_anx = np.mean([float(elem) for elem in df_anx['anxiety_premeasure'].to_numpy()])

    uid_to_group = {}
    for index, row in df_anx.iterrows():
        uid = row['uid']
        anx = float(row['anxiety_premeasure'])
        if anx >= mean_anx:
            group = 1
        else:
            group = 2

        uid_to_group[uid] = group

    return uid_to_group

def create_pos_group_r1():
    df_pos = pd.read_csv('../minimap_data/positive_emotions.csv')
    df_pos = df_pos.replace(r'^s*$', float('NaN'), regex=True)
    df_pos = df_pos.replace(r' ', float('NaN'), regex=True)
    df_pos = df_pos[df_pos['positive_premeasure'].notna()]

    mean_pos = np.mean([float(elem) for elem in df_pos['positive_premeasure'].to_numpy()])

    uid_to_group = {}
    for index, row in df_pos.iterrows():
        uid = row['uid']
        pos = float(row['positive_premeasure'])
        if pos >= mean_pos:
            group = 1
        else:
            group = 2

        uid_to_group[uid] = group

    return uid_to_group

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_random_group_r1():
    df_pos = pd.read_csv('../minimap_data/positive_emotions.csv')
    df_pos = df_pos.replace(r'^s*$', float('NaN'), regex=True)
    df_pos = df_pos.replace(r' ', float('NaN'), regex=True)
    df_pos = df_pos[df_pos['positive_premeasure'].notna()]

    mean_pos = np.mean([float(elem) for elem in df_pos['positive_premeasure'].to_numpy()])

    indices_list = list(range(len(df_pos)))
    random.shuffle(indices_list)

    chunk_1 = indices_list[:int(len(df_pos)/2)]
    chunk_2 = indices_list[int(len(df_pos)/2):]

    uid_to_group = {}
    for index, row in df_pos.iterrows():
        uid = row['uid']
        # pos = np.random.uniform(0,1)
        if index in chunk_1:
            group = 1
        else:
            group = 2

        uid_to_group[uid] = group

    return uid_to_group

def create_agg_group_r1():
    df_pos = pd.read_csv('../minimap_data/positive_emotions.csv')
    df_pos = df_pos.replace(r'^s*$', float('NaN'), regex=True)
    df_pos = df_pos.replace(r' ', float('NaN'), regex=True)
    df_pos = df_pos[df_pos['positive_premeasure'].notna()]

    mean_pos = np.mean([float(elem) for elem in df_pos['positive_premeasure'].to_numpy()])

    indices_list = list(range(len(df_pos)))
    random.shuffle(indices_list)

    chunk_1 = indices_list[:int(len(df_pos)/2)]
    chunk_2 = indices_list[int(len(df_pos)/2):]

    uid_to_group = {}
    for index, row in df_pos.iterrows():
        uid = row['uid']
        # pos = np.random.uniform(0,1)
        group = 1
        uid_to_group[uid] = group

    return uid_to_group


def merge_r1_groups_single_emotion():
    uid_to_anger = create_anger_group_r1()
    uid_to_anx = create_anx_group_r1()
    uid_to_pos = create_pos_group_r1()
    uid_to_rand = create_random_group_r1()
    uid_to_anger_anx = create_anger_anx_group_r1()
    uid_to_agg = create_agg_group_r1()

    merged_uid_to_groups = {}
    for uid in uid_to_anger:
        if uid in uid_to_anx and uid in uid_to_pos:
            merged_uid_to_groups[uid] = {'anger': uid_to_anger[uid],
                                         'anx': uid_to_anx[uid],
                                         'pos': uid_to_pos[uid],
                                         'random': uid_to_rand[uid],
                                         'anger-anx': uid_to_anger_anx[uid],
                                         'agg': uid_to_agg[uid]}

    return merged_uid_to_groups

def create_anger_anx_group_r2():
    df_anx = pd.read_csv('../minimap_data/minimap_hmm.csv')
    df_anx = df_anx.replace(r'^s*$', float('NaN'), regex=True)
    df_anx = df_anx.replace(r' ', float('NaN'), regex=True)
    df_anx = df_anx[df_anx['anger_premeasure'].notna()]
    df_anx = df_anx[df_anx['anxiety_premeasure'].notna()]
    # uids_list = df_anx['uid'].unique()

    mean_anger = np.mean([float(elem) for elem in df_anx['anger_premeasure'].to_numpy()])
    mean_anx = np.mean([float(elem) for elem in df_anx['anxiety_premeasure'].to_numpy()])


    df_r2_anx = pd.read_csv('../minimap_data/round2_minimap_data.csv')
    df_r2_anx = df_r2_anx.replace(r'^s*$', float('NaN'), regex=True)
    df_r2_anx = df_r2_anx.replace(r' ', float('NaN'), regex=True)

    df_r2_anx = df_r2_anx[df_r2_anx['anger_pep1'].notna()]
    df_r2_anx = df_r2_anx[df_r2_anx['anxiety_pep1'].notna()]

    # mean_anger = np.mean([float(elem) for elem in df_r2_anx['anger_pep1'].to_numpy()])
    # mean_anx = np.mean([float(elem) for elem in df_r2_anx['anxiety_pep1'].to_numpy()])

    r2_uid_to_anger_anx = {}
    r2_uid_to_group = {}
    for index, row in df_r2_anx.iterrows():
        uid = row['uid']
        anger = float(row['anger_pep1'])
        anx = float(row['anxiety_pep1'])
        if anger >= mean_anger:
            if anx >= mean_anx:
                group = 1
            else:
                group = 3
        else:
            if anx >= mean_anx:
                group = 2
            else:
                group = 4

        r2_uid_to_anger_anx[uid] = {'anger': anger, 'anxiety': anx, 'group': group}
        r2_uid_to_group[uid] = group

    return r2_uid_to_group










