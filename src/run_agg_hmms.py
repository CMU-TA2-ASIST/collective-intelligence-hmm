import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import itertools
import random
from hmm import unsupervised_HMM, visualize_O, visualize_sparsities
from create_groups import merge_r1_groups_single_emotion
from scipy import stats

uid_to_groups = merge_r1_groups_single_emotion()

def run_hmms(group_to_state_id_data_w_pid, group_to_binary_state_data, group_to_state_mapping, group_num_to_title, n_states, savename=''):
    group_no_to_hmm = {}
    for group_no in group_to_state_id_data_w_pid:
        group_no_to_hmm[group_no] = {}

        state_id_data = group_to_state_id_data_w_pid[group_no]
        state_binary_data = group_to_binary_state_data[group_no]

        state_id_to_state = group_to_state_mapping[group_no]['id_to_vec']
        state_to_state_id = group_to_state_mapping[group_no]['vec_to_id']

        hmm_input_X = []
        index_to_team = {}
        counter = 0
        for p_uid in state_id_data:
            hmm_input_X.append(state_id_data[p_uid])
            index_to_team[counter] = p_uid
            counter += 1

        hmm_input_X = np.array(hmm_input_X)

        N_iters = 100
        # n_states = 6

        test_unsuper_hmm = unsupervised_HMM(hmm_input_X, n_states, N_iters)

        # print('emission', test_unsuper_hmm.generate_emission(10))
        hidden_seqs = {}
        team_num_to_seq_probs = {}
        max_probs = []
        for j in range(len(hmm_input_X)):
            # print("team", team_numbers[j])
            # print("reindex", X[j][:50])
            team_id = index_to_team[j]
            viterbi_output, all_sequences_and_probs = test_unsuper_hmm.viterbi_all_probs(hmm_input_X[j])
            alphas = test_unsuper_hmm.probability_alphas(hmm_input_X[j])
            # pdb.set_trace()
            team_num_to_seq_probs[team_id] = all_sequences_and_probs
            hidden_seqs[team_id] = [int(x) for x in viterbi_output]
            max_probs.append(alphas)

        group_no_to_hmm[group_no]['hmm'] = test_unsuper_hmm
        print("max_probs: ", np.mean(max_probs))

        # hidden_state_to_mean_value = {1:[], 2:[], 3:[], 4:[]}
        hidden_state_to_mean_value_list = {}

        # team_id_map_to_state_id_sequence
        for team_id in hidden_seqs:
            for i in range(len(hidden_seqs[team_id])):
                hidden_state = hidden_seqs[team_id][i]
                state = state_binary_data[team_id][i]
                mean_value = sum(list(state))

                if hidden_state not in hidden_state_to_mean_value_list:
                    hidden_state_to_mean_value_list[hidden_state] = []

                hidden_state_to_mean_value_list[hidden_state].append(mean_value)

        hidden_state_to_mean_value = {}
        for hidden_state in hidden_state_to_mean_value_list:
            hidden_state_to_mean_value[hidden_state] = np.mean(hidden_state_to_mean_value_list[hidden_state])

        # print("hidden_state_to_mean_value", hidden_state_to_mean_value)
        sorted_hidden_states = {k: v for k, v in sorted(hidden_state_to_mean_value.items(), key=lambda item: item[1])}

        new_hidden_state_to_old = dict(enumerate(sorted_hidden_states.keys()))
        old_hidden_state_to_new = {v: k for k, v in new_hidden_state_to_old.items()}

        team_id_to_new_hidden = {}
        for team_id in hidden_seqs:
            team_id_to_new_hidden[team_id] = [old_hidden_state_to_new[x] for x in hidden_seqs[team_id]]

        # Plot



        for team_id in hidden_seqs:
        # for i in range(len(index_to_team_map)):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # legend_labels = []
        #     print("index_to_team_map = ", index_to_team_map)
        #     print("team_id_map_to_new_hidden = ", team_id_map_to_new_hidden)
        # for i in range(4):
        #     team_id = index_to_team[i]
            # legend_labels.append(team_id)
            #         if team_id not in team_id_map_to_new_hidden:
            #             continue
            plt.plot(range(len(team_id_to_new_hidden[team_id])), team_id_to_new_hidden[team_id])
            plt.title(f"CI (Hidden) States for Team: {team_id}")
            # plt.xlabel("Collective Intelligence")

        # plt.legend(legend_labels)

            y = list(range(n_states))
            labels = ['State 1: Low CI', 'State 2', 'State 3: High CI', 'State 4', 'State 5', 'State 6: Good']
            labels = labels[:n_states]
            plt.yticks(y, labels, rotation='horizontal')

            ax.set_yticklabels(labels)

            plt.ylabel("Hidden State (Collective Intelligence)")
            plt.xlabel("Time (Minute)")
            # plt.title(group_num_to_title[group_no])

            plt.savefig(f'{savename}_N={n_states}_agg_seq_hidden_states_{team_id}.png')
            plt.close()

        A = np.array(test_unsuper_hmm.A)
        O = np.array(test_unsuper_hmm.O)

        new_A = []
        new_O = []
        for new_h_i in range((A.shape[0])):
            new_row = []
            for new_h_j in range(A.shape[0]):
                old_h_i = new_hidden_state_to_old[new_h_i]
                old_h_j = new_hidden_state_to_old[new_h_j]

                new_row.append(A[old_h_i, old_h_j])
            new_A.append(new_row)

            new_O.append(O[old_h_i, :])

        group_no_to_hmm[group_no]['new_A'] = np.array(new_A)
        group_no_to_hmm[group_no]['new_O'] = np.array(new_O)
        group_no_to_hmm[group_no]['new_hidden_state_to_old'] = new_hidden_state_to_old
        group_no_to_hmm[group_no]['old_hidden_state_to_new'] = old_hidden_state_to_new

        #     state_id_to_state = dict(enumerate(unique_states_list))
        #     state_to_state_id = {v: k for k, v in state_id_to_state.items()}
        group_no_to_hmm[group_no]['state_id_to_state'] = state_id_to_state
        group_no_to_hmm[group_no]['state_to_state_id'] = state_to_state_id
        group_no_to_hmm[group_no]['old_hidden_state_to_mean_value'] = hidden_state_to_mean_value

        group_no_to_hmm[group_no]['hidden_seqs'] = hidden_seqs

        visualize_sparsities(new_A, new_O, savename)
        visualize_O(group_no_to_hmm[group_no], savename)

    with open(f'{savename}_minimap_group_no_to_hmm_{savename}.pickle', 'wb') as handle:
        pickle.dump(group_no_to_hmm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return group_no_to_hmm

def compute_loss(group_no_to_hmm, group_to_state_id_data_w_pid, group_to_binary_state_data, group_to_state_mapping, group_num_to_title, savename=''):
    '''
    This function computes the loss for each group and the total loss for all groups.
    :param group_no_to_hmm:  dictionary of group number to hmm
    :param group_to_state_id_data_w_pid: dictionary of group number to state id data with pid
    :param group_to_binary_state_data: dictionary of group number to binary state data
    :param group_to_state_mapping: dictionary of group number to state mapping
    :param group_num_to_title: dictionary of group number to title
    :param savename:   name of the file to save the results
    :return: loss for each group and the total loss for all groups

    '''
    group_to_loss = {}
    total_loss = []

    group_to_likelihood = {}
    total_likelihood = []
    for group_no in group_to_state_id_data_w_pid:
        group_hmm = group_no_to_hmm[group_no]['hmm']
        A = np.array(group_hmm.A)
        O = np.array(group_hmm.O)


        team_id_to_state_id_sequence = group_to_state_id_data_w_pid[group_no]
        state_binary_data = group_to_binary_state_data[group_no]

        state_id_to_state = group_to_state_mapping[group_no]['id_to_vec']
        state_to_state_id = group_to_state_mapping[group_no]['vec_to_id']

        average_loss = []
        average_likelihood = []

        for team_id in team_id_to_state_id_sequence:
            seq = team_id_to_state_id_sequence[team_id]

            seq = np.array(seq)

            alphas = group_hmm.probability_alphas(seq)

            # recommendations = []
            for t in range(1, len(seq)):
                partial_seq = seq[:t]
                viterbi_output, all_sequences_and_probs = group_hmm.viterbi_all_probs(partial_seq)

                # Get max likelihood
                current_hidden = int(viterbi_output[-1])

                curr_obs_state = state_id_to_state[seq[t - 1]]

                normalized_hidden_probs = A[current_hidden, :] / sum(A[current_hidden, :])
                next_hidden_predicted, next_hidden_prob = np.argmax(normalized_hidden_probs), max(
                    normalized_hidden_probs)

                valid_obs = []
                for j in range(O.shape[1]):
                    if j not in state_id_to_state:
                        print("continuing")
                        continue
                    obs = state_id_to_state[j]
                    if obs[3:] == curr_obs_state[0:3]:
                        valid_obs.append(O[current_hidden, j])
                    else:
                        valid_obs.append(0)

                if sum(valid_obs) > 0:
                    valid_obs /= sum(valid_obs)

                next_obs_predicted_idx, next_obs_prob = np.argmax(valid_obs), max(valid_obs)
                next_obs_predicted_state = state_id_to_state[next_obs_predicted_idx]

                true_next_obs_state = state_id_to_state[seq[t]]

                loss = np.array(next_obs_predicted_state[0:3]) - np.array(true_next_obs_state[0:3])
                loss = sum([abs(elem) for elem in loss])
                average_loss.append(loss)
                average_likelihood.append(alphas)


        group_to_loss[group_no] = average_loss
        total_loss.extend(average_loss)

        group_to_likelihood[group_no] = average_likelihood
        total_likelihood.extend(average_likelihood)

    return group_to_loss, total_loss, group_to_likelihood, total_likelihood




def compute_hmm_all_affect(n_states):
    # n_states = 10
    print("n_states = ", n_states)
    # random_states = [25, 7399, 1383, 7867, 6502, 6046, 7226, 7520, 8200, 9448, 3904, 4982, 4121, 4283, 6473, 9619, 9223,
    #                 2900, 5242, 7647, 5703, 9115, 1231, 381, 5208, 4601, 4499, 6070, 8916, 2082, 1319, 9970, 5029, 2388,
    #                 4174, 1622, 8374, 4394, 3543, 2687, 1386, 6219, 5387, 5110, 5910, 8265, 4139, 8728, 4977, 7715,
    #                 9933, 6287, 3742, 7111, 5446, 673, 8329, 6524, 3503, 862, 8302, 9845, 220, 1406, 534, 3839, 7851,
    #                 6631, 6289, 4774, 5086, 1896, 4288, 6946, 5173, 7891, 222, 4304, 5898, 3257, 2993, 5719, 8295,
    #                 5596, 9810, 6342, 9119, 1816, 92, 2679, 9063, 6622, 9056, 9925, 6995, 7743, 287, 6082, 4992, 3018, 1424]
    random_states = [25, 7399, 1383, 7867, 6502, 6046, 7226, 7520, 8200, 9448, 3904, 4982]


    successes = []
    fails = []
    for r_state in random_states:
        print("r_state = ", r_state)

        try:
        # for t in range(1):
            for savename in [f'AGG_RAND{r_state}_TRAIN']:
                # print("AFFECT: ", savename)
                group_num_to_title = {
                    1: 'High Anger, High Anxiety',
                    2: 'Low Anger, High Anxiety',
                    3: 'High Anger, Low Anxiety',
                    4: 'Low Anger, Low Anxiety',
                }

                with open(f'minimap_data_aug15/minimap_group_to_state_data_{savename}.pickle', 'rb') as handle:
                    group_to_state_id_data = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_state_data_w_pid_{savename}.pickle', 'rb') as handle:
                    group_to_state_id_data_w_pid = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_binary_state_data_{savename}.pickle', 'rb') as handle:
                    group_to_binary_state_data = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_state_mapping_{savename}.pickle', 'rb') as handle:
                    group_to_state_mapping = pickle.load(handle)

                savename = f'{savename}_N={n_states}'
                group_no_to_hmm = run_hmms(group_to_state_id_data_w_pid, group_to_binary_state_data, group_to_state_mapping, group_num_to_title, n_states, savename=savename)

                group_to_loss, total_loss, group_to_likelihood, total_likelihood = compute_loss(group_no_to_hmm,
                                                                                                group_to_state_id_data_w_pid,
                                                                                                group_to_binary_state_data,
                                                                                                group_to_state_mapping,
                                                                                                group_num_to_title,
                                                                                                savename=savename)
                print("\n\n\nTRAIN")
                print("AFFECT: ", savename)
                print("MEAN Loss: ", np.mean(total_loss))
                print("STD Loss: ", np.std(total_loss))
                print()

                # print("AFFECT: ", savename)
                print("MEAN Likelihood: ", np.mean(total_likelihood))
                print("STD Likelihood: ", np.std(total_likelihood))
                print()

                with open(f'minimap_data_aug15/total_loss_{savename}.pickle', 'wb') as handle:
                    pickle.dump(total_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/group_to_loss_{savename}.pickle', 'wb') as handle:
                    pickle.dump(group_to_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/group_to_likelihood_{savename}.pickle', 'wb') as handle:
                    pickle.dump(group_to_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/total_likelihood_{savename}.pickle', 'wb') as handle:
                    pickle.dump(total_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)


                test_openname =  f'AGG_RAND{r_state}_TEST'
                test_savename = f'AGG_RAND{r_state}_TEST_N={n_states}'

                # savename = 'AGG_TEST'
                with open(f'minimap_data_aug15/minimap_group_to_state_data_{test_openname}.pickle', 'rb') as handle:
                    group_to_state_id_data = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_state_data_w_pid_{test_openname}.pickle', 'rb') as handle:
                    group_to_state_id_data_w_pid = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_binary_state_data_{test_openname}.pickle', 'rb') as handle:
                    group_to_binary_state_data = pickle.load(handle)

                with open(f'minimap_data_aug15/minimap_group_to_state_mapping_{test_openname}.pickle', 'rb') as handle:
                    group_to_state_mapping = pickle.load(handle)


                group_to_loss, total_loss, group_to_likelihood, total_likelihood = compute_loss(group_no_to_hmm, group_to_state_id_data_w_pid, group_to_binary_state_data, group_to_state_mapping, group_num_to_title, savename=test_savename)
                print("\n\n\nTEST")
                print("AFFECT: ", savename)
                print("MEAN Loss: ", np.mean(total_loss))
                print("STD Loss: ", np.std(total_loss))
                print()

                # print("AFFECT: ", savename)
                print("MEAN Likelihood: ", np.mean(total_likelihood))
                print("STD Likelihood: ", np.std(total_likelihood))
                print()

                with open(f'minimap_data_aug15/total_loss_{test_savename}.pickle', 'wb') as handle:
                    pickle.dump(total_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/group_to_loss_{test_savename}.pickle', 'wb') as handle:
                    pickle.dump(group_to_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/group_to_likelihood_{test_savename}.pickle', 'wb') as handle:
                    pickle.dump(group_to_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'minimap_data_aug15/total_likelihood_{test_savename}.pickle', 'wb') as handle:
                    pickle.dump(total_likelihood, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"SUCCEEDED at random seed: {r_state}")
            successes.append(r_state)
            # if len(successes) > 10:
            #     break
        except:

            print(f"FAILED at random seed: {r_state}")
            fails.append(r_state)

    print('successes = ', successes)
    print('fails = ', fails)

def plot_losses():
    colors=['red', 'blue', 'green', 'orange']
    affect_to_loss = {}
    affects = ['ANGERANX_TEST', 'AGG_TEST']
    for i in range(len(affects)):
        savename  = affects[i]
        # print("AFFECT: ", savename)
        group_num_to_title = {
            1: 'High Anger, High Anxiety',
            2: 'Low Anger, High Anxiety',
            3: 'High Anger, Low Anxiety',
            4: 'Low Anger, Low Anxiety',
        }

        with open(f'minimap_data/minimap_group_to_state_data_{savename}.pickle', 'rb') as handle:
            group_to_state_id_data = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_state_data_w_pid_{savename}.pickle', 'rb') as handle:
            group_to_state_id_data_w_pid = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_binary_state_data_{savename}.pickle', 'rb') as handle:
            group_to_binary_state_data = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_state_mapping_{savename}.pickle', 'rb') as handle:
            group_to_state_mapping = pickle.load(handle)

        with open(f'minimap_data/minimap_group_no_to_hmm_{savename.split("_")[0]}_TRAIN.pickle', 'rb') as handle:
            group_no_to_hmm = pickle.load(handle)

        group_to_loss, total_loss = compute_loss(group_no_to_hmm, group_to_state_id_data_w_pid, group_to_binary_state_data, group_to_state_mapping, group_num_to_title, savename=savename)
        # print("AFFECT: ", savename)
        print("AFFECT: ", savename)
        print("MEAN Loss: ", np.mean(total_loss))
        print("STD Loss: ", np.std(total_loss))
        print()
        affect_to_loss[savename] = total_loss

    # Create lists for the plot
    groups = [affect for affect in affect_to_loss]
    x_pos = np.arange(len(groups))
    means = [np.mean(affect_to_loss[group]) for group in affect_to_loss]
    std_devs = [stats.sem(affect_to_loss[group]) for group in affect_to_loss]

    # Build the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x_pos, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)
    ax.set_ylabel('L1 Loss in Predicting Next State')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_title('Loss of HMM Predictions by Affect Split')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('minimap_data/minimap_losses_by_affect_with_error_bars_with_anganx_v_agg_test_1.png')
    plt.close()

    affects = ['ANGERANX_TRAIN', 'AGG_TRAIN']
    affect_to_loss = {}
    for i in range(len(affects)):
        savename = affects[i]
        # print("AFFECT: ", savename)
        group_num_to_title = {
            1: 'High Anger, High Anxiety',
            2: 'Low Anger, High Anxiety',
            3: 'High Anger, Low Anxiety',
            4: 'Low Anger, Low Anxiety',
        }

        with open(f'minimap_data/minimap_group_to_state_data_{savename}.pickle', 'rb') as handle:
            group_to_state_id_data = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_state_data_w_pid_{savename}.pickle', 'rb') as handle:
            group_to_state_id_data_w_pid = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_binary_state_data_{savename}.pickle', 'rb') as handle:
            group_to_binary_state_data = pickle.load(handle)

        with open(f'minimap_data/minimap_group_to_state_mapping_{savename}.pickle', 'rb') as handle:
            group_to_state_mapping = pickle.load(handle)

        with open(f'minimap_data/minimap_group_no_to_hmm_{savename.split("_")[0]}_TRAIN.pickle', 'rb') as handle:
            group_no_to_hmm = pickle.load(handle)

        group_to_loss, total_loss = compute_loss(group_no_to_hmm, group_to_state_id_data_w_pid,
                                                 group_to_binary_state_data, group_to_state_mapping, group_num_to_title,
                                                 savename=savename)
        # print("AFFECT: ", savename)
        print("AFFECT: ", savename)
        print("MEAN Loss: ", np.mean(total_loss))
        print("STD Loss: ", np.std(total_loss))
        print()
        affect_to_loss[savename] = total_loss

    # Create lists for the plot
    groups = [affect for affect in affect_to_loss]
    x_pos = np.arange(len(groups))
    means = [np.mean(affect_to_loss[group]) for group in affect_to_loss]
    std_devs = [stats.sem(affect_to_loss[group]) for group in affect_to_loss]

    # Build the plot
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x_pos, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10, color=colors)
    ax.set_ylabel('L1 Loss in Predicting Next State')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_title('Loss of HMM Predictions by Affect Split')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('minimap_data/minimap_losses_by_affect_with_error_bars_with_anganx_v_agg_train_1.png')
    plt.close()

if __name__ == "__main__":
    for n_state in range(2, 11):
        compute_hmm_all_affect(n_state)
    # plot_losses()



