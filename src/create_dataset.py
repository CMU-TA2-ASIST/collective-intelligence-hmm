import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from create_groups import merge_r1_groups_single_emotion
import pickle

from sklearn.model_selection import train_test_split

def get_robot_traj():
    robot_X =[84, 84, 84, 84, 84, 84, 83, 83, 82, 82, 81, 81, 80, 80, 79, 79, 78, 78, 77, 77, 76, 76, 75, 75, 74, 74, 73, 73, 72, 72, 72, 72, 72, 72, 71, 71, 70, 70, 69, 69, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 71, 71, 72, 72, 72, 72, 72, 72, 71, 71, 70, 70, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 68, 68, 68, 68, 67, 67, 66, 66, 65, 65, 64, 64, 63, 63, 62, 62, 61, 61, 60, 60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 62, 62, 62, 62, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 82, 82, 82, 82, 81, 81, 81, 81, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 80, 80, 80, 80, 80, 80, 79, 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 82, 82, 81, 81, 80, 80, 79, 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 77, 77, 77, 77, 77, 77, 77, 77, 76, 76, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 74, 74, 73, 73, 72, 72, 71, 71, 70, 70, 69, 69, 68, 68, 67, 67, 66, 66, 65, 65, 64, 64, 63, 63, 62, 62, 61, 61, 60, 60, 59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 57, 57, 58, 58, 58, 58, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 58, 58, 57, 57, 56, 56, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83, 83, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 88, 88, 87, 87, 86, 86, 86, 86, 86, 86, 86, 86, 85, 85, 84, 84, 83, 83, 82, 82, 81, 81, 80, 80, 79, 79, 78, 78, 77, 77, 77, 77, 76, 76, 76, 76, 75, 75, 75, 75, 74, 74, 74, 74, 73, 73, 72, 72, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 71, 71, 70, 70, 70, 70, 69, 69, 68, 68, 67, 67, 66, 66, 65, 65, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69]
    robot_Y =[38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 42, 42, 41, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 42, 42, 41, 41, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 45, 45, 45, 45, 45, 45, 46, 46, 47, 47, 47, 47, 47, 47, 46, 46, 45, 45, 44, 44, 43, 43, 42, 42, 42, 42, 41, 41, 40, 40, 39, 39, 38, 38, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32, 31, 31, 30, 30, 29, 29, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 34, 34, 34, 34, 34, 34, 35, 35, 36, 36, 37, 37, 37, 37, 37, 37, 36, 36, 35, 35, 34, 34, 33, 33, 32, 32, 31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, 9, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 28, 28, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]


    N = len(robot_X)
    print(N)
    difference = int(N/10)
    print(difference)

    min_to_robot_traj = {(5, 0): []}
    counter = 0
    for minute in [4, 3, 2, 1, 0]:
        for second in [30, 0]:
            counter += 1
            keyname = (minute, second)
            t_robot_x = robot_X[difference * (counter - 1): difference * (counter)]
            t_robot_y = robot_Y[difference * (counter - 1): difference * (counter)]
            traj_robot = [(t_robot_x[i], t_robot_y[i]) for i in range(len(t_robot_x))]
            min_to_robot_traj[keyname] = traj_robot

    return min_to_robot_traj


def compute_minute_to_traj(df_traj):
    traj = df_traj['trajectory'].to_numpy()
    times = df_traj['time_spent'].to_numpy()
    sent_msgs = df_traj['sent_messages'].to_numpy()
    victims = df_traj['target'].to_numpy()

    to_include = True

    minute_to_traj = {}
    minute_to_msgs = {}
    minute_to_victims = {}

    curr_min = 5
    curr_sec = 0

    where_start = np.where(times == 'start')[0]
    if len(where_start) == 0:
        minute_to_traj[(curr_min, curr_sec)] = []
    else:
        #         print("where_start = ", where_start)
        start_idx = where_start[0]
        t = str(times[start_idx])
        traj_t = str(traj[start_idx])
        msgs_t = str(sent_msgs[start_idx])
        vic_t = str(victims[start_idx])

        if traj_t == 'nan':
            curr_traj = []
        else:
            curr_traj = [eval(x) for x in traj_t.split(';')]

        if msgs_t == 'nan':
            curr_msgs = []
        else:
            curr_msgs = msgs_t

        minute_to_traj[(curr_min, curr_sec)] = curr_traj
    #     minute_to_msgs[(curr_min, curr_sec)] = minute_to_msgs
    #     minute_to_victims[(curr_min, curr_sec)] = []

    if curr_sec == 0:
        curr_min -= 1
        curr_sec = 30

    curr_traj = []
    prev_min = 5
    prev_seconds = 60

    for i in range(len(traj)):
        t = str(times[i])
        traj_t = str(traj[i])
        msgs_t = sent_msgs[i]
        vic_t = victims[i]

        if ':' not in t:
            continue

        t_min = int(t.split(":")[0])
        t_sec = int(t.split(":")[1])

        if traj_t != 'nan':
            curr_t_traj = [eval(x) for x in traj_t.split(';')]
            #             print("INPUTS TO ROUND")
            #             print("t = ", t)
            #             print("current = ", (curr_min, curr_sec))
            #             print("previous = ", (prev_min, prev_seconds))
            #             print()

            if t_min == curr_min:
                if t_sec == curr_sec:
                    curr_traj.extend(curr_t_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    if curr_sec == 0:
                        curr_min -= 1
                        curr_sec = 30
                    else:
                        curr_sec = 0
                    curr_traj = []

                elif t_sec > curr_sec:
                    curr_traj.extend(curr_t_traj)

                elif t_sec < curr_sec and curr_sec == 30:
                    diff_in_past_section = abs(prev_seconds - curr_sec)  # 2:45-2:30
                    diff_in_next_section = abs(curr_sec - t_sec)  # 2:30-2
                    # 2- 1:30

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_next_section))
                    next_section_idx = past_section_idx + 1

                    curr_traj.extend(curr_t_traj[:past_section_idx + 1])

                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = []
                    curr_traj.extend(curr_t_traj[next_section_idx:])

                elif t_sec == 0 and curr_sec == 30:
                    diff_in_past_section = abs(prev_seconds - curr_sec)  # 2:45 - 2:30
                    diff_in_next_section = abs(curr_sec - t_sec)  # 2:30- 2
                    # 2 - 1:30

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_next_section))
                    next_section_idx = past_section_idx + 1

                    curr_traj.extend(curr_t_traj[:past_section_idx + 1])

                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = []
                    curr_traj.extend(curr_t_traj[next_section_idx:])

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []

                elif t_sec == 30 and curr_sec == 0:
                    diff_in_past_section = abs(prev_seconds - curr_sec)  # 2:15 - 2:00
                    diff_in_next_section = abs(curr_sec - t_sec)  # 2:00- 1:30
                    # 1:30-1

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_next_section))
                    next_section_idx = past_section_idx + 1

                    curr_traj.extend(curr_t_traj[:past_section_idx + 1])

                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []
                    curr_traj.extend(curr_t_traj[next_section_idx:])

                    curr_sec = 0
                    curr_traj = []

            elif t_min == curr_min - 1:
                if t_sec > curr_sec and curr_sec == 0:

                    diff_in_past_section = abs(prev_seconds - curr_sec)
                    diff_in_next_section = abs(curr_sec - t_sec)

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_next_section))
                    next_section_idx = past_section_idx + 1

                    curr_traj.extend(curr_t_traj[:past_section_idx + 1])

                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []

                elif t_sec == 30 and curr_sec == 30:
                    diff_in_past_section = abs(prev_seconds - 30)  # 2:40-2:30
                    diff_in_mid_section = 30  # 2:30-2
                    diff_in_next_section = 30  # 2-1:30
                    # 1:30-1

                    past_section_idx = int(
                        diff_in_past_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:next_section_idx + 1]
                    next_section_traj = curr_t_traj[next_section_idx + 1:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = next_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = []

                elif t_sec == 0 and curr_sec == 0:
                    diff_in_past_section = abs(prev_seconds - 0)  # 2:04 -2
                    diff_in_mid_section = 30  # 2 - 1:30
                    diff_in_next_section = 30  # 1:30 - 1
                    # 1-0:30

                    past_section_idx = int(
                        diff_in_past_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:next_section_idx + 1]
                    next_section_traj = curr_t_traj[next_section_idx + 1:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = next_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []

                elif t_sec == 0 and curr_sec == 30:
                    # To Do

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:45-3:30
                    diff_in_mid_section = 30  # 3:30 - 3
                    #                     diff_in_next_section = abs(30 - t_sec)  # 3 - 2:30
                    # 2:30 - 2

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_mid_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section))
                    #                     next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:]
                    #                     next_section_traj = curr_t_traj[next_section_idx+1:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0

                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    #                     curr_traj = next_section_traj
                    #                     minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    #                     curr_sec = 0
                    curr_traj = []


                elif t_sec == 30 and curr_sec == 0:
                    # To Do

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:15-3
                    diff_in_mid_section = 30  # 3 - 2:30
                    #                     diff_in_next_section = abs(30 - t_sec)  # 2:30 - 2
                    # 2 - 1:30

                    past_section_idx = int(diff_in_past_section / (diff_in_past_section + diff_in_mid_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section))
                    #                     next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:]
                    #                     next_section_traj = curr_t_traj[next_section_idx+1:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    #                     curr_traj = next_section_traj
                    #                     minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    #                     curr_min -= 1
                    #                     curr_sec = 30
                    curr_traj = []

                elif t_sec > curr_sec and curr_sec == 30:
                    # 2 sections off
                    diff_in_past_section = abs(prev_seconds - curr_sec)
                    diff_in_mid_section = 30
                    diff_in_next_section = abs(60 - t_sec)

                    past_section_idx = int(
                        diff_in_past_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:mid_section_idx + 1]
                    next_section_traj = curr_t_traj[next_section_idx:]

                    #                 print("past_section_traj = ", past_section_traj)
                    #                 print("mid_section_traj = ", mid_section_traj)
                    #                 print("next_section_traj = ", next_section_traj)

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = next_section_traj

                elif t_sec == 0 and curr_sec == 30:
                    diff_in_past_section = abs(prev_seconds - 0)
                    diff_in_mid_section = 30
                    diff_in_next_section = abs(60 - t_sec)

                    past_section_idx = int(
                        diff_in_past_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    mid_section_idx = past_section_idx + 1 + int(
                        diff_in_mid_section / (diff_in_past_section + diff_in_mid_section + diff_in_next_section))
                    next_section_idx = mid_section_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_traj = curr_t_traj[past_section_idx + 1:]
                    next_section_traj = []

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = next_section_traj


                elif t_sec < curr_sec and curr_sec == 30:
                    # 3 sections off, 2:30 --> 1:14, 2:40-2:30, 2:30-2, 2-1:30, 1:30-1:14
                    #                     print("PROBLEMMM")
                    #                     print("t = ", t)
                    #                     print("current = ", (curr_min, curr_sec))
                    #                     print("previous = ", (prev_min, prev_seconds))
                    #                     print()
                    diff_in_past_section = abs(prev_seconds - 30)
                    diff_in_mid_1_section = 30
                    diff_in_mid_2_section = 30
                    diff_in_next_section = abs(30 - t_sec)

                    past_section_idx = int(diff_in_past_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_1_idx = past_section_idx + 1 + int(diff_in_mid_1_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_2_idx = mid_section_1_idx + 1 + int(diff_in_mid_2_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    next_section_idx = mid_section_2_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_1_traj = curr_t_traj[past_section_idx + 1:mid_section_1_idx + 1]
                    mid_section_2_traj = curr_t_traj[mid_section_1_idx + 1:mid_section_2_idx + 1]
                    next_section_traj = curr_t_traj[mid_section_2_idx:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_1_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30

                    curr_traj = mid_section_2_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = next_section_traj


            #                 elif t_sec <= curr_sec and curr_sec == 0:
            #                     # 3 sections off, 2:30 --> 1:14, 2:40-2:30, 2:30-2, 2-1:30, 1:30-1:14
            #                     print("PROBLEMMM 2")
            #                     print("t = ", t)
            #                     print("current = ", (curr_min, curr_sec))
            #                     print("previous = ", (prev_min, prev_seconds))
            #                     print()

            elif t_min == curr_min - 2:
                ### 4:30 --> 2:30, 4:00 --> 2:00, 4:00 --> 2:30
                #                 if t_sec > curr_sec and curr_sec == 0:
                #                 print("MAJOR PROBLEM 2: ", t)
                #                 print("t = ", t)
                #                 print("current = ", (curr_min, curr_sec))
                #                 print("previous = ", (prev_min, prev_seconds))
                #                 print()
                if t_sec == 30 and curr_sec == 0:

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:15-3
                    diff_in_mid_1_section = 30  # 3-2:30
                    diff_in_mid_2_section = 30  # 2:30- 2
                    diff_in_next_section = abs(30 - t_sec)  # 2 - 1:30
                    # 1:30-1

                    past_section_idx = int(diff_in_past_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_1_idx = past_section_idx + 1 + int(diff_in_mid_1_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_2_idx = mid_section_1_idx + 1 + int(diff_in_mid_2_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    next_section_idx = mid_section_2_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_1_traj = curr_t_traj[past_section_idx + 1:mid_section_1_idx + 1]
                    mid_section_2_traj = curr_t_traj[mid_section_1_idx + 1:mid_section_2_idx + 1]
                    next_section_traj = curr_t_traj[mid_section_2_idx:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_1_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_2_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = next_section_traj

                    curr_sec = 0
                    curr_traj = []


                elif t_sec == 0 and curr_sec == 30:
                    # To Do

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:45-3:30
                    diff_in_mid_1_section = 30  # 3:30-3
                    diff_in_mid_2_section = 30  # 3- 2:30
                    diff_in_next_section = abs(30 - t_sec)  # 2:30 - 2
                    # 2-1:30

                    past_section_idx = int(diff_in_past_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_1_idx = past_section_idx + 1 + int(diff_in_mid_1_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    mid_section_2_idx = mid_section_1_idx + 1 + int(diff_in_mid_2_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_next_section))
                    next_section_idx = mid_section_2_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_1_traj = curr_t_traj[past_section_idx + 1:mid_section_1_idx + 1]
                    mid_section_2_traj = curr_t_traj[mid_section_1_idx + 1:mid_section_2_idx + 1]
                    next_section_traj = curr_t_traj[mid_section_2_idx:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_1_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_2_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = next_section_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []


                elif t_sec == 30 and curr_sec == 30:
                    # To Do

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:45-3:30
                    diff_in_mid_1_section = 30  # 3:30-3
                    diff_in_mid_2_section = 30  # 3- 2:30
                    diff_in_mid_3_section = 30  # 2:30 - 2
                    diff_in_next_section = abs(30 - t_sec)  # 2 - 1:30
                    # 1:30-1

                    past_section_idx = int(diff_in_past_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_1_idx = past_section_idx + 1 + int(diff_in_mid_1_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_2_idx = mid_section_1_idx + 1 + int(diff_in_mid_2_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_3_idx = mid_section_2_idx + 1 + int(diff_in_mid_3_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    next_section_idx = mid_section_3_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_1_traj = curr_t_traj[past_section_idx + 1:mid_section_1_idx + 1]
                    mid_section_2_traj = curr_t_traj[mid_section_1_idx + 1:mid_section_2_idx + 1]
                    mid_section_3_traj = curr_t_traj[mid_section_2_idx + 1:mid_section_3_idx + 1]
                    next_section_traj = curr_t_traj[next_section_idx:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_1_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_2_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_3_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = next_section_traj

                    curr_sec = 0
                    curr_traj = []


                elif t_sec == 0 and curr_sec == 0:
                    # To Do

                    diff_in_past_section = abs(prev_seconds - 30)  # 3:15-3:00
                    diff_in_mid_1_section = 30  # 3:0-2:30
                    diff_in_mid_2_section = 30  # 2:30- 2
                    diff_in_mid_3_section = 30  # 2 - 1:30
                    diff_in_next_section = abs(30 - t_sec)  # 1:30 - 1
                    # 1 - 0:30

                    past_section_idx = int(diff_in_past_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_1_idx = past_section_idx + 1 + int(diff_in_mid_1_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_2_idx = mid_section_1_idx + 1 + int(diff_in_mid_2_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    mid_section_3_idx = mid_section_2_idx + 1 + int(diff_in_mid_3_section / (
                                diff_in_past_section + diff_in_mid_1_section + diff_in_mid_2_section + diff_in_mid_3_section + diff_in_next_section))
                    next_section_idx = mid_section_3_idx + 1

                    past_section_traj = curr_t_traj[:past_section_idx + 1]
                    mid_section_1_traj = curr_t_traj[past_section_idx + 1:mid_section_1_idx + 1]
                    mid_section_2_traj = curr_t_traj[mid_section_1_idx + 1:mid_section_2_idx + 1]
                    mid_section_3_traj = curr_t_traj[mid_section_2_idx + 1:mid_section_3_idx + 1]
                    next_section_traj = curr_t_traj[next_section_idx:]

                    curr_traj.extend(past_section_traj)
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_1_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = mid_section_2_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = mid_section_3_traj
                    minute_to_traj[(curr_min, curr_sec)] = curr_traj

                    curr_sec = 0
                    curr_traj = next_section_traj

                    curr_min -= 1
                    curr_sec = 30
                    curr_traj = []


            else:
                #                 print("MAJOR PROBLEM: ", t)
                #                 print("t = ", t)
                #                 print("current = ", (curr_min, curr_sec))
                #                 print("previous = ", (prev_min, prev_seconds))
                #                 print()
                to_include = False

        prev_min = t_min
        prev_seconds = t_sec

    stop_indices = np.where(times == 'stop')[0]
    if len(stop_indices) > 0:
        stop_idx = stop_indices[0]
        t = str(times[start_idx])
        traj_t = str(traj[start_idx])
        msgs_t = str(sent_msgs[start_idx])
        vic_t = str(victims[start_idx])

        if traj_t != 'nan':
            curr_traj.extend([eval(x) for x in traj_t.split(';')])

    minute_to_traj[(curr_min, curr_sec)] = curr_traj

    return minute_to_traj, to_include


def compute_minute_to_msgs(df_traj):
    traj = df_traj['trajectory'].to_numpy()
    times = df_traj['time_spent'].to_numpy()
    sent_msgs = df_traj['sent_messages'].to_numpy()
    victims = df_traj['target'].to_numpy()

    to_include = True

    minute_to_msgs = {}

    curr_min = 5
    curr_sec = 0

    #     minute_to_msgs[(curr_min, curr_sec)] = []
    #     minute_to_victims[(curr_min, curr_sec)] = []

    prev_time = (5, 0)
    for i in range(len(traj)):
        t = str(times[i])
        msgs_t = str(sent_msgs[i])

        if ':' in t:
            t_min = int(t.split(":")[0])
            t_sec = int(t.split(":")[1])
            prev_time = (t_min, t_sec)

        if msgs_t == 'nan':
            continue

        prev_t_min = prev_time[0]
        prev_t_sec = prev_time[1]

        window_sec = 0
        window_min = prev_t_min
        if prev_t_sec > 30:
            window_sec = 30

        #         print((prev_t_min, prev_t_sec), 't = ', (window_min, window_sec))
        #         print("msgs_t: ", msgs_t)
        #         print()

        keyname = (window_min, window_sec)
        if keyname not in minute_to_msgs:
            minute_to_msgs[keyname] = []

        minute_to_msgs[keyname].append(msgs_t)
    #     minute_to_traj[(curr_min, curr_sec)] = curr_traj

    return minute_to_msgs


def compute_minute_to_victims(df_traj):
    traj = df_traj['trajectory'].to_numpy()
    times = df_traj['time_spent'].to_numpy()
    sent_msgs = df_traj['sent_messages'].to_numpy()
    victims = df_traj['target'].to_numpy()

    to_include = True

    minute_to_victims = {}

    curr_min = 5
    curr_sec = 0

    #     minute_to_msgs[(curr_min, curr_sec)] = []
    minute_to_victims[(curr_min, curr_sec)] = []

    prev_time = (5, 0)
    for i in range(len(traj)):
        t = str(times[i])
        victims_t = str(victims[i])

        if ':' in t:
            t_min = int(t.split(":")[0])
            t_sec = int(t.split(":")[1])
            prev_time = (t_min, t_sec)

        if victims_t in ['nan', 'door']:
            continue

        prev_t_min = prev_time[0]
        prev_t_sec = prev_time[1]

        window_sec = 0
        window_min = prev_t_min
        if prev_t_sec > 30:
            window_sec = 30

        #         print((prev_t_min, prev_t_sec), 't = ', (window_min, window_sec))
        #         print("victims_t: ", victims_t)
        #         print()

        keyname = (window_min, window_sec)
        if keyname not in minute_to_victims:
            minute_to_victims[keyname] = []

        minute_to_victims[keyname].append(victims_t)

    return minute_to_victims


def l2(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)






def compute_effort(traj_list, robot_list):
    eff = 0
    for i in range(len(traj_list) - 1):
        eff += l2(traj_list[i], traj_list[i + 1])

    eff_robot = 0
    for i in range(len(robot_list) - 1):
        eff_robot += l2(robot_list[i], robot_list[i + 1])

    eff = eff + eff_robot
    return eff





def get_r1_data(valid_uids):
    min_to_robot_traj = get_robot_traj()
    df = pd.read_csv('../minimap_data/minimap_study_3_data.csv')
    uid_to_groups = merge_r1_groups_single_emotion()
    # valid_uids = list(uid_to_groups.keys())

    uid_to_minute_victims = {}
    for i in range(len(valid_uids)):
        p_uid = valid_uids[i]
        df_traj = df[(df['userid']==p_uid) & (df['episode']==1)]

        df_traj = df_traj.sort_values(by=['created_at'], ascending=False)

        minute_to_victims = compute_minute_to_victims(df_traj)
    #     if to_include:
        uid_to_minute_victims[p_uid] = minute_to_victims

    uid_to_minute_msgs = {}
    for i in range(len(valid_uids)):
        p_uid = valid_uids[i]
        df_traj = df[(df['userid']==p_uid) & (df['episode']==1)]

        df_traj = df_traj.sort_values(by=['created_at'], ascending=False)

        minute_to_msgs = compute_minute_to_msgs(df_traj)
    #     if to_include:
        uid_to_minute_msgs[p_uid] = minute_to_msgs



    uid_to_minute_traj = {}
    for i in range(len(valid_uids)):
        p_uid = valid_uids[i]
        df_traj = df[(df['userid']==p_uid) & (df['episode']==1)]

        df_traj = df_traj.sort_values(by=['time_spent'], ascending=False)

        minute_to_traj, to_include = compute_minute_to_traj(df_traj)
        if to_include:
            uid_to_minute_traj[p_uid] = minute_to_traj

    uid_min_to_data = {}

    for puid in uid_to_minute_traj:
        uid_min_to_data[puid] = {}
        human_traj = uid_to_minute_traj[puid]
        robot_traj = min_to_robot_traj

        human_msgs = uid_to_minute_msgs[puid]
        human_victims = uid_to_minute_victims[puid]
        counter = 0
        for minute in [4, 3, 2, 1, 0]:
            for second in [30, 0]:
                keyname = (minute, second)
                uid_min_to_data[puid][counter] = {}
                if keyname not in human_traj:
                    curr_human_traj = []
                else:
                    curr_human_traj = human_traj[keyname]
                curr_robot_traj = robot_traj[keyname]
                effort = compute_effort(curr_human_traj, curr_robot_traj)

                uid_min_to_data[puid][counter]['effort'] = effort

                if keyname not in human_msgs:
                    num_msgs = 0
                else:
                    num_msgs = len(human_msgs[keyname])
                uid_min_to_data[puid][counter]['msgs'] = num_msgs

                if keyname not in human_victims:
                    num_victims = 0
                else:
                    num_victims = len(human_victims[keyname])
                uid_min_to_data[puid][counter]['victims'] = num_victims

                counter += 1

    return uid_min_to_data


def split_data_by_group(uid_min_to_data, uid_to_group):
    data_by_group = {x: {} for x in uid_to_group.values()}
    for p_uid in uid_min_to_data:
        group_no = uid_to_group[p_uid]
        data_by_group[group_no][p_uid] = uid_min_to_data[p_uid]
    return data_by_group

def binarize_data(data_by_group, uid_to_group, savename=''):
    group_to_means = {x: {} for x in uid_to_group.values()}

    for group in data_by_group:
        group_data = data_by_group[group]

        all_effort = {i: [] for i in range(10)}
        all_victims = {i: [] for i in range(10)}
        all_msgs = {i: [] for i in range(10)}

        for p_uid in group_data:
            for i in range(10):
                all_effort[i].append(group_data[p_uid][i]['effort'])
                all_msgs[i].append(group_data[p_uid][i]['msgs'])
                all_victims[i].append(group_data[p_uid][i]['victims'])

        #     print(all_victims)
        final_effort = {i: np.mean(all_effort[i]) for i in range(10)}
        final_victims = {i: np.mean(all_victims[i]) for i in range(10)}
        final_msgs = {i: np.mean(all_msgs[i]) for i in range(10)}

        group_to_means[group]['effort'] = final_effort
        group_to_means[group]['victims'] = final_victims
        group_to_means[group]['msgs'] = final_msgs

    group_to_binary_data = {}

    for group in data_by_group:
        group_data = data_by_group[group]

        mean_effort = group_to_means[group]['effort']
        mean_victims = group_to_means[group]['victims']
        mean_msgs = group_to_means[group]['msgs']

        new_group_data = {}
        for p_uid in group_data:
            p_uid_data = []
            for i in range(10):
                effort_binary = 0 if group_data[p_uid][i]['effort'] < mean_effort[i] else 1
                msgs_binary = 0 if group_data[p_uid][i]['msgs'] < mean_msgs[i] else 1
                victims_binary = 0 if group_data[p_uid][i]['victims'] < mean_victims[i] else 1

                p_uid_data.append((effort_binary, msgs_binary, victims_binary))

            new_group_data[p_uid] = p_uid_data

        group_to_binary_data[group] = new_group_data

    group_to_binary_state_data = {}
    group_to_state_list = {}

    for group in data_by_group:
        group_data = data_by_group[group]
        group_to_state_list[group] = []

        mean_effort = group_to_means[group]['effort']
        mean_victims = group_to_means[group]['victims']
        mean_msgs = group_to_means[group]['msgs']

        new_group_data = {}
        for p_uid in group_data:
            p_uid_data = []
            for i in range(9):
                effort_binary = 0 if group_data[p_uid][i]['effort'] < mean_effort[i] else 1
                msgs_binary = 0 if group_data[p_uid][i]['msgs'] < mean_msgs[i] else 1
                victims_binary = 0 if group_data[p_uid][i]['victims'] < mean_victims[i] else 1

                effort_binary_next = 0 if group_data[p_uid][i + 1]['effort'] < mean_effort[i + 1] else 1
                msgs_binary_next = 0 if group_data[p_uid][i + 1]['msgs'] < mean_msgs[i + 1] else 1
                victims_binary_next = 0 if group_data[p_uid][i + 1]['victims'] < mean_victims[i + 1] else 1

                state_vector = (
                effort_binary_next, msgs_binary_next, victims_binary_next, effort_binary, msgs_binary, victims_binary)
                p_uid_data.append(state_vector)

                if state_vector not in group_to_state_list[group]:
                    group_to_state_list[group].append(state_vector)

            new_group_data[p_uid] = p_uid_data

        group_to_binary_state_data[group] = new_group_data

    group_to_state_mapping = {}

    for group in group_to_state_list:
        group_to_state_mapping[group] = {}
        state_id_to_state = dict(enumerate(group_to_state_list[group]))
        state_to_state_id = {v: k for k, v in state_id_to_state.items()}

        group_to_state_mapping[group]['id_to_vec'] = state_id_to_state
        group_to_state_mapping[group]['vec_to_id'] = state_to_state_id

    group_to_state_id_data_w_pid = {}
    group_to_state_id_data = {}

    for group in group_to_binary_state_data:
        group_data = group_to_binary_state_data[group]
        state_to_state_id = group_to_state_mapping[group]['vec_to_id']

        all_data = []
        all_data_w_pid = {}
        for p_uid in group_data:
            state_data = [state_to_state_id[elem] for elem in group_data[p_uid]]
            all_data.append(state_data)
            all_data_w_pid[p_uid] = state_data

        group_to_state_id_data[group] = all_data
        group_to_state_id_data_w_pid[group] = all_data_w_pid

    with open(f'minimap_data/minimap_group_to_state_data_{savename}.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data/minimap_group_to_state_data_w_pid_{savename}.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data_w_pid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data/minimap_group_to_binary_state_data_{savename}.pickle', 'wb') as handle:
        pickle.dump(group_to_binary_state_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data/minimap_group_to_state_mapping_{savename}.pickle', 'wb') as handle:
        pickle.dump(group_to_state_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

def binarize_data_train_test(data_by_group, uid_to_group, train_uids, test_uids, savename=''):
    group_to_means = {x: {} for x in uid_to_group.values()}

    for group in data_by_group:
        group_data = data_by_group[group]

        all_effort = {i: [] for i in range(10)}
        all_victims = {i: [] for i in range(10)}
        all_msgs = {i: [] for i in range(10)}

        for p_uid in group_data:
            for i in range(10):
                all_effort[i].append(group_data[p_uid][i]['effort'])
                all_msgs[i].append(group_data[p_uid][i]['msgs'])
                all_victims[i].append(group_data[p_uid][i]['victims'])

        #     print(all_victims)
        final_effort = {i: np.mean(all_effort[i]) for i in range(10)}
        final_victims = {i: np.mean(all_victims[i]) for i in range(10)}
        final_msgs = {i: np.mean(all_msgs[i]) for i in range(10)}

        group_to_means[group]['effort'] = final_effort
        group_to_means[group]['victims'] = final_victims
        group_to_means[group]['msgs'] = final_msgs

    group_to_binary_data = {}

    for group in data_by_group:
        group_data = data_by_group[group]

        mean_effort = group_to_means[group]['effort']
        mean_victims = group_to_means[group]['victims']
        mean_msgs = group_to_means[group]['msgs']

        new_group_data = {}
        for p_uid in group_data:
            p_uid_data = []
            for i in range(10):
                effort_binary = 0 if group_data[p_uid][i]['effort'] < mean_effort[i] else 1
                msgs_binary = 0 if group_data[p_uid][i]['msgs'] < mean_msgs[i] else 1
                victims_binary = 0 if group_data[p_uid][i]['victims'] < mean_victims[i] else 1

                p_uid_data.append((effort_binary, msgs_binary, victims_binary))

            new_group_data[p_uid] = p_uid_data

        group_to_binary_data[group] = new_group_data


    group_to_binary_state_data = {}
    group_to_state_list = {}

    for group in data_by_group:
        group_data = data_by_group[group]
        group_to_state_list[group] = []

        mean_effort = group_to_means[group]['effort']
        mean_victims = group_to_means[group]['victims']
        mean_msgs = group_to_means[group]['msgs']

        new_group_data = {}
        for p_uid in group_data:
            if p_uid not in train_uids:
                continue
            p_uid_data = []
            for i in range(9):
                effort_binary = 0 if group_data[p_uid][i]['effort'] < mean_effort[i] else 1
                msgs_binary = 0 if group_data[p_uid][i]['msgs'] < mean_msgs[i] else 1
                victims_binary = 0 if group_data[p_uid][i]['victims'] < mean_victims[i] else 1

                effort_binary_next = 0 if group_data[p_uid][i + 1]['effort'] < mean_effort[i + 1] else 1
                msgs_binary_next = 0 if group_data[p_uid][i + 1]['msgs'] < mean_msgs[i + 1] else 1
                victims_binary_next = 0 if group_data[p_uid][i + 1]['victims'] < mean_victims[i + 1] else 1

                state_vector = (
                effort_binary_next, msgs_binary_next, victims_binary_next, effort_binary, msgs_binary, victims_binary)
                p_uid_data.append(state_vector)

                if state_vector not in group_to_state_list[group]:
                    group_to_state_list[group].append(state_vector)

            new_group_data[p_uid] = p_uid_data

        group_to_binary_state_data[group] = new_group_data

    group_to_state_mapping = {}

    for group in group_to_state_list:
        group_to_state_mapping[group] = {}
        state_id_to_state = dict(enumerate(group_to_state_list[group]))
        state_to_state_id = {v: k for k, v in state_id_to_state.items()}

        group_to_state_mapping[group]['id_to_vec'] = state_id_to_state
        group_to_state_mapping[group]['vec_to_id'] = state_to_state_id


    # FOR TRAINING SET
    group_to_state_id_data_w_pid = {}
    group_to_state_id_data = {}
    group_to_binary_state_data_new = {}

    for group in group_to_binary_state_data:
        group_to_binary_state_data_new[group] = []

        group_data = group_to_binary_state_data[group]
        state_to_state_id = group_to_state_mapping[group]['vec_to_id']

        all_data = []
        all_data_w_pid = {}
        for p_uid in group_data:
            if p_uid not in train_uids:
                continue
            state_data = [state_to_state_id[elem] for elem in group_data[p_uid]]
            all_data.append(state_data)
            all_data_w_pid[p_uid] = state_data
            group_to_binary_state_data_new[group].extend(group_data[p_uid])

        group_to_state_id_data[group] = all_data
        group_to_state_id_data_w_pid[group] = all_data_w_pid

    with open(f'minimap_data_aug15/minimap_group_to_state_data_{savename}_TRAIN.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_state_data_w_pid_{savename}_TRAIN.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data_w_pid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_binary_state_data_{savename}_TRAIN.pickle', 'wb') as handle:
        pickle.dump(group_to_binary_state_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_state_mapping_{savename}_TRAIN.pickle', 'wb') as handle:
        pickle.dump(group_to_state_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # FOR TEST SET
    group_to_binary_state_data = {}
    group_to_state_list = {}

    for group in data_by_group:
        group_data = data_by_group[group]
        group_to_state_list[group] = []

        mean_effort = group_to_means[group]['effort']
        mean_victims = group_to_means[group]['victims']
        mean_msgs = group_to_means[group]['msgs']

        new_group_data = {}
        for p_uid in group_data:
            if p_uid not in test_uids:
                continue
            p_uid_data = []
            for i in range(9):
                effort_binary = 0 if group_data[p_uid][i]['effort'] < mean_effort[i] else 1
                msgs_binary = 0 if group_data[p_uid][i]['msgs'] < mean_msgs[i] else 1
                victims_binary = 0 if group_data[p_uid][i]['victims'] < mean_victims[i] else 1

                effort_binary_next = 0 if group_data[p_uid][i + 1]['effort'] < mean_effort[i + 1] else 1
                msgs_binary_next = 0 if group_data[p_uid][i + 1]['msgs'] < mean_msgs[i + 1] else 1
                victims_binary_next = 0 if group_data[p_uid][i + 1]['victims'] < mean_victims[i + 1] else 1

                state_vector = (
                    effort_binary_next, msgs_binary_next, victims_binary_next, effort_binary, msgs_binary,
                    victims_binary)
                p_uid_data.append(state_vector)

                if state_vector not in group_to_state_list[group]:
                    group_to_state_list[group].append(state_vector)

            new_group_data[p_uid] = p_uid_data

        group_to_binary_state_data[group] = new_group_data



    group_to_state_id_data_w_pid = {}
    group_to_state_id_data = {}
    group_to_binary_state_data_new = {}

    for group in group_to_binary_state_data:
        group_to_binary_state_data_new[group] = []

        group_data = group_to_binary_state_data[group]
        state_to_state_id = group_to_state_mapping[group]['vec_to_id']

        all_data = []
        all_data_w_pid = {}
        for p_uid in group_data:
            if p_uid not in test_uids:
                continue
            state_data = []
            for elem in group_data[p_uid]:
                if elem in state_to_state_id:
                    state_data.append(state_to_state_id[elem])
                else:
                    print("skipped")
            # state_data = [state_to_state_id[elem] for elem in group_data[p_uid]]
            all_data.append(state_data)
            all_data_w_pid[p_uid] = state_data
            group_to_binary_state_data_new[group].extend(group_data[p_uid])

        group_to_state_id_data[group] = all_data
        group_to_state_id_data_w_pid[group] = all_data_w_pid

    with open(f'minimap_data_aug15/minimap_group_to_state_data_{savename}_TEST.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_state_data_w_pid_{savename}_TEST.pickle', 'wb') as handle:
        pickle.dump(group_to_state_id_data_w_pid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_binary_state_data_{savename}_TEST.pickle', 'wb') as handle:
        pickle.dump(group_to_binary_state_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'minimap_data_aug15/minimap_group_to_state_mapping_{savename}_TEST.pickle', 'wb') as handle:
        pickle.dump(group_to_state_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_datasets_single_emotion():
    merged_uid_to_data = merge_r1_groups_single_emotion()

    uid_to_anger = {pid: merged_uid_to_data[pid]['anger'] for pid in merged_uid_to_data}
    uid_to_anx = {pid: merged_uid_to_data[pid]['anx'] for pid in merged_uid_to_data}
    uid_to_pos = {pid: merged_uid_to_data[pid]['pos'] for pid in merged_uid_to_data}
    uid_to_rand = {pid: merged_uid_to_data[pid]['random'] for pid in merged_uid_to_data}

    uids = list(merged_uid_to_data.keys())
    uid_min_to_data = get_r1_data(uids)

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_anger)
    binarize_data(data_by_group, uid_to_anger, savename='ANGER')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_anx)
    binarize_data(data_by_group, uid_to_anx, savename='ANX')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_pos)
    binarize_data(data_by_group, uid_to_pos, savename='POS')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_rand)
    binarize_data(data_by_group, uid_to_rand, savename='RANDOM')

    print("DONE")


def create_datasets_single_emotion_train_test():
    merged_uid_to_data = merge_r1_groups_single_emotion()

    uids = list(merged_uid_to_data.keys())
    training_data, testing_data = train_test_split(uids, test_size=0.2, random_state=25)
    # print(training_data)
    # print(testing_data)

    uid_to_anger = {pid: merged_uid_to_data[pid]['anger'] for pid in merged_uid_to_data}
    uid_to_anx = {pid: merged_uid_to_data[pid]['anx'] for pid in merged_uid_to_data}
    uid_to_pos = {pid: merged_uid_to_data[pid]['pos'] for pid in merged_uid_to_data}
    uid_to_rand = {pid: merged_uid_to_data[pid]['random'] for pid in merged_uid_to_data}
    uid_to_anger_anx = {pid: merged_uid_to_data[pid]['anger-anx'] for pid in merged_uid_to_data}
    uid_to_agg = {pid: merged_uid_to_data[pid]['agg'] for pid in merged_uid_to_data}

    uids = list(merged_uid_to_data.keys())
    uid_min_to_data = get_r1_data(uids)

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_agg)
    binarize_data_train_test(data_by_group, uid_to_agg, training_data, testing_data, savename='AGG')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_anger_anx)
    binarize_data_train_test(data_by_group, uid_to_anger_anx, training_data, testing_data, savename='ANGERANX')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_anger)
    binarize_data_train_test(data_by_group, uid_to_anger, training_data, testing_data, savename='ANGER')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_anx)
    binarize_data_train_test(data_by_group, uid_to_anx, training_data, testing_data, savename='ANX')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_pos)
    binarize_data_train_test(data_by_group, uid_to_pos, training_data, testing_data, savename='POS')

    data_by_group = split_data_by_group(uid_min_to_data, uid_to_rand)
    binarize_data_train_test(data_by_group, uid_to_rand, training_data, testing_data, savename='RANDOM')


    print("DONE")

def create_kfold_agg_datasets():
    random_states = np.random.randint(0, 10000, size=100)
    print("random_states", list(random_states))
    test_percentage = 0.2

    for r_state in [25]:
        print("RUNING: ", r_state)
        merged_uid_to_data = merge_r1_groups_single_emotion()

        uids = list(merged_uid_to_data.keys())


        training_data, testing_data = train_test_split(uids, test_size=test_percentage, random_state=r_state)

        uid_to_agg = {pid: merged_uid_to_data[pid]['agg'] for pid in merged_uid_to_data}

        uids = list(merged_uid_to_data.keys())
        uid_min_to_data = get_r1_data(uids)

        data_by_group = split_data_by_group(uid_min_to_data, uid_to_agg)
        binarize_data_train_test(data_by_group, uid_to_agg, training_data, testing_data, savename=f'AGG_RAND{r_state}')


    print("DONE")


if __name__ == "__main__":
    # create_datasets_single_emotion_train_test()
    create_kfold_agg_datasets()




