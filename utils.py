import pandas_queries, plot_graph
import numpy as np
from enum import Enum
import datetime

file_name = 'results_tests_shortproc1'
DIRECTORY = 'C:/Users/AVIADFUX/Desktop/Projects/Mor_project/analyze/results/%s_directory/' % file_name

class Algo(Enum):
    SCOUT = 'SCOUT'
    Entropy = 'Entropy'
    MaxProb = 'MaxProb'
    Rand = 'Rand'

def comparing_algorithms_fixed_player(data_base, fixed_player, fixed_num, dp):
    if fixed_player == "uav":
        scout = data_base[Algo.SCOUT.value][dp][:, fixed_num - 1]
        entropy = data_base[Algo.Entropy.value][dp][:, fixed_num - 1]
        maxProb = data_base[Algo.MaxProb.value][dp][:, fixed_num - 1]
        rand = data_base[Algo.Rand.value][dp][:, fixed_num - 1]
    else:
        scout = data_base[Algo.SCOUT.value][dp][fixed_num - 1, :]
        entropy = data_base[Algo.Entropy.value][dp][fixed_num - 1, :]
        maxProb = data_base[Algo.MaxProb.value][dp][fixed_num - 1, :]
        rand = data_base[Algo.Rand.value][dp][fixed_num - 1, :]

    plot_graph.comparing_algorithms_bars_fixed_scenarios(directory=DIRECTORY, SCOUT_results=scout, ENTROPY_results=entropy,
                                              MAX_PROB_results=maxProb, RANDOM_results=rand, IsSuccessRate=True,
                                              fixed_player=fixed_player, fixed_num=fixed_num, dp=dp)

def comparing_algorithms(results_csv, directory, DET_range, THR_range, delay, algo, path_type, general_comparison=True, enemy_count=0, uav_count=0):
    '''
    Generate comparison between all algorithms via
    :param results_csv: The csv file for analyzing
    :param general_comparison: When true so check all scenarios
    :param enemy_count:
    :param uav_count:
    :return:
    '''
    list_detection_success_rate_scout, list_detection_duration_scout, thr_scout = \
        determined_SCOUT_results(DET_range, pandas_queries.create_success_rates_table_SCOUT(directory, results_csv, DET_range, THR_range, general_comparison, enemy_count, uav_count),
                                 pandas_queries.create_duration_table_SCOUT(directory, results_csv, DET_range, THR_range, delay, general_comparison, enemy_count, uav_count))

    # Calculate success rate of all algorithms
    maxProb_result = pandas_queries.create_success_rates_table(DET_range, results_csv, 'MaxProb', general_comparison, enemy_count, uav_count)
    entropy_result = pandas_queries.create_success_rates_table(DET_range, results_csv, 'Entropy', general_comparison, enemy_count, uav_count)
    random_result = pandas_queries.create_success_rates_table(DET_range, results_csv, 'Rand', general_comparison, enemy_count, uav_count)
    plot_graph.comparing_algorithms_bars(DET_range, directory, list_detection_success_rate_scout, thr_scout, entropy_result, maxProb_result, random_result,
                                         delay, algo, path_type,
                                        IsSuccessRate=True, is_genaral_comparison=general_comparison, enemyCount=enemy_count, uavCount=uav_count)

    # calculate the mean of duration to enemy catched for all algorithms
    maxProb_result = pandas_queries.create_duration_table(DET_range, results_csv, 'MaxProb', delay, general_comparison, enemy_count, uav_count)
    entropy_result = pandas_queries.create_duration_table(DET_range, results_csv, 'Entropy', delay, general_comparison, enemy_count, uav_count)
    random_result = pandas_queries.create_duration_table(DET_range, results_csv, 'Rand', delay, general_comparison, enemy_count, uav_count)
    plot_graph.comparing_algorithms_bars(DET_range, directory, list_detection_duration_scout, thr_scout, entropy_result, maxProb_result, random_result,
                                         delay, algo, path_type,
                                        IsSuccessRate=False, is_genaral_comparison=general_comparison, enemyCount=enemy_count, uavCount=uav_count)

def determined_SCOUT_results(DET_range, success_rate_list_and_best_thr, duration_heatmap):
    success_rate_list, best_thr = success_rate_list_and_best_thr

    list_detection_success_rate = np.zeros((len(DET_range)))
    list_detection_duration = np.zeros((len(DET_range)))
    list_final_coordinates = []

    for i in range(np.size(duration_heatmap, 0)):
        row_i = duration_heatmap[i, :]

        #Collect all columns with the high value of success rate
        high_success_rate_column = []
        [high_success_rate_column.append(coordinate[0][1]) for coordinate in success_rate_list if coordinate[0][0] == i]

        #Collect the duration values for the coordinate with high success rate values
        duration_for_high_success = []
        [duration_for_high_success.append(duration_heatmap[i, col]) for col in high_success_rate_column]

        #Collect the low duration with high success rate for each raw
        list_detection_duration[i] = np.min(duration_for_high_success)

        #Collect the high success rate value for each raw
        high_value_success_rate = []
        [high_value_success_rate.append(coordinate_with_value[1]) for coordinate_with_value in success_rate_list if coordinate_with_value[0][0] == i]
        list_detection_success_rate[i] = high_value_success_rate[0]

    return list_detection_success_rate, list_detection_duration, best_thr

def min_values_for_each_row(heatmap):
    list_of_coordinates_for_each_row = []
    for i in range(np.size(heatmap, 0)):
        row_i = heatmap[i, :]
        min_value = np.min(row_i)
        index_of_min_value_in_row_i = np.where(row_i == np.amin(row_i))
        list_of_coordinates_for_each_row.append(((i, index_of_min_value_in_row_i[0][0]), min_value))

    return list_of_coordinates_for_each_row
