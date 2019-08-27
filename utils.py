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

def comparing_algorithms(results_csv, directory, general_comparison=True, enemy_count=0, uav_count=0):
    '''
    Generate comparison between all algorithms via
    :param results_csv: The csv file for analyzing
    :param general_comparison: When true so check all scenarios
    :param enemy_count:
    :param uav_count:
    :return:
    '''
    list_detection_success_rate_scout, list_detection_duration_scout = \
        determined_SCOUT_results(pandas_queries.create_success_rates_table_SCOUT(directory, results_csv, general_comparison, enemy_count, uav_count),
                                 create_duration_table_SCOUT(directory, results_csv, general_comparison, enemy_count, uav_count))

    # Calculate success rate of all algorithms
    maxProb_result = pandas_queries.create_success_rates_table(results_csv, 'MaxProb', general_comparison, enemy_count, uav_count)
    entropy_result = pandas_queries.create_success_rates_table(results_csv, 'Entropy', general_comparison, enemy_count, uav_count)
    random_result = pandas_queries.create_success_rates_table(results_csv, 'Rand', general_comparison, enemy_count, uav_count)
    plot_graph.comparing_algorithms_bars(directory, list_detection_success_rate_scout, entropy_result, maxProb_result, random_result,
                              IsSuccessRate=True, is_genaral_comparison=general_comparison, enemyCount=enemy_count, uavCount=uav_count)

    # calculate the mean of duration to enemy catched for all algorithms
    maxProb_result = pandas_queries.create_duration_table(results_csv, 'MaxProb', general_comparison, enemy_count, uav_count)
    entropy_result = pandas_queries.create_duration_table(results_csv, 'Entropy', general_comparison, enemy_count, uav_count)
    random_result = pandas_queries.create_duration_table(results_csv, 'Rand', general_comparison, enemy_count, uav_count)
    plot_graph.comparing_algorithms_bars(directory, list_detection_duration_scout, entropy_result, maxProb_result, random_result,
                              IsSuccessRate=False, is_genaral_comparison=general_comparison, enemyCount=enemy_count, uavCount=uav_count)

def determined_SCOUT_results(success_rate_list, duration_heatmap):
    list_detection_success_rate = np.zeros((6))
    list_detection_duration = np.zeros((6))
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

    return list_detection_success_rate, list_detection_duration

def min_values_for_each_row(heatmap):
    list_of_coordinates_for_each_row = []
    for i in range(np.size(heatmap, 0)):
        row_i = heatmap[i, :]
        min_value = np.min(row_i)
        index_of_min_value_in_row_i = np.where(row_i == np.amin(row_i))
        list_of_coordinates_for_each_row.append(((i, index_of_min_value_in_row_i[0][0]), min_value))

    return list_of_coordinates_for_each_row



def create_duration_table_SCOUT(directory, CSV_file, is_genaral_comparison=True, enemyCount=0, uavCount=0):
    algorithm = 'SCOUT'

    if is_genaral_comparison:
        pdf_name = directory + '_duration_' + algorithm + '-algorithm_general_' + str(
            datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter  by algorithm kind
        filtered_file = CSV_file.loc[lambda df: df.Algorithm == algorithm]

    else:
        pdf_name = directory + '_duration_' + algorithm + '-algorithm_[' + str(
            enemyCount) + ']-enemy_[' + str(uavCount) + ']-UAV_' + str(datetime.datetime.today()).replace(' ',
                                                                                                                   '_').replace(
            ':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        filtered_file = \
        CSV_file.loc[lambda df: df.initialEnemyCount == enemyCount].loc[lambda df: df.initialUavCount == uavCount].loc[
            lambda df: df.Algorithm == algorithm]

    DET_SIZE = 6
    THR_SIZE = 9
    min_range = 1
    max_range = 10
    DET_range = np.round(np.arange(0.5, 1.09, 0.1), 1)
    THR_range = np.round(np.arange(0.1, 1.0, 0.1), 1)

    heatmap = np.zeros((DET_SIZE, THR_SIZE))
    for i in range(5, 11):
        DETECTION_PROB = i / 10
        if DETECTION_PROB not in filtered_file[['DETECTION_PROBABILITY']].values: continue
        for j in range(min_range, max_range):
            THR = j / 10
            if THR not in filtered_file[['CELL_CHOOSE_PROB_THRESHOLD']].values: continue

            # filter by detection prob and threshold per all success scenarios
            filter = filtered_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 1]
            if len(filter[['duration']].values) > 0:
                heatmap[i - 5][j - 1] = filter[['duration']].values.mean(axis=0)

    plot_graph.create_heatmap(heatmap, np.around(THR_range, 1), np.around(DET_range, 1), pdf_name, 'Duration until enemy cathed')
    return heatmap
    # min_values_for_each_row(heatmap)
    # createCsvFile(heatmap, THR_range, DET_range, pdf_name)
    # createPolysTable(heatmap, THR_range, min(set(filtered_file[['duration']].values.flatten())), max(set(filtered_file[['duration']].values.flatten())))
    # create_table(pdf_name, heatmap, max(set(filtered_file[['duration']].values.flatten())), min(set(filtered_file[['duration']].values.flatten())), 'Catched Enemy Duration', DET_range, THR_range, True)

