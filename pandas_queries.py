from pickletools import decimalnl_long
import plot_graph

import numpy as np
import datetime
import operator

def create_data_base(csv):
    algo_name = ['SCOUT', 'Entropy', 'MaxProb', 'Rand']
    algo_dict = {}
    THR = {}

    for algo in algo_name:
        csv_algo = csv.loc[lambda df: df.Algorithm == algo]
        if algo == 'SCOUT':
            algo_dict[algo], THR = create_data_base_scout(csv_algo)
        else:
            algo_dict[algo] = create_data_base_algo(csv_algo)

    return algo_dict, THR

def create_data_base_algo(csv):
    """
    create table by DP, for all DP handle matrix with enemy count in raws and uav count in columns
    :param csv: parsed csv for specific algo
    :return: Dict for specific algo
    """
    algo_data_base = {}
    # detection_probability
    dp_name = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    enemy_size, uav_size = 6, 6

    for dp in dp_name:
        csv_dp = csv.loc[lambda df: df.DETECTION_PROBABILITY == dp]
        Matrix = [[0 for x in range(enemy_size)] for y in range(uav_size)]
        for enemy_i in range(1, enemy_size):
            csv_dp_enemy = csv_dp.loc[lambda df: df.initialEnemyCount == enemy_i]
            for uav_i in range(1, uav_size):
                csv_dp_enemy_uav = csv_dp_enemy.loc[lambda df: df.initialUavCount == uav_i]

                # filter by detection prob and threshold per all success scenarios
                success = csv_dp_enemy_uav.loc[lambda df: df.success == 1]
                success_num = len(success[['success']].values)

                failures = csv_dp_enemy_uav.loc[lambda df: df.success == 0]
                failures_num = len(failures[['success']].values)

                if success_num + failures_num > 0:
                    Matrix[enemy_i - 1][uav_i - 1] = (success_num / (success_num + failures_num)) * 100
        algo_data_base[dp] = np.array(Matrix)

    return algo_data_base


def create_data_base_scout(csv):
    """ For SCOUT algo:
        create table by DP, for all DP handle matrix with enemy count in raws and uav count in columns
        :param csv: parsed csv for SCOUT
        :return: Dict for SCOUT
        """
    algo_data_base = {}
    THR = {}

    enemy_size, uav_size = 6, 6

    # detection_probability
    dp_name = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for dp in dp_name:
        algo_data_base[dp] = np.array([[0 for x in range(enemy_size)] for y in range(uav_size)])
        THR[dp] = np.array([[0 for x in range(enemy_size)] for y in range(uav_size)])
    dp_name_dict = {0.5: 0, 0.6: 1, 0.7: 2, 0.8: 3, 0.9: 4, 1: 5}

    for dp in dp_name:
        for enemy_i in range(1, enemy_size):
            for uav_i in range(1, uav_size):
                try:
                    coordinate_list = create_success_rates_table_SCOUT\
                        (" ", csv, is_genaral_comparison=False, enemyCount=enemy_i, uavCount=uav_i)

                    for item in coordinate_list:
                        if item[0][0] == dp_name_dict[dp]:
                            algo_data_base[dp][enemy_i - 1][uav_i - 1] = item[1]
                            THR[dp][enemy_i - 1][uav_i - 1] = item[0][1]
                except:
                    #No enemy count or uav count
                    b=9
    return algo_data_base, THR

def create_success_rates_table_SCOUT(directory, CSV_file, DET_range, THR_range, is_genaral_comparison=True, enemyCount=0, uavCount=0):
    algorithm = 'SCOUT'

    if is_genaral_comparison:
        pdf_name = directory + '_successRates' + '_' + algorithm + '-algorithm_general_' + str(datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by algorithm kind
        CSV_file = CSV_file.loc[lambda df: df.Algorithm == algorithm]

    else:
        pdf_name = directory +'_successRates' + '_' + algorithm + '-algorithm_[' + str(
            enemyCount) + ']-enemy_[' + str(
            uavCount) + ']-UAV_' + str(datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        CSV_file = \
            CSV_file.loc[lambda df: df.initialEnemyCount == enemyCount].loc[
                lambda df: df.initialUavCount == uavCount].loc[
                lambda df: df.Algorithm == algorithm]

    best_thr = [find_best_thr_by_dp(CSV_file, THR_range, i) for i in DET_range]

    heatmap = np.zeros((len(DET_range), len(THR_range)))

    for i, DETECTION_PROB in enumerate(DET_range):
        if DETECTION_PROB not in CSV_file[['DETECTION_PROBABILITY']].values: continue

        for j, THR in enumerate(THR_range):
            if THR not in CSV_file[['CELL_CHOOSE_PROB_THRESHOLD']].values: continue

            # filter by detection prob and threshold per all success scenarios
            success = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[
                lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 1]

            success_num = len(success[['success']].values)

            failures = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[
                lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 0]

            failures_num = len(failures[['success']].values)
            if success_num + failures_num > 0:

                heatmap[i][j] = (success_num / (success_num + failures_num)) * 100

    plot_graph.create_heatmap(heatmap, THR_range, DET_range, pdf_name,

                              'Success Rate')
    # create_heatmap(heatmap, np.around(THR_range, 1), np.around(DET_range, 1), pdf_name, 'Success rates')
    # createCsvFile(heatmap, THR_range, DET_range, pdf_name)
    # create_table(pdf_name, heatmap, 100, 30, 'success rate percent', DET_range, THR_range)
    return max_values(heatmap), best_thr

def max_values(heatmap):
    """
    This function take the success rate heatmap and collect all max value of Threshold per detection prob
    :param heatmap:
    :return: list of coordinates and max value, in the left we find (x,y) when x - DP and y - THR.
    """
    list_of_coordinates_for_each_row = []
    for i in range(np.size(heatmap, 0)):
        row_i = heatmap[i, :]
        max_value = np.around(np.max(row_i), 1)
        index_of_max_value_in_row_i = np.where(row_i == np.amax(row_i))
        [list_of_coordinates_for_each_row.append(((i, coordinate), max_value))
         for coordinate in index_of_max_value_in_row_i[0]]

    return list_of_coordinates_for_each_row

def create_duration_table(DET_range, CSV_file, algorithm, delay, is_genaral_comparison=True, enemyCount=0, uavCount=0):
    path = 'C:/Users/AVIADFUX/Desktop/Projects/Mor_project/analyze/results/'

    if is_genaral_comparison:
        pdf_name = path + '_successRates' + '_' + algorithm + '-algorithm_generalPlayers_' + str(
            datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        CSV_file = \
            CSV_file.loc[lambda df: df.Algorithm == algorithm]
    else:
        pdf_name = path + '_duration_' + algorithm + '-algorithm' + str(
            enemyCount) + '-enemiesCount_' + str(
            uavCount) + '-uavCount_' + str(datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        CSV_file = \
            CSV_file.loc[lambda df: df.initialEnemyCount == enemyCount].loc[
                lambda df: df.initialUavCount == uavCount].loc[
                lambda df: df.Algorithm == algorithm]


    list_duration_detection = np.zeros((len(DET_range)))
    for i, DETECTION_PROB in enumerate(DET_range):
        if DETECTION_PROB not in CSV_file[['DETECTION_PROBABILITY']].values: continue

        # filter by detection prob and threshold per all success scenarios
        success_filter = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[lambda df: df.success == 1]
        if len(success_filter[['duration']].values) > 0:
            list_duration_detection[i] = success_filter[['duration']].values.mean(axis=0) - delay

    return list_duration_detection

def create_success_rates_table(DET_range, CSV_file, algorithm, is_genaral_comparison=True, enemyCount=0, uavCount=0):
    path = 'C:/Users/AVIADFUX/Desktop/Projects/Mor_project/analyze/results/'

    if is_genaral_comparison:
        pdf_name = path + '_successRates' + '_' + algorithm + '-algorithm_generalPlayers_' + str(
            datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        CSV_file = \
            CSV_file.loc[lambda df: df.Algorithm == algorithm]
    else:
        pdf_name = path + '_successRates' + '_' + algorithm + '-algorithm' + str(
            enemyCount) + '-enemiesCount_' + str(
            uavCount) + '-uavCount_' + str(datetime.datetime.today()).replace(' ', '_').replace(':', '-') + '.'

        # filter by enemies and UAVs count and by algorithm kind
        CSV_file = \
            CSV_file.loc[lambda df: df.initialEnemyCount == enemyCount].loc[
                lambda df: df.initialUavCount == uavCount].loc[
                lambda df: df.Algorithm == algorithm]


    list_detection_success_rate = np.zeros((len(DET_range)))
    for i, DETECTION_PROB in enumerate(DET_range):
        if DETECTION_PROB not in CSV_file[['DETECTION_PROBABILITY']].values: continue

        # filter by detection prob and threshold per all success scenarios
        success = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[lambda df: df.success == 1]

        success_num = len(success[['success']].values)

        failures = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[lambda df: df.success == 0]

        failures_num = len(failures[['success']].values)
        if success_num + failures_num > 0:
            list_detection_success_rate[i] = (success_num / (success_num + failures_num)) * 100

    return list_detection_success_rate


def find_best_thr_by_dp(csv_file, THR_range, dp):
    """
    This method find the best THR for specific detection probability
    :param csv_file:
    :param dp:
    :return: The best THR
    """

    filter = csv_file.loc[lambda df: df.Algorithm == 'SCOUT']
    filter = filter.loc[lambda df: df.DETECTION_PROBABILITY == dp]

    if filter.empty:
        return 0
    thr_dict = {}
    for j, THR in enumerate(THR_range):
        failures = filter.loc[lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 0]
        success = filter.loc[lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 1]

        failures_num = len(failures[['success']].values)
        success_num = len(success[['success']].values)

        if success_num + failures_num == 0: continue

        thr_dict[THR] = success_num / (success_num + failures_num)

    return max(thr_dict.items(), key=operator.itemgetter(1))[0]


def create_duration_table_SCOUT(directory, CSV_file, DET_range, THR_range, delay, is_genaral_comparison=True,
                                enemyCount=0, uavCount=0):
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
            CSV_file.loc[lambda df: df.initialEnemyCount == enemyCount].loc[
                lambda df: df.initialUavCount == uavCount].loc[
                lambda df: df.Algorithm == algorithm]

    heatmap = np.zeros((len(DET_range), len(THR_range)))

    for i, DETECTION_PROB in enumerate(DET_range):
        if DETECTION_PROB not in filtered_file[['DETECTION_PROBABILITY']].values: continue

        for j, THR in enumerate(THR_range):
            if THR not in filtered_file[['CELL_CHOOSE_PROB_THRESHOLD']].values: continue

            # filter by detection prob and threshold per all success scenarios
            filter = filtered_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[
                lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR].loc[lambda df: df.success == 1]
            if len(filter[['duration']].values) > 0:
                heatmap[i][j] = filter[['duration']].values.mean(axis=0) - delay

    # plot_graph.create_heatmap(heatmap, np.around(THR_range, 1), np.around(DET_range, 1), pdf_name, 'Duration until enemy cathed')
    plot_graph.create_heatmap(heatmap, THR_range, DET_range, pdf_name,
                              'Duration until enemy cathed')
    return heatmap
    # min_values_for_each_row(heatmap)
    # createCsvFile(heatmap, THR_range, DET_range, pdf_name)
    # createPolysTable(heatmap, THR_range, min(set(filtered_file[['duration']].values.flatten())), max(set(filtered_file[['duration']].values.flatten())))
    # create_table(pdf_name, heatmap, max(set(filtered_file[['duration']].values.flatten())), min(set(filtered_file[['duration']].values.flatten())), 'Catched Enemy Duration', DET_range, THR_range, True)
