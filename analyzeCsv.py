import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import seaborn as sns
import os
import pandas_queries
import utils
import sys
import argparse


file_name = 'run___i40_e5_u2_d2000___short'

OnAllCatched = 'OnAllCatched%s|'
MAX_enemy = 10
IS_OUTPUT_PDF = False
IS_SHOW_PLT = True

def find_best_thr(csv_file):
    results = {}

    for dp in np.arange(0.5, 1, 0.1):
        dp = round(dp, 2)
        results[dp] = pandas_queries.find_best_thr_by_dp(csv_file, dp)
    results[1] = pandas_queries.find_best_thr_by_dp(csv_file, 1)

    return results

def main():

    parser = argparse.ArgumentParser(
        description='Analyze CSV arguments')

    parser.add_argument('--path_table', type=str)
    parser.add_argument('--enemy_count', type=int)
    parser.add_argument('--uav_count', type=int)
    parser.add_argument('--delay', type=int)
    parser.add_argument('--dp', type=float)
    parser.add_argument('--exploration_algo', type=str) #m/a/h
    parser.add_argument('--fixed_player', type=str)
    parser.add_argument('--fixed_num', type=str)
    parser.add_argument('--graph_type', type=str, default="compare_algo")

    args = parser.parse_args()

    results_csv = pd.read_csv(args.path_table, error_bad_lines=False)

    results_csv = results_csv.loc[lambda df: df.delay == args.delay]

    if args.graph_type == "compare_algo":
        utils.comparing_algorithms(results_csv, args.path_table, general_comparison=False, enemy_count=args.enemy_count, uav_count=args.uav_count)
    elif args.graph_type == "general_compare_algo":
        utils.comparing_algorithms(results_csv, args.path_table)
    elif args.graph_type == "fixed_player":
        data_base, THR = pandas_queries.create_data_base(results_csv)
        utils.comparing_algorithms_fixed_player(data_base, dp=args.dp, fixed_player=args.fixed_player, fixed_num=args.fixed_num)

    best_thr = find_best_thr(results_csv)


if __name__ == '__main__':
    main()



def createTableByDelay(CSV_file, delay):

    path = 'C:/Users/AVIADFUX/Desktop/Projects/Mor_project/analyze/results/'
    with PdfPages(path + file_name + '_' + str(delay) + '-delay_' + str(datetime.datetime.today()).replace(' ','_').replace(':','-') + '.pdf') as pdf:

        DET_SIZE = 6
        THR_SIZE = 11
        heatmap = np.zeros((DET_SIZE, THR_SIZE))
        for i in range(5, 11):
            DETECTION_PROB = i / 10
            if DETECTION_PROB not in CSV_file[['DETECTION_PROBABILITY']].values: continue
            for j in range(0, 11):
                THR = j / 10
                if THR not in CSV_file[['CELL_CHOOSE_PROB_THRESHOLD']].values: continue

                filter = CSV_file.loc[lambda df: df.DETECTION_PROBABILITY == DETECTION_PROB].loc[lambda df: df.CELL_CHOOSE_PROB_THRESHOLD == THR]

                catchedNum = 0
                for w in range(1, MAX_enemy):
                    filterByEnemy = filter.loc[lambda df: df.initialEnemyCount == w]
                    if len(filterByEnemy[['initialEnemyCount']].values) > 0:
                        catchedNum += len(filterByEnemy.loc[lambda df: df.eventStr == OnAllCatched % w][['eventStr']].values)

                enemyArivedNum = len(filter.loc[lambda df: df.eventStr == 'OnEnemyArrive|'][['eventStr']].values)
                if catchedNum + enemyArivedNum == 0: continue
                heatmap[i-5][j] = catchedNum * 100 / (catchedNum + enemyArivedNum)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        DET = np.arange(0.5, 1.09, 0.1)
        THR = np.arange(0.0, 1.1, 0.1)
        X, Y = np.meshgrid(DET, THR)

        surf = ax.contour3D(X, Y, np.transpose(heatmap), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.set_xlabel('DETECTION PROBABILITY')
        ax.set_ylabel('THRESHOLD')
        ax.set_zlabel('Catched Enemy Percent')

        # Customize the z axis.
        ax.set_zlim(0, 100)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        pdf.savefig()

        plt.show()

    f = 7

def create_table(pdf_name, z_axis, max_z_axis, min_z_axis, z_label, DET=np.arange(0.5, 1.09, 0.1),
                 THR=np.arange(0.0, 1.1, 0.1), reverse=False):
    with PdfPages(pdf_name + 'pdf') as pdf:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(DET, THR)

        surf = ax.plot_surface(X, Y, np.transpose(z_axis), cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.set_xlabel('DETECTION PROBABILITY')
        ax.set_ylabel('THRESHOLD')
        ax.set_zlabel(z_label)

        # Customize the z axis, determined max z axis as max duration.
        if reverse: ax.set_zlim(max_z_axis, min_z_axis)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        if IS_OUTPUT_PDF: pdf.savefig()
        if IS_SHOW_PLT: plt.show()


def createCsvFile(heatmap, THR, DET, file_name):
    THR = np.append(np.zeros(1), THR)

    DET = DET.reshape(len(DET), 1)
    heatmap = np.append(DET, heatmap, axis=1)

    heatmap = np.vstack((THR, heatmap))

    np.savetxt(file_name + 'csv', heatmap, delimiter=",")

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.1)

def createPolysTable(heatmap, THR_range, min_z, max_z):
    verts = []

    xs = np.arange(0.5, 1.1, 0.1)
    for raw in heatmap.T:
        b = list(zip(xs, raw))
        b.insert(0, (0.4, min_z))
        b.insert(len(xs) + 2, (1.1, min_z))
        verts.append(b)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    poly = PolyCollection(verts, facecolors=[cc('peachpuff'), cc('cyan'), cc('lime'), cc('orange'), cc('grey'),
                                             cc('darkred'), cc('darkblue'), cc('darkgreen'), cc('black')])
    poly.set_alpha(0.6)
    ax.add_collection3d(poly, zs=THR_range, zdir='y')

    ax.set_xlabel('X')
    ax.set_xlim3d(0.4, 1.1)
    ax.set_ylabel('Y')
    ax.set_ylim3d(0.0, 1.1)
    ax.set_zlabel('Z')
    ax.set_zlim3d(min_z, 3000)

    if IS_SHOW_PLT: plt.show()