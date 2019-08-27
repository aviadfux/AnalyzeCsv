import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import seaborn as sns
import pandas as pd

IS_OUTPUT_PDF = False
IS_SHOW_PLT = True

def comparing_algorithms_bars_fixed_scenarios(directory, SCOUT_results, ENTROPY_results, MAX_PROB_results, RANDOM_results, IsSuccessRate, fixed_player, fixed_num, dp):
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()

    ind = np.arange(6)
    scout = ax.bar(ind + width / 4, SCOUT_results, width/2,
                     label='Scout')
    entropy = ax.bar(ind - 3*width / 4, ENTROPY_results, width/2,
                    label='Entropy')
    maxProb = ax.bar(ind + 3*width / 4, MAX_PROB_results, width/2,
                    label='Max Prob')
    random = ax.bar(ind - width / 4, RANDOM_results, width/2,
                     label='Random')

    if IsSuccessRate:
        comparison_name = ylabel = 'Success Rate'
        title = "Graph for success rate - DP: [{0}], fixed [{1}]: [{2}]".format(dp, fixed_player, fixed_num)
        ax.set_ylim(0, 100)
    else:
        comparison_name = ylabel = 'Duration'
        title = "Graph for Duration - DP: [{0}], fixed [{1}]: [{2}]".format(dp, fixed_player, fixed_num)
        ax.set_ylim(1500, 3500)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    x_label = "Enemy Count" if fixed_player == "uav" else "UAV Count"
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(np.around(ind + 1))
    ax.legend()

    with PdfPages(
            directory + "_success rate__DP_[{0}]__fixed_[{1}]_[{2}]__".format(dp, fixed_player, fixed_num) + str(datetime.datetime.today()).replace(' ', '_').replace(
                    ':', '-') + '.pdf') as pdf:
        fig.tight_layout()
        if IS_OUTPUT_PDF: pdf.savefig()

    if IS_SHOW_PLT: plt.show()

def comparing_algorithms_bars(directory, SCOUT_results, ENTROPY_results, MAX_PROB_results, RANDOM_results, IsSuccessRate, is_genaral_comparison=True, enemyCount=0, uavCount=0):
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()

    ind = np.arange(6)
    scout = ax.bar(ind + width / 4, SCOUT_results, width/2,
                     label='Scout')
    entropy = ax.bar(ind - 3*width / 4, ENTROPY_results, width/2,
                    label='Entropy')
    maxProb = ax.bar(ind + 3*width / 4, MAX_PROB_results, width/2,
                    label='Max Prob')
    random = ax.bar(ind - width / 4, RANDOM_results, width/2,
                     label='Random')

    if is_genaral_comparison:
        scenario = 'General'
    else:
        scenario = "[{0}]-Enemy and [{1}]-UAV".format(enemyCount, uavCount)

    if IsSuccessRate:
        comparison_name = ylabel = 'Success Rate'
        title = "Graph for success rate by detection probability for {0} scenario".format(scenario)
        ax.set_ylim(0, 100)
    else:
        comparison_name = ylabel = 'Duration'
        title = "Graph for Duration by detection probability for {0} scenario".format(scenario)
        ax.set_ylim(1500, 3500)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Detection Probability")
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(np.around(np.arange(0.5, 1.01, 0.1), 1))
    ax.legend()

    with PdfPages(
            directory + '_compare algorithms_' + comparison_name + '_' + scenario + '_' + str(datetime.datetime.today()).replace(' ', '_').replace(
                    ':', '-') + '.pdf') as pdf:
        fig.tight_layout()
        if IS_OUTPUT_PDF: pdf.savefig()

    if IS_SHOW_PLT: plt.show()

def create_heatmap(heatmap, THR, DET, file_name, title):
    with PdfPages(file_name + 'pdf') as pdf:
        fig = plt.figure()
        sns.set()
        sns.set(font_scale=0.8)

        THR = np.append(np.zeros(1), THR)

        DET = DET.reshape(len(DET), 1)
        heatmap = np.append(DET, heatmap, axis=1)

        heatmap = np.vstack((THR, heatmap))

        pandasTable = pd.DataFrame(data=heatmap[1:, 1:], index=heatmap[1:, 0], columns=heatmap[0, 1:])

        ax = sns.heatmap(pandasTable, annot=True, fmt=".1f", label='Duration')
        ax.set_title(title)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Detection Probability')

        if IS_OUTPUT_PDF: pdf.savefig()
        if IS_SHOW_PLT: plt.show()