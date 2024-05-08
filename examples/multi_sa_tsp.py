import time

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sko.SA import SA_TSP

file_name = sys.argv[1] if len(sys.argv) > 1 else 'data/nctu.csv'
points_coordinate = np.loadtxt(file_name, delimiter=',')
num_points = points_coordinate.shape[0]
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
distance_matrix = distance_matrix * 111000  # 1 degree of lat/lon ~ = 111000m


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def plot(results, titles, legends):
    # %% Plot the best routine
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    num_col = len(results)+1
    fig, ax = plt.subplots(1, num_col, figsize=(5 * num_col, 5))

    for i, result in enumerate(results):
        best_points, best_distance, sa_tsp, T_max, T_min, L, cal_time = result
        best_distance = round(best_distance, 2)
        cal_time = round(cal_time, 2)
        local_variables = {'T_max': T_max, 'T_min': T_min, 'L': L}
        title = ", ".join(["{}:{}".format(var, local_variables.get(var)) for var in titles])
        var_info = ", ".join(["{}:{}".format(var, local_variables.get(var)) for var in legends])
        label = var_info + f", time:{cal_time}"
        ax[0].plot(sa_tsp.best_y_history, label=label)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Distance")
        ax[0].set_ylim(5300, 7000)
        ax[0].set_title(title, fontsize=14)
        ax[0].legend(loc='best', fontsize=12)

        best_points_ = np.concatenate([best_points, [best_points[0]]])
        best_points_coordinate = points_coordinate[best_points_, :]
        ax[i+1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
                   marker='o', markerfacecolor='b', color='c', linestyle='-',
                   label=f"Best Distance={best_distance}")
        ax[i+1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[i+1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax[i+1].set_title(var_info, fontsize=14)
        ax[i+1].set_xlabel("Longitude")
        ax[i+1].set_ylabel("Latitude")
        ax[i+1].legend(loc='best', fontsize=12)
    plt.tight_layout()
    var_name = "_".join([var for var in legends])
    plt.savefig(f'image/sa-tsp-{var_name}.png', dpi=500)
    plt.show()


def generate_parameter_range(start, stop, num):
    if num == 1:
        return np.array([(start + stop) / 2])
    else:
        return np.linspace(start, stop, num)


def main():
    results = []

    T_maxs = generate_parameter_range(90, 110, 3)
    T_mins = generate_parameter_range(1, 3, 1)
    Ls = generate_parameter_range(4, 16, 1) * num_points

    titles = []
    legends = []
    if len(T_maxs) > 1:
        legends.append('T_max')
    else:
        titles.append('T_max')

    if len(T_mins) > 1:
        legends.append('T_min')
    else:
        titles.append('T_min')

    if len(Ls) > 1:
        legends.append('L')
    else:
        titles.append('L')

    for T_max in T_maxs:
        for T_min in T_mins:
            for L in Ls:
                start_time = time.time()
                sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=T_max, T_min=T_min, L=L)
                best_points, best_distance = sa_tsp.run()
                end_time = time.time()
                results.append((best_points, best_distance, sa_tsp, T_max, T_min, L, (end_time - start_time)))
                # print(best_points, best_distance, cal_total_distance(best_points))
    plot(results, titles, legends)


if __name__ == '__main__':
    main()
