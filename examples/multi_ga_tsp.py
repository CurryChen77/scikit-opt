import time

import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sko.GA import GA_TSP

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
        best_points, best_distance, ga_tsp, size_pop, max_iter, prob_mut, cal_time = result
        best_distance = round(best_distance, 2)
        cal_time = round(cal_time, 2)
        local_variables = {'size_pop': size_pop, 'max_iter': max_iter, 'prob_mut': prob_mut}
        title = ", ".join(["{}:{}".format(var, local_variables.get(var)) for var in titles])
        var_info = ", ".join(["{}:{}".format(var, local_variables.get(var)) for var in legends])
        label = var_info + f", time:{cal_time}"
        ax[0].plot(ga_tsp.generation_best_Y, label=label)
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("Distance")
        ax[0].set_ylim(5300, 9000)
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
    plt.savefig(f'image/ga-tsp-{var_name}.png', dpi=500)
    plt.show()


def generate_parameter_range(start, stop, num):
    if num == 1:
        return np.array([(start + stop) / 2])
    else:
        return np.linspace(start, stop, num)


def main():
    results = []

    size_pops = generate_parameter_range(80, 100, 1)
    max_iters = generate_parameter_range(2000, 3000, 1)
    prob_muts = generate_parameter_range(1, 3, 3)

    titles = []
    legends = []
    if len(size_pops) > 1:
        legends.append('size_pop')
    else:
        titles.append('size_pop')

    if len(max_iters) > 1:
        legends.append('max_iter')
    else:
        titles.append('max_iter')

    if len(prob_muts) > 1:
        legends.append('prob_mut')
    else:
        titles.append('prob_mut')

    for size_pop in size_pops:
        for max_iter in max_iters:
            for prob_mut in prob_muts:
                start_time = time.time()
                ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=int(size_pop), max_iter=int(max_iter), prob_mut=int(prob_mut))
                best_points, best_distance = ga_tsp.run()
                end_time = time.time()
                results.append((best_points, best_distance[0], ga_tsp, size_pop, max_iter, prob_mut, (end_time - start_time)))
                # print(best_points, best_distance, cal_total_distance(best_points))
    plot(results, titles, legends)


if __name__ == '__main__':
    main()
