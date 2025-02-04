import os
import statistics
from Constants import *
import pickle
from Plots import *

# NOT THE BEST way to do this, but I'm reusing old code
types_of_search = ["systematic", "local", "annealing forward", "annealing reverse", "penalty"]
types_of_search_dict = { }
index = 0
for search_type in types_of_search:
    types_of_search_dict[f"{search_type} search"] = index
    index += 1

# extract general data
coverages = [[] for _ in range(len(types_of_search))]
exploration_levels = [[] for _ in range(len(types_of_search))]
times = [[] for _ in range(len(types_of_search))]
for type_of_search, type_value in types_of_search_dict.items():
    for j in range(NUM_OF_SIMULATIONS):
        coverages[type_value].append(pickle.load(open(f"Experiment results/experiment2/{type_of_search}/{j}/coverages.p","rb")))
        times[type_value].append(pickle.load(open(f"Experiment results/experiment2/{type_of_search}/{j}/time_elapsed.p", "rb")))
        exploration_levels[type_value].append(pickle.load(open(f"Experiment results/experiment2/{type_of_search}/{j}/exploration_levels.p", "rb")))

# get average time foreach type of search
times_avg = [statistics.mean(time) for time in times]
for type_of_search in types_of_search_dict.keys():
    os.makedirs(f'Experiment results/experiment2/{type_of_search}/average', exist_ok=True)
    with open(f"Experiment results/experiment2/{type_of_search}/average_time_elapsed.txt", "w") as f:
        f.write(str(times_avg[types_of_search_dict[type_of_search]]) + "\n")

# get final values
final_coverages = [[0 for i in range(len(coverages[0]))] for j in range(len(coverages))]
final_explorations = [[0 for i in range(len(exploration_levels[0]))] for j in range(len(exploration_levels))]
for i in range(len(coverages)):
    for j in range(len(coverages[i])):
        final_coverages[i][j] = coverages[i][j][-1]
        final_explorations[i][j] = exploration_levels[i][j][-1]

mean_covs = [statistics.mean(final_coverage) for final_coverage in final_coverages]
max_covs = [max(final_coverage) for final_coverage in final_coverages]
min_covs = [min(final_coverage) for final_coverage in final_coverages]
std_devs_covs = [statistics.stdev(final_coverage) for final_coverage in final_coverages]
for type_of_search in types_of_search_dict.keys():
    with open(f"Experiment results/experiment2/{type_of_search}/average/mean_coverage.txt", "w") as f:
        f.write(str(mean_covs[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/max_coverage.txt", "w") as f:
        f.write(str(max_covs[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/min_coverage.txt", "w") as f:
        f.write(str(min_covs[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/std_dev_coverage.txt", "w") as f:
        f.write(str(std_devs_covs[types_of_search_dict[type_of_search]]) + "\n")


mean_expls = [statistics.mean(final_exploration) for final_exploration in final_explorations]
max_expls = [max(final_exploration) for final_exploration in final_explorations]
min_expls = [min(final_exploration) for final_exploration in final_explorations]
std_devs_expls = [statistics.stdev(final_exploration) for final_exploration in final_explorations]
for type_of_search in types_of_search_dict.keys():
    with open(f"Experiment results/experiment2/{type_of_search}/average/mean_exploration.txt", "w") as f:
        f.write(str(mean_expls[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/max_exploration.txt", "w") as f:
        f.write(str(max_expls[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/min_exploration.txt", "w") as f:
        f.write(str(min_expls[types_of_search_dict[type_of_search]]) + "\n")
    with open(f"Experiment results/experiment2/{type_of_search}/average/std_dev_exploration.txt", "w") as f:
        f.write(str(std_devs_expls[types_of_search_dict[type_of_search]]) + "\n")


# get averages
average_coverages = [[0 for i in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(types_of_search))]
average_explorations = [[0 for i in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(types_of_search))]
for i in range(len(types_of_search)):
    for k in range(NUM_OF_ITERATIONS + 1):
        for j in range(NUM_OF_SIMULATIONS):
            average_coverages[i][k] += coverages[i][j][k]
            average_explorations[i][k] += exploration_levels[i][j][k]
        average_coverages[i][k] /= NUM_OF_SIMULATIONS
        average_explorations[i][k] /= NUM_OF_SIMULATIONS

for type_of_search in types_of_search_dict.keys():
    plot_coverage(average_coverages[types_of_search_dict[type_of_search]]
                  , times_avg[types_of_search_dict[type_of_search]]
                  , type_of_search.replace(" search", "")
                  , None
                  , "average"
                  , None
                  , None)
    plot_exploration(average_explorations[types_of_search_dict[type_of_search]]
                  , times_avg[types_of_search_dict[type_of_search]]
                  , type_of_search.replace(" search", "")
                  , None
                  , "average"
                  , None)

plot_coverages_comparison(average_coverages, types_of_search)
plot_exploration_comparison(average_explorations, types_of_search)