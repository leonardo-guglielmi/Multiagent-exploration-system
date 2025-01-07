import pickle
import statistics
from Plots import plot_coverage, plot_coverages_comparison, plot_exploration_comparison, plot_scatter_regression
from datetime import datetime as date

from Simulate import simulate
from User import User
from Constants import *
from Sensor import Sensor


def main():
    try:
        types_of_search = ["systematic", "local", "annealing forward", "annealing reverse", "penalty"]
        expl_weights = ["constant", "decrescent"]

        print(f"Simulations begin: {date.now()}\n")
        for i in range(NUM_OF_SIMULATIONS):
            deserialize = False
            for type_of_search in types_of_search:
                for expl_weight in expl_weights:
                    print(f'----- Starting simulation [{type_of_search}-{expl_weight}] : {i} -----')
                    simulate(type_of_search, expl_weight, i, deserialize)
                    Sensor.id = 0
                    User.id = 0
                    deserialize = True
        print("Simulations completed")

        # this dictionary is used to convert types of search into numbers for indexing
        types_of_search_dict = {"systematic search": 0, "local search": 1, "annealing forward search": 2,
                                "annealing reverse search": 3, "penalty search": 4}

        # load results into arrays
        coverages = [[] for _ in range(len(types_of_search))]
        times = [[] for _ in range(len(types_of_search))]
        exploration_levels = [[] for _ in range(len(types_of_search))]
        for type_of_search, type_value in types_of_search_dict.items():
            for expl_weight in expl_weights:
                for j in range(NUM_OF_SIMULATIONS):
                    # 3D matrix, for each type of search, for each simulation load coverage history of that simulation
                    coverages[type_value].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/coverages.p", "rb")))
                    times[type_value].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/time_elapsed.p", "rb")))
                    exploration_levels[type_value].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/exploration_levels.p", "rb")))

        times_avg = [statistics.mean(time) for time in times]
        # ------
        # statistic indices for coverage
        # ------

        # extract final coverages of each simulation
        final_coverages = [[0 for _ in range(len(coverages[0]))] for _ in range(len(coverages))]
        for i in range(len(coverages)):
            for j in range(len(coverages[i])):
                final_coverages[i][j] = coverages[i][j][-1]  # get only the final coverage at the end of the simulation

        mean_final_coverage = [statistics.mean(final_reward) for final_reward in final_coverages]
        max_final_coverage = [max(final_reward) for final_reward in final_coverages]
        min_final_coverage = [min(final_reward) for final_reward in final_coverages]
        std_devs_coverage = [statistics.stdev(final_reward) for final_reward in final_coverages]

        # if the simulation stopped earlier, add elements to adapt list length
        for i in range(len(types_of_search)):
            for j in range(NUM_OF_SIMULATIONS):
                if len(coverages[i][j]) < NUM_OF_ITERATIONS + 1:
                    for k in range(NUM_OF_ITERATIONS + 1 - len(coverages[i][j])):
                        coverages[i][j].append(1.0)

        # for each type_of_search, get the average coverage of all simulations for specific time
        average_coverages = [[0 for _ in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(types_of_search))]
        for i in range(len(types_of_search)):
            for k in range(NUM_OF_ITERATIONS + 1):
                for j in range(NUM_OF_SIMULATIONS):
                    average_coverages[i][k] += coverages[i][j][k]
                average_coverages[i][k] /= NUM_OF_SIMULATIONS

        # ------
        # statistic indices for exploration
        # ------
        final_expl_levels = [[0 for _ in range(len(exploration_levels[0]))] for _ in range(len(exploration_levels))]
        for i in range(len(exploration_levels)):
            for j in range(len(exploration_levels[i])):
                final_expl_levels[i][j] = exploration_levels[i][j][-1]  # get only the final coverage at the end of the simulation

        mean_final_expl = [statistics.mean(final_expl) for final_expl in final_expl_levels]
        max_final_expl = [max(final_expl) for final_expl in final_expl_levels]
        min_final_expl = [min(final_expl) for final_expl in final_expl_levels]
        std_devs_expl = [statistics.stdev(final_expl) for final_expl in final_expl_levels]

        # if the simulation stopped earlier, add elements to adapt list length
        for i in range(len(types_of_search)):
            for j in range(NUM_OF_SIMULATIONS):
                if len(exploration_levels[i][j]) < NUM_OF_ITERATIONS + 1:
                    for k in range(NUM_OF_ITERATIONS + 1 - len(exploration_levels[i][j])):
                        exploration_levels[i][j].append(1.0)

        # for each type_of_search, get the average coverage of all simulations for specific time
        average_expl_levels = [[0 for _ in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(types_of_search))]
        for i in range(len(types_of_search)):
            for k in range(NUM_OF_ITERATIONS + 1):
                for j in range(NUM_OF_SIMULATIONS):
                    average_expl_levels[i][k] += exploration_levels[i][j][k]
                average_expl_levels[i][k] /= NUM_OF_SIMULATIONS

        # QUESTI LI STO SPERIMENTANDO
        #avg_der = [[] for _ in range(len(types_of_search))]
        #for i in range(len(exploration_levels)):
            #for j in range(len(exploration_levels[i])):
                #simu_expl_lvl = exploration_levels[i][j]
                #der = []
                #for k in range(1, len(simu_expl_lvl)):
                    #der.append(simu_expl_lvl[k]-simu_expl_lvl[k-1])
                #avg_der[i][j] = statistics.mean(der)

        # -----
        # storing results
        # -----
        for type_of_search in types_of_search_dict.keys():
            for expl_weight in expl_weights:
                plot_coverage(average_coverages[types_of_search_dict[type_of_search]], 0,
                          type_of_search.replace(" search", ""), expl_weight, "average")

                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/average_time_elapsed.txt", "w") as f:
                    f.write(str(times_avg[types_of_search_dict[type_of_search]]) + "\n")

                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/mean_coverage.txt", "w") as f:
                    f.write(str(mean_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/max_coverage.txt", "w") as f:
                    f.write(str(max_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/min_coverage.txt", "w") as f:
                    f.write(str(min_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/std_dev_coverage.txt", "w") as f:
                    f.write(str(std_devs_coverage[types_of_search_dict[type_of_search]]) + "\n")

                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/mean_exploration.txt", "w") as f:
                    f.write(str(mean_final_expl[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/max_exploration.txt", "w") as f:
                    f.write(str(max_final_expl[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/min_exploration.txt", "w") as f:
                    f.write(str(min_final_expl[types_of_search_dict[type_of_search]]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/std_dev_exploration.txt", "w") as f:
                    f.write(str(std_devs_expl[types_of_search_dict[type_of_search]]) + "\n")

        plot_coverages_comparison(average_coverages)
        plot_exploration_comparison(average_expl_levels)
        #plot_scatter_regression(final_expl_levels, avg_der)

    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(str(e))
        raise e

if __name__ == '__main__':
    main()
