import pickle
import statistics

from Plots import *
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
        expl_weights_dict = {"constant": 0, "decrescent": 1}

        # load results into 4D matrix, for each type of search, for each exploration weight, for each simulation load coverage history of that simulation
        coverages: list[list[list[list[float]]]] = [[[] for _ in range(len(expl_weights))] for _ in range(len(types_of_search))]
        times: list[list[list[list[float]]]] = [[[] for _ in range(len(expl_weights))] for _ in range(len(types_of_search))]
        exploration_levels: list[list[list[list[float]]]] = [[[] for _ in range(len(expl_weights))] for _ in range(len(types_of_search))]

        for type_of_search, search_index in types_of_search_dict.items():
            for expl_weight, weight_index in expl_weights_dict.items():
                for j in range(NUM_OF_SIMULATIONS):
                    coverages[search_index][weight_index].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/coverages.p", "rb")))
                    times[search_index][weight_index].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/time_elapsed.p", "rb")))
                    exploration_levels[search_index][weight_index].append(pickle.load(open(f"Simulations output/{type_of_search}/{expl_weight} weight/{j}/exploration_levels.p", "rb")))

        # 2D matrix sorted by [type_of_search][expl_weight]
        times_avg = [[statistics.mean(time) for time in times_by_search] for times_by_search in times]

        # --------------------------------------------------------------------------------------------------------------
        # statistic indices for coverage
        # --------------------------------------------------------------------------------------------------------------

        final_coverages = [[[0 for _ in range(len(coverages[0][0]))] for _ in range(len(coverages[0]))] for _ in range(len(coverages))]
        for i in range(len(coverages)):
            for j in range(len(coverages[i])):
                for k in range(len(coverages[i][j])):
                    final_coverages[i][j][k] = coverages[i][j][k][-1]  # get only the final coverage at the end of the simulation

        mean_final_coverages = [[statistics.mean(final_cov_by_weight) for final_cov_by_weight in final_cov_by_search] for final_cov_by_search in final_coverages]
        max_final_coverages = [[max(final_cov_by_weight) for final_cov_by_weight in final_cov_by_search] for final_cov_by_search in final_coverages]
        min_final_coverages = [[min(final_cov_by_weight) for final_cov_by_weight in final_cov_by_search] for final_cov_by_search in final_coverages]
        std_devs_coverages = [[statistics.stdev(final_cov_by_weight) for final_cov_by_weight in final_cov_by_search] for final_cov_by_search in final_coverages]

        # if the simulation stopped earlier, add elements to adapt list length
        for i in range(len(types_of_search)):   # iter through searches
            for j in range(len(expl_weights)):      # iter through weights
                for k in range(NUM_OF_SIMULATIONS):     # iter through simulations
                    if len(coverages[i][j][k]) < NUM_OF_ITERATIONS + 1:
                        for _ in range(NUM_OF_ITERATIONS + 1 - len(coverages[i][j])):
                            coverages[i][j][k].append(1)

        # for each type_of_search, get the average coverage of all simulations for specific time
        average_coverages = [[[0 for _ in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(expl_weights))] for _ in range(len(types_of_search))]
        for i in range(len(types_of_search)):
            for j in range(len(expl_weights)):
                for k in range(NUM_OF_ITERATIONS + 1):
                    for l in range(NUM_OF_SIMULATIONS):
                        average_coverages[i][j][k] += coverages[i][j][l][k]
                    average_coverages[i][j][k] /= NUM_OF_SIMULATIONS

        # --------------------------------------------------------------------------------------------------------------
        # statistic indices for exploration
        # --------------------------------------------------------------------------------------------------------------
        final_expl_levels: list[list[list[int]]] = [[[0 for _ in range(len(exploration_levels[0][0]))] for _ in range(len(exploration_levels[0]))] for _ in range(len(exploration_levels))]
        for i in range(len(exploration_levels)):
            for j in range(len(exploration_levels[i])):
                for k in range(len(exploration_levels[i][j])):
                    final_expl_levels[i][j][k] = exploration_levels[i][j][k][-1]  # get only the final coverage at the end of the simulation

        mean_final_expls = [[statistics.mean(final_expl_by_weight) for final_expl_by_weight in final_expl_by_search] for final_expl_by_search in final_expl_levels]
        max_final_expls = [[max(final_expl_by_weight) for final_expl_by_weight in final_expl_by_search] for final_expl_by_search in final_expl_levels]
        min_final_expls = [[min(final_expl_by_weight) for final_expl_by_weight in final_expl_by_search] for final_expl_by_search in final_expl_levels]
        std_devs_expls = [[statistics.stdev(final_expl_by_weight) for final_expl_by_weight in final_expl_by_search] for final_expl_by_search in final_expl_levels]

        # if the simulation stopped earlier, add elements to adapt list length
        for i in range(len(types_of_search)):  # iter through searches
            for j in range(len(expl_weights)):  # iter through weights
                for k in range(NUM_OF_SIMULATIONS):  # iter through simulations
                    if len(exploration_levels[i][j][k]) < NUM_OF_ITERATIONS + 1:
                        for _ in range(NUM_OF_ITERATIONS + 1 - len(coverages[i][j])):
                            exploration_levels[i][j][k].append(1)

        # for each type_of_search, get the average coverage of all simulations for specific time
        average_expl_levels = [[[0 for _ in range(NUM_OF_ITERATIONS + 1)]
                                for _ in range(len(expl_weights))]
                               for _ in range(len(types_of_search))]
        for i in range(len(types_of_search)):
            for j in range(len(expl_weights)):
                for k in range(NUM_OF_ITERATIONS + 1):
                    for l in range(NUM_OF_SIMULATIONS):
                        average_expl_levels[i][j][k] += exploration_levels[i][j][l][k]
                    average_expl_levels[i][j][k] /= NUM_OF_SIMULATIONS



        # --------------------------------------------------------------------------------------------------------------
        # storing results
        # --------------------------------------------------------------------------------------------------------------

        # store statistic indices for each type of search, for each exploration weight
        for type_of_search, search_index in types_of_search_dict.items():

            plot_coverage_weight_coverage_comparison([average_coverages[search_index][expl_weights_dict["constant"]]
                                                     , average_coverages[search_index][expl_weights_dict["decrescent"]]]
                                                     , type_of_search)
            plot_exploration_weight_coverage_comparison([average_expl_levels[search_index][expl_weights_dict["constant"]]
                                                     , average_expl_levels[search_index][expl_weights_dict["decrescent"]]]
                                                     , type_of_search)

            for expl_weight, weight_index in expl_weights_dict.items():
                plot_coverage(average_coverages[search_index][weight_index], 0, type_of_search.replace(" search", ""), expl_weight, "average")
                plot_exploration(average_expl_levels[search_index][weight_index], 0, type_of_search.replace(" search", ""), expl_weight, "average")

                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/average_time_elapsed.txt", "w") as f:
                    f.write(str(times_avg[search_index][weight_index]) + "\n")

                # todo: continua a fare questa modifica dell'indice dopo, ma il filone è questo
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/mean_coverage.txt", "w") as f:
                    f.write(str(mean_final_coverages[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/max_coverage.txt", "w") as f:
                    f.write(str(max_final_coverages[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/min_coverage.txt", "w") as f:
                    f.write(str(min_final_coverages[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/std_dev_coverage.txt", "w") as f:
                    f.write(str(std_devs_coverages[search_index][weight_index]) + "\n")

                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/mean_exploration.txt", "w") as f:
                    f.write(str(mean_final_expls[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/max_exploration.txt", "w") as f:
                    f.write(str(max_final_expls[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/min_exploration.txt", "w") as f:
                    f.write(str(min_final_expls[search_index][weight_index]) + "\n")
                with open(f"Simulations output/{type_of_search}/{expl_weight} weight/std_dev_exploration.txt", "w") as f:
                    f.write(str(std_devs_expls[search_index][weight_index]) + "\n")

        plot_coverages_comparison(average_coverages)
        plot_exploration_comparison(average_expl_levels)
        # plot_scatter_regression(final_expl_levels, avg_der)

    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(str(e))
        raise e

if __name__ == '__main__':
    main()
