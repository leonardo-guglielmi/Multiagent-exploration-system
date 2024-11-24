import pickle
import statistics

from Plots import plot_coverage, plot_coverages_comparison
from Simulate import simulate
from User import User
from Constants import *
from Sensor import Sensor


def main():
    try:
        num_of_simulations = 30
        types_of_search = ["systematic", "local", "annealing forward", "annealing reverse", "penalty"]

        for i in range(num_of_simulations):
            deserialize = False
            for type_of_search in types_of_search:
                print("----- Starting simulation ", type_of_search, ":", i, " -----")
                simulate(type_of_search, i, deserialize)
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
        for type_of_search, type_value in types_of_search_dict.items():
            for j in range(num_of_simulations):
                # 3D matrix, for each type of search, for each simulation load coverage history of that simulation
                coverages[type_value].append(pickle.load(open(f"Plots/{type_of_search}/{j}/rewards.p", "rb")))
                times[type_value].append(pickle.load(open(f"Plots/{type_of_search}/{j}/time_elapsed.p", "rb")))

        # ------
        # calculation of some statistic indices
        # ------
        # statistics about time
        times_avg = [statistics.mean(time) for time in times]

        # statistics about coverage
        final_coverages = [[0 for _ in range(len(coverages[0]))] for _ in range(len(coverages))]  # just to transpose the matrix
        for i in range(len(coverages)):
            for j in range(len(coverages[i])):
                final_coverages[i][j] = coverages[i][j][-1]  # get only the final reward at the end of the simulation

        # this statistics are for each type of search
        mean_final_coverage = [statistics.mean(final_reward) for final_reward in final_coverages]
        max_final_coverage = [max(final_reward) for final_reward in final_coverages]
        min_final_coverage = [min(final_reward) for final_reward in final_coverages]
        std_devs = [statistics.stdev(final_reward) for final_reward in final_coverages]

        # if the simulation stopped earlier, add elements to adapt list length
        for i in range(len(types_of_search)):
            for j in range(num_of_simulations):
                if len(coverages[i][j]) < NUM_OF_ITERATIONS + 1:
                    for k in range(NUM_OF_ITERATIONS + 1 - len(coverages[i][j])):
                        coverages[i][j].append(1.0)

        # for each type_of_search, get the average coverage of all simulations for specific time
        average_coverages = [[0 for _ in range(NUM_OF_ITERATIONS + 1)] for _ in range(len(types_of_search))]
        for i in range(len(types_of_search)):
            for k in range(NUM_OF_ITERATIONS + 1):
                for j in range(num_of_simulations):
                    average_coverages[i][k] += coverages[i][j][k]
                average_coverages[i][k] /= num_of_simulations

        for type_of_search in types_of_search_dict.keys():
            plot_coverage(average_coverages[types_of_search_dict[type_of_search]], 0,
                          type_of_search.replace(" search", ""), "average")

            with open(f"Plots/{type_of_search}/mean_reward.txt", "w") as f:
                f.write(str(mean_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
            with open(f"Plots/{type_of_search}/max_reward.txt", "w") as f:
                f.write(str(max_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
            with open(f"Plots/{type_of_search}/min_reward.txt", "w") as f:
                f.write(str(min_final_coverage[types_of_search_dict[type_of_search]]) + "\n")
            with open(f"Plots/{type_of_search}/std_dev.txt", "w") as f:
                f.write(str(std_devs[types_of_search_dict[type_of_search]]) + "\n")
            with open(f"Plots/{type_of_search}/average_time_elapsed.txt", "w") as f:
                f.write(str(times_avg[types_of_search_dict[type_of_search]]) + "\n")

        plot_coverages_comparison(average_coverages)

    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(str(e))
        raise e


if __name__ == '__main__':
    main()
