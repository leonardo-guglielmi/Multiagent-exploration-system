from Constants import *
import pickle
import statistics

from Plots import plot_coverage, plot_coverages_comparison, plot_exploration_comparison

custom_probs = [True, False]

# extract general data
coverages = {True: [], False:[]}
exploration_levels = {True: [], False:[]}
#times = {True: [], False:[]}

for prob in custom_probs:
    for j in range(NUM_OF_SIMULATIONS):
        coverages[prob].append(pickle.load(open(f"Experiment results/experiment3/custom prob {prob}/{j}/coverages.p","rb")))
        #times[prob].append(pickle.load(open(f"Experiment results/experiment3/custom prob {prob}/{j}/time_elapsed.p", "rb")))
        exploration_levels[prob].append(pickle.load(open(f"Experiment results/experiment3/custom prob {prob}/{j}/exploration_levels.p", "rb")))

average_coverages = {True: [0 for _ in range(NUM_OF_ITERATIONS + 1)], False: [0 for _ in range(NUM_OF_ITERATIONS + 1)]}
average_explorations = {True: [0 for _ in range(NUM_OF_ITERATIONS + 1)], False: [0 for _ in range(NUM_OF_ITERATIONS + 1)]}
for prob in custom_probs:
    for k in range(NUM_OF_ITERATIONS + 1):
        for j in range(NUM_OF_SIMULATIONS):
            average_coverages[prob][k] += coverages[prob][j][k]
            average_explorations[prob][k] += exploration_levels[prob][j][k]
        average_coverages[prob][k] /= NUM_OF_SIMULATIONS
        average_explorations[prob][k] /= NUM_OF_SIMULATIONS

plot_coverages_comparison(average_coverages.values(), ['high variability', 'low variability'] ,path="Experiment results/experiment3/")
plot_exploration_comparison(average_explorations.values(), ['high variability', 'low variability'] ,path="Experiment results/experiment3/")