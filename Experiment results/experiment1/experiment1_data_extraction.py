import pickle
from matplotlib import pyplot as plt
import statistics
from Constants import NUM_OF_SIMULATIONS, NUM_OF_ITERATIONS

coverages = {True:{True: [], False: []}, False:{True: [], False: []}}
explorations = {True: [], False: []}

# extract all datas
for expl in [True, False]:
    for bs in [True, False]:
        for i in range(NUM_OF_SIMULATIONS):
            coverages[expl][bs].append(pickle.load(open(f"Experiment results/experiment1/expl {expl}/BS {bs}/{i}/coverages.p", 'rb')))
            if expl:
                explorations[bs].append(pickle.load(open(f"Experiment results/experiment1/expl {expl}/BS {bs}/{i}/exploration_levels.p", 'rb')))

# get final statistics
final_cov = {True: {True: [0 for _ in range(NUM_OF_SIMULATIONS)], False: [0 for _ in range(NUM_OF_SIMULATIONS)]}
            , False: {True: [0 for _ in range(NUM_OF_SIMULATIONS)], False: [0 for _ in range(NUM_OF_SIMULATIONS)]}}
final_expl = {True: [0 for _ in range(NUM_OF_SIMULATIONS)], False: [0 for _ in range(NUM_OF_SIMULATIONS)]}
for expl in [True, False]:
    for bs in [True, False]:
        for i in range(NUM_OF_SIMULATIONS):
            final_cov[expl][bs][i] = coverages[expl][bs][i][-1]
            if expl:
                final_expl[bs][i] = explorations[bs][i][-1]
for expl in [True, False]:
    for bs in [True, False]:
        with (open(f"Experiment results/experiment1/expl {expl}/BS {bs}/final_coverage_statistics.txt", 'w') as f):
            output = f"max: {max(final_cov[expl][bs])} " + \
                     f"min: {min(final_cov[expl][bs])} "  + \
                     f"avg: {statistics.mean(final_cov[expl][bs])} " + \
                     f"std_dev: {statistics.stdev(final_cov[expl][bs])} "
            f.write(output)
        if expl:
            with (open(f"Experiment results/experiment1/expl {expl}/BS {bs}/final_exploration_statistics.txt", 'w') as f):
                output = f"max: {max(final_expl[bs])} " + \
                         f"min: {min(final_expl[bs])} " + \
                         f"avg: {statistics.mean(final_expl[bs])} " + \
                         f"std_dev: {statistics.stdev(final_expl[bs])}"
                f.write(output)

# get overhaul statistics
avg_cov = {True: {True: [0 for _ in range(NUM_OF_ITERATIONS)], False: [0 for _ in range(NUM_OF_ITERATIONS)]}
            , False: {True: [0 for _ in range(NUM_OF_ITERATIONS)], False: [0 for _ in range(NUM_OF_ITERATIONS)]}}
avg_expl = {True: [0 for _ in range(NUM_OF_ITERATIONS)]
            , False: [0 for _ in range(NUM_OF_ITERATIONS)]}

min_starting_coverages = []
for expl in [True, False]:
    for bs in [True, False]:
        for j in range(NUM_OF_ITERATIONS):
            avg_cov_iter = 0
            avg_expl_iter = 0

            for i in range(NUM_OF_SIMULATIONS):
                avg_cov_iter += coverages[expl][bs][i][j]
                if expl:
                    avg_expl_iter += explorations[bs][i][j]

            avg_cov_iter /= NUM_OF_SIMULATIONS
            avg_cov[expl][bs][j] = avg_cov_iter
            if j == 0:
                min_starting_coverages.append(avg_cov_iter)
            if expl:
                avg_expl_iter /= NUM_OF_SIMULATIONS
                avg_expl[bs][j] = avg_expl_iter
min_starting_cov = min(min_starting_coverages)

# plot coverage comparison
fig, ax = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Coverage")
ax[0].set_ylim(min_starting_cov,1)
ax[0].plot(range(len(avg_cov[True][True])), avg_cov[True][True], label="with exploration")
ax[0].plot(range(len(avg_cov[False][True])), avg_cov[False][True], label="without exploration")
ax[0].legend(loc='lower right')
ax[0].set_title(f"Coverage comparison with Base Stations")

ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Coverage")
ax[1].set_ylim(min_starting_cov,1)
ax[1].plot(range(len(avg_cov[True][False])), avg_cov[True][False], label="with exploration")
ax[1].plot(range(len(avg_cov[False][False])), avg_cov[False][False], label="without exploration")
ax[1].legend(loc='lower right')
ax[1].set_title(f"Coverage comparison without Base Stations")

fig.savefig("Experiment results/experiment1/coverage_comparison.png", bbox_inches='tight')

# plot exploration comparison
fig, ax = plt.subplots()
ax.set_xlabel("Iteration")
ax.set_ylabel("Exploration level")
ax.plot(avg_expl[True], label="with BS")
ax.plot(avg_expl[False], label="without BS")
ax.legend(loc='lower right')
ax.set_title(f"Exploration comparison")

fig.savefig("Experiment results/experiment1/exploration_comparison.png", bbox_inches='tight')