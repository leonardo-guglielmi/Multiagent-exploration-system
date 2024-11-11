import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from Constants import *
import os

user_scatter = []
agent_scatter = []
prob_matrix_scatter = []


# todo: start using fig. instead of plt. for function calling
def plot_area(area, users, base_stations, agents, type_of_search, num_of_iter, prob_matrix_history):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.8)

    users_x, users_y = zip(*[user.get_position() for user in users])
    base_stations_x, base_stations_y = zip(*[base_station.get_2D_position() for base_station in base_stations])

    plt.scatter(base_stations_x, base_stations_y, color='blue')

    plt.xlim(0, area.width)
    plt.ylim(0, area.length)

    users_label = Line2D([0], [0], marker='x', color='red', label='Utenti non coperti', markerfacecolor='red',
                         markersize=6, linestyle='None')
    covered_users_label = Line2D([0], [0], marker='^', color='green', label='Utenti coperti', markerfacecolor='green',
                                 markersize=6, linestyle='None')
    base_stations_label = Line2D([0], [0], marker='o', color='blue', label='Base stations', markerfacecolor='blue',
                                 markersize=6, linestyle='None')
    agents_label = Line2D([0], [0], marker='o', color='black', label='Agenti', markerfacecolor='black',
                          markersize=6, linestyle='None')
    plt.legend(handles=[users_label, covered_users_label, base_stations_label, agents_label], bbox_to_anchor=(0.7, 1.3),
               loc='upper right')

    trajectories = [agent.trajectory for agent in agents]
    lines = [ax.plot([], [], lw=0.7)[0] for _ in
             trajectories]  # [0] allow to work directly with Line2D objects, not with list of lines

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # todo: add statically patches and update them with patch.set()
    def init_prob():
        for line in lines:
            line.set_data([], [])

        matrix = prob_matrix_history[0]
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                ax.add_patch(Rectangle((j * EXPLORATION_REGION_WIDTH, k * EXPLORATION_REGION_HEIGTH),
                                       EXPLORATION_REGION_WIDTH, EXPLORATION_REGION_HEIGTH, color="orange"))
        return lines  # + patches

    def animate(i):
        # this two ifs are used to clear the plot from the precedent frame
        global user_scatter
        for scatter in user_scatter:
            scatter.remove()
            user_scatter.remove(scatter)

        global agent_scatter
        for scatter in agent_scatter:
            scatter.remove()
            agent_scatter.remove(scatter)

        colors = ['green' if user.coverage_history[i] else 'red' for user in users]
        markers = ['^' if user.coverage_history[i] else 'x' for user in users]

        # draw users' markers
        for xu, yu, color2, marker in zip(users_x, users_y, colors, markers):
            user_scatter.append(plt.scatter(xu, yu, color=color2, marker=marker))

        # draw users' trajectory
        for agent, trajectory in zip(agents, trajectories):
            xa, ya = trajectory[i]
            agent_scatter.append(plt.scatter(xa, ya, color='black'))

        # used for the final coverage image
        if i == 0:
            plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/initial coverage.png')

        if i == len(trajectories[0]) - 1:
            for line, trajectory in zip(lines, trajectories):
                x_coord = [coord[0] for coord in trajectory[:i + 1]]
                y_coord = [coord[1] for coord in trajectory[:i + 1]]
                line.set_data(x_coord, y_coord)
            plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/final coverage.png')
        return lines

    def animate_prob(i):
        global user_scatter
        for scatter in user_scatter[:]:
            scatter.remove()
            user_scatter.remove(scatter)

        global agent_scatter
        for scatter in agent_scatter[:]:
            scatter.remove()
            agent_scatter.remove(scatter)

        global prob_matrix_scatter
        for scatter in prob_matrix_scatter[:]:
            scatter.remove()
            prob_matrix_scatter.remove(scatter)

        matrix = prob_matrix_history[i]
        print("prob animation iter " + str(i))
        for j in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                prob_matrix_scatter.append(
                    ax.add_patch(Rectangle((k * EXPLORATION_REGION_WIDTH, j * EXPLORATION_REGION_HEIGTH),
                                           EXPLORATION_REGION_WIDTH, EXPLORATION_REGION_HEIGTH, alpha=matrix[j][k],
                                           color="orange")))

        colors = ['green' if user.coverage_history[i] else 'red' for user in users]
        markers = ['^' if user.coverage_history[i] else 'x' for user in users]
        # draw users' markers
        for xu, yu, color, marker in zip(users_x, users_y, colors, markers):
            user_scatter.append(plt.scatter(xu, yu, color=color, marker=marker))

        # draw users' trajectory
        for agent, trajectory in zip(agents, trajectories):
            xa, ya = trajectory[i]
            print(f"agent position: {xa},{ya}")
            agent_scatter.append(plt.scatter(xa, ya, color='black'))


        # used for the final coverage image
        if i == 0:
            plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/initial coverage prob.png')

        if i == len(trajectories[0]) - 1:
            for line, trajectory in zip(lines, trajectories):
                x_coord = [coord[0] for coord in trajectory[:i + 1]]
                y_coord = [coord[1] for coord in trajectory[:i + 1]]
                line.set_data(x_coord, y_coord)
            plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/final coverage prob.png')
        return lines

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trajectories[0]), interval=200, blit=True)
    ani.save(f'Plots/{type_of_search} search/{num_of_iter}/animation.mp4', writer='ffmpeg')

    ani_prob = animation.FuncAnimation(fig, animate_prob, init_func=init, frames=len(trajectories[0]), interval=200,
                                       blit=True)
    ani_prob.save(f'Plots/{type_of_search} search/{num_of_iter}/animation_prob.mp4', writer='ffmpeg')

    plt.show()


def plot_rewards(coverages, time_elapsed, type_of_search, num_of_iter):
    plt.subplots()
    plt.plot(range(len(coverages)), coverages)
    plt.xlabel('Iterations')
    plt.ylabel('Coverage')
    plt.text(1.1, 1.1, f'Time elapsed: {time_elapsed}', horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes)
    os.makedirs(os.path.dirname(f'Plots/{type_of_search} search/{num_of_iter}/'), exist_ok=True)
    plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/coverage_graphic.png')
    pickle.dump(coverages, open(f'Plots/{type_of_search} search/{num_of_iter}/rewards.p', 'wb'))
    plt.show()


def plot_exploration(exploration_levels, time_elapsed, type_of_search, num_of_iter):
    plt.subplots()
    plt.plot(range(len(exploration_levels)), exploration_levels)
    plt.xlabel('Iters')
    plt.ylabel('Exploration')
    plt.text(1.1, 1.1, f'Time elapsed: {time_elapsed}', horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes)
    os.makedirs(os.path.dirname(f'Plots/{type_of_search} search/{num_of_iter}/'), exist_ok=True)
    plt.savefig(f'Plots/{type_of_search} search/{num_of_iter}/exploration_graphic.png')
    pickle.dump(exploration_levels, open(f'Plots/{type_of_search} search/{num_of_iter}/exploration_level.p', 'wb'))
    plt.show()


def plot_rewards_comparison(rewards):
    plt.subplots()
    for reward in rewards:
        plt.plot(range(len(reward)), reward)
    plt.legend(["Systematic", "Local", "Annealing forward", "Annealing reverse", "Penalty"])
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.savefig(f'Plots/rewards_graphic_comparison.png')
    plt.show()
