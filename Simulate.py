import os
import pickle

from Plots import plot_area, plot_coverage, plot_exploration
from Control_function import Control_function
from Sensor import Agent, Base_station
from Area import Area
from User import User
from Constants import *
from timeit import default_timer as timer
from Control_function_config_DTO import Control_function_DTO as DTO

from AgentProcess import AgentProcess
from multiprocessing import SimpleQueue

def simulate(type_of_search, expl_weight, num_of_iter, deserialize):
    # -----------------------------------------------
    #     1° step: simulation's environment creation
    # -----------------------------------------------

    area = Area(AREA_WIDTH, AREA_LENGTH)

    agents = [
        Agent(area, COMMUNICATION_RADIUS, TRANSMITTING_POWER, ALTITUDE + i * SENSOR_HEIGHT + MIN_VERTICAL_DISTANCE,
              deserialize) for i in range(N)]

    # by default a base station does not interfere with the communication of the other agents
    b1 = Base_station(area, COMMUNICATION_RADIUS, 1 / 4 * area.width, 1 / 4 * area.length, TRANSMITTING_POWER)
    b2 = Base_station(area, COMMUNICATION_RADIUS, 1 / 4 * area.width, 3 / 4 * area.length, TRANSMITTING_POWER)
    b3 = Base_station(area, COMMUNICATION_RADIUS, 3 / 4 * area.width, 1 / 4 * area.length, TRANSMITTING_POWER)
    b4 = Base_station(area, COMMUNICATION_RADIUS, 3 / 4 * area.width, 3 / 4 * area.length, TRANSMITTING_POWER)
    base_stations = [b1, b2, b3, b4]

    users = [User(area, DESIRED_COVERAGE_LEVEL, deserialize) for _ in range(M)]

    # -----------------------------------------------
    #     2° step: working variables setup
    # -----------------------------------------------

    # coverage history
    coverage_levels = []
    # exploration level history
    exploration_levels = []
    # probability distribution matrix history
    prob_matrix_history = []

    dto = DTO(type_of_search=type_of_search,
              type_of_exploration="PCINCC",
              expl_weight=expl_weight,
              is_concurrent=True,
              backhaul_network_available = True)
    cf = Control_function(area, base_stations, agents, users, dto)

    # starting points for coverage & exploration levels
    current_reward = cf.RCR_after_move()
    coverage_levels.append(current_reward)

    cf.update_probability_distribution_matrix()
    current_expl = cf.get_exploration_level()
    exploration_levels.append(current_expl)
    prob_matrix_history.append(cf.get_prob_matrix_snapshot())

    print("Start coverage level: ", current_reward)
    with open("output_log.txt", 'a') as f:
        f.write(f"Start coverage level: {current_reward}\n")
    print("Start exploration level: ", current_expl)
    with open("output_log.txt", 'a') as f:
        f.write(f"Start exploration level: {current_expl}\n")

    # -----------------------------------------------
    #     3° step: simulation start
    # -----------------------------------------------

    start = timer()

    # control function continue to iterate until all users are cover or reach the limit of iterations (NUM_OF_ITERATIONS)
    t = 0  # iter counter (it's the time variable in the mathematical model)
    while current_reward != 1.0 and t < NUM_OF_ITERATIONS:

        if dto.is_concurrent:
            proc_list = []
            queue = SimpleQueue()

            for ag in agents:
                proc_list.append(AgentProcess(cf, ag, t, queue))
            for proc in proc_list:
                proc.start()
            for proc in proc_list:
                proc.join()
                proc.close()

            output_list = [(0,0) for _ in range(N+B)]
            while not queue.empty():
                ag = queue.get()
                output_list[ag.id] = ag.goal_point
            queue.close()
            for agent in agents:
                agent.goal_point = output_list[agent.id]

        else:
            for agent in agents:
                other_agents = [a for a in agents if a.id != agent.id]
                agent.goal_point = cf.find_goal_point_for_agent(agent, other_agents, t)

        # every t the agents are moved in the direction of the goal point calculated by the control function
        # and the exploration matrix is updated
        cf.move_agents()

        # at the end RCR and exploration level are updated, each user's is_covered flag is assigned
        current_reward = cf.RCR_after_move()
        coverage_levels.append(current_reward)

        cf.update_probability_distribution_matrix()
        prob_matrix_history.append(cf.get_prob_matrix_snapshot())

        current_expl = cf.get_exploration_level()
        exploration_levels.append(current_expl)

        t += 1
        if type_of_search == "mixed" and t == int(NUM_OF_ITERATIONS / 2):
            type_of_search = "systematic mixed"

        print(type_of_search, "iteration: ", t, " coverage level: ", current_reward, " exploration_level: ",
              current_expl)
        with open("output_log.txt", 'a') as f:
            f.write(f"{type_of_search} iteration: {t} | coverage level: {current_reward} | exploration level: {current_expl}\n")

        # UNCOMMENT THIS FOR DEBUG
        # print(f"is sensor graph connected? {cf.get_agents_graph_connection()}")

    end = timer()

    time_elapsed = end - start
    # final CLI output
    print("Time elapsed: ", time_elapsed)
    print("Final coverage level: ", current_reward)
    print("Final exploration level: ", current_expl)
    if type_of_search == "systematic mixed":
        type_of_search = "mixed"

    # saving results with pickle files
    print("Saving simulation data...")
    os.makedirs(os.path.normpath(f'Simulations output/{type_of_search} search/{expl_weight} weight/{num_of_iter}'), exist_ok=True)
    # noinspection PyTypeChecker
    pickle.dump(time_elapsed, open(f"Simulations output/{type_of_search} search/{expl_weight} weight/{num_of_iter}/time_elapsed.p", "wb"))
    # noinspection PyTypeChecker
    pickle.dump(coverage_levels, open(f'Simulations output/{type_of_search} search/{expl_weight} weight/{num_of_iter}/coverages.p', 'wb'))
    # noinspection PyTypeChecker
    pickle.dump(exploration_levels, open(f'Simulations output/{type_of_search} search/{expl_weight} weight/{num_of_iter}/exploration_levels.p', 'wb'))

    # plotting results
    print("Plotting results...")
    plot_area(area, users, base_stations, agents, type_of_search, num_of_iter, prob_matrix_history, expl_weight)
    plot_coverage(coverage_levels, time_elapsed, type_of_search, expl_weight, num_of_iter)
    plot_exploration(exploration_levels, time_elapsed, type_of_search, expl_weight, num_of_iter)
