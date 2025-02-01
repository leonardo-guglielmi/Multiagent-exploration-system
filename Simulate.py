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

from multiprocessing import Process
from multiprocessing import Manager

def simulate(type_of_search, expl_weight, num_of_iter, deserialize, use_expl=True):
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
              type_of_exploration="LCIENCC",
              expl_weight=expl_weight,
              is_concurrent=True,
              backhaul_network_available = True
              , use_expl=True)
    cf = Control_function(area, base_stations, agents, users, dto)

    # starting points for coverage & exploration levels
    current_reward = cf.RCR_after_move()
    coverage_levels.append(current_reward)

    if use_expl:
        cf.update_probability_distribution_matrix(init=True)
        current_expl = cf.get_exploration_level()
        exploration_levels.append(current_expl)
        prob_matrix_history.append(cf.get_prob_matrix_snapshot())

    print("Start coverage level: ", current_reward)
    with open("logs/output_log.txt", 'a') as f:
        f.write(f"Start coverage level: {current_reward}\n")

    if use_expl:
        print("Start exploration level: ", current_expl)
        with open("logs/output_log.txt", 'a') as f:
            f.write(f"Start exploration level: {current_expl}\n")

    # -----------------------------------------------
    #     3° step: simulation start
    # -----------------------------------------------

    start = timer()

    # control function continue to iterate until all users are cover or reach the limit of iterations (NUM_OF_ITERATIONS)
    t = 0  # iter counter (it's the time variable in the mathematical model)
    while current_reward != 1.0 and t < NUM_OF_ITERATIONS:

        if dto.is_concurrent:
            with Manager() as manager:
                shared_dict = manager.dict()
                processes = [Process( target=concurrent_find_goal_point
                                      , args=(cf, ag, t, shared_dict, ))
                                    for ag in agents ]
                for p in processes:
                    p.start()
                for p in processes:
                    p.join()
                    p.close()
                for ag in agents:
                    ag.goal_point = shared_dict[ag.id]
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

        if use_expl:
            cf.update_probability_distribution_matrix()
            prob_matrix_history.append(cf.get_prob_matrix_snapshot())

            current_expl = cf.get_exploration_level()
            exploration_levels.append(current_expl)

        t += 1
        if type_of_search == "mixed" and t == int(NUM_OF_ITERATIONS / 2):
            type_of_search = "systematic mixed"

        print(type_of_search, "iteration: ", t, " coverage level: ", current_reward, f" exploration_level: {current_expl}" if use_expl else "")
        with open("logs/output_log.txt", 'a') as f:
            f.write(f"{type_of_search} iteration: {t} | coverage level: {current_reward} | " + f"exploration_level: {current_expl}" if use_expl else "")

        # UNCOMMENT THIS FOR DEBUG
        # print(f"is sensor graph connected? {cf.get_agents_graph_connection()}")

    end = timer()

    time_elapsed = end - start
    # final CLI output
    print("Time elapsed: ", time_elapsed)
    print("Final coverage level: ", current_reward)
    if use_expl:
        print("Final exploration level: ", current_expl)
    if type_of_search == "systematic mixed":
        type_of_search = "mixed"

    # saving results with pickle files
    print("Saving simulation data...")
    os.makedirs(os.path.normpath(f'Simulations output/{type_of_search} search/{expl_weight} weight/expl {use_expl}/{num_of_iter}'), exist_ok=True)
    # noinspection PyTypeChecker
    pickle.dump(time_elapsed, open(f"Simulations output/{type_of_search} search/{expl_weight} weight/expl {use_expl}/{num_of_iter}/time_elapsed.p", "wb"))
    # noinspection PyTypeChecker
    pickle.dump(coverage_levels, open(f'Simulations output/{type_of_search} search/{expl_weight} weight/expl {use_expl}/{num_of_iter}/coverages.p', 'wb'))
    if use_expl:
        # noinspection PyTypeChecker
        pickle.dump(exploration_levels, open(f'Simulations output/{type_of_search} search/{expl_weight} weight/expl {use_expl}/{num_of_iter}/exploration_levels.p', 'wb'))

    # plotting results
    print("Plotting results...")
    plot_area(area, users, base_stations, agents, type_of_search, num_of_iter, prob_matrix_history, expl_weight, use_expl=use_expl)
    plot_coverage(coverage_levels, time_elapsed, type_of_search, expl_weight, num_of_iter, use_expl=use_expl)
    if use_expl:
        plot_exploration(exploration_levels, time_elapsed, type_of_search, expl_weight, num_of_iter)


def concurrent_find_goal_point(cf, agent, t, output_dict):
    other_agents = []
    for ag in cf.agents:
        if ag.id != agent.id:
            other_agents.append(ag)
    output_dict[agent.id] = cf.find_goal_point_for_agent(agent, other_agents, t, print_expl_eval=False)