import copy
import math
import numpy
import Exploration
from Constants import *
import scipy
from Sensor import Base_station
from User import *


class Control_function:
    def __init__(self, area, base_stations, agents, users):
        self.area = area
        self.base_stations = base_stations
        self.agents = agents
        self.users = users

        # used for agents connectivity test
        self.__sensors_graph = None
        self.__is_connected_flag = False
        self.__update_sensors_graph()

        self.max_dist_for_coverage = (PATH_GAIN / (DESIRED_COVERAGE_LEVEL * PSDN * BANDWIDTH / TRANSMITTING_POWER)) ** 0.5

        # used for exploration
        self.pd_matrix = Exploration.Probability_distribution_matrix(EXPLORATION_REGION_WIDTH, EXPLORATION_REGION_LENGTH)
        self.pd_matrix.update(self)

    # ---------------------------------
    # Methods for agents connectivity
    # ---------------------------------
    def __update_sensors_graph(self):
        self.__sensors_graph = self.__calculate_graph()
        self.__is_connected_flag = self.__is_connected(self.__sensors_graph)

    def __calculate_graph(self):
        sensors = self.base_stations + self.agents

        graph = numpy.zeros((len(sensors), len(sensors)))
        for i in range(len(sensors)):
            for j in range(len(sensors)):
                # question: non c'é una potenza/radice di troppo
                graph[i][j] = 1 if i != j and (
                        (sensors[i].get_x() - sensors[j].get_x()) ** 2 +
                        (sensors[i].get_y() - sensors[j].get_y()) ** 2 +
                        (sensors[i].get_z() - sensors[j].get_z()) ** 2) ** 0.5 <= COMMUNICATION_RADIUS ** 2 \
                    else 0
        return graph

    # old code: this method was private
    # interface function for __is_connected, now only works on control function's graph
    def connection_test(self):
        return self.__is_connected(self.__calculate_graph())

    @staticmethod
    def __is_connected(graph):
        # uses algebraic connectivity
        laplacian = numpy.zeros((len(graph), len(graph)))
        for i in range(len(graph)):
            for j in range(len(graph)):
                laplacian[i][j] = -graph[i][j] if i != j else sum(graph[i])
        return True if sorted(scipy.linalg.eigvals(laplacian))[1] > 0 else False

    def move_agents(self):
        for agent in self.agents:
            # update the position of the agent. If the displacement is too big, it is limited to MAX_DISPLACEMENT
            delta_x = agent.goal_point[0] - agent.get_x()
            delta_y = agent.goal_point[1] - agent.get_y()
            distance = math.dist(agent.goal_point, agent.get_2D_position())

            if EPSILON * distance < MAX_DISPLACEMENT:
                agent.set_x(agent.get_x() + EPSILON * delta_x)
                agent.set_y(agent.get_y() + EPSILON * delta_y)
            else:
                agent.set_x(agent.get_x() + (MAX_DISPLACEMENT * delta_x) / distance)
                agent.set_y(agent.get_y() + (MAX_DISPLACEMENT * delta_y) / distance)
            agent.trajectory.append(agent.get_2D_position())
            self.__update_sensors_graph()

    # ---------------------------------
    # methods for the signal analysis
    # ---------------------------------

    # old code: this method was private
    @staticmethod
    def channel_gain(current_sensor, current_user):
        # from file:///C:/Users/andrea/OneDrive/Desktop/uni/Tesi/Dynamic_Coverage_Control_of_Multi_Agent_Systems_v1.pdf
        # return the channel gain between user and agent
        # todo: chiedi al prof se va bene modificare questo metodo, mi dava problemi nel caso di allineamento
        if current_sensor.get_3D_position() == current_user.get_position()+(0,):
            return PATH_GAIN
        else:
            return PATH_GAIN / math.pow(math.dist(current_sensor.get_3D_position(), current_user.get_position() + (0,)), 2)

    # old code: this was private
    # returns the total power of interferences that disturbs the sensor signal
    def interference_power(self, sensor, user, other_agents):
        interference_power = 0
        for other_sensor in other_agents + self.base_stations:
            if other_sensor.id != sensor.id:  # this is necessary because other_agents contains also the target sensor when called
                if isinstance(other_sensor, Base_station) and not other_sensor.interference_by_bs:
                    continue
                else:
                    interference_power += self.channel_gain(other_sensor, user) * other_sensor.transmitting_power
        return interference_power

    # returns a matrix that associate at each user the SINR of each agent
    def __SINR(self, interference_powers):
        SINR_matrix = numpy.zeros((len(self.agents)+len(self.base_stations), len(self.users)))

        for sensor in self.agents + self.base_stations:
            for user in self.users:
                SINR_matrix[sensor.id][user.id] = (self.channel_gain(sensor, user) * sensor.transmitting_power) / (
                        interference_powers[sensor.id][user.id] + PSDN * BANDWIDTH)
        return SINR_matrix

    # ---------------------------------
    # methods for the RCR
    # ---------------------------------

    def __RCR(self, SINR_matrix, set_flag=False):
        RCR = 0
        if self.connection_test():
            total_SINR_per_user = [max(col) for col in zip(*SINR_matrix)]
            for user in self.users:
                if total_SINR_per_user[user.id] - user.desired_coverage_level > 0:
                    RCR += 1
                    if set_flag:
                        user.set_is_covered(True)
                else:
                    if set_flag:
                        user.set_is_covered(False)
        return RCR / len(self.users)

    def RCR_after_move(self):
        interference_powers = [[0 for _ in range(len(self.users))] for _ in
                               range(len(self.agents) + len(self.base_stations))]
        for user in self.users:
            for sensor in self.agents + self.base_stations:
                interference_powers[sensor.id][user.id] = self.interference_power(sensor, user, self.agents)

        SINR_matrix = self.__SINR(interference_powers)
        return self.__RCR(SINR_matrix, True)

    # ----------------------------------
    # method that samples new points
    # ----------------------------------
    def get_points(self, agent, other_agents, type_of_search, t):

        points_x = []
        points_y = []

        # Campiona NUM_OF_SAMPLES punti con distribuzione gaussiana centrata in agent.get_position() e
        # varianza area.width/3 e area.length/3, entro i limiti dell'area
        while len(points_x) < NUM_OF_SAMPLES:
            sample_x = scipy.stats.norm.rvs(agent.get_x(), self.area.width / 3)
            # se l'area ha una forma diversa da un rettangolo va
            # gestito diversamente questo if (servirà un metodo area.is_valid_x(x))
            if 0 <= sample_x <= self.area.width:
                points_x.append(sample_x)

        while len(points_y) < NUM_OF_SAMPLES:
            sample_y = scipy.stats.norm.rvs(agent.get_y(), self.area.length / 3)
            # se l'area ha una forma diversa da un rettangolo va gestito diversamente
            # questo if (servirà un metodo area.is_valid_y(y))
            if 0 <= sample_y <= self.area.length:
                points_y.append(sample_y)

        points = list(zip(points_x, points_y))

        if type_of_search == "local" or type_of_search == "mixed":
            # elimina i punti che sono più vicini a un altro agente rispetto all'agente corrente
            # problemi: questa metodologia predilige la ricerca locale, senza possibilità di cercare lontano
            new_points = copy.deepcopy(points)
            for point in points:
                for other_agent in other_agents:
                    if math.dist(point, other_agent.get_2D_position()) < math.dist(point, agent.get_2D_position()):
                        new_points.remove(point)
                        break
            points = new_points

        if type_of_search == "annealing forward" or type_of_search == "annealing reverse":
            # tanto più è la distanza tra il punto campionato e l'agente corrente, tanto più è bassa la probabilità di
            # accettare il punto, oltre al fatto che la prob si riduce con il passare delle iterazioni
            # simulated annealing
            new_points = copy.deepcopy(points)
            for point in points:
                for other_agent in other_agents:
                    delta_distance = math.dist(point, other_agent.get_2D_position()) - math.dist(point,
                                                                                                 agent.get_2D_position())
                    # man mano che t aumenta, la probabilità di rimuovere un punto lontano diminuisce
                    if delta_distance < 0 and random.random() < (
                            t / NUM_OF_ITERATIONS if type_of_search == "annealing forward" else 1 - t / NUM_OF_ITERATIONS):
                        new_points.remove(point)
                        break
            points = new_points
        return points

    # ---------------------------------
    # method that choose between the sampled points in the method above
    # ---------------------------------
    def find_goal_point_for_agent(self, agent, other_agents, type_of_search, t):
        best_point = None
        best_cost_function = -1  # old code: best_coverage = -1

        # store starting interferences power
        partial_interference_powers = [[0 for _ in range(len(self.users))] for _ in
                                       range(len(other_agents) + len(self.base_stations) + 1)]
        for user in self.users:
            for sensor in [agent] + other_agents + self.base_stations:
                partial_interference_powers[sensor.id][user.id] = self.interference_power(sensor, user, other_agents)

        # iters through new sampled point and the actual position (it may don't move)
        i = 0
        for point in [agent.get_2D_position()] + self.get_points(agent, other_agents, type_of_search, t):

            # question: perché faccio la deep copy se tanto poi sovrascrivo tutto?
            temporary_interference_powers = copy.deepcopy(partial_interference_powers)

            # move the agent and store its old position
            original_position = agent.get_2D_position()
            agent.set_2D_position(point[0], point[1])

            # update interferences power with new agent position
            for user in self.users:
                for sensor in other_agents + self.base_stations:
                    temporary_interference_powers[sensor.id][user.id] += agent.transmitting_power * self.channel_gain(agent, user)

            SINR_matrix = self.__SINR(temporary_interference_powers)
            total_coverage_level = self.__RCR(SINR_matrix)

            # skip: questo non ti riguarda
            if type_of_search == "penalty":
                # se il punto è troppo vicino a un punto in cui c'è gia un altro agente -> penalità
                for other_agent in other_agents:
                    if math.dist(point, other_agent.get_2D_position()) < math.dist(point, agent.get_2D_position()):
                        # per ogni agente più vicino al punto campionato, decrementa la copertura totale di 1/len(users)
                        total_coverage_level -= PENALTY
                        break

            # create a simulation environment to calculate exploration level with new agent position
            new_expl_level = self.test_calcolo_parziale(self.pd_matrix, agent)
            i += 1

            agent.set_2D_position(original_position[0], original_position[1])

            # todo: attento qua la coverage è impostata a 0
            cost_function_under_test = 0 * total_coverage_level + EXPLORATION_FACTOR * new_expl_level
            if cost_function_under_test > best_cost_function or (cost_function_under_test == best_cost_function and
                                                                 math.dist(agent.get_2D_position(), point) > math.dist(agent.get_2D_position(), best_point)):
                best_cost_function = cost_function_under_test
                best_point = point

            # old code
            # if total_coverage_level > best_coverage or total_coverage_level == best_coverage and math.dist(
            #        agent.get_2D_position(), point) < math.dist(agent.get_2D_position(), best_point):
            #    best_coverage = total_coverage_level
            #    best_point = point

        return best_point

    # --------------------------
    # Methods for exploration level
    # --------------------------

    @staticmethod
    # per ora può essere un'idea, todo: chiedi al prof se può tornare
    # metodo finale per il calcolo dell' expl_level totale
    def exploration_level(pd_matrix):
        expl = pd_matrix.matrix.size
        for i in range(pd_matrix.matrix.shape[0]):
            for j in range(pd_matrix.matrix.shape[1]):
                expl -= pd_matrix.matrix[i, j]
        return expl/pd_matrix.matrix.size

    # just checks if one region is covered
    def is_region_cover(self, region_x, region_y):
        result = False
        user = User(None, DESIRED_COVERAGE_LEVEL, is_fake=True)
        user.set_position(region_x * EXPLORATION_REGION_WIDTH + EXPLORATION_REGION_WIDTH / 2,
                          region_y * EXPLORATION_REGION_LENGTH + EXPLORATION_REGION_LENGTH / 2)

        sensors_interference = [0 for _ in self.agents + self.base_stations]
        for sensor in self.agents + self.base_stations:
            sensors_interference[sensor.id] = self.interference_power(sensor, user, self.agents + self.base_stations)

        sensors_signal = [0 for _ in self.agents + self.base_stations]
        if self.connection_test():
            for sensor in self.agents + self.base_stations:
                sensors_signal[sensor.id] = (self.channel_gain(sensor, user) * sensor.transmitting_power) / (
                        sensors_interference[sensor.id] + PSDN * BANDWIDTH)

            max_signal = max(sensors_signal)
            if max_signal - user.desired_coverage_level > 0:
                result = True

        return result

    def test_calcolo_parziale(self, pd_matrix, agent):
        inf_x = int((agent.get_x() - agent.communication_radius) / EXPLORATION_REGION_WIDTH)
        if inf_x < 0:
            inf_x = 0

        inf_y = int((agent.get_y() - agent.communication_radius) / EXPLORATION_REGION_LENGTH)
        if inf_y < 0:
            inf_y = 0

        sup_x = int((agent.get_x() + agent.communication_radius) / EXPLORATION_REGION_WIDTH)
        if sup_x >= int(AREA_WIDTH/EXPLORATION_REGION_WIDTH):
            sup_x = int(AREA_WIDTH/EXPLORATION_REGION_WIDTH)-1

        sup_y = int((agent.get_y() + agent.communication_radius) / EXPLORATION_REGION_LENGTH)
        if sup_y >= int(AREA_LENGTH/EXPLORATION_REGION_LENGTH):
            sup_y = int(AREA_LENGTH/EXPLORATION_REGION_LENGTH)-1

        # todo: semplifica ulteriormente andando ad aggiungere l'utente solo se quella cella non è già coperta
        fake_users = [None for _ in range((sup_x - inf_x) * (sup_y - inf_y))]
        k = 0
        for i in range(inf_x, sup_x):
            for j in range(inf_y, sup_y):
                fake_users[k] = Fake_user(None, DESIRED_COVERAGE_LEVEL,
                                          i * EXPLORATION_REGION_WIDTH + EXPLORATION_REGION_WIDTH / 2,
                                          j * EXPLORATION_REGION_LENGTH + EXPLORATION_REGION_LENGTH / 2,
                                          pd_matrix.matrix[i][j])
                fake_users[k].id = k
                k += 1

        interference_powers = [[0 for _ in range(len(fake_users))] for _ in
                               range(len(self.agents) + len(self.base_stations))]
        for user in self.users:
            for sensor in self.agents + self.base_stations:
                interference_powers[sensor.id][user.id] = self.interference_power(sensor, user, self.agents)

        SINR_matrix = numpy.zeros((len(self.agents) + len(self.base_stations), len(fake_users)))
        for sensor in self.agents + self.base_stations:
            for user in fake_users:
                SINR_matrix[sensor.id][user.id] = (self.channel_gain(sensor, user) * sensor.transmitting_power) / (
                        interference_powers[sensor.id][user.id] + PSDN * BANDWIDTH)

        expl_level = 0
        total_SINR_per_user = [max(col) for col in zip(*SINR_matrix)]
        for user in fake_users:
            if total_SINR_per_user[user.id] - user.desired_coverage_level > 0:
                expl_level += user.probability
        return expl_level





