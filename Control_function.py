import copy
import math
import numpy
from Constants import *
import scipy
from Sensor import Base_station, Agent
from User import *


class Control_function:
    def __init__(self, area, base_stations, agents, users, dto):
        # initialization of all actors in the simulation
        self.area = area
        self.base_stations = base_stations
        self.agents = agents
        self.users = users

        # declaration of variables used to check connectivity between agents
        self.__sensors_graph = None
        self.__is_connected_flag = False
        self.__update_sensors_graph()

        self.max_dist_for_coverage = (PATH_GAIN / (
                DESIRED_COVERAGE_LEVEL * PSDN * BANDWIDTH / TRANSMITTING_POWER)) ** 0.5

        # ----
        # attributes used for exploration
        # ----

        # extracting dto information
        self.type_of_search = dto.type_of_search
        self.type_of_coverage = dto.type_of_coverage
        self.type_of_exploration = dto.type_of_exploration
        self.expl_weight = dto.expl_weight
        self.concurrent_mode = dto.is_concurrent

        # Matrix that correlates each cell with the likelihood of a user in that area
        self.__prob_matrix = numpy.zeros( (int(AREA_WIDTH / EXPLORATION_REGION_WIDTH),
                                            int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH)) )

        # List of bools, used just to see if a user pass from "covered" to "uncovered" and modify correctly the probability in that cell
        self.__user_coverage_list = []

    # ==================================================================================================================
    # Methods for agent connectivity
    # ==================================================================================================================

    # Used just to update sensor_graph and is_connected_flag all in once
    def __update_sensors_graph(self):
        self.__sensors_graph = self.__calculate_graph()
        self.__is_connected_flag = self.is_connected(self.__sensors_graph)

    # Generate the connection graph between sensors
    def __calculate_graph(self):
        sensors = self.base_stations + self.agents

        graph = numpy.zeros((len(sensors), len(sensors)))
        for i in range(len(sensors)):
            for j in range(len(sensors)):
                #
                # question: non c'é una potenza/radice di troppo?
                graph[i][j] = 1 if i != j and (
                        (sensors[i].get_x() - sensors[j].get_x()) ** 2 +
                        (sensors[i].get_y() - sensors[j].get_y()) ** 2 +
                        (sensors[i].get_z() - sensors[j].get_z()) ** 2) ** 0.5 <= COMMUNICATION_RADIUS ** 2 \
                    else 0
        return graph

    # Used to test connection between sensors after an agent moves
    def __connection_test(self):
        return self.is_connected(self.__calculate_graph())

    # Tests graph connectivity using laplacian connectivity
    @staticmethod
    def is_connected(graph):
        # uses algebraic connectivity
        laplacian = numpy.zeros((len(graph), len(graph)))
        for i in range(len(graph)):
            for j in range(len(graph)):
                laplacian[i][j] = -graph[i][j] if i != j else sum(graph[i])
        return True if sorted(scipy.linalg.eigvals(laplacian))[1] > 0 else False

    # Moves agents in their goal point
    def move_agents(self):
        for agent in self.agents:
            delta_x = agent.goal_point[0] - agent.get_x()
            delta_y = agent.goal_point[1] - agent.get_y()
            distance = math.dist(agent.goal_point, agent.get_2D_position())

            # if the displacement is too big, it is limited to MAX_DISPLACEMENT
            if EPSILON * distance < MAX_DISPLACEMENT:
                agent.set_x(agent.get_x() + EPSILON * delta_x)
                agent.set_y(agent.get_y() + EPSILON * delta_y)
            else:
                agent.set_x(agent.get_x() + (MAX_DISPLACEMENT * delta_x) / distance)
                agent.set_y(agent.get_y() + (MAX_DISPLACEMENT * delta_y) / distance)
            agent.trajectory.append(agent.get_2D_position())
            self.__update_sensors_graph()

    # ==================================================================================================================
    # methods for the signal analysis
    # ==================================================================================================================

    @staticmethod
    # return the channel gain between agent and user
    def channel_gain(current_sensor, current_user):
        # from file:///C:/Users/andrea/OneDrive/Desktop/uni/Tesi/Dynamic_Coverage_Control_of_Multi_Agent_Systems_v1.pdf
        return PATH_GAIN / math.pow(math.dist(current_sensor.get_3D_position(), current_user.get_position() + (0,)), 2)

    @staticmethod
    # returns the channel gain between two point p1 and p2
    def channel_gain_by_position(p1, p2):
        # adjusting dimension to avoid errors
        if len(p1) <= 2:
            p1 += (0,)
        if len(p2) <= 2:
            p2 += (0,)
        return PATH_GAIN / math.pow(math.dist(p1, p2), 2)

    # returns the total power of interferences that disturbs the signal between sensor and user
    def __interference_power(self, sensor, user, other_agents):
        interference_power = 0
        for other_sensor in other_agents + self.base_stations:
            if other_sensor.id != sensor.id:  # this is necessary because other_agents may contain also the target sensor when called
                if isinstance(other_sensor, Base_station) and not other_sensor.interference_by_bs:
                    continue
                else:
                    interference_power += self.channel_gain(other_sensor, user) * other_sensor.transmitting_power
        return interference_power

    # return the total power of interferences that disturbs the sensor's signal in some point of space
    def __interference_powers_by_position(self, sensor, point, other_sensors):
        interference_pow = 0
        for other_sensor in other_sensors:
            if other_sensor.id != sensor.id:  # this is necessary because other_agents may contain also the target sensor when called
                if  (isinstance(sensor, Agent) and isinstance(other_sensor, Base_station) and not other_sensor.interference_by_bs) \
                        or (isinstance(sensor, Base_station) and isinstance(other_sensor, Agent) and not sensor.interference_by_bs):
                    continue
                else:
                    interference_pow += self.channel_gain_by_position(other_sensor.get_3D_position(), point) * other_sensor.transmitting_power
        return interference_pow

    # returns a matrix that associate at each user the SINR of each agent
    def __SINR(self, interference_powers):
        SINR_matrix = numpy.zeros((len(self.agents) + len(self.base_stations), len(self.users)))

        for sensor in self.agents + self.base_stations:
            for user in self.users:
                SINR_matrix[sensor.id][user.id] = (self.channel_gain(sensor, user) * sensor.transmitting_power) / (
                        interference_powers[sensor.id][user.id] + PSDN * BANDWIDTH)
        return SINR_matrix

    # ==================================================================================================================
    # methods for the RCR
    # ==================================================================================================================

    def __RCR_interference(self, SINR_matrix, set_flag=False):
        RCR = 0
        if self.__connection_test():
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

    def __RCR_no_interference(self, set_flag=False):
        RCR = 0
        if self.__connection_test():
            for user in self.users:
                user_covered_flag = False
                for sensor in self.agents + self.base_stations:
                    if math.dist(sensor.get_3D_position(), user.get_3D_position()) < sensor.communication_radius:
                        RCR += 1
                        user_covered_flag = True
                        break
                if set_flag:
                    user.set_is_covered(user_covered_flag)
        return RCR / len(self.users)

    # returns the RCR after an agent moves
    def RCR_after_move(self):

        if self.type_of_coverage == "simple":
            return self.__RCR_no_interference(True)

        elif self.type_of_coverage == "interference":
            interference_powers = [[0 for _ in range(len(self.users))] for _ in
                                   range(len(self.agents) + len(self.base_stations))]
            for user in self.users:
                for sensor in self.agents + self.base_stations:
                    interference_powers[sensor.id][user.id] = self.__interference_power(sensor, user, self.agents)

            SINR_matrix = self.__SINR(interference_powers)
            return self.__RCR_interference(SINR_matrix, True)

        else:
            raise Exception("Invalid type_of_coverage")

    # ==================================================================================================================
    # method that samples new points
    # ==================================================================================================================
    def get_points(self, agent, other_agents, t):

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

        if self.type_of_search == "local" or self.type_of_search == "mixed":
            # elimina i punti che sono più vicini a un altro agente rispetto all'agente corrente
            # problemi: questa metodologia predilige la ricerca locale, senza possibilità di cercare lontano
            new_points = copy.deepcopy(points)
            for point in points:
                for other_agent in other_agents:
                    if math.dist(point, other_agent.get_2D_position()) < math.dist(point, agent.get_2D_position()):
                        new_points.remove(point)
                        break
            points = new_points

        if self.type_of_search == "annealing forward" or self.type_of_search == "annealing reverse":
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
                            t / NUM_OF_ITERATIONS if self.type_of_search == "annealing forward" else 1 - t / NUM_OF_ITERATIONS):
                        new_points.remove(point)
                        break
            points = new_points
        return points

    # ==================================================================================================================
    # method that choose between the sampled points of an agent
    # ==================================================================================================================

    # this function detects if the specified agent is coupled to another one
    def agent_coupling_detection(self, agent):
        result = False
        if len(agent.trajectory) > 2:
            for other_agent in self.agents:
                if (other_agent != agent
                        and math.dist(agent.get_2D_position(), other_agent.get_2D_position()) < 2*MAX_DISPLACEMENT):
                    result = True
                    break
        return result

    def find_goal_point_for_agent(self, agent, other_agents, t, print_expl_eval=False):
        best_point = None
        best_reward = -1

        if self.type_of_coverage == "interference":
            # store powers of the actual interference
            partial_interference_powers = [[0 for _ in range(len(self.users))] for _ in
                                       range(len(other_agents) + len(self.base_stations) + 1)]
            for user in self.users:
                for sensor in [agent] + other_agents + self.base_stations:
                    partial_interference_powers[sensor.id][user.id] = self.__interference_power(sensor, user, other_agents)

        best_expl_evaluation = 0 # used for debug
        # iters through new sampled points and the actual position (it may don't move)
        i = 0
        for point in [agent.get_2D_position()] + self.get_points(agent, other_agents, t):

            # move the agent and store its old position
            original_position = agent.get_2D_position()
            agent.set_2D_position(point[0], point[1])

            if self.type_of_coverage == "interference":
                temporary_interference_powers = copy.deepcopy(partial_interference_powers)

                # update interferences power with new agent position
                for user in self.users:
                    for sensor in other_agents + self.base_stations:
                        # question: non dovrei prima rimuoverlo? inoltre non dovrei ricalcolarmi le interferenze sul segnale di agent?
                        temporary_interference_powers[sensor.id][user.id] += agent.transmitting_power * self.channel_gain(
                            agent, user)

                SINR_matrix = self.__SINR(temporary_interference_powers)
                new_coverage_level = self.__RCR_interference(SINR_matrix)

            elif self.type_of_coverage == "simple":
                new_coverage_level = self.__RCR_no_interference()

            else:
                raise Exception("Invalid type_of_coverage")

            new_expl_level = self.__evaluate_new_exploration(agent)

            if self.type_of_search == "penalty":
                # se il punto è troppo vicino a un punto in cui c'è gia un altro agente -> penalità
                for other_agent in other_agents:
                    if math.dist(point, other_agent.get_2D_position()) < math.dist(point, agent.get_2D_position()):
                        # per ogni agente più vicino al punto campionato, decrementa la copertura totale di 1/len(users)
                        new_coverage_level -= PENALTY
                        break

            i += 1

            reward_under_test = new_coverage_level + self.exploration_weight(self.expl_weight) * new_expl_level
            if self.agent_coupling_detection(agent):
                reward_under_test *= AGENTS_COUPLING_PENALTY

            if reward_under_test > best_reward or (reward_under_test == best_reward and
                                                   math.dist(agent.get_2D_position(), point) > math.dist(
                                                    agent.get_2D_position(), best_point)):
                best_reward = reward_under_test
                best_point = point
                best_expl_evaluation = new_expl_level

            agent.set_2D_position(original_position[0], original_position[1])
        if print_expl_eval:
            print(f"DEBUG: Agent {agent.id} best exploration evaluation {best_expl_evaluation}") # uncomment this for DEBUG
        return best_point

    # ==================================================================================================================
    # Methods for exploration
    # ==================================================================================================================

    @staticmethod
    # given indices of probability matrix, returns the coordinates of cell center (created for code clarity)
    def get_cell_center(cell_x, cell_y):
        return (cell_x * EXPLORATION_REGION_WIDTH + EXPLORATION_REGION_WIDTH / 2,
                cell_y * EXPLORATION_REGION_HEIGTH + EXPLORATION_REGION_HEIGTH / 2)

    @staticmethod
    # used to elaborate global exploration level
    def exploration_level(prob_matrix):
        expl = prob_matrix.size
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                expl -= prob_matrix[i, j]
        return expl / prob_matrix.size

    # used to call this function on the cf's prob matrix from outside
    def get_exploration_level(self):
        return self.exploration_level(self.__prob_matrix)

    # tests if one cell for exploration is covered
    def __is_cell_covered(self, cell_x, cell_y):
        result = False
        point = self.get_cell_center(cell_x, cell_y) + (0,)

        # using flag because __connection_test() already called in __RCR(), if it's not the case the connection test must
        # be called here
        if self.__is_connected_flag:

            # simpler method, an exploration cell is considered explored when it's center is near to some sensor
            if self.type_of_exploration == "simple":
                for agent in self.agents + self.base_stations:
                    if math.dist(agent.get_3D_position(), point) < agent.communication_radius:
                        result = True
                        break

            # this method takes into account interferences from other sensor to decide if a cell is explored or not
            elif self.type_of_exploration == "PSI" \
                    or self.type_of_exploration == "PSINCC" \
                    or self.type_of_exploration == "PCI" \
                    or self.type_of_exploration == "PCINCC":

                sensors_interference = [0 for _ in self.agents + self.base_stations]
                for sensor in self.agents + self.base_stations:
                    sensors_interference[sensor.id] = self.__interference_powers_by_position(sensor, point,
                                                                                             self.agents + self.base_stations)
                sensors_SINR = [0 for _ in self.agents + self.base_stations]

                # TODO se ho una BS come sensore, non ho interferenze in generale giusto? quindi in questo caso dovrei ignorare anche le interferenze degli agenti giusto?
                for sensor in self.agents + self.base_stations:
                    sensors_SINR[sensor.id] = (self.channel_gain_by_position(sensor.get_3D_position(),
                                                                             point) * sensor.transmitting_power) / (
                                                      sensors_interference[sensor.id] + PSDN * BANDWIDTH)
                    if sensors_SINR[sensor.id] > DESIRED_COVERAGE_LEVEL:
                        result = True
                        break

            else:
                raise Exception("Invalid type_of_exploration")

        return result

    # used to elaborate exploration gain after agent's movement
    def __evaluate_new_exploration(self, agent):
        exploration_level = 0

        # examines how agent's movement modifies probability
        if self.type_of_exploration == "simple":
            tmp_matrix = copy.deepcopy(self.__prob_matrix)
            self.__update_prob_matrix(tmp_matrix)
            exploration_level = self.exploration_level(tmp_matrix)

        # only examines local impacts of agent's movement: selects a square of cells centered in agent's position, and
        # uses only those cells to evaluate exploration gain
        elif self.type_of_exploration == "PSI": # Proximity Square Interference

            # control to not exceed area limits
            inf_x = int((agent.get_x() - agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if inf_x < 0:
                inf_x = 0
            inf_y = int((agent.get_y() - agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if inf_y < 0:
                inf_y = 0
            sup_x = int((agent.get_x() + agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if sup_x >= int(AREA_WIDTH / EXPLORATION_REGION_WIDTH):
                sup_x = int(AREA_WIDTH / EXPLORATION_REGION_WIDTH) - 1
            sup_y = int((agent.get_y() + agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if sup_y >= int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH):
                sup_y = int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH) - 1

            cells = []  # this list it will contain both coordinates and probability of a cell
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    cells.append((self.get_cell_center(i, j) + (0,), self.__prob_matrix[i, j]))

            # select only those agents that are sufficiently close to the agent I'm watching
            relevant_agents = []
            for sensor in self.agents + self.base_stations:
                if (sensor != agent
                        and math.dist(sensor.get_2D_position(),
                                      agent.get_2D_position()) <= agent.communication_radius + sensor.communication_radius):
                    relevant_agents.append(sensor)
            relevant_agents.append(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))

            # excluding cells which have probability =0 (eg are covered) from exploration
            for k in range(len(cells)):
                if cells[k][1] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position(relevant_agents[j],
                                                                                           cells[k][0], relevant_agents)

            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k][1] != 0:
                        for j in range(len(relevant_agents)):
                            SINR_matrix[k][j] = ((self.channel_gain_by_position(relevant_agents[j].get_3D_position(),
                                                                                cells[k][0]) * relevant_agents[
                                                      j].transmitting_power) /
                                                 (interference_powers[k][j] + PSDN * BANDWIDTH))

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k][1]
            exploration_level /= len(cells)

        # only examines local impacts of agent's movement: selects a square of cells centered in agent's position, and
        # uses only those cells to evaluate exploration gain
        elif self.type_of_exploration == "PSINCC": # Proximity Square Interference, Neighbour Cell Check

            # control to not exceed area limits
            inf_x = int((agent.get_x() - agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if inf_x < 0:
                inf_x = 0
            inf_y = int((agent.get_y() - agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if inf_y < 0:
                inf_y = 0
            sup_x = int((agent.get_x() + agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if sup_x >= int(AREA_WIDTH / EXPLORATION_REGION_WIDTH):
                sup_x = int(AREA_WIDTH / EXPLORATION_REGION_WIDTH) - 1
            sup_y = int((agent.get_y() + agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if sup_y >= int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH):
                sup_y = int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH) - 1

            cells = []  # this list it will contain both coordinates and probability of a cell
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    cells.append((self.get_cell_center(i, j) + (0,), self.__prob_matrix[i, j]))

            # select only those agents that are sufficiently close to the agent I'm watching
            relevant_agents = []
            for sensor in self.agents + self.base_stations:
                if (sensor != agent
                        and math.dist(sensor.get_2D_position(), agent.get_2D_position()) <= agent.communication_radius + sensor.communication_radius):
                    relevant_agents.append(sensor)
            relevant_agents.append(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))

            # excluding cells which have probability =0 (eg are covered) from exploration
            for k in range(len(cells)):
                if cells[k][1] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position(relevant_agents[j], cells[k][0], relevant_agents)

            already_checked_cells = []  # in this list I put those cells that have neighbor with high SINR
            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k][1] != 0:
                    if k not in already_checked_cells:
                        for j in range(len(relevant_agents)):
                            SINR_matrix[k][j] = ((self.channel_gain_by_position(relevant_agents[j].get_3D_position(), cells[k][0]) * relevant_agents[j].transmitting_power) /
                                                         (interference_powers[k][j] + PSDN * BANDWIDTH))

                            # if I get high SINR, mark also neighbor cells as relevant and exit from cycle
                            if SINR_matrix[k][j] >= 0.85:
                                if (k+1) % (sup_y -inf_y) != 0 and cells[k+1] != 0:
                                    SINR_matrix[k+1][j] = 1
                                    already_checked_cells.append(k + 1)
                                if (k+1) <= (sup_x -inf_x -1)*(sup_y -inf_y) and cells[k + sup_y - inf_y] != 0:
                                    SINR_matrix[k + sup_y - inf_y][j] = 1
                                    already_checked_cells.append(k + sup_y - inf_y)
                                break
                # using some values to be safe: SINR=0, the cells doesn't contribute to exploration, SINR=1 the cells contribute

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k][1]
            exploration_level /= len(cells)

        # only examines local impacts of agent's movement: selects a square of cells centered in agent's position, and
        # uses only those cells to evaluate exploration gain
        elif self.type_of_exploration == "PCI":  # Proximity Circle Interference

            # control to not exceed area limits
            inf_x = int((agent.get_x() - agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if inf_x < 0:
                inf_x = 0
            inf_y = int((agent.get_y() - agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if inf_y < 0:
                inf_y = 0
            sup_x = int((agent.get_x() + agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if sup_x >= int(AREA_WIDTH / EXPLORATION_REGION_WIDTH):
                sup_x = int(AREA_WIDTH / EXPLORATION_REGION_WIDTH) - 1
            sup_y = int((agent.get_y() + agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if sup_y >= int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH):
                sup_y = int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH) - 1

            cells = []  # this list it will contain both coordinates and probability of a cell
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    if math.dist(self.get_cell_center(i, j), agent.get_2D_position()) <= agent.communication_radius:
                        cells.append((self.get_cell_center(i, j) + (0,), self.__prob_matrix[i, j]))

            # select only those agents that are sufficiently close to the agent I'm watching
            relevant_agents = []
            for sensor in self.agents + self.base_stations:
                if (sensor != agent
                        and math.dist(sensor.get_2D_position(),
                                      agent.get_2D_position()) <= agent.communication_radius + sensor.communication_radius):
                    relevant_agents.append(sensor)
            relevant_agents.append(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))

            # excluding cells which have probability =0 (eg are covered) from exploration
            for k in range(len(cells)):
                if cells[k][1] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position(relevant_agents[j],
                                                                                           cells[k][0], relevant_agents)

            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k][1] != 0:
                    for j in range(len(relevant_agents)):
                        SINR_matrix[k][j] = ((self.channel_gain_by_position(relevant_agents[j].get_3D_position(),
                                                                            cells[k][0]) * relevant_agents[
                                                  j].transmitting_power) /
                                             (interference_powers[k][j] + PSDN * BANDWIDTH))

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k][1]
            exploration_level /= len(cells)

        elif self.type_of_exploration == "PCINCC":

            # control to not exceed area limits
            inf_x = int((agent.get_x() - agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if inf_x < 0:
                inf_x = 0
            inf_y = int((agent.get_y() - agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if inf_y < 0:
                inf_y = 0
            sup_x = int((agent.get_x() + agent.communication_radius) / EXPLORATION_REGION_WIDTH)
            if sup_x >= int(AREA_WIDTH / EXPLORATION_REGION_WIDTH):
                sup_x = int(AREA_WIDTH / EXPLORATION_REGION_WIDTH) - 1
            sup_y = int((agent.get_y() + agent.communication_radius) / EXPLORATION_REGION_HEIGTH)
            if sup_y >= int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH):
                sup_y = int(AREA_LENGTH / EXPLORATION_REGION_HEIGTH) - 1

            cells = []  # this list it will contain both coordinates and probability of a cell
            num_cells = 0
            for i in range(inf_x, sup_x):
                cells_column = []
                for j in range(inf_y, sup_y):
                    if math.dist(self.get_cell_center(i, j), agent.get_2D_position()) <= agent.communication_radius:
                        cells_column.append({"pos": self.get_cell_center(i, j) + (0,), "prob": self.__prob_matrix[i, j]})
                        num_cells += 1
                if len(cells_column) > 0:
                    cells.append(cells_column)

            # select only those agents that are sufficiently close to the agent I'm watching
            relevant_agents = []
            for sensor in self.agents + self.base_stations:
                if sensor != agent and math.dist(sensor.get_2D_position(), agent.get_2D_position()) <= agent.communication_radius + sensor.communication_radius:
                    relevant_agents.append(sensor)
            relevant_agents.append(agent)

            interference_powers = []
            # excluding cells which have probability =0 (eg are covered) from exploration
            for i in range(len(cells)):
                interference_powers_column = []
                for j in range(len(cells[i])):
                    interference_powers_for_cell = []
                    if cells[i][j]["prob"] != 0:
                        for k in range(len(relevant_agents)):
                            interference_powers_for_cell.append(self.__interference_powers_by_position(relevant_agents[k],cells[i][j]["pos"], relevant_agents))
                    interference_powers_column.append(interference_powers_for_cell)
                interference_powers.append(interference_powers_column)

            SINR_matrix = []

            checked_cells = []
            for i in range(len(cells)):
                SINR_matrix_column = []
                for j in range(len(cells[i])):
                    SINR_for_cell = []
                    if cells[i][j]["pos"] in checked_cells:
                        SINR_for_cell.append(1)
                    else:
                        if cells[i][j]["prob"] != 0:
                            for k in range(len(relevant_agents)):
                                SINR_tmp = (self.channel_gain_by_position(relevant_agents[k].get_3D_position(), cells[i][j]["pos"]) * relevant_agents[k].transmitting_power /
                                                 (interference_powers[i][j][k] + PSDN * BANDWIDTH))
                                SINR_for_cell.append(SINR_tmp)

                                # if I get high SINR, mark also neighbor cells as relevant and exit from cycle
                                if SINR_tmp >= 0.85:
                                    if j+1 < len(cells[i]) and cells[i][j+1]["prob"] != 0: # check for upper cell
                                        checked_cells.append(cells[i][j]["pos"])

                                    if i+1 < len(cells) and j < len(cells[i+1]) and cells[i+1][j]["prob"] != 0 and cells[i][j]["pos"] == tuple(map(sum, zip(cells[i+1][j]["pos"],(EXPLORATION_REGION_WIDTH, 0)))): # check for lateral
                                        checked_cells.append(cells[i][j]["pos"])
                                    break
                    SINR_matrix_column.append(SINR_for_cell)
                SINR_matrix.append(SINR_matrix_column)

            max_SINR_per_cell = [[max(SINR_matrix[i][j]) if len(SINR_matrix[i][j]) != 0 else 0 for j in range(len(cells[i]))] for i in range(len(cells))]
            for i in range(len(max_SINR_per_cell)):
                for j in range(len(max_SINR_per_cell[i])):
                    if max_SINR_per_cell[i][j] > DESIRED_COVERAGE_LEVEL:
                        exploration_level += cells[i][j]["prob"]
            exploration_level /= num_cells

        else:
            raise Exception("Invalid type_of_exploration")

        return exploration_level

    # return the weight of exploration in cost function
    def exploration_weight(self, type_of_weight):
        # constant weight
        if type_of_weight == "constant":
            return EXPLORATION_WEIGHT

        # weight that decrease based on the number of covered users
        elif type_of_weight == "decrescent":
            num_user_covered = 0
            for cov in self.__user_coverage_list:
                if cov:
                    num_user_covered += 1
            return 1 if num_user_covered <= 1 else 2/num_user_covered

        else:
            raise Exception("Invalid expl_weight")

    def __update_prob_matrix(self, matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i, j] = 0 if self.__is_cell_covered(i, j) \
                    else (1 - matrix[i, j]) * USER_APPEARANCE_PROBABILITY + matrix[i, j] * (
                        1 - USER_DISCONNECTION_PROBABILITY)

        # old_user_cov is bool, new_user is an object (otherwise it's necessary to do a deepcopy of users list, not so efficient)
        for old_user_cov, actual_user in zip(self.__user_coverage_list, self.users):
            if old_user_cov and not actual_user.is_covered:
                user_x, user_y = actual_user.get_position()
                cell_x = int(user_x / EXPLORATION_REGION_WIDTH)
                cell_y = int(user_y / EXPLORATION_REGION_HEIGTH)
                matrix[cell_x][cell_y] = 1

    # used to automatically update the prob_matrix from outside
    def update_probability_distribution_matrix(self):
        self.__update_prob_matrix(self.__prob_matrix)
        self.__user_coverage_list = self.get_user_coverage_list()

    # returns a snapshot of prob_matrix
    def get_prob_matrix_snapshot(self):
        return copy.deepcopy(self.__prob_matrix)

    def get_user_coverage_list(self):
        return [user.is_covered for user in self.users]
