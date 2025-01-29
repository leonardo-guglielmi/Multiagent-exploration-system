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

        # --------------------------------------------------------------------------------------------------------------
        # extracting dto information and load simulation config
        # --------------------------------------------------------------------------------------------------------------
        self.type_of_search = dto.type_of_search
        self.type_of_exploration = dto.type_of_exploration
        self.expl_weight = dto.expl_weight
        self.backhaul_network_available = dto.backhaul_network_available

        # --------------------------------------------------------------------------------------------------------------
        # attributes for network CONNECTIVITY (useful only if backhaul network isn't available)
        # --------------------------------------------------------------------------------------------------------------
        if not self.backhaul_network_available:
            self.__sensors_graph = None
            self.__is_connected_flag = False
            self.__update_sensors_graph()

        self.max_dist_for_coverage = (PATH_GAIN / (
                DESIRED_COVERAGE_LEVEL * PSDN * BANDWIDTH / TRANSMITTING_POWER)) ** 0.5

        # --------------------------------------------------------------------------------------------------------------
        # attributes for EXPLORATION
        # --------------------------------------------------------------------------------------------------------------

        # Matrix that correlates each cell with the likelihood of a user in that area
        self.__prob_matrix = numpy.zeros((int(AREA_WIDTH / EXPLORATION_CELL_WIDTH),
                                            int(AREA_LENGTH / EXPLORATION_CELL_HEIGTH)))

    # ==================================================================================================================
    # Methods for agents connectivity
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
                graph[i][j] = 1 if i != j and (
                        (sensors[i].get_x() - sensors[j].get_x()) ** 2 +
                        (sensors[i].get_y() - sensors[j].get_y()) ** 2 +
                        (sensors[i].get_z() - sensors[j].get_z()) ** 2) <= COMMUNICATION_RADIUS ** 2 \
                    else 0
        return graph

    # Used to test connection between sensors after an agent moves
    def __connection_test(self):
        return self.is_connected(self.__calculate_graph())

    # Return the actual connectivity status, used only for debug purposes
    def get_agents_graph_connection(self):
        return self.__is_connected_flag

    # Tests graph connectivity using laplacian connectivity
    @staticmethod
    def is_connected(graph):
        # uses algebraic connectivity
        laplacian = numpy.zeros((len(graph), len(graph)))
        for i in range(len(graph)):
            for j in range(len(graph)):
                laplacian[i][j] = -graph[i][j] if i != j else sum(graph[i])
        return True if sorted(scipy.linalg.eigvals(laplacian))[1] > 0 else False

    # ==================================================================================================================
    # Methods for agents movement
    # ==================================================================================================================

    # Moves agents in their goal point
    def __move_agents(self):
        for agent in self.agents:
            coupling_deviation = self.__agent_coupling_detection(agent)
            delta_x = agent.goal_point[0] - agent.get_x() + coupling_deviation[0]
            delta_y = agent.goal_point[1] - agent.get_y() + coupling_deviation[1]
            distance = math.dist(agent.goal_point, agent.get_2D_position())

            # if the displacement is too big, it is limited to MAX_DISPLACEMENT
            if EPSILON * distance < MAX_DISPLACEMENT:
                agent.set_x(agent.get_x() + EPSILON * delta_x)
                agent.set_y(agent.get_y() + EPSILON * delta_y)
            else:
                agent.set_x(agent.get_x() + (MAX_DISPLACEMENT * delta_x) / distance)
                agent.set_y(agent.get_y() + (MAX_DISPLACEMENT * delta_y) / distance)
            agent.trajectory.append(agent.get_2D_position())

        if not self.backhaul_network_available:
            self.__update_sensors_graph()


    # Detects if the specified agent is coupled with another agents, and returns the proper deviation to move it away
    def __agent_coupling_detection(self, agent):
        deviation = (0,0)
        if len(agent.trajectory) > DECOUPLING_HISTORY_DEPTH:
            for other_agent in self.agents:
                if other_agent != agent:
                    distance_history = [ ]
                    for i in range(DECOUPLING_HISTORY_DEPTH):
                        distance_history.append( math.dist(agent.trajectory[i], other_agent.trajectory[i]))
                    if sum(distance_history) <= len(distance_history)*COUPLING_DISTANCE:
                        deviation += ( ((agent.trajectory[0])[0] - (other_agent.trajectory[0])[0])
                                     , ((agent.trajectory[0])[1] - (other_agent.trajectory[0])[1]) )
        return deviation

    # ==================================================================================================================
    # Methods for the SIGNAL analysis
    # ==================================================================================================================

    @staticmethod
    # Return the channel gain between agent and user
    def channel_gain(current_sensor, current_user):
        return PATH_GAIN / math.pow(math.dist(current_sensor.get_3D_position(), current_user.get_position() + (0,)), 2)

    @staticmethod
    # Returns the channel gain between two points p1 and p2
    def channel_gain_by_position(p1, p2):
        # adjusting dimension to avoid errors
        if len(p1) <= 2:
            p1 += (0,)
        if len(p2) <= 2:
            p2 += (0,)
        return PATH_GAIN / math.pow(math.dist(p1, p2), 2)

    # Returns the total power of interferences that disturbs the signal between sensor and user
    def __interference_power(self, sensor, user, other_agents):
        interference_power = 0
        for other_sensor in other_agents + self.base_stations:
            if other_sensor.id != sensor.id:  # this is necessary because other_agents may contain also
                                              # the target sensor when called
                if ( isinstance(sensor, Agent)
                    and isinstance(other_sensor,Base_station)
                    and not other_sensor.interference_by_bs
                    ) or (
                    isinstance(sensor, Base_station)
                    and isinstance(other_sensor,Agent)
                    and not sensor.interference_by_bs
                ):
                    continue
                else:
                    interference_power += self.channel_gain(other_sensor, user) * other_sensor.transmitting_power
        return interference_power

    # Return the total power of interferences that disturbs the sensor's signal in some point of space
    def __interference_powers_by_position(self, sensor, point, other_sensors):
        interference_pow = 0
        for other_sensor in other_sensors:
            if other_sensor.id != sensor.id:  # this is necessary because other_agents may contain also
                                              # the target sensor when called
                if (isinstance(sensor, Agent)
                    and isinstance(other_sensor, Base_station)
                    and not other_sensor.interference_by_bs
                ) or (
                        isinstance(sensor, Base_station)
                        and isinstance(other_sensor, Agent)
                        and not sensor.interference_by_bs
                ):
                    continue
                else:
                    interference_pow += ( self.channel_gain_by_position(other_sensor.get_3D_position(), point) *
                                         other_sensor.transmitting_power )
        return interference_pow

    # returns a matrix that associate at each user the SINR of each agent
    def __SINR(self, interference_powers):
        SINR_matrix = numpy.zeros((len(self.agents) + len(self.base_stations), len(self.users)))

        for sensor in self.agents + self.base_stations:
            for user in self.users:
                SINR_matrix[sensor.id][user.id] = (self.channel_gain(sensor, user) * sensor.transmitting_power) / (
                        interference_powers[sensor.id][user.id] + PSDN * BANDWIDTH )
        return SINR_matrix

    # ==================================================================================================================
    # Methods for RCR
    # ==================================================================================================================

    def __RCR_interference(self, SINR_matrix, set_flag=False):
        RCR = 0

        # only if the backhaul network is not available, check for agents' connectivity
        if not self.backhaul_network_available:
            is_graph_connected = self.__connection_test()

        total_SINR_per_user = [max(col) for col in zip(*SINR_matrix)]
        for user in self.users:
            user_covered_flag = False
            if total_SINR_per_user[user.id] - user.desired_coverage_level > 0 \
                    and (self.backhaul_network_available or is_graph_connected): # lazy evaluation
                RCR += 1
                user_covered_flag = True
            if set_flag:
                user.set_is_covered(user_covered_flag)

        return RCR / len(self.users)


    # Returns the RCR value after the agents' movement
    def RCR_after_move(self):
        interference_powers = [ [0 for _ in range(len(self.users))]
                                for _ in range(len(self.agents) + len(self.base_stations))
                              ]
        for user in self.users:
            for sensor in self.agents + self.base_stations:
                interference_powers[sensor.id][user.id] = self.__interference_power(sensor, user, self.agents)

        SINR_matrix = self.__SINR(interference_powers)
        return self.__RCR_interference(SINR_matrix, True)

    # ==================================================================================================================
    # Method that SAMPLES new points
    # ==================================================================================================================
    def __get_points(self, agent, other_agents, t):

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
                    delta_distance = ( math.dist(point, other_agent.get_2D_position())
                                      - math.dist(point,agent.get_2D_position())
                                    )
                    # man mano che t aumenta, la probabilità di rimuovere un punto lontano diminuisce
                    if delta_distance < 0 and random.random() < (
                            t / NUM_OF_ITERATIONS
                            if self.type_of_search == "annealing forward"
                            else 1 - t / NUM_OF_ITERATIONS
                    ):
                        new_points.remove(point)
                        break
            points = new_points
        return points

    # ==================================================================================================================
    # Method that choose between the sampled points of an agent
    # ==================================================================================================================
    def find_goal_point_for_agent(self, agent, other_agents, t, print_expl_eval=False):
        best_point = None
        best_reward = -1

        # store powers of the actual interference
        partial_interference_powers = [ [0 for _ in range(len(self.users))] for _ in
                                        range(len(other_agents) + len(self.base_stations) + 1)
                                    ]
        for user in self.users:
            for sensor in [agent] + other_agents + self.base_stations:
                partial_interference_powers[sensor.id][user.id] = self.__interference_power(sensor, user, other_agents)

        best_expl_evaluation = 0 # used for DEBUG

        # iters through new sampled points and the actual position (it may don't move)
        i = 0
        for point in [agent.get_2D_position()] + self.__get_points(agent, other_agents, t):

            # move the agent and store its old position
            original_position = agent.get_2D_position()
            agent.set_2D_position(point[0], point[1])

            interference_powers_new_position = copy.deepcopy(partial_interference_powers)

            # update interferences power with new agent position
            for user in self.users:
                for sensor in other_agents + self.base_stations:
                    interference_powers_new_position[sensor.id][user.id] += ( agent.transmitting_power *
                                                                            self.channel_gain(agent, user) )

            SINR_matrix = self.__SINR(interference_powers_new_position)
            new_coverage_level = self.__RCR_interference(SINR_matrix)

            new_expl_level = self.__evaluate_new_exploration(agent)

            if self.type_of_search == "penalty":
                # se il punto è troppo vicino a un punto in cui c'è gia un altro agente -> penalità
                for other_agent in other_agents:
                    if math.dist(point, other_agent.get_2D_position()) < math.dist(point, agent.get_2D_position()):
                        # per ogni agente più vicino al punto campionato, decrementa la copertura totale di 1/len(users)
                        new_coverage_level -= PENALTY
                        new_expl_level -= PENALTY
                        break

            i += 1

            reward_under_test = new_coverage_level + self.__exploration_weight() * new_expl_level
            if reward_under_test > best_reward or (reward_under_test == best_reward
                                                   and math.dist(agent.get_2D_position(), point) >
                                                   math.dist(agent.get_2D_position(), best_point)
            ):
                best_reward = reward_under_test
                best_point = point
                best_expl_evaluation = new_expl_level

            agent.set_2D_position(original_position[0], original_position[1])
        if print_expl_eval:
            # output for DEBUG
            print(f"DEBUG: Agent {agent.id} best exploration evaluation {best_expl_evaluation}")

        return best_point

    # ==================================================================================================================
    # Methods for EXPLORATION
    # ==================================================================================================================

    @staticmethod
    # Given indices of probability matrix, returns the coordinates of cell center (created for code clarity)
    def get_cell_center(cell_x, cell_y):
        return (cell_x * EXPLORATION_CELL_WIDTH + EXPLORATION_CELL_WIDTH / 2,
                cell_y * EXPLORATION_CELL_HEIGTH + EXPLORATION_CELL_HEIGTH / 2)

    @staticmethod
    # Get the exploration level of the given matrix
    def exploration_level(prob_matrix):
        expl = prob_matrix.size
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                expl -= prob_matrix[i, j]
        return expl / prob_matrix.size

    # Returns actual global exploration level
    def get_exploration_level(self):
        return self.exploration_level(self.__prob_matrix)

    # Return the given exploration cell is covered
    def __is_cell_covered(self, cell_x, cell_y):
        result = False
        point = self.get_cell_center(cell_x, cell_y) + (0,)

        # if not using backhaul network, checks for network connectivity
        # uses the cf's flag just because __connection_test() should be already called in __RCR(), if
        # it's not the case the connection function must be called here
        if self.backhaul_network_available or self.__is_connected_flag: # lazy evaluation

            # simpler method, an exploration cell is considered explored when it's center is near to some sensor
            if self.type_of_exploration == "simple":
                for agent in self.agents + self.base_stations:
                    if math.dist(agent.get_3D_position(), point) < agent.communication_radius:
                        result = True
                        break

            # these methods take into account interferences from other sensor to decide if a cell is explored or not
            elif self.type_of_exploration == "LSIE" \
                    or self.type_of_exploration == "LSIENCC" \
                    or self.type_of_exploration == "LCIE" \
                    or self.type_of_exploration == "LCIENCC":

                sensors_interference = [0 for _ in self.agents + self.base_stations]
                for sensor in self.agents + self.base_stations:
                    sensors_interference[sensor.id] = self.__interference_powers_by_position(sensor, point
                                                                                             , self.agents
                                                                                             + self.base_stations)
                sensors_SINR = [0 for _ in self.agents + self.base_stations]

                for sensor in self.agents + self.base_stations:
                    sensors_SINR[sensor.id] = (
                                            self.channel_gain_by_position(sensor.get_3D_position(), point)
                                            * sensor.transmitting_power
                                            / (sensors_interference[sensor.id] + PSDN * BANDWIDTH)
                                        )
                    if sensors_SINR[sensor.id] > DESIRED_COVERAGE_LEVEL:
                        result = True
                        break

            else:
                raise Exception("Invalid type_of_exploration")

        return result

    @staticmethod
    # Return the x,y bounds to select local exploration area cells (improves code clarity)
    def __get_local_bounds(x, y):

        inf_x = int((x - EXPLORATION_RADIUS) / EXPLORATION_CELL_WIDTH)
        if inf_x < 0:
            inf_x = 0

        inf_y = int((y - EXPLORATION_RADIUS) / EXPLORATION_CELL_HEIGTH)
        if inf_y < 0:
            inf_y = 0

        sup_x = int((x + EXPLORATION_RADIUS) / EXPLORATION_CELL_WIDTH)
        if sup_x >= int(AREA_WIDTH / EXPLORATION_CELL_WIDTH):
            sup_x = int(AREA_WIDTH / EXPLORATION_CELL_WIDTH) - 1

        sup_y = int((y + EXPLORATION_RADIUS) / EXPLORATION_CELL_HEIGTH)
        if sup_y >= int(AREA_LENGTH / EXPLORATION_CELL_HEIGTH):
            sup_y = int(AREA_LENGTH / EXPLORATION_CELL_HEIGTH) - 1

        return inf_x, inf_y, sup_x, sup_y

    # Returns the list of those sensor whose interference is not negligible
    def __get_relevant_agents(self, agent):
        relevant_agents = []
        for sensor in self.agents + self.base_stations:
            if (sensor != agent
                    and math.dist(sensor.get_2D_position(), agent.get_2D_position()) <= agent.communication_radius * 2
            ):
                relevant_agents.append(sensor)
        relevant_agents.append(agent)
        return relevant_agents

    # Evaluate the exploration gain after agent's movement
    def __evaluate_new_exploration(self, agent):
        exploration_level = 0

        # --------------------------------------------------------------------------------------------------------------
        # examines how agent's movement modifies global exploration matrix
        if self.type_of_exploration == "simple":
            tmp_matrix = copy.deepcopy(self.__prob_matrix)
            self.__update_prob_matrix(tmp_matrix)
            exploration_level = self.exploration_level(tmp_matrix)

        # --------------------------------------------------------------------------------------------------------------
        # only examines local impacts of agent's movement: selects a square of cells centered in agent's position, and
        # evaluate the exploration level only on those cells
        elif self.type_of_exploration == "LSIE": # Local Square Interference Exploration

            inf_x, inf_y, sup_x, sup_y = self.__get_local_bounds(agent.get_x(), agent.get_y())
            cells = []  # this list will contain both coordinates and probability of the cells
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    cells.append({"pos": self.get_cell_center(i, j) + (0,), "prob": self.__prob_matrix[i, j]})

            relevant_agents = self.__get_relevant_agents(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))
            # excluding cells which have probability =0 (e.g. are covered) from exploration
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position( relevant_agents[j]
                                                                                            , cells[k]["pos"]
                                                                                            , relevant_agents )

            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                        for j in range(len(relevant_agents)):
                            SINR_matrix[k][j] = (self.channel_gain_by_position( relevant_agents[j].get_3D_position()
                                                                                , cells[k]["pos"] )
                                                  * relevant_agents[j].transmitting_power
                                                  / (interference_powers[k][j] + PSDN * BANDWIDTH)
                                                 )

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k]["prob"]
            exploration_level /= len(cells)

        # --------------------------------------------------------------------------------------------------------------
        # adds SINR prediction on neighbour cells to the LSIE method
        elif self.type_of_exploration == "LSIENCC": # Local Square Interference Exploration, Neighbour Cell Check

            inf_x, inf_y, sup_x, sup_y = self.__get_local_bounds(agent.get_x(), agent.get_y())
            cells = []  # this list will contain both coordinates and probability of the cells
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    cells.append({"pos": self.get_cell_center(i, j) + (0,), "prob": self.__prob_matrix[i, j]})

            relevant_agents = self.__get_relevant_agents(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))
            # excluding cells which have probability =0 (e.g. are covered) from exploration
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position(relevant_agents[j]
                                                                                           , cells[k]["pos"]
                                                                                           , relevant_agents)

            already_checked_cells = []  # in this list I put those cells that have neighbor with high SINR
            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                    if k in already_checked_cells:
                        SINR_matrix[k][0] = 1
                    else:
                        for j in range(len(relevant_agents)):
                            SINR_matrix[k][j] = (self.channel_gain_by_position( relevant_agents[j].get_3D_position()
                                                                               , cells[k]["pos"] )
                                                 * relevant_agents[j].transmitting_power
                                                 / (interference_powers[k][j] + PSDN * BANDWIDTH)
                                                 )

                            # if I get high SINR, mark also neighbor cells as relevant and exit from cycle
                            if SINR_matrix[k][j] >= NEIGHBOUR_SINR_THRESHOLD:
                                # check and mark upper cell
                                if (k+1) % (sup_y -inf_y) != 0 and cells[k+1] != 0:
                                    already_checked_cells.append(k + 1)
                                # check and mark right cell
                                if (k+1) <= (sup_x -inf_x -1)*(sup_y -inf_y) and cells[k + sup_y - inf_y] != 0:
                                    already_checked_cells.append(k + sup_y - inf_y)
                                break
                # using some reference values for SINR:
                # SINR=0, the cells doesn't contribute to exploration
                # SINR=1 the cells contribute

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k]["prob"]
            exploration_level /= len(cells)

        # --------------------------------------------------------------------------------------------------------------
        # changes the area shape of LSIE, passing from square to circle
        elif self.type_of_exploration == "LCIE":  # Local Circle Interference Exploration

            inf_x, inf_y, sup_x, sup_y = self.__get_local_bounds(agent.get_x(), agent.get_y())
            cells = []  # this list it will contain both coordinates and probability of the cells
            for i in range(inf_x, sup_x):
                for j in range(inf_y, sup_y):
                    if math.dist(self.get_cell_center(i, j), agent.get_2D_position()) <= agent.communication_radius:
                        cells.append({"pos": self.get_cell_center(i, j) + (0,), "prob": self.__prob_matrix[i, j]})

            relevant_agents = self.__get_relevant_agents(agent)

            interference_powers = numpy.zeros((len(cells), len(relevant_agents)))
            # excluding cells which have probability =0 (e.g. are covered) from exploration
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                    for j in range(len(relevant_agents)):
                        interference_powers[k][j] = self.__interference_powers_by_position(relevant_agents[j]
                                                                                           , cells[k]["pos"]
                                                                                           , relevant_agents)

            SINR_matrix = numpy.zeros((len(cells), len(relevant_agents)))
            for k in range(len(cells)):
                if cells[k]["prob"] != 0:
                    for j in range(len(relevant_agents)):
                        SINR_matrix[k][j] = (self.channel_gain_by_position( relevant_agents[j].get_3D_position()
                                                                           , cells[k]["pos"] )
                                             * relevant_agents[j].transmitting_power
                                             / (interference_powers[k][j] + PSDN * BANDWIDTH)
                                             )

            max_SINR_per_cell = [max(cell_SINR) for cell_SINR in SINR_matrix]

            for k in range(len(max_SINR_per_cell)):
                if max_SINR_per_cell[k] > DESIRED_COVERAGE_LEVEL:
                    exploration_level += cells[k]["prob"]
            exploration_level /= len(cells)

        # --------------------------------------------------------------------------------------------------------------
        elif self.type_of_exploration == "LCIENCC":  # Local Circle Interference Exploration, Neighbour Cell Control

            inf_x, inf_y, sup_x, sup_y = self.__get_local_bounds(agent.get_x(), agent.get_y())
            cells = []  # this "matrix" will contain both coordinates and probability of the cell
            num_cells = 0
            for i in range(inf_x, sup_x):
                cells_column = []
                for j in range(inf_y, sup_y):
                    if math.dist(self.get_cell_center(i, j), agent.get_2D_position()) <= agent.communication_radius:
                        cells_column.append({"pos": self.get_cell_center(i, j) +(0,), "prob": self.__prob_matrix[i, j]})
                        num_cells += 1
                if len(cells_column) > 0:
                    cells.append(cells_column)

            relevant_agents = self.__get_relevant_agents(agent)

            interference_powers = [[[] for _ in range(len(cells[i]))] for i in range(len(cells))]
            for i in range(len(cells)):
                for j in range(len(cells[i])):
                    if cells[i][j]["prob"] != 0:
                        for k in range(len(relevant_agents)):
                            interference_powers[i][j] .append( self.__interference_powers_by_position(relevant_agents[k]
                                                                                               , cells[i][j]["pos"]
                                                                                               , relevant_agents) )

            SINR_matrix = [[[] for _ in range(len(cells[i]))] for i in range(len(cells))]
            checked_cells = []
            for i in range(len(cells)):
                for j in range(len(cells[i])):
                    if cells[i][j]["prob"] != 0:
                        if cells[i][j]["pos"] in checked_cells:
                            SINR_matrix.append(1)
                        else:
                            for k in range(len(relevant_agents)):
                                SINR = (self.channel_gain_by_position(relevant_agents[k].get_3D_position()
                                                                                   , cells[i][j]["pos"])
                                                     * relevant_agents[k].transmitting_power
                                                     / (interference_powers[i][j][k] + PSDN * BANDWIDTH)
                                        )
                                SINR_matrix[i][j].append(SINR)

                                if SINR >= NEIGHBOUR_SINR_THRESHOLD:
                                    # check and mark upper cell
                                    if j + 1 < len(cells[i]) \
                                            and cells[i][j + 1]["prob"] != 0:  # check for upper cell
                                        checked_cells.append(cells[i][j+1]["pos"])

                                    # first check to mark right cell
                                    if i + 1 < len(cells):
                                        # search if the right cell it's inside local area
                                        for cell in cells[i+1]:
                                            if cell["pos"] == self.__sum_triple( cells[i][j]["pos"],
                                                                                (EXPLORATION_CELL_WIDTH, 0, 0)
                                                ):
                                                if cell["prob"] != 0:
                                                    checked_cells.append(cell["pos"])
                                                break # exit from right cell search
                                    break # exit from agents cycle

            max_SINR_per_cell = [ [ max(SINR_matrix[i][j]) if len(SINR_matrix[i][j]) != 0
                                  else 0 for j in range(len(cells[i]))
                                  ]
                                for i in range(len(cells))
                                ]

            for i in range(len(max_SINR_per_cell)):
                for j in range(len(max_SINR_per_cell[i])):
                    if max_SINR_per_cell[i][j] > DESIRED_COVERAGE_LEVEL:
                        exploration_level += cells[i][j]["prob"]
            exploration_level /= num_cells

        else:
            raise Exception("Invalid type_of_exploration")

        return exploration_level

    # Returns the exploration weight in objective function
    def __exploration_weight(self):
        # constant weight
        if self.expl_weight == "constant":
            return EXPLORATION_WEIGHT

        # weight that decreases as the number of covered users increases
        elif self.expl_weight == "decrescent":
            num_user_covered = 0
            for user in self.users:
                if user.is_covered:
                    num_user_covered += 1
            return 1 if num_user_covered <= 1 else 2/num_user_covered

        else:
            raise Exception("Invalid expl_weight")

    def __update_prob_matrix(self, matrix, init=False):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if self.__is_cell_covered(i, j):
                    matrix[i, j] = 0
                elif init:
                    matrix[i, j] = INIT_PROBABLITY
                else:
                    matrix[i, j] =(1 - matrix[i, j]) * USER_APPEARANCE_PROBABILITY \
                            + matrix[i, j] * (1 - USER_DISCONNECTION_PROBABILITY)

        for user in self.users:
            if len(user.coverage_history) >= 2 \
                    and not user.coverage_history[-1] and user.coverage_history[-2]:
                user_x, user_y = user.get_position()
                cell_x = int(user_x / EXPLORATION_CELL_WIDTH)
                cell_y = int(user_y / EXPLORATION_CELL_HEIGTH)
                matrix[cell_x][cell_y] = 1

    # Used to automatically update the prob_matrix from outside
    def update_probability_distribution_matrix(self):
        self.__update_prob_matrix(self.__prob_matrix)

    # Returns a snapshot of prob_matrix
    def get_prob_matrix_snapshot(self):
        return copy.deepcopy(self.__prob_matrix)

    @staticmethod
    def __sum_triple(t1, t2):
        return t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2]

