AREA_WIDTH = 1000  # in meters
AREA_LENGTH = 1000  # in meters

# constants for the simulation taken from ///C:/Users/andrea/OneDrive/Desktop/uni/Tesi/Deep_Reinforcement_Learning-Based_Effective_Coverage_Control_With_Connectivity_Constraints%20(1)%20(1).pdf
# and from file:///C:/Users/andrea/OneDrive/Desktop/uni/Tesi/Dynamic_Coverage_Control_of_Multi_Agent_Systems_v1.pdf
NUM_OF_SAMPLES = 250  # number of points each agent generates as potentially new positions (default: 250, test: 25)
EPSILON = 0.1  # percentage of how the agent moves in the chosen direction
COMMUNICATION_RADIUS = 200  # of the agent (default: 200)
DESIRED_COVERAGE_LEVEL = 0.5  # by the user
MAX_DISPLACEMENT = 10  # max distance an agent can move from its actual position
NUM_OF_ITERATIONS = 100  # max num of iterations before the algorithm stops (default: 100, test: 30)
MIN_VERTICAL_DISTANCE = 0.15  # in meters
SENSOR_HEIGHT = 0.15  # in meters

M = 30  # number of users
N = 10  # number of agents
B = 4  # number of base stations
PENALTY = 1/M  # const for penalty search

""" Power Spectral Density Noise """
PSDN = 7.164E-16  # =-174dBm/Hz
BANDWIDTH = 2000000  # in Hz

""" PATH_GAIN = lambda^2/(4*pi)^2, where lambda = c/f is the wavelength of the signal."""
PATH_GAIN = 0.0001

"""Altitude of the sensors"""
ALTITUDE = 50  # in meters

"""Transmit Power"""
TRANSMITTING_POWER = 0.2  # in Watts

# constants for exploration
EXPLORATION_FACTOR = 0.4  # weight of exploration in total cost-function (rho in th mathematical model)
USER_DISCONNECTION_PROBABILITY = 0.01  # (Pd in the model)
USER_APPEARANCE_PROBABILITY = 0.1  # (Pb in the model)
EXPLORATION_REGION_WIDTH = 20  # in meters (default: 20, test: 50)
EXPLORATION_REGION_HEIGTH = 20  # in meters (default: 20, test: 50)
