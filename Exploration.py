import numpy
from Constants import *
from User import User


# ------------------
# ASSUZIONI:
# - all'inizio non conosco il coverage richiesto dall'utente, perciò mi baso solo sul range degli agenti
# ------------------

# just checks if one region is covered (fixme: un pò bovino passare la cf)
# todo: check del ragionamento
def is_region_cover(region_x, region_y, sensors, cf):
    result = False
    user = User(None, DESIRED_COVERAGE_LEVEL, is_fake=True)
    user.__x = region_x
    user.__y = region_y

    sensors_interference = [0 for _ in sensors]
    for sensor in sensors:
        other_sensors = sensors.remove(sensor)
        sensors_interference[sensor.id] = cf.__interference_power(sensor, user, other_sensors)

    sensors_signal = [0 for _ in sensors]
    if cf.__connection_test():
        for sensor in sensors:
            sensors_signal[sensor.id] = (cf.__channel_gain(sensor, user) * sensor.transmitting_power) / (
                        sensors_interference[sensor.id] + PSDN * BANDWIDTH)
        max_signal = max(sensors_signal)
        if max_signal - user.desired_coverage_level > 0:
            result = True

    return result


# keep memory of how probably there is an uncovered user in that region
class Probability_distribution_matrix:

    def __init__(self, region_width, region_length):
        self.matrix = numpy.zeros((int(AREA_WIDTH/region_width), int(AREA_LENGTH/region_length)))

    def update(self, cf):
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                self.matrix[i, j] = 0 if is_region_cover(i, j, cf.base_stations + cf.agents, cf) \
                    else USER_APPEARANCE_PROBABILITY / self.matrix.size + self.matrix[i, j] * USER_DISCONNECTION_PROBABILITY
