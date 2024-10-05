import numpy
from Constants import *


# keep memory of how probably there is an uncovered user in that region
class Probability_distribution_matrix:

    def __init__(self, region_width, region_length, users_list):
        self.matrix = numpy.zeros((int(AREA_WIDTH/region_width), int(AREA_LENGTH/region_length)))
        self.users_list = users_list

    def update(self, cf):
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                self.matrix[i, j] = 0 if cf.is_cell_covered(i, j) \
                    else USER_APPEARANCE_PROBABILITY / self.matrix.size + self.matrix[i, j] * USER_DISCONNECTION_PROBABILITY

        for old_user, new_user in zip(self.users_list, cf.total_users):
            if old_user.is_covered and not new_user.is_covered:
                user_x, user_y = new_user.get_position()
                cell_x = int(user_x / EXPLORATION_REGION_WIDTH)
                cell_y = int(user_y / EXPLORATION_REGION_HEIGTH)
                self.matrix[cell_x][cell_y] = 1
        self.users_list = cf.users
