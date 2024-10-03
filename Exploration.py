import numpy
from Constants import *


# ------------------
# ASSUZIONI:
# - all'inizio non conosco il coverage richiesto dall'utente, perciò mi baso solo sul range degli agenti
# ------------------

# keep memory of how probably there is an uncovered user in that region
class Probability_distribution_matrix:

    def __init__(self, region_width, region_length):
        self.matrix = numpy.zeros((int(AREA_WIDTH/region_width), int(AREA_LENGTH/region_length)))

    def update(self, cf):
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                # print("iter "+str(i+j))
                self.matrix[i, j] = 0 if cf.is_region_cover(i, j) \
                    else USER_APPEARANCE_PROBABILITY / self.matrix.size + self.matrix[i, j] * USER_DISCONNECTION_PROBABILITY
