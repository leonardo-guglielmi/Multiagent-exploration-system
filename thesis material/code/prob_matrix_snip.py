# probability distribution matrix definition
self.prob_matrix = numpy.zeros((int(AREA_WIDTH / EXPLORATION_CELL_WIDTH),
                                int(AREA_LENGTH / EXPLORATION_CELL_HEIGTH))
                            )

# ...

def update_prob_matrix(self, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if self.__is_cell_covered(i, j):
                matrix[i, j] = 0
            
            # ... initialization case

            else:
                matrix[i, j] =(1 - matrix[i, j]) * USER_APPEARANCE_PROBABILITY \
                        + matrix[i, j] * (1 - USER_DISCONNECTION_PROBABILITY)

        for user in self.users:
            if len(user.coverage_history) >= 2 \
                    and not user.coverage_history[-1] \
                    and user.coverage_history[-2]:
                user_x, user_y = user.get_position()
                cell_x = int(user_x / EXPLORATION_CELL_WIDTH)
                cell_y = int(user_y / EXPLORATION_CELL_HEIGTH)
                matrix[cell_x][cell_y] = 1