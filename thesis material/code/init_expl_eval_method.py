# inside LCIENCC method ...
inf_x, inf_y, sup_x, sup_y = self.__get_local_bounds(agent.get_x()
                                                    , agent.get_y()
                                                    )
cells = []  # this "matrix" will contain both 
            # coordinates and probability of the cell
num_cells = 0
for i in range(inf_x, sup_x):
    cells_column = []
    for j in range(inf_y, sup_y):

        if math.dist(self.get_cell_center(i, j), agent.get_2D_position()) 
                <= agent.communication_radius:
            
            cells_column.append(
                {"pos": self.get_cell_center(i, j) +(0,)
                , "prob": self.__prob_matrix[i, j]}
            )
            num_cells += 1
    
    if len(cells_column) > 0:
        cells.append(cells_column)

relevant_agents = self.__get_relevant_agents(agent)