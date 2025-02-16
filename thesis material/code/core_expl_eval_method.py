# .. iter throught cells ...
if cells[i][j]["prob"] != 0:
    
    # if this cell's coverage is predicted, mark it as covered
    # and skip it
    if cells[i][j]["pos"] in checked_cells:
        SINR_matrix.append(1)
    else:
        for k in range(len(relevant_agents)):
            SINR = (
                self.channel_gain_by_position(
                    relevant_agents[k].get_3D_position()
                    , cells[i][j]["pos"]
                )
                * relevant_agents[k].transmitting_power
                / (interference_powers[i][j][k] + PSDN * BANDWIDTH)
            )
            SINR_matrix[i][j].append(SINR)

            if SINR >= NEIGHBOUR_SINR_THRESHOLD:
                # check and mark upper cell
                if j + 1 < len(cells[i]) \
                        and cells[i][j + 1]["prob"] != 0:
                    checked_cells.append(cells[i][j+1]["pos"])

                # check if there exist the right column
                if i + 1 < len(cells):
                    # search if that cell it's inside the
                    # right row
                    for cell in cells[i+1]:
                        if cell["pos"] == \
                                self.__sum_triple( 
                                    cells[i][j]["pos"] ,
                                    (EXPLORATION_CELL_WIDTH, 0, 0)
                            ):
                            # if that cell esists, check if it's 
                            # not already covered, and mark it 
                            if cell["prob"] != 0:
                                checked_cells.append(cell["pos"])
                            break # exit from right cell search
                break # exit from agents cycle
