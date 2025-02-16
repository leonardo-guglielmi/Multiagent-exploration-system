max_SINR_per_cell = [ 
        [ max(SINR_matrix[i][j]) if len(SINR_matrix[i][j]) != 0
            else 0 

        for j in range(len(cells[i])) 
    ]
    for i in range(len(cells))
]

for i in range(len(max_SINR_per_cell)):
    for j in range(len(max_SINR_per_cell[i])):
        if max_SINR_per_cell[i][j] > DESIRED_COVERAGE_LEVEL:
            exploration_level += cells[i][j]["prob"]
exploration_level /= num_cells