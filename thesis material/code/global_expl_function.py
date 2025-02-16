@staticmethod
# used to elaborate global exploration level
def exploration_level(prob_matrix):
    expl = prob_matrix.size
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            expl -= prob_matrix[i, j]
    return expl / prob_matrix.size        