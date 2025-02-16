# code that samples points
# ...
if self.type_of_search == "annealing forward" \
    or self.type_of_search == "annealing reverse":

    new_points = copy.deepcopy(points)
    for point in points:
        for other_agent in other_agents:
            delta_distance = (
                math.dist(point, other_agent.get_2D_position())  
                - math.dist(point,agent.get_2D_position())
            )

            if delta_distance < 0 and random.random() < (
                    t / NUM_OF_ITERATIONS 
                    if self.type_of_search == "annealing forward" 
                    else 1 - t / NUM_OF_ITERATIONS
                ):
                new_points.remove(point)
                break
    points = new_points
# ...