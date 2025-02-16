# code that samples points
# ...
if self.type_of_search == "local":
    new_points = copy.deepcopy(points)
    for point in points:
        for other_agent in other_agents:
            if ( math.dist(point, other_agent.get_2D_position()) < 
                math.dist(point, agent.get_2D_position())
            ):
                new_points.remove(point)
                break
    points = new_points
# ...