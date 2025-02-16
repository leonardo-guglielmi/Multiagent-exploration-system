def move_agents(self):
    for agent in self.agents:
        # ...
        delta_x = agent.goal_point[0] - agent.get_x() # ...
        delta_y = agent.goal_point[1] - agent.get_y() # ...
        distance =math.dist(agent.goal_point, agent.get_2D_position())

        # if the displacement is too big, it 
        # is limited to MAX_DISPLACEMENT
        if EPSILON * distance < MAX_DISPLACEMENT:
            agent.set_x(agent.get_x() + EPSILON * delta_x)
            agent.set_y(agent.get_y() + EPSILON * delta_y)
        else:
            agent.set_x(
                agent.get_x() + (MAX_DISPLACEMENT * delta_x) 
                / distance
            )
            agent.set_y(
                agent.get_y() + (MAX_DISPLACEMENT * delta_y) 
                / distance
                )
        agent.trajectory.append(agent.get_2D_position())
