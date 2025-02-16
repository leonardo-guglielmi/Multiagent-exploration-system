def agent_coupling_detection(self, agent):
    deviation = (0,0)
    if len(agent.trajectory) > DECOUPLING_HISTORY_DEPTH:
        for other_agent in self.agents:
            if other_agent != agent:
                distance_history = [ ]
                for i in range(DECOUPLING_HISTORY_DEPTH):
                    distance_history.append( 
                        math.dist(agent.trajectory[i]
                            , other_agent.trajectory[i]
                            ) )
                if sum(distance_history) <= len(distance_history)*COUPLING_DISTANCE:
                    deviation += ( 
                        ( (agent.trajectory[0])[0] 
                            - (other_agent.trajectory[0])[0] )
                        , ( (agent.trajectory[0])[1] 
                            - (other_agent.trajectory[0])[1] ) 
                    )
    return deviation

def move_agents(self):
    for agent in self.agents:
        coupling_deviation = self.__agent_coupling_detection(agent)
        delta_x = ( agent.goal_point[0] 
                    - agent.get_x() 
                    + coupling_deviation[0] 
                )   
        delta_y = ( agent.goal_point[1] 
                    - agent.get_y() 
                    + coupling_deviation[1] 
                )
    # ...
        