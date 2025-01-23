from multiprocessing import Process

class AgentProcess(Process):

    def __init__(self, cf, agent, t, output_queue):
        super().__init__()
        self.cf = cf
        self.agent = agent
        self.other_agents = []
        for ag in self.cf.agents:
            if ag.id != agent.id:
                self.other_agents.append(ag)
        self.t = t
        self.point = self.agent.get_2D_position()
        self.output_queue = output_queue

    def run(self):
        self.agent.goal_point = self.cf.find_goal_point_for_agent(self.agent, self.other_agents, self.t, print_expl_eval=False)
        self.output_queue.put(self.agent)