class Control_function_DTO:
    def __init__(self, type_of_search, type_of_exploration, expl_weight, is_concurrent, backhaul_network_available, use_expl, use_custom_prob):
        self.type_of_search = type_of_search
        self.type_of_exploration = type_of_exploration
        self.expl_weight = expl_weight # Now useless
        self.is_concurrent = is_concurrent
        self.backhaul_network_available = backhaul_network_available
        self.use_expl = use_expl
        self.use_custom_prob = use_custom_prob