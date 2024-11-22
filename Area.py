class Area:
    """ This class represents the 2D area where agents, base stations and users are displaced (simple data class) """
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def get_area(self):
        return self.width * self.length
