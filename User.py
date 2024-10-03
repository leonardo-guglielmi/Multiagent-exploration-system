import pickle
import random


class User:
    id = 0  # incremental id attribute, static attribute

    def __init__(self, area, desired_coverage_level, deserialize=False, is_fake=False):
        self.__x = None
        self.__y = None
        if not is_fake:
            self.id = User.id
            User.id += 1

            # if deserialize is True, the position of the user is loaded from serialized file (using pickle module)
            # else assign random position, after that serialize the object and save it in the specified file
            if deserialize:
                self.__x, self.__y = pickle.load(open("User position/user" + str(self.id) + ".p", "rb"))
            else:
                self.__x, self.__y = random.uniform(0, area.length), random.uniform(0, area.width)
                pickle.dump((self.__x, self.__y), open("User position/user" + str(self.id) + ".p", "wb"))

        # defining other attributes
        self.area = area
        self.desired_coverage_level = desired_coverage_level
        self.is_covered = None
        self.coverage_history = []

    def set_is_covered(self, is_covered):
        self.is_covered = is_covered
        self.coverage_history.append(is_covered)  # also updates user's coverage history

    def get_position(self):
        return self.__x, self.__y

    def set_position(self, x, y):
        self.__x = x
        self.__y = y


class Fake_user(User):
    def __init__(self, area, desired_coverage_level, x, y, probability, deserialize=False, is_fake=True):
        User.__init__(self, area, desired_coverage_level, deserialize, is_fake)
        self.set_position(x, y)
        self.probability = probability
