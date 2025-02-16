def exploration_weight(self):
    # constant weight
    if self.expl_weight == "constant":
        return EXPLORATION_WEIGHT

    # Weight that decreases as the number of covered users increases
    elif self.expl_weight == "decrescent":
        num_user_covered = 0
        for user in self.users:
            if user.is_covered:
                num_user_covered += 1
        return 1 if num_user_covered <= 1 else 2/num_user_covered