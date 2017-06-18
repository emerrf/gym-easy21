import gym
from gym import spaces
import numpy as np


class CardHandler:

    def __init__(self):
        self.hand = list()
        self.hand_sum = 0
        self._is_stick = False

    def reset(self):
        self.__init__()

    def hit(self, first=False):
        card_num = np.random.random_integers(1, 10)

        if first:
            card_colour = "B"
        else:
            card_colour = ("B", "R")[np.random.binomial(1, 1 / 3.0)]

        self.hand.append((card_num, card_colour))

        if card_colour == "B":
            self.hand_sum += card_num
        else:
            self.hand_sum -= card_num

    def stick(self):
        self._is_stick = True

    def is_busted(self):
        return not (1 <= self.hand_sum <= 21)

    def is_stick(self):
        return self._is_stick


class Easy21(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):

        # Observation Space
        # Dealer's first card [1, 10] and player's sum [1, 21]
        self.observation_space = spaces.Tuple((
            spaces.Box(low=1, high=10, shape=(1,)),
            spaces.Box(low=1, high=21, shape=(1,))
        ))

        # Action Space
        #   0 = sticks (no further cards)
        #   1 = hits (draws another card)
        self.action_space = spaces.Discrete(2)

        self.dim = (10, 21, 2)  # First card, player's sum, num actions

        # Dealer's hand
        self.dealer = CardHandler()

        # Player's hand
        self.player = CardHandler()

    def get_score(self):
        are_both_stick = self.dealer.is_stick() and self.player.is_stick()
        if self.dealer.is_busted() or \
                (are_both_stick and self.player.hand_sum > self.dealer.hand_sum):
                return 1
        elif self.player.is_busted() or \
                (are_both_stick and self.player.hand_sum < self.dealer.hand_sum):
                return -1
        else:
            return 0

    def _step(self, action):
        assert self.action_space.contains(action)

        # Player Action
        if action == 0:  # stick
            self.player.stick()
            while not (self.dealer.is_busted() or self.dealer.is_stick()):
                self.dealer.hit()
                if self.dealer.hand_sum >= 17:
                    self.dealer.stick()

        elif action == 1:  # hit
            self.player.hit()

        self.make_info(action)
        return self.make_observation()

    def make_observation(self):
        return (
            (self.dealer.hand[0][0], self.player.hand_sum),
            self.get_score(),
            self.dealer.is_busted() or self.player.is_busted() or (
                self.dealer.is_stick() and self.dealer.is_stick()),
            self.info
        )

    def make_info(self, action):
        self.info["dealer_hand"] = self.dealer.hand
        self.info["player_hand"] = self.player.hand
        self.info["dealer_sum"] = self.dealer.hand_sum
        self.info["player_sum"] = self.player.hand_sum
        self.info["actions"].append(("stick", "hit")[action])
        self.info["rewards"].append(self.get_score())

    def _reset(self):
        self.player.reset()
        self.player.hit(first=True)
        self.dealer.reset()
        self.dealer.hit(first=True)
        self.info = {
            "dealer_hand": self.dealer.hand,
            "player_hand": self.player.hand,
            "dealer_sum": self.dealer.hand_sum,
            "player_sum": self.player.hand_sum,
            "actions": list(),
            "rewards": list()
        }
        return self.make_observation()

    def _render(self, mode='human', close=False):
        pass
