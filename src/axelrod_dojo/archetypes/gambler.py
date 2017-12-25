import random

from axelrod import Action, Gambler
from axelrod.strategies.lookerup import create_lookup_table_keys
from axelrod_dojo.utils import Params

C, D = Action.C, Action.D


class GamblerParams(Params):
    """
    A class for a Gambler archetype used to train Gambler strategies using 
    various optimization algorithms. 
    
    More details for the Gambler player can be found in the axelrod 
    documentation: http://axelrod.readthedocs.io/ 
    
    Parameters
    ----------
    plays : integer
        The number of player's last moves to remember
    op_plays : integer
        The number of opponent's last moves to remember
    op_start_plays : integer
        The number of opponent's initial moves to remember   
    """
    def __init__(self, plays, op_plays, op_start_plays, pattern=None):
        self.PlayerClass = Gambler
        self.plays = plays
        self.op_plays = op_plays
        self.op_start_plays = op_start_plays

        if pattern is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.pattern = pattern

    def player(self):
        """Turns the attribute vector in to a Gambler player instance."""
        player = self.PlayerClass(pattern=self.vector)
        return player

    def receive_vector(self, vector):
        """Receives a vector and creates an instance attribute called
        vector.  Ignores extra parameters."""
        self.vector = vector

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables.  Ignores extra
        parameters."""
        size = len(self.pattern)
        lb = [0.0] * size
        ub = [1.0] * size

        return lb, ub

    @staticmethod
    def random_params(plays, op_plays, op_start_plays):
        keys = create_lookup_table_keys(plays, op_plays, op_start_plays)

        pattern = [random.random() for _ in keys]
        return pattern

    def randomize(self):
        pattern = self.random_params(
            self.plays, self.op_plays, self.op_start_plays)
        self.pattern = pattern

    def __repr__(self):
        return "{}:{}:{}:{}".format(
            self.plays,
            self.op_plays,
            self.op_start_plays,
            '|'.join(str(v) for v in self.pattern)
        )
