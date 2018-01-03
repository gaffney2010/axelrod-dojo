import random
from random import randrange

import numpy as np
from axelrod.strategies.hmm import HMMPlayer
from axelrod import Action
from axelrod_dojo.utils import Params

C, D = Action.C, Action.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows

def random_vector(size):
    """Create a random vector of values in [0, 1] that sums to 1."""
    vector = []
    s = 1
    for _ in range(size - 1):
        r = s * random.random()
        vector.append(r)
        s -= r
    vector.append(s)
    return vector

def normalize_vector(vec):
    s = sum(vec)
    if s == 0.0:
        n = len(vec)
        return [1 / n for v in vec]
    vec = [v / s for v in vec]
    return vec

def mutate_row(row, mutation_probability):
    """
    Given a row of probabilities, randomly change each entry with probability
    `mutation_probability` (a value between 0 and 1).  If changing, then change
    by a value randomly (uniformly) chosen from [-0.25, 0.25] bounded by 0 and
    100%.
    """
    randoms = np.random.random(len(row))
    for i in range(len(row)):
        if randoms[i] < mutation_probability:
            ep = random.uniform(-1, 1) / 4
            row[i] += ep
            if row[i] < 0:
                row[i] = 0
            if row[i] > 1:
                row[i] = 1
    return row


class HMMParams(Params):

    def __init__(self, num_states, mutation_probability=None, transitions_C=None,
                 transitions_D=None, emission_probabilities=None,
                 initial_state=0, initial_action=C):
        self.PlayerClass = HMMPlayer
        self.num_states = num_states
        if mutation_probability is None:
            self.mutation_probability = 10 / (num_states ** 2)
        else:
            self.mutation_probability = mutation_probability
        if transitions_C is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.transitions_C = copy_lists(transitions_C)
            self.transitions_D = copy_lists(transitions_D)
            self.emission_probabilities = list(emission_probabilities)
            self.initial_state = initial_state
            self.initial_action = initial_action


    def player(self):
        player = self.PlayerClass(self.transitions_C, self.transitions_D,
                                  self.emission_probabilities,
                                  self.initial_state, self.initial_action)
        return player

    def copy(self):
        return HMMParams(self.num_states, self.mutation_probability,
                         self.transitions_C, self.transitions_D,
                         self.emission_probabilities,
                         self.initial_state, self.initial_action)

    @staticmethod
    def random_params(num_states):
        t_C = []
        t_D = []
        p = []
        for _ in range(num_states):
            t_C.append(random_vector(num_states))
            t_D.append(random_vector(num_states))
            p.append(random.random())
        initial_state = randrange(num_states)
        initial_action = C
        return p, t_C, t_D, initial_state, initial_action

    def randomize(self):
        p, t_C, t_D, initial_state, initial_action = self.random_params(self.num_states)
        self.emission_probabilities = p
        self.transitions_C = t_C
        self.transitions_D = t_D
        self.initial_state = initial_state
        self.initial_action = initial_action

    @staticmethod
    def mutate_rows(rows, mutation_probability):
        for i, row in enumerate(rows):
            row = mutate_row(row, mutation_probability)
            rows[i] = normalize_vector(row)
        return rows

    def mutate(self):
        self.transitions_C = self.mutate_rows(
            self.transitions_C, self.mutation_probability)
        self.transitions_D = self.mutate_rows(
            self.transitions_D, self.mutation_probability)
        self.emission_probabilities = mutate_row(
            self.emission_probabilities, self.mutation_probability)
        if random.random() < self.mutation_probability / 10:
            self.initial_action = self.initial_action.flip()
        if random.random() < self.mutation_probability / (10 * self.num_states):
            self.initial_state = randrange(self.num_states)
        # Change node size?

    @staticmethod
    def crossover_rows(rows1, rows2):
        num_states = len(rows1)
        crosspoint = randrange(num_states)
        new_rows = copy_lists(rows1[:crosspoint])
        new_rows += copy_lists(rows2[crosspoint:])
        return new_rows

    @staticmethod
    def crossover_weights(w1, w2):
        crosspoint = random.randrange(len(w1))
        new_weights = list(w1[:crosspoint]) + list(w2[crosspoint:])
        return new_weights

    def crossover(self, other):
        # Assuming that the number of states is the same
        t_C = self.crossover_rows(self.transitions_C, other.transitions_C)
        t_D = self.crossover_rows(self.transitions_D, other.transitions_D)
        emissions = self.crossover_weights(
            self.emission_probabilities, other.emission_probabilities)
        return HMMParams(self.num_states, self.mutation_probability,
                         t_C, t_D, emissions,
                         self.initial_state, self.initial_action)

    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return "|".join(ss)

    def __repr__(self):
        return "{}:{}:{}:{}:{}".format(
            self.initial_state,
            self.initial_action,
            self.repr_rows(self.transitions_C),
            self.repr_rows(self.transitions_D),
            self.repr_rows([self.emission_probabilities])
        )

    @classmethod
    def parse_repr(cls, s):
        def parse_vector(line):
            row = line.split('_')
            row = list(map(float, row))
            return row
        def parse_matrix(sm):
            rows = []
            lines = sm.split('|')
            for line in lines:
                rows.append(parse_vector(line))
            return rows
        lines = s.split(':')
        initial_state = int(lines[0])
        initial_action = Action.from_char(lines[1])
        t_C = parse_matrix(lines[2])
        t_D = parse_matrix(lines[3])
        p = parse_vector(lines[4])
        num_states = len(t_C)
        return cls(num_states=num_states,
                   transitions_C=t_C,
                   transitions_D=t_D,
                   emission_probabilities=p,
                   initial_state=initial_state,
                   initial_action=initial_action)

    def receive_vector(self, vector):
        """
        Read a serialized vector into the set of HMM parameters (less initial
        state).  Then assign those HMM parameters to this class instance.

        Assert that the vector has the right number of elements for an HMMParams
        class with self.num_states.

        Assume the first num_states^2 entries are the transitions_C matrix.  The
        next num_states^2 entries are the transitions_D matrix.  Then the next
        num_states entries are the emission_probabilities vector.  Finally the last
        entry is the initial_action.
        """

        assert(len(vector) == 2 * self.num_states ** 2 + self.num_states + 1)

        def deserialize(vector):
            matrix = []
            for i in range(self.num_states):
                row = vector[self.num_states * i: self.num_states * (i + 1)]
                row = normalize_vector(row)
                matrix.append(row)
            return matrix

        break_tc = self.num_states ** 2
        break_td = 2 * self.num_states ** 2
        break_ep = 2 * self.num_states ** 2 + self.num_states
        self.transitions_C = deserialize(vector[0:break_tc])
        self.transitions_D = deserialize(vector[break_tc:break_td])
        self.emission_probabilities = normalize_vector(vector[break_td:break_ep])
        self.initial_action = C if round(vector[-1]) == 0 else D
        self.initial_state = 0

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""

        vec_len = 2 * self.num_states ** 2 + self.num_states + 1

        lb = [0.0] * vec_len
        ub = [1.0] * vec_len

        return lb, ub
