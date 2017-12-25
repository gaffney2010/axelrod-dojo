import unittest
import random
import axelrod as axl
from axelrod_dojo.archetypes.hmm import random_vector

from axelrod_dojo import HMMParams

C, D = axl.Action.C, axl.Action.D


class TestHMMParams(unittest.TestCase):
    def test_init(self):
        # Test with TFT.
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)
        self.assertEqual(hmm_params.PlayerClass, axl.HMMPlayer)
        self.assertEqual(hmm_params.num_states, num_states)
        self.assertEqual(hmm_params.mutation_probability, 2.5)
        self.assertEqual(hmm_params.transitions_C, t_C)
        self.assertEqual(hmm_params.transitions_D, t_D)
        self.assertEqual(hmm_params.emission_probabilities, p)
        self.assertEqual(hmm_params.initial_state, 0)
        self.assertEqual(hmm_params.initial_action, C)

    def test_init_without_defaults(self):
        """
        Check that by default have random rows.
        """
        num_states = 2
        transitions_Cs, transitions_Ds = [], []
        for seed in range(2):
            axl.seed(seed)
            hmm_params = HMMParams(num_states=num_states)
            transitions_Cs.append(hmm_params.transitions_C)
            transitions_Ds.append(hmm_params.transitions_D)
        self.assertNotEqual(*transitions_Cs)
        self.assertNotEqual(*transitions_Ds)

    def test_player(self):
        # Test with TFT.
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)
        self.assertEqual(hmm_params.player(),
                         axl.HMMPlayer(transitions_C=t_C,
                                       transitions_D=t_D,
                                       emission_probabilities=p,
                                       initial_state=0,
                                       initial_action=C))

    def test_copy(self):
        num_states = 2
        hmm_params = HMMParams(num_states=num_states)
        copy_hmm_params = hmm_params.copy()
        # Not the same
        self.assertIsNot(hmm_params, copy_hmm_params)

        # Have same attributes
        self.assertEqual(hmm_params.PlayerClass, copy_hmm_params.PlayerClass)
        self.assertEqual(hmm_params.num_states, copy_hmm_params.num_states)
        self.assertEqual(hmm_params.transitions_C,
                         copy_hmm_params.transitions_C)
        self.assertEqual(hmm_params.transitions_D,
                         copy_hmm_params.transitions_D)
        self.assertEqual(hmm_params.emission_probabilities,
                         copy_hmm_params.emission_probabilities)
        self.assertEqual(hmm_params.initial_action,
                         copy_hmm_params.initial_action)
        self.assertEqual(hmm_params.initial_state,
                         copy_hmm_params.initial_state)

    def test_random_params(self):
        num_states = 2
        hmm_params = HMMParams(num_states=num_states)
        p, t_C, t_D, initial_state, initial_action = hmm_params.random_params(num_states)

        self.assertEqual(len(p), num_states)
        self.assertEqual(len(t_C), num_states)
        self.assertEqual(len(t_D), num_states)
        for s1 in range(num_states):
            self.assertEqual(len(t_C[s1]), num_states)
            self.assertEqual(len(t_D[s1]), num_states)
            self.assertTrue(0.0 <= p[s1] <= 1.0)
            for s2 in range(num_states):
                self.assertTrue(0.0 <= t_C[s1][s2] <= 1.0)
                self.assertTrue(0.0 <= t_D[s1][s2] <= 1.0)
        self.assertIn(initial_state, range(num_states))
        self.assertIn(initial_action, (C, D))

    def test_randomize(self):
        hmm_params = HMMParams(num_states=2)

        p, t_C, t_D = [], [], []
        axl.seed(5)
        for _ in range(2):
            t_C_elem = [random.random()]
            t_C_elem.append(1.0 - t_C_elem[0])
            t_C.append(t_C_elem)
            t_D_elem = [random.random()]
            t_D_elem.append(1.0 - t_D_elem[0])
            t_D.append(t_D_elem)
            p.append(random.random())
        initial_state = random.randrange(2)

        axl.seed(5)
        hmm_params.randomize()
        self.assertEqual(hmm_params.transitions_C, t_C)
        self.assertEqual(hmm_params.transitions_D, t_D)
        self.assertEqual(hmm_params.emission_probabilities, p)
        self.assertEqual(hmm_params.initial_state, initial_state)
        self.assertEqual(hmm_params.initial_action, C)

    def test_crossover_rows(self):
        num_states = 2
        hmm_params = HMMParams(num_states=num_states)

        t_C1 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        t_C2 = [[0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5], [0., 0., 0.5, 0.5]]

        axl.seed(0)
        t_C = hmm_params.crossover_rows(t_C1, t_C2)
        self.assertEqual(t_C, t_C1[:3] + t_C2[3:])

        axl.seed(1)
        t_C = hmm_params.crossover_rows(t_C1, t_C2)
        self.assertEqual(t_C, t_C1[:1] + t_C2[1:])

    def test_crossover(self):
        num_states = 4

        t_C1 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        t_D1 = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
        p1 = [1., 1., 1., 1.]
        hmm_params1 = HMMParams(num_states=num_states,
                                transitions_C=t_C1,
                                transitions_D=t_D1,
                                emission_probabilities=p1,
                                initial_state=0,
                                initial_action=C)

        t_C2 = [[0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5], [0., 0., 0.5, 0.5]]
        t_D2 = [[0.5, 0.5, 0., 0.], [0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5], [0., 0., 0.5, 0.5]]
        p2 = [0., 0., 0., 0.]
        hmm_params2 = HMMParams(num_states=num_states,
                                transitions_C=t_C2,
                                transitions_D=t_D2,
                                emission_probabilities=p2,
                                initial_state=0,
                                initial_action=C)

        axl.seed(1)
        hmm_params = hmm_params1.crossover(hmm_params2)
        self.assertEqual(hmm_params.transitions_C,
                         hmm_params1.transitions_C[:1] + hmm_params2.transitions_C[1:])
        self.assertEqual(hmm_params.transitions_D,
                         hmm_params1.transitions_D[:0] + hmm_params2.transitions_D[0:])
        self.assertEqual(hmm_params.emission_probabilities,
                         hmm_params1.emission_probabilities[:2]
                         + hmm_params2.emission_probabilities[2:])

        self.assertEqual(hmm_params.PlayerClass, hmm_params1.PlayerClass)
        self.assertEqual(hmm_params.num_states, hmm_params1.num_states)
        self.assertEqual(hmm_params.initial_action,
                         hmm_params1.initial_action)
        self.assertEqual(hmm_params.initial_state,
                         hmm_params1.initial_state)

    def test_repr_rows(self):
        num_states = 2
        hmm_params = HMMParams(num_states=num_states)

        t_C = [[1, 0], [1, 0]]
        string_repr = hmm_params.repr_rows(t_C)
        self.assertEqual(string_repr, '1_0|1_0')

    def test_repr(self):
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)

        self.assertEqual(hmm_params.__repr__(),
                         '0:C:1_0|1_0:0_1|0_1:1_0')

    def test_parse_repr(self):
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        string_repr = '0:C:1_0|1_0:0_1|0_1:1_0'
        parameters = HMMParams.parse_repr(string_repr)
        self.assertEqual(parameters.initial_state, 0)
        self.assertEqual(parameters.initial_action, C)
        self.assertEqual(parameters.transitions_C, t_C)
        self.assertEqual(parameters.transitions_D, t_D)
        self.assertEqual(parameters.emission_probabilities, p)

    def test_vector_to_instance(self):
        num_states = 4
        
        vector = []
        for _ in range(2 * num_states):
            vector += random_vector(num_states)
        for _ in range(num_states + 1):
            vector.append(random.random())
            
        hmm_params = HMMParams(num_states=num_states)
        hmm_params.receive_vector(vector=vector)

        instance = hmm_params.player()
        self.assertIsInstance(instance, axl.HMMPlayer)

    def test_create_vector_bounds(self):
        num_states = 4
        size = 2 * num_states ** 2 + num_states + 1
        
        hmm_params = HMMParams(num_states=num_states)
        lb, ub = hmm_params.create_vector_bounds()

        self.assertIsInstance(lb, list)
        self.assertEqual(len(lb), size)
        self.assertIsInstance(ub, list)
        self.assertEqual(len(ub), size)
