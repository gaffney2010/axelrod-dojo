import axelrod as axl
import random
import unittest

from axelrod_dojo import GamblerParams

C, D = axl.Action.C, axl.Action.D


class TestGamblerParams(unittest.TestCase):
    def test_init(self):
        plays = 2
        op_plays = 2
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertEqual(gambler_params.PlayerClass, axl.Gambler)
        self.assertEqual(gambler_params.plays, plays)
        self.assertEqual(gambler_params.op_plays, op_plays)
        self.assertEqual(gambler_params.op_start_plays, op_start_plays)

    def test_receive_vector(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertRaises(AttributeError, GamblerParams.__getattribute__,
                          *[gambler_params, 'vector'])

        vector = [random.random() for _ in range(6)]
        gambler_params.receive_vector(vector)

        self.assertEqual(gambler_params.vector, vector)

    def test_vector_to_instance(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        gambler_params.receive_vector([random.random() for _ in range(8)])
        instance = gambler_params.player()

        self.assertIsInstance(instance, axl.Gambler)

    def test_create_vector_bounds(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)
        lb, ub = gambler_params.create_vector_bounds()

        self.assertIsInstance(lb, list)
        self.assertIsInstance(ub, list)
        self.assertEqual(len(lb), 8)
        self.assertEqual(len(ub), 8)

    def test_random_params(self):
        plays = 2
        op_plays = 2
        op_start_plays = 2

        table = GamblerParams.random_params(plays=plays, op_plays=op_plays,
                                            op_start_plays=op_start_plays)

        self.assertIsInstance(table, list)
        self.assertEqual(len(table), 64)

    def test_randomize(self):
        plays = 2
        op_plays = 2
        op_start_plays = 2

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays)

        self.assertIsInstance(gambler_params.pattern, list)
        self.assertEqual(len(gambler_params.pattern), 64)

    def test_repr(self):
        plays = 1
        op_plays = 1
        op_start_plays = 1

        pattern = [0.05, 0.07, 0.70, 0.03,
                   0.60, 0.70, 0.45, 0.10]

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays,
                                       pattern=pattern)

        self.assertEqual(gambler_params.__repr__(),
                         '1:1:1:0.05|0.07|0.7|0.03|0.6|0.7|0.45|0.1')
