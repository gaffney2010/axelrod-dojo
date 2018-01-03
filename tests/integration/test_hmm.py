import unittest
import tempfile
import csv

import axelrod as axl
import axelrod_dojo as dojo

C, D = axl.Action.C, axl.Action.D


class TestHMM(unittest.TestCase):
    temporary_file = tempfile.NamedTemporaryFile()

    def test_score(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     bottleneck=2,
                                     mutation_probability=.01,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from temp file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.HMMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.HMMParams)

        # Test that can use these loaded params in a new algorithm instance
        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     population=best,
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)
        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

    def test_score_with_weights(self):
        name = "score"
        turns = 5
        noise = 0
        repetitions = 5
        num_states = 3
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     weights=[5, 1, 1, 1, 1],
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from temp file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.HMMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.HMMParams)

            self.assertEqual(best[0].__repr__(), best_params)

    def test_score_with_sample_count(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     sample_count=2,  # Randomly sample 2 opponents at each step
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from temp file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.HMMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.HMMParams)

            self.assertEqual(best[0].__repr__(), best_params)

    def test_score_with_sample_count_and_weights(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     sample_count=2,  # Randomly sample 2 opponents at each step
                                     weights=[5, 1, 1, 1, 1],
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from tempo file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.HMMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.HMMParams)

            self.assertEqual(best[0].__repr__(), best_params)

    def test_score_with_particular_players(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        opponents = [s() for s in axl.basic_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=0)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

    def test_population_init_with_given_rate(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states,
                                                    "mutation_probability": .5},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)

        for p in population.population:
            self.assertEqual(p.mutation_probability, .5)
        generations = 1
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 1)

    def test_score_pso(self):
        name = "score"
        turns = 5
        noise = 0
        repetitions = 2
        num_states = 4
        opponents = [s() for s in axl.demo_strategies]
        size = 30
        generations = 5

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        pso = dojo.PSO(dojo.HMMParams,
                       params_kwargs={"num_states": num_states},
                       objective=objective,
                       opponents=opponents,
                       population=size,
                       generations=generations)

        axl.seed(0)
        xopt, fopt = pso.swarm()
        
        self.assertTrue(len(xopt) == 2 * num_states ** 2 + num_states + 1)

        # You can put the optimal vector back into a HMM.
        simple_hmm_opt = dojo.HMMParams(num_states=num_states)
        simple_hmm_opt.receive_vector(xopt)
        simple_player = simple_hmm_opt.player()
        self.assertTrue(simple_player.hmm.is_well_formed())  # This should get asserted in initialization anyway
        
        self.assertIsInstance(simple_hmm_opt, dojo.HMMParams)
        print(xopt)  # As a vector still
        print(simple_hmm_opt)  # As a HMM

    def test_pso_to_ea(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 2
        num_states = 3
        opponents = [s() for s in axl.demo_strategies]
        size = 30
        generations = 3

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        winners = []

        for _ in range(5):

            axl.seed(_)

            pso = dojo.PSO(dojo.HMMParams,
                           params_kwargs={"num_states": num_states},
                           objective=objective,
                           opponents=opponents,
                           population=size,
                           generations=generations)

            xopt, fopt = pso.swarm()

            # You can put the optimal vector back into a HMM.
            hmm_opt = dojo.HMMParams(num_states=num_states)
            hmm_opt.receive_vector(xopt)

            winners.append(hmm_opt)
            
        # Put the winners of the PSO into an EA.
                                             
        population = dojo.Population(params_class=dojo.HMMParams,
                                     params_kwargs={"num_states": num_states},
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     population=winners,
                                     bottleneck=2,
                                     mutation_probability = .01,
                                     processes=1)

        axl.seed(0)
        population.run(generations)
        
        # Block resource (?)
        with open(self.temporary_file.name, "w") as f:
            pass

        scores = population.score_all()
        record, record_holder = 0, -1
        for i, s in enumerate(scores):
            if s >= record:
                record = s
                record_holder = i
        xopt, fopt = population.population[record_holder], record

        print(xopt)
