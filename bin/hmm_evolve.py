"""
Hidden Markov Model Evolver

Usage:
    hmm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES] [--algorithm ALGORITHM]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: hmm_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 5]
    --algorithm ALGORITHM       Which algorithm to use (GA for genetic algorithm or PSO for
                                particle swarm algorithm) [default: GA]
"""

from docopt import docopt

from axelrod_dojo import HMMParams, Population, prepare_objective
from axelrod_dojo.algorithms.particle_swarm_optimization import PSO
from axelrod_dojo.archetypes.hmm import read_vector


if __name__ == '__main__':
    arguments = docopt(__doc__, version='HMM Evolver 0.3')
    print(arguments)
    processes = int(arguments['--processes'])

    # Vars for the genetic algorithm
    population = int(arguments['--population'])
    mutation_probability = float(arguments['--mu'])
    generations = int(arguments['--generations'])
    bottleneck = int(arguments['--bottleneck'])
    output_filename = arguments['--output']

    # Objective
    name = str(arguments['--objective'])
    repetitions = int(arguments['--repetitions'])
    turns = int(arguments['--turns'])
    noise = float(arguments['--noise'])
    nmoran = int(arguments['--nmoran'])

    # HMM
    num_states = int(arguments['--states'])
    param_kwargs = {"num_states": num_states}

    if arguments['--algorithm'] == "PSO":
        objective = prepare_objective(name, turns, noise, repetitions, nmoran)
        pso = PSO(HMMParams, param_kwargs, objective=objective,
                  population=population, generations=generations)

        xopt_helper, fopt = pso.swarm()
        t_C, t_D, p, starting_move = read_vector(xopt_helper, num_states)
        xopt = HMMParams(num_states=num_states,
                         transitions_C=t_C,
                         transitions_D=t_D,
                         emission_probabilities=p,
                         initial_state=0,
                         initial_action=starting_move)
    else:
        objective = prepare_objective(name, turns, noise, repetitions, nmoran)
        population = Population(HMMParams, param_kwargs, population, objective,
                                output_filename, bottleneck, mutation_probability,
                                processes=processes)
        population.run(generations)
        
        # Get the best member of the population to output.
        scores = population.score_all()
        record, record_holder = 0, -1
        for i, s in enumerate(scores):
            if s >= record:
                record = s
                record_holder = i
        xopt, fopt = population.population[record_holder], record
    
    print("Best Score: {} {}".format(fopt, xopt))
