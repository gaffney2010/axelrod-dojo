"""
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code

Usage:
    pso_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--output OUTPUT_FILE] [--objective OBJECTIVE]
    [--repetitions REPETITIONS] [--turns TURNS] [--noise NOISE]
    [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 100]
    --population POPULATION     Starting population size  [default: 40]
    --output OUTPUT_FILE        File to write data to [default: pso_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS     Number of opponent starting plays in the lookup table [default: 2]
"""

from docopt import docopt

from axelrod_dojo.archetypes.gambler import GamblerParams
from axelrod_dojo.algorithms.particle_swarm_optimization import PSO


from axelrod_dojo.utils import prepare_objective

if __name__ == "__main__":
    arguments = docopt(__doc__, version='PSO Evolve 0.3')
    print(arguments)

    # Population Args
    population = int(arguments['--population'])
    generations = int(arguments['--generations'])

    # Objective
    name = str(arguments['--objective'])
    repetitions = int(arguments['--repetitions'])
    turns = int(arguments['--turns'])
    noise = float(arguments['--noise'])
    nmoran = int(arguments['--nmoran'])

    # Lookup Tables
    plays = int(arguments['--plays'])
    op_plays = int(arguments['--op_plays'])
    op_start_plays = int(arguments['--op_start_plays'])
    params_kwargs = {"plays": plays,
                    "op_plays": op_plays,
                    "op_state_plays": op_start_plays}

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)

    pso = PSO(GamblerParams, params_kwargs, objective=objective,
              population=population, generations=generations)

    xopt, fopt = pso.swarm()

    print(xopt)
    print(fopt)
