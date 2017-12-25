import axelrod as axl
import pyswarm

from axelrod_dojo.utils import score_params
from axelrod_dojo.utils import PlayerInfo
from multiprocessing import cpu_count


class PSO(object):
    """PSO class that implements a particle swarm optimization algorithm."""
    def __init__(self, params_class, params_kwargs, objective, opponents=None,
                 population=1, generations=1, debug=True, phip=0.8, phig=0.8,
                 omega=0.8, weights=None, sample_count=None, processes=1):

        self.params_class = params_class
        self.params_kwargs = params_kwargs
        self.objective = objective
        if opponents is None:
            self.opponents_information = [
                    PlayerInfo(s, {}) for s in axl.short_run_time_strategies]
        else:
            self.opponents_information = [
                    PlayerInfo(p.__class__, p.init_kwargs) for p in opponents]
        self.population = population
        self.generations = generations
        self.debug = debug
        self.phip = phip
        self.phig = phig
        self.omega = omega
        self.weights = weights
        self.sample_count = sample_count
        if processes == 0:
            self.processes = cpu_count()
        else:
            self.processes = processes

    def swarm(self):

        params = self.params_class(**self.params_kwargs)
        lb, ub = params.create_vector_bounds()

        def objective_function(vector):
            params.receive_vector(vector=vector)
            instance_generation_function = 'player'

            return - score_params(params=params, objective=self.objective,
                                  opponents_information=self.opponents_information,
                                  weights=self.weights,
                                  sample_count=self.sample_count,
                                  instance_generation_function=instance_generation_function
                                 )

        # TODO remove check once v 0.7 is pip installable
        # There is a multiprocessing version (0.7) of pyswarm available at
        # https://github.com/tisimst/pyswarm, just pass processes=X
        # Pip installs version 0.6
        if pyswarm.__version__ == "0.7":
            xopt, fopt = pyswarm.pso(objective_function, lb, ub,
                                     swarmsize=self.population,
                                     maxiter=self.generations, debug=self.debug,
                                     phip=self.phip, phig=self.phig,
                                     omega=self.omega, processes=self.processes)
        else:
            xopt, fopt = pyswarm.pso(objective_function, lb, ub,
                                     swarmsize=self.population,
                                     maxiter=self.generations, debug=self.debug,
                                     phip=self.phip, phig=self.phig,
                                     omega=self.omega)
        return xopt, fopt
