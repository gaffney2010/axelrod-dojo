Genetic Algorithm
=================

A genetic algorithm aims to mimic evolutionary processes so as to optimise a
particular function on some space of candidate solutions.

The process can be described by assuming that there is a function
:math:`f:V\to \mathbb{R}`, where :math:`V` is some vector space. 
In the case of the Prisoner's dilemma,
the vector space :math:`V` corresponds to some representation of a
particular archetype (which might not actually be a numeric vector space) and
the function :math:`f` corresponds to some measure of performance/fitness of the
strategy in question.

In this setting a candidate solution :math:`x\in\mathbb{R}^m` corresponds to a
chromosome with each :math:`x_i` corresponding to a gene.

The genetic algorithm has three essential parameters:

- The population size: the algorithm makes use of a number of candidate
  solutions at each stage.
- The bottle neck parameter: at every stage the candidates in the population are
  ranked according to their fitness, only a certain number are kept (the best
  performing ones) from one generation to the next. This number is referred to
  as the bottle neck.
- The mutation probability: from one stage to the next when new individuals are
  added to the population (more about this process shortly) there is a
  probability with which each gene randomly mutates.

New individuals are added to the population (so as to ensure that the population
size stays constant from one stage to the next) using a process of "crossover".
Two high performing individuals are paired and according to some predefined
procedure, genes from both these individuals are combined to create a new
individual.

For each strategy archetype, this library thus defines a process for mutation as
well as for crossover.

Finite state machines
---------------------

A finite state machine is made up of the following:

- a mapping from a state/action pair to another target state/action pair
- an initial state/action pair.

(See [Harper2017]_ for more details.)

The crossover and mutation are implemented in the following way:

- Crossover: this is done by taking a randomly selected number of target
  state/actions
  pairs from one individual and the rest from the other.
- Mutation: given a mutation probability :math:`\delta` each target state/action
  has a probability :math:`\delta` of being randomly changed to one of the other
  states or actions. Furthermore the **initial** action has a probability of
  being swapped of :math:`\delta\times 10^{-1}` and the **initial** state has a
  probability of being changed to another random state of :math:`\delta \times
  10^{-1} \times N` (where :math:`N` is the number of states).

Hidden Markov models
---------------------

A hidden Markov model is made up of the following:

- a mapping from a state/action pair to a probability of defect or cooperation.
- a cooperation transition matrix, the probability of transitioning to each
  state, given current state and an opponent cooperation.
- a defection transition matrix, the probability of transitioning to each
  state, given current state and an opponent defection.
- an initial state/action pair.

(See [Harper2017]_ for more details.)

The crossover and mutation are implemented in the following way:

- Crossover: this is done by taking a randomly selected number of rows from
  one cooperation transition matrix and the rest from the other to form a target
  cooperation transition matrix; then a different number of randomly selected
  rows from one defection transition matrix and the rest from the other; and
  then a randomly select number of entries from one state/part -> probability
  mapping and the rest from the other.
- Mutation: given a mutation probability :math:`delta` each cell of both
  transition matrices and the state/part -> probability mapping have probability
  :math:`delta` of being increased by :math:`varepsilon`, where
  :math:`varepsilon` is randomly drawn uniformly from :math:`[-0.25, 0.25]`
  (A negative number would decrease.)  Then the transition matrices and mapping
  are adjusted so that no cell is outside :math:`[0, 1]` and the transition
  matrices are normalized so that each row adds to 1. Furthermore the
  **initial** action has a probability of being swapped of
  :math:`\delta\times 10^{-1}` and the **initial** state has a probability of
  being changed to another random state of
  :math:`\delta \times 10^{-1} \times N` (where :math:`N` is the number of
  states).
