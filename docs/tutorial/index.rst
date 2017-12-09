Tutorial
========

In this tutorial we will aim to find the best Finite State Machine against a
collection of other strategies from the Axelrod library [Harper2017]_.

First let us get the collection of opponents against which we aim to train::

    >>> import axelrod as axl
    >>> opponents = [axl.TitForTat(), axl.Alternator(), axl.Defector()]
    >>> opponents
    [Tit For Tat, Alternator, Defector]

We are now going to prepare the training algorithm. First of all, we need to
prepare the objective of our strategy. In this case we will aim to maximise
:code:`score` in a match with :code:`10` turns over :code:`1` repetition::

    >>> import axelrod_dojo as dojo
    >>> objective = dojo.prepare_objective(name="score", turns=10, repetitions=1)

The algorithm we are going to use is a genetic algorithm which requires a
population of individuals. Let us set up the inputs::

    >>> params_class = dojo.FSMParams
    >>> params_kwargs = {"num_states": 2}

Using this we can now create our Population (with :code:`20` individuals) for a
genetic algorithm::

    >>> axl.seed(1)
    >>> population = dojo.Population(params_class=params_class,
    ...                              params_kwargs=params_kwargs,
    ...                              size=20,
    ...                              objective=objective,
    ...                              output_filename="training_output.csv",
    ...                              opponents=opponents,
    ...                              bottleneck=2,
    ...                              mutation_probability=.1)


We can now evolve our population::

    >>> generations = 4
    >>> population.run(generations)
    Scoring Generation 1
    Generation 1 | Best Score: 2.1 0:C:0_C_0_C:0_D_1_C:1_C_1_D:1_D_1_D
    Scoring Generation 2
    Generation 2 | Best Score: 2.1 0:C:0_C_0_C:0_D_1_C:1_C_1_D:1_D_1_D
    Scoring Generation 3
    Generation 3 | Best Score: 2.1 0:C:0_C_0_C:0_D_1_C:1_C_1_D:1_D_1_D
    Scoring Generation 4
    Generation 4 | Best Score: 2.1 0:C:0_C_0_C:0_D_1_C:1_C_1_D:1_D_1_D

The :code:`run` command prints out the progress of the algorithm and this is
also written to the output file (we passed :code:`output_filename` as an
argument earlier).

The last best score is a finite state machine with representation
:code:`0:C:0_C_0_C:0_D_1_D:1_C_1_D:1_D_1_D` which corresponds to a strategy that
stays in state :code:`0` as long as the opponent cooperates but otherwise goes
to state :code:`1` and defects forever. Indeed, if the strategy is playing
:code:`Defector` or :code:`Alternator` then it should just defect, otherwise it
should cooperate.
