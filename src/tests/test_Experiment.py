from ..paper_experiments.experiments.experiment import CautiousDecider
from argparse import Namespace
import random

def test_init():
    # Choose custom args
    #
    args = Namespace()
    args.cautious_decider = True
    args.half_cheetah = False
    args.pendulum = False
    args.config = './config_files/cautious_decider.cfg'
    args.epochs = 10
    args.output = '../../TestExperiment'
    exp_args = Namespace()
    exp_args.tolerance = 1
    exp_args.difficulty = 10
    exp_args.choices = 2
    exp_args.distribution = 'uniform'
    exp_args.lsvi_ucb = True
    exp_args.q_learning = False
    # Temporal class
    class SimpleAlgorithm:
        def __init__(self):
            pass
        def __str__(self):
            return 'Testing algorithm placeholder'

        def decide(self, state):
            return -1 if random.random() < 0.5 else +1

    algorithms = [SimpleAlgorithm(),]

    experiment = CautiousDecider(args, exp_args, algorithms)
    experiment.run()

    assert True
