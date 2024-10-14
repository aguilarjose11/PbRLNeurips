""" Execute experiments used in paper

The `run.py` script contains the code used for generating the
empirical results of our published research paper. See README.md for
more information on using this code, the experiments, setup,
and author contact.

@author Jose E. Aguilar Escamilla
@date June 18th, 2024
@email aguijose(at)oregonstate.edu
"""

import argparse
import pickle
import time

from experiments import CLIParser, CLIArguments
from experiments.experiment import exps as EXPERIMENT_LIST
from experiments.algorithms import LSVIUCB, QLearning

### Experiment Specifics (used by CLI documentation) ###
_NAME = 'python -m run'
_ABOUT = 'experiments used in research paper. Feel free to inspect ' \
         'our code and use this code to replicate our results.'
_TIPS = 'For questions about this codebase, contact Jose E. Aguilar ' \
        'Escamilla (aguijose[at]oregonstate.edu)'


########################################################################
######################## CLI Parsing Functions #########################
########################################################################

def get_params() -> CLIParser:
    """Construct main the CLI parser"""
    parser = argparse.ArgumentParser(prog=_NAME,
                                     description=_ABOUT,
                                     epilog=_TIPS)
    # Nickname for adding new argument groups for parsing CLI
    new_group = parser.add_argument_group

    ''' Experiment Parameters'''
    # Do not miss that this variable is really a nickname for
    # `add_argument` method from parser
    experiments = new_group(
        title='Experiment Choice (Only Use One Experiment Flag)',
        description='experiments are selected by choosing one of the '
                    'flags. The --config parameter allows a file to '
                    'be passed for further configuration of the '
                    'experiments.'
    ).add_argument
    experiments('--cautious_decider', action='store_true',
                default=False,
                help='Executes the "Cautious Decider" experiment.'
    )
    experiments('--half_cheetah', action='store_true',
                default=False,
                help='Executes the "Half-Cheetahv4" from '
                     'MuJoCo experiment.'
    )
    experiments('--pendulum', action='store_true',
                default=False,
                help='Executes the "Inverted-Pendulum" '
                     'from MuJoCo experiment.'
    )
    experiments('--config', type=str,
                required=True,
                help='Experiment-specific configuration file.'
    )
    experiments('--experiment_help', action='store_true',
                default=False,
                help='Display selected experiment help.')

    ''' Hyperparameter Parameters '''
    # Do not miss that this variable is really a nickname for
    # `add_argument` method from parser
    hyperparams = parser.add_argument_group(
        title='Hyper-parameter Choices for experiments',
        description='Basic experiment hyperparameter configuration. '
                    'These parameters only deal with the learning  '
                    'behaviour and experiment length. All other '
                    'configurations are handled in a different parser.'
    ).add_argument

    hyperparams('--epochs', type=int, default=10,
                help='Number of epochs')
    hyperparams('--output', type=str,
                default='Unnamed Experiment - <date>',
                help='The output folder name will contain ll of the '
                     'data collected for the experiment. If the folder '
                     'name has the token <date>, then, it will be '
                     'replaced by the date and time of experiment.')

    return parser


def parse_args(parser: CLIParser) -> CLIArguments:
    """Parse CLI options passed to script"""

    args = parser.parse_args()
    # Manual parsing:
    # Replace <date> with today's date
    args.output = args.output.replace('<date>', time.ctime())

    if args.cautious_decider:
        args.experiment_str = 'cautious_decider'
    elif args.half_cheetah:
        args.experiment_str = 'half_cheetah'
    elif args.pendulum:
        args.experiment_str = 'pendulum'
    else:
        assert False, 'No experiment was chosen! See help menu for ' \
                      'more.'

    return args


def get_parser_exp():
    """ Parse basic arguments for experiments

    The parser generated will be 'augmented' by the experiment class
    itself (parser_augment())
    """

    parser = argparse.ArgumentParser()
    # Shorthand for function
    new_group = parser.add_argument_group
    learning = new_group(
        title='Learning hyperparameters',
        description='Learning hyperparameters for learning '
                    'algorithm').add_argument
    learning('--lr', type=float,
             default=1e-3,
             help='Learning rate for learning algorithm.')
    return parser


########################################################################
####################### Miscellaneous Functions ########################
########################################################################

def get_algorithms(args_exp: tuple[dict, None]) -> list:
    algos = []
    if args_exp.lsvi_ucb:
        algos.append(LSVIUCB())
    if args_exp.q_learning:
        algos.append(QLearning())
    return algos


########################################################################
############################ Main Function #############################
########################################################################

if __name__ == '__main__':
    ''' Parsing of CLI and configuration file parameters '''
    # Get parameters
    parser = get_params()
    args = parse_args(parser)

    # If configuration file passed, open; else, pass error (we must
    # have configuration file!).
    parser_exp = get_parser_exp()

    # Get chosen experiment's class
    experiment_class = EXPERIMENT_LIST[args.experiment_str]

    # Parser augmentation is unique to each experiment. See
    # documentation for more.
    parser_exp = experiment_class.parser_augment(parser_exp)

    if args.experiment_help:
        parser_exp.parse_args(['--help'])

    # Parse arguments of configuration file
    with open(args.config, 'r') as config_file:
        # Create list of params to parse
        cli_input = config_file.read().split()
        args_exp = parser_exp.parse_args(cli_input)


    ''' Algorithm and Environments Instantiation and Preparation'''

    # Create list of algorithms
    algos = get_algorithms(args_exp)
    # Create experiment
    experiment = experiment_class(args, args_exp, algos)

    ''' Experiment Execution Code'''
    data = experiment.run()

    # Store data and model in specified location
    with open(args.output, 'wb') as jar:
        pickle.dump(data, jar)

    exit(0)
