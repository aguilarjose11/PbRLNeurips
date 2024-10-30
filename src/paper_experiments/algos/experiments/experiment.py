'''

@author Jose E. Aguilar Escamilla
@date June 12th, 2024
@email aguijose(at)oregonstate.edu

'''

from abc import ABC, abstractmethod

from data_collector import Datum

class Experiment(ABC):
    '''Reinforcement Learning experiment '''

    @abstractmethod
    def execute(self) -> list[Datum]:
        '''Run experiment'''
        pass

    def __call__(self, *args, **kwargs):
        return self.execute(*args, **kwargs)


# # [NOTE] List of environments can be found at the end of the script.
# class __Experiment(ABC):
#     ''' Base class for experiments'''
#
#     def __init__(self,
#                  args: CLIArguments,
#                  algorithms: list[RLAlgorithm, ...]):
#         self.args = args
#         self.epochs = args.epochs
#         self.algorithms = algorithms
#     #@abstractmethod
#     def run(self):
#         # for every epoch
#             # for every algorithm in self._algorithms
#                 # For every timestep (Flip with above if instantiating
#                 # all envs at once.)
#                     # get algorithm's corresponding state (pop it)
#                     # get algorithm's choice on state
#                     # Get reward after applying algorithm's choice
#                     # Save new state as current state for the algorithm
#                     # save reward cummulatitevally
#         # Return list of cummulative reward per epoch, and learnt model
#         # (pull as algorithm.model).
#         pass
#
#     @staticmethod
#     @abstractmethod
#     def parser_augment(parser: CLIParser):
#         ''' Argument parses augmentation for experiment attributes '''
#         #parser.add_argument
#         pass #return parser
#
#     @abstractmethod
#     def reset(self):
#         pass
#
#     ### Class Properties ###
#
#     @property
#     def data(self):
#         return self._data
#     @data.setter
#     @abstractmethod
#     def set_datum(self, datum) -> None:
#         pass
#     @data.getter
#     @abstractmethod
#     def get_data(self) -> dict:
#         pass
#
#     @property
#     def policy(self):
#         return self._policy
#     @policy.setter
#     @abstractmethod
#     def set_policy(self) -> None:
#         pass
#     @policy.getter
#     @abstractmethod
#     def get_policy(self) -> dict:
#         pass




