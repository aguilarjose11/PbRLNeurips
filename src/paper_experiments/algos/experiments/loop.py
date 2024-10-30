from abc import ABC, abstractmethod

from data_collector import Datum

class TrainingLoop(ABC):
    '''Training loop'''

    @abstractmethod
    def loop(self) -> Datum:
        '''Single training loop'''
        pass