from abc import ABC, abstractmethod

class Datum(ABC):
    '''Individual experiment datum'''
    pass

class DataCollector(ABC):
    ''' Experiment data collector'''
    @abstractmethod
    def inspect(self,
                **kwargs) -> Datum:
        ''' Collects and create datum from experiment '''
        pass