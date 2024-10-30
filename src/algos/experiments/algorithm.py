from abc import ABC, abstractmethod
from typing import Any

from .environment import State, Action
from .data_collector import Datum

class Algorithm(ABC):

    # Methods

    @abstractmethod
    def act(self,
            state: State) -> Action:
        '''Algorithm decision making'''
        pass

    @abstractmethod
    def learn(self) -> None:
        '''Learning algorithm implementation'''
        pass

    @abstractmethod
    def inspect(self) -> dict | Datum | Any:
        ''' Inspect algorithm for data'''
        pass

    def __call__(self, *args, **kwargs):
        return self.act(*args, **kwargs)