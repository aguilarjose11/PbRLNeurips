from typing import Any

from ..experiments.algorithm import Algorithm
from ..experiments.data_collector import Datum
from ..experiments.environment import State, Action

class RAPbRL(Algorithm):
    def __init__(self,
                 ):
        # self.pi_1

        self.trajectory_1 = []
        self.trajectory_2 = []

    def act(self,
            state: State) -> Action:
        pass

    def learn(self) -> None:
        pass

    def inspect(self) -> dict | Datum | Any:
        pass
