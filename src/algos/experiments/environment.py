""" Implement basic mechanisms for environments """

# Environment information will be contained in a dictionary structure
# where any extra information can be passed from the environment.
from abc import ABC, abstractmethod

# Reward assumed to be just a float value
Reward = float

class Action(ABC):
    '''Abstract class of Action'''
    pass


class State(ABC):
    '''Abstract class of State'''
    pass


class EnvInfo(ABC):
    '''Environment information data structure'''
    pass


class Environment(ABC):
    '''Abstract class of Environment
    Properties
    ==========
    current_state: State
        - The current state of the environment. Will be affected by
        the step method.
        - For more advanced use, Setter/Getter can be overriden but
        given by default.
    Methods
    =======
    step(Action): tuple[State, Reward]
        - Applies the passed action and returns the subsequent
        State and Reward generated by the effect of the action in
        the environment.
    reset: State
        - Resets the environment and return the initial state.
    '''

    ##################
    ### Properties ###
    ##################

    @property
    def current_state(self) -> State:
        ''' Current state property
        Stored in self._state, the property tracks the current state
        of the environment. Setter and Getters are provided but not
        required to be used. For more advanced use, setter and
        getters can be overloaded by using the same decorator.
        '''
        return self._state

    @current_state.setter
    def set_current_state(self,
                          state: State) -> None:
        '''Ibid'''
        self._state = state

    @current_state.getter
    def get_current_state(self) -> State:
        '''Ibid'''
        return self._state

    ###############
    ### Methods ###
    ###############

    @abstractmethod
    def step(self,
             action: Action) -> tuple[State, Reward, EnvInfo]:
        '''Apply action to environment'''
        pass

    @abstractmethod
    def reset(self) -> State:
        '''Reset environment'''
        pass