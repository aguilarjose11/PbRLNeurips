class CautiousDeciderEnvironment(Environment):
    """ Cautious Decider Environment Implementation

    The cautious decider environment is a simple MDP implementation of
    a n-horizon mdp. The state is represented by a vector of length
    equal to the horizon. There are 2 actions an agent can take, and
    each action is depicted in the state as a -1 or +1.

    Attributes
    ----------
    theta
    phi
    delta
    tau
    kappa
    """

    ''' Definition of State and Action '''
    EnvAction = torch.IntTensor # Will be a 1x1 vector

    def __init__(self,
                 reward_param: Vector,
                 state_action_map: Callable[[State, Action], Vector],
                 /, # Keyword or order
                 transition_map: Callable[[State, Action], State],
                 *, # Keyword only
                 difficulty: int,
                 complexity: int=2,):
        """ Environment Constructor

        Parameters
        ----------

        reward_param: Vector

        state_action_map: Callable[[State, Action], Vector]

        transition_map: float

        difficulty: int
            aka horizon

        complexity: int

        """

        ''' Placement '''
        self.theta: Vector = reward_param
        self.phi: Callable[[State, Action], Vector] = state_action_map

        ''' Placement or Keyword '''
        self.T = transition_map

        ''' Keyword '''
        self.tau: int = difficulty
        self.kappa: int = complexity

        ''' Environment Attributes '''
        # State, which is a 2d vector
        self.state: Vector = torch.zeros([self.tau, 1])
        # Timestep, which counts from 0 to the horizon (tau)
        self.t: int = None

    def reset(self,) -> tuple[State]:
        """ Reset environment """

        ''' (Re-)Construct Environment'''
        # Set state
        self.t = 0
        self.state = torch.zeros([self.tau, 1])

        ''' Return State '''
        return self.state


    def take_action(self,
                    action: Action) -> tuple[State, EnvInfo]:
        """ Take action and return state and environment info """
        info = dict()
        # Apply action. -2 to account for yet-tp-make-choice.
        if self.t < (self.tau ):
            self.T(self.state, action)
            self.t += 1
            info = {
                'reward': self.state.T @ self.theta,
                'done': not (self.t < self.tau - 1),
            }
        else:
            info = {
                'reward': float(self.state.T @ self.theta),
                'done': True,
            }

        return self.state, info

