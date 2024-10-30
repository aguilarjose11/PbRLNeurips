########################################################################
###################### Experiment Implementations ######################
########################################################################


class CautiousDecider(Experiment):
    """ Implement Cautious Decider Experiment

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self,
                 args: CLIArguments,
                 args_exp: CLIArguments,
                 algorithms: list[RLAlgorithm, ...]):

        # Add base experiment arguments
        super().__init__(args=args,
                         algorithms=algorithms)

        # Add experiment-unique configuration
        self.args_exp: CLIArguments = args_exp
        self.tolerance: float = args_exp.tolerance
        self.difficulty: int = args_exp.difficulty
        self.choices: int = args_exp.choices
        self.distribution: str = args_exp.distribution
        if self.distribution.lower() == 'uniform':
            self.noise_dist: Callable[[], list[float]] = lambda : \
                scipy.stats.uniform().rvs() * 2 - 1
        # elif self.distribution.lower() == 'gaussian':
        #     pass
        else:
            assert False, 'Distribution specified does not exist or is not implemented!'
        # self.noise
        self.algorithms = algorithms

        # Randomly choose optimal policy/optimal state
        self.seed = 42
        random.seed(42)
        self.theta = torch.Tensor([-1 if random.random() < 0.5 else +1 for _ in range(self.difficulty)])

        # Create functions used by problem environment
        self.exploration = 0.1
        def transition_map(state, action):
            action_ = np.sign(action)
            if random.random() < self.exploration:
                action_ *= -1
            t = int(state.abs().sum()) # assuming 2 actions
            state[t] = action_
            return state

        def state_action(state, action):
            state_ = state.clone().detach()
            t = int(state_.abs().sum())  # assuming 2 actions
            state_[t] = action
            return state_

        self.env = CautiousDeciderEnvironment(
            self.theta,
            state_action,
            transition_map=transition_map,
            difficulty=self.difficulty,
            complexity=self.choices)

    def run(self):
        # Data loging
        data = []
        # for every epoch
        for epoch in range(self.epochs):
            datum = {str(alg): [] for alg in self.algorithms}
            # for every algorithm in self._algorithms
            for algorithm in self.algorithms:
                # For every timestep (Flip with above if instantiating all envs at once.)
                state = self.env.reset()
                for step in range(self.difficulty):
                    # get algorithm's corresponding state (pop it)
                    # get algorithm's choice on state
                    action = algorithm.decide(state)
                    # Get reward after applying algorithm's choice
                    state_, info = self.env.take_action(action)
                    # Save new state as current state for the algorithm
                    state = state_
                    # save reward cummulatitevally
                    datum[str(algorithm)].append(float(info['reward']))
            data.append(datum)# Return list of cummulative reward per epoch, and learnt model (pull as algorithm.model).
        return data

    def get_data(self) -> dict:
        pass

    def get_policy(self) -> dict:
        pass

    def set_datum(self, datum) -> None:
        pass

    def set_policy(self) -> None:
        pass

    def reset(self):
        pass

    def parser_augment(parser: CLIParser):
        # short name for parser method
        new_arg = parser.add_argument

        new_arg('--tolerance', type=int,
                default=1,
                help='Tolerance of MDP to mistakes. This is the '
                     'number of mistakes after which no "redo" is '
                     'possible.'
        )
        new_arg('--difficulty', type=int,
                default=4,
                help='The difficulty level is the same as the number '
                     'of steps. It is the number of times choices must '
                     'be made.'
        )
        new_arg('--choices', type=int,
                default=2,
                help='Number of choices to choose from.'
        )
        new_arg('--distribution', type=str, default='uniform',
                help='Distribution of rewards of arms. Other choices '
                     'are "gaussian"'
        )

        new_algo = parser.add_argument_group(
            title='Algorithms',
            description='Implemented algorithms for CautiousDecider '
                        'Experiment').add_argument
        new_algo('--q_learning', action='store_true',
                default=False,
                help='Use q-learning algorithm.'
        )
        new_algo('--lsvi_ucb', action='store_true',
                default=False,
                help='Use Least Squares Value Iteration with Upper '
                     'Confidence Bound (LSVI-UCB) algorithm.'
        )

        return parser

class HalfCheetah(Experiment):
    pass

class InvertedPendulum(Experiment):
    pass


''' Dictionary of experiments with their respective class '''
exps = {
    'cautious_decider': CautiousDecider,
    'half_cheetah': HalfCheetah,
    'inverted_pendulum': InvertedPendulum,
}
