import numpy as np

class EnvWorker:
    """
    Environment worker class. It represents a single environment.
    """
    def __init__(self, env):
        """
        EnvWorker initializer.
        :param env: Environment class object.
        """
        self.env = env
        # Result stores the information received from the environment
        self.result = None

    def reset(self):
        """
        Reset the environment.
        :param kwargs: Environment keyword arguments.
        """
        # Get observation from environment
        obs = self.env.reset()
        self.result = obs

    def send(self, action : list[int]):
        """
        Perform an action in the environment.
        :param action: Action to apply into the environment.
        """
        next_obs, reward, terminated, truncated = self.env.step(action)
        self.result = (next_obs, reward, terminated, truncated)

    def receive(self):
        """
        Return the information received from the environment.
        :return: Tuple of environment data.
        return only a torch.tensor if recive() is called after reset()
        otherwise it return -> tuple[torch.tensor, int, bool, bool]
        """
        return self.result
    

class VectorEnv():
    """
    Pool of enviroment
    """
    def __init__(self, env: list):

        self.envs = env
        self.workers = [EnvWorker(i) for i in env]

    def __len__(self) -> int:
        """
        Return the number of environments.
        :return: The number of the environments.
        """
        return len(self.envs)
    
    def reset(self, indices=None) -> np.array:
        """
        Reset the all or the specified environments.
        :param indices: Multiple environment indices must be given as an iterable. A single
        environment index can also be provided as a scalar. Passing None means all the environments. Default to None.
        :param kwargs: Environment keyword arguments.
        :return: Stacked environment observations.
        """
        # Get an iterable of environment indices
        if indices is None:
            # Every environment
            workers = list(range(len(self.envs)))
        elif np.isscalar(indices):
            # A single environment index is given as input
            # Convert to a list with a single index
            workers = [indices]
        else:
            # The indices are already given as an iterable, e.g. list, numpy array
            workers = indices

        for index in workers:
            # Reset the single environment
            self.workers[index].reset()
        # Stack every received observation by the EnvWorkers
        obses = [self.workers[index].receive() for index in workers]            # List of (#index) array (type = [torch.tensor,])
        
        print(obses)
        print("")
        print(type(obses))
        return np.stack(obses)                                                  # Matrix (#index)_rows x (state_size)_col
    
    def step(self, actions, indices=None):
        """
        Perform actions into all or the specified environments.
        :param actions: List of actions to perform.
        :param indices: Multiple environment indices must be given as an iterable. A single
        environment index can also be provided as a scalar. Passing None means all the environments. Default to None.
        :return: Stacked environment responses.
        """

        if indices is None:
            # Every environment
            workers = list(range(len(self.envs)))
        elif np.isscalar(indices):
            # A single environment index is given as input
            # Convert to a list with a single index
            workers = [indices]
        else:
            # The indices are already given as an iterable, e.g. list, numpy array
            workers = indices

        # Perform a step into every EnvWorker
        for i, j in enumerate(workers):
            self.workers[j].send(actions[i])

        # Prepare the results
        result = []
        for j in workers:
            # Get environment step return
            env_return = self.workers[j].receive()
            result.append(env_return)

        # Unpack result        
        next_obses, rewards, terms, truncs = tuple(zip(*result))
        
        
        
        return (
            np.stack(next_obses),
            np.stack(rewards),
            np.stack(terms),
            np.stack(truncs)) 
