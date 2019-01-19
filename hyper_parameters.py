


from ddqn_globals import DdqnGlobals 


class AgentParameters:
    def __init__(self,
                 state_size,
                 epsilon_start, 
                 epsilon_min, 
                 epsilon_decay_step,
                 delayTraining,
                 update_target_rate,
                 gamma,
                 learning_rate,
                 fit_frequency):
        self.state_size = state_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_step = epsilon_decay_step
        self.delayTraining = delayTraining
        self.update_target_rate = update_target_rate
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.fit_frequency = fit_frequency
        
StandardAgentParameters = AgentParameters(
        state_size = DdqnGlobals.STATE_DIMENSIONS,
        epsilon_start = 1.0,
        epsilon_min = 0.05,
        epsilon_decay_step = 0.000002,
        delayTraining = 20000,
        update_target_rate = 10000,
        gamma = 0.99,
        learning_rate = 0.00025,
        fit_frequency = 10
        )

class MiscParameters:
    def __init__(self, 
                 createGifEveryXEpisodes, 
                 batchSize):
        self.createGifEveryXEpisodes = createGifEveryXEpisodes
        self.batchSize = batchSize


ShortEpisodeParameters = MiscParameters(createGifEveryXEpisodes=500, batchSize = 128)
LongEpisodeParameters = MiscParameters(createGifEveryXEpisodes=100, batchSize = 128)


