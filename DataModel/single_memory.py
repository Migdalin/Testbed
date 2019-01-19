

class SingleMemory():
    def __init__(self, FirstFrameId, Action, Reward, EpisodeOver):
        self.FirstFrameId = FirstFrameId
        self.Action = Action
        self.Reward = Reward
        self.EpisodeOver = EpisodeOver
