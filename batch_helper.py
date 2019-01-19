
import numpy as np

class BatchHelper:
    def __init__(self, memory, batchSize, actionSize):
        self.Memory = memory
        self.BatchSize = batchSize
        self.ActionSize = actionSize

    def GetFrames(self, stateId):
        return  self.Memory.GetFramesForState(stateId)

    def GetCurrentState(self):
        singletonBatch = self.Memory.GetCurrentState()
        return np.reshape(singletonBatch, (1,) + singletonBatch.shape)

    def GetBatch(self):
        stateIndexes = np.random.randint(low=0, high=len(self.Memory.Memories), size=self.BatchSize)
        startStateList = []
        nextStateList = []
        actionList = []
        rewards = []
        gameOvers = []
        
        for i in stateIndexes:
            curMemory = self.Memory.GetMemory(i)
            startStateList.append(self.GetFrames(curMemory.FirstFrameId))
            nextStateList.append(self.GetFrames(curMemory.FirstFrameId+1))
            actionList.append(curMemory.Action)
            rewards.append(curMemory.Reward)
            gameOvers.append(curMemory.EpisodeOver)
            
        startStates = np.stack(startStateList, axis=0)
        nextStates = np.stack(nextStateList, axis=0)
        actions = np.eye(self.ActionSize)[np.array(actionList)]
        return startStates, nextStates, np.array(actions), np.array(rewards), np.array(gameOvers)

