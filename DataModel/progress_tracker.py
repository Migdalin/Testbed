
from collections import namedtuple
from time import time
import numpy as np

ProgressTrackerParms= namedtuple(
        'ProgressTrackerParms',
        'avgPerXEpisodes, longAvgPerXEpisodes')

class ProgressTracker:
    def __init__(self, parms):
        self.Parms = parms
        self.rewardHistory = []
        self.totalEpisodes = 0
        self.totalSteps = 0
        self.startTime = 0
        self.maxReward = None
        
    def GetAverageReward(self):
        return np.mean(self.rewardHistory[-self.Parms.avgPerXEpisodes:])
        
    def GetLongAverageReward(self):
        return np.mean(self.rewardHistory)
    
    def GetMaxReward(self):
        return self.maxReward
    
    def OnEpisodeStart(self):
        self.startTime = time()
        
    def OnEpisodeOver(self, reward, steps):
        self.totalEpisodes += 1
        self.totalSteps += steps
        if(self.maxReward == None):
            self.maxReward = reward
        else:
            self.maxReward = max(self.maxReward, reward)
        
        self.rewardHistory.append(reward)
        if(len(self.rewardHistory) > self.Parms.longAvgPerXEpisodes):
            self.rewardHistory.pop(0)
        
        self.PrintEpisodeInfo(reward, steps)
        if((self.totalEpisodes % self.Parms.avgPerXEpisodes) == 0):
            self.PrintShortAverage()
            
        if((self.totalEpisodes % self.Parms.longAvgPerXEpisodes) == 0):
            self.PrintLongAverage()
        
    def PrintEpisodeInfo(self, reward, steps):
        episode = self.totalEpisodes
        elapsedTime = time() - self.startTime
        print(f"Episode: {episode};  Score: {reward};  Steps: {steps}; Time: {elapsedTime:.2f}")
        
    def PrintShortAverage(self):
        self.DoPrintAverage(self.GetAverageReward())
        
    def PrintLongAverage(self):
        self.DoPrintAverage(self.GetLongAverageReward(), "Long ")
        
    def DoPrintAverage(self, average, prefix=" "):
        print(f"******* {prefix}Average Score: {average};  Total Steps:  {self.totalSteps} ********")
        