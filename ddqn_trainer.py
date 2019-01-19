
import os.path
import numpy as np
import gym

from active_memory import ActiveMemory
from batch_helper import BatchHelper
from ddqn_agent import DdqnAgent
from ddqn_globals import DdqnGlobals
from gif_saver import GifSaver
from DataModel.progress_tracker import ProgressTracker, ProgressTrackerParms
from hyper_parameters import StandardAgentParameters, ShortEpisodeParameters, LongEpisodeParameters

'''
 Based on agents from rlcode, keon, A.L.Ecoffet, and probably several others
'''

class ImagePreProcessor:
    def to_grayscale(img):
        return np.mean(img, axis=2).astype(np.uint8)
    
    def downsample(img):
        return img[::2, ::2]
    
    def Preprocess(img):
        shrunk = ImagePreProcessor.downsample(img)
        return ImagePreProcessor.to_grayscale(shrunk)

class EpisodeManager:
    def __init__(self, environment, memory, action_size, miscParameters):
        self._environment = environment
        self._memory = memory
        batchHelper = BatchHelper(memory, miscParameters.batchSize, action_size)
        self.progressTracker = ProgressTracker(
                ProgressTrackerParms(avgPerXEpisodes=10, longAvgPerXEpisodes=100))

        self._agent = DdqnAgent(StandardAgentParameters, 
                               action_size, 
                               batchHelper, 
                               self.progressTracker)
        self._gifSaver = GifSaver(memory, 
                                  self._agent, 
                                  save_every_x_episodes=miscParameters.createGifEveryXEpisodes)
        
    def ShouldStop(self):
        return os.path.isfile("StopTraining.txt")
        
    def Run(self):
        while(self.ShouldStop() == False):
            self.progressTracker.OnEpisodeStart()
            score, steps = self.RunOneEpisode()
            self.progressTracker.OnEpisodeOver(score, steps)
            self._agent.OnGameOver(steps)
            self._gifSaver.OnEpisodeOver()
        self._agent.OnExit()

    def OnNextEpisode(self):
        self._environment.reset()
        info = None
        for _ in range(np.random.randint(DdqnGlobals.FRAMES_PER_STATE, DdqnGlobals.MAX_NOOP)):
            frame, _, done, info = self.NextStep(self._agent.GetNoOpAction())
            self._memory.AddFrame(frame)
        return info
            
    def NextStep(self, action):
        rawFrame, reward, done, info = self._environment.step(action)
        processedFrame = ImagePreProcessor.Preprocess(rawFrame)
        return processedFrame, reward, done, info
            
    def RunOneEpisode(self):
        info = self.OnNextEpisode()
        done = False
        stepCount = 0
        score = 0
        livesLeft = info['ale.lives']
        while not done:            
            action = self._agent.GetAction()
            frame, reward, done, info = self.NextStep(action)
            score += reward
            if(info['ale.lives'] < livesLeft):
                reward = -1
                livesLeft = info['ale.lives']
            self._memory.AddMemory(frame, action, reward, done)
            self._agent.Replay()
            stepCount += 1
        return score, stepCount
             
class Trainer:
    def Run(self, whichGame, miscParams):
        env = gym.make(whichGame)
        print(env.unwrapped.get_action_meanings())
        memory = ActiveMemory()
        num_actions = env.action_space.n
        if('Pong' in whichGame):
            num_actions = 4  # Don't need RIGHTFIRE or LEFTFIRE (do we?)
        mgr = EpisodeManager(env, memory, action_size = num_actions, miscParameters=miscParams)
        mgr.Run()

def Main(whichGame, miscParams):
    trainer = Trainer()
    trainer.Run(whichGame, miscParams)

#cProfile.run("Main('PongDeterministic-v4', )", "profilingResults.cprof")
#Main('PongDeterministic-v4', LongEpisodeParameters)
Main('BreakoutDeterministic-v4', ShortEpisodeParameters)

