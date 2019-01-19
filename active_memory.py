
import numpy as np
from DataModel.single_memory import SingleMemory
from DataModel.single_frame import SingleFrame
from ddqn_globals import DdqnGlobals

class ActiveMemory:
    def __init__(self):
        self.MaxActiveMemories = 100000
        self.Frames = dict()
        self.Memories = []
        self.FramesPerState = DdqnGlobals.FRAMES_PER_STATE
        self.MaxFrameId = 0  # We add new frames based on this
        self.MinFrameId = 1  # We remove old frames based on this

    def Forget(self):
        unusedFrameId = 0
        while(len(self.Memories) > self.MaxActiveMemories):
            oldMemory = self.Memories.pop(0)
            unusedFrameId = oldMemory.FirstFrameId
        
        while(self.MinFrameId <= unusedFrameId):
            self.Frames.pop(self.MinFrameId)
            self.MinFrameId += 1
        
    def AddMemory(self, newFrameData, action, reward, gameOver):
        newFrameId = self.AddFrame(newFrameData)
        startFrameId = newFrameId - self.FramesPerState
        assert (startFrameId in self.Frames.keys()), "Frame buffer out of sync with memory state."
        theMemory = SingleMemory(int(startFrameId), int(action), int(reward), gameOver)
        self.Memories.append(theMemory)
        self.Forget()

    '''
    "Active" memories are referenced by their transient index, not by their
    persistent ID.
    '''
    def GetMemory(self, index):
        return self.Memories[index]

    '''
    Because one memory state can contain multiple frames, we need a way to add frames
    before we start adding new state memories.
    '''
    def AddFrame(self, newFrameData):
        self.MaxFrameId += 1
        theFrame = SingleFrame(self.MaxFrameId, newFrameData)
        self.Frames[theFrame.id] = theFrame
        return theFrame.id

    def Normalize(self, frames):
        normalized = np.float32(frames) / 255.
        return normalized

    def GetCurrentState(self):
        currentStateId = (max(self.Frames.keys()) - self.FramesPerState) + 1
        assert (currentStateId > 0), f"Not enough frames to define a complete state."
        return self.GetFramesForState(currentStateId)
    
    def GetFramesForState(self, stateId):
        tempTuple = ()
        for i in range(self.FramesPerState):
            frameData = self.Frames[stateId+i].Contents
            tempTuple = tempTuple + (frameData.reshape(
                    DdqnGlobals.FRAME_DIMENSIONS[0], 
                    DdqnGlobals.FRAME_DIMENSIONS[1], 
                    1),)
        
        frameStack = np.concatenate(tempTuple, axis=-1)
        return self.Normalize(frameStack)

