
from collections import namedtuple
import numpy as np
from time import time
import random
import tensorflow as tf
from tensorflow.layers import Dense

from ddqn_globals import DdqnGlobals



'''
 Based on agents from rlcode, keon, A.L.Ecoffet, and probably several others
'''

ModelInfo = namedtuple('ModelInfo', 
                       'modelName, frames, actionsMask, filteredOutput, targetQ, cost, optimizer')
ModelInfo.__new__.__defaults__ = (None,) * len(ModelInfo._fields)

ConvArgs = namedtuple('ConvArgs',
                      'layerInput, numFilters, filterSize, stride, init, namePrefix')
ConvArgs.__new__.__defaults__ = (None,) * len(ConvArgs._fields)


class DdqnAgent():
    def __init__(self, agentParams, action_size, batchHelper, progressTracker):
        self.session = tf.Session()
        self.Params = agentParams
        self.BatchHelper = batchHelper
        self.SetDefaultParameters(action_size)
        self.progressTracker = progressTracker
        self.trainingModel = self.BuildModel('online')
        self.targetModel = self.BuildModel('target')
        self.InitStatsWriter()
        #self.LoadModelInfo()
        self.session.run(tf.global_variables_initializer())
        self.UpdateTargetModel()
    
    def SetDefaultParameters(self, action_size):
        self.action_size = action_size
        self.epsilon = self.Params.epsilon_start
        self.current_step_count = 0
        self.total_step_count = 0
        self.total_episodes = 0

    def InitStatsWriter(self):        
        self.statsWriter = tf.summary.FileWriter(f"tensorboard/{int(time())}")
        tf.summary.scalar("Loss", self.trainingModel.cost)
        self.writeStatsOp = tf.summary.merge_all()
        self.next_summary_checkpoint = self.Params.delayTraining
        self.SaveWeightsFilename = "DdqnWeights.h5"
    
    def BuildConv2D(self, convArgs, modelName):
        with tf.variable_scope(modelName):
            channelAxis = 3
            filterShape = [convArgs.filterSize, 
                           convArgs.filterSize, 
                           convArgs.layerInput.get_shape()[channelAxis], 
                           convArgs.numFilters]
    
            filters = tf.get_variable(shape=filterShape, 
                                      dtype=tf.float32,
                                      initializer=convArgs.init,
                                      name=convArgs.namePrefix + 'filters')
            
            conv = tf.nn.conv2d(input=convArgs.layerInput,
                                filter=filters, 
                                strides=[1,convArgs.stride,convArgs.stride,1], 
                                padding='VALID')
    
            activated = tf.nn.relu(conv)
            return activated
        
    
    def BuildModel(self, modelName):
        with tf.variable_scope(modelName):
            kernelInit = tf.contrib.layers.xavier_initializer()
        
            frames = tf.placeholder(dtype=tf.float32, 
                                    shape=(None,) + DdqnGlobals.STATE_DIMENSIONS, 
                                    name=modelName+'frames')
        
            conv_1 = self.BuildConv2D(
                    ConvArgs(layerInput = frames, 
                             numFilters = 32,
                             filterSize = 8,
                             stride = 4,
                             init = kernelInit,
                             namePrefix=modelName+'c1'),
                             modelName)
               
            conv_2 = self.BuildConv2D(
                    ConvArgs(layerInput = conv_1,
                             numFilters = 64, 
                             filterSize = 4, 
                             stride = 2,
                             init = kernelInit,
                             namePrefix=modelName+'c2'),
                             modelName)
            
            conv_3 = self.BuildConv2D(
                    ConvArgs(layerInput = conv_2,
                             numFilters = 64,
                             filterSize = 3,
                             stride = 1,
                             init = kernelInit,
                             namePrefix=modelName+'c3'),
                             modelName)
    
            conv_flattened = tf.layers.Flatten()(conv_3)
            
            #  Split into dueling networks
            xavierInit = tf.contrib.layers.xavier_initializer()
            
            #ADVANTAGE
            advantageInput = Dense(units = 512, 
                                   activation='relu', 
                                   kernel_initializer=kernelInit,
                                   name=modelName+'advantageInput')(conv_flattened)

            advantage = Dense(self.action_size, 
                              activation='relu',
                              kernel_initializer=xavierInit,
                              name=modelName+'advantage')(advantageInput)

            # VALUE
            valueInput = Dense(units = 512, 
                               activation='relu', 
                               kernel_initializer=kernelInit,
                               name=modelName+'valueInput')(conv_flattened)
            
            value = Dense(1, 
                          kernel_initializer=xavierInit,
                          name=modelName+'value')(valueInput)
            
            # Rejoin into single network
            advantageDiff = tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))
            policy = advantageDiff + value
            
            # "The output layer is a fully-connected linear layer with a single output for each valid action."
            rawOutput = Dense(self.action_size, 
                              kernel_initializer=kernelInit,
                              name=modelName+'rawOutput')(policy)

            # Finally, we multiply the output by the mask!
            actionsMask = tf.placeholder(dtype=tf.float32,
                                         shape=((None, self.action_size)), 
                                         name=modelName+'actionsMask')
            filteredOutput = tf.multiply(rawOutput, actionsMask)
            
            targetQ = tf.placeholder(dtype=tf.float32,
                                     shape=((None, self.action_size)), 
                                     name=modelName+'targetQ')
            cost = tf.losses.huber_loss(targetQ, filteredOutput)
            optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.Params.learning_rate).minimize(cost)

            modelInfo = ModelInfo(modelName=modelName,
                                  frames=frames, 
                                  actionsMask=actionsMask, 
                                  filteredOutput=filteredOutput, 
                                  targetQ=targetQ, 
                                  cost=cost, 
                                  optimizer=optimizer)
        
            return modelInfo
        
    # Write TF Summaries
    def WriteStats(self, feedDict):
        summary = self.session.run(self.writeStatsOp, feed_dict=feedDict)
        self.statsWriter.add_summary(summary, self.total_step_count)

        summary = tf.Summary()
        avgEpisodes = self.progressTracker.Parms.avgPerXEpisodes
        summary.value.add(tag=f'Average Reward ({avgEpisodes} episodes)', 
                          simple_value=self.progressTracker.GetAverageReward())
        
        avgEpisodes = self.progressTracker.Parms.longAvgPerXEpisodes
        summary.value.add(tag=f'Average Reward ({avgEpisodes} episodes)', 
                          simple_value=self.progressTracker.GetLongAverageReward())
        
        summary.value.add(tag='Max Reward', simple_value=self.progressTracker.GetMaxReward())

        summary.value.add(tag='Epsilon', simple_value=self.epsilon)
        summary.value.add(tag='Learning Rate', simple_value=self.Params.learning_rate)
        summary.value.add(tag='Fit Frequency', simple_value=self.Params.fit_frequency)
        summary.value.add(tag='Update Target Rate', simple_value=self.Params.update_target_rate)
        summary.value.add(tag='Delay Training', simple_value=self.Params.delayTraining)
        summary.value.add(tag='Memory Size', simple_value=self.BatchHelper.Memory.MaxActiveMemories)
        summary.value.add(tag='Batch Size', simple_value=self.BatchHelper.BatchSize)

        self.statsWriter.add_summary(summary, self.total_step_count)

        self.statsWriter.flush()

    def Replay(self):
        if(self.total_step_count < self.Params.delayTraining):
            return
        
        if((self.total_step_count % self.Params.fit_frequency) != 0):
            return
        
        start_states, next_states, actions, rewards, gameOvers = self.BatchHelper.GetBatch()

        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.session.run(self.targetModel.filteredOutput,
                                         feed_dict={self.targetModel.frames: next_states,
                                                    self.targetModel.actionsMask: np.ones(actions.shape)})
        
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[gameOvers] = 0
        
        # The Q values of each start state is the reward + gamma * the max next state Q value
        Q_values = rewards + (self.Params.gamma * np.max(next_Q_values, axis=1))
        targetQ = actions * Q_values[:,None]
        
        # Fit the keras model. Note how we are passing the actions as the mask and multiplying
        # the targets by the actions.
        feedDict = {self.trainingModel.frames: start_states,
                    self.trainingModel.actionsMask: actions,
                    self.trainingModel.targetQ: targetQ}
        
        self.session.run([self.trainingModel.optimizer, self.trainingModel.cost],
                         feed_dict=feedDict)
        
        if(self.total_step_count > self.next_summary_checkpoint):
            self.WriteStats(feedDict)
            self.next_summary_checkpoint = self.Params.update_target_rate + self.total_step_count
    
    # get action from model using epsilon-greedy policy
    def GetAction(self):
        self.total_step_count += 1
        self.current_step_count += 1
        
        if(self.epsilon > self.Params.epsilon_min):
            self.epsilon -= self.Params.epsilon_decay_step
        
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        curState = self.BatchHelper.GetCurrentState()
        q_value = self.session.run(self.trainingModel.filteredOutput,
                                   feed_dict = {self.trainingModel.frames: curState,
                                                self.trainingModel.actionsMask: np.ones((1,self.action_size))})
        
        return np.argmax(q_value[0])

    def UpdateTargetModelInternal(self):
        # Get the parameters of our training network
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.trainingModel.modelName)
        
        # Get the parameters of our target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.targetModel.modelName)
    
        op_holder = []
        
        # Update our target_network parameters with training_network parameters
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder
    
    def UpdateTargetModel(self):
        updateTarget = self.UpdateTargetModelInternal()
        self.session.run(updateTarget)

    def GetNoOpAction(self):
        return 0

    #    def LoadModelInfo(self):
    #        weightsFile = Path(self.SaveWeightsFilename)
    #        if(weightsFile.is_file()):
    #            self.trainingModel.load_weights(self.SaveWeightsFilename)
    #            print("*** Model Weights Loaded ***")

    #def SaveModelInfo(self):
    #    self.targetModel.save_weights(self.SaveWeightsFilename)

    def UpdateAndSave(self):
        self.UpdateTargetModel()
        self.current_step_count = 0
        #self.SaveModelInfo()

    def OnExit(self):
        self.UpdateAndSave()

    def OnGameOver(self, steps):
        self.total_episodes += 1
        if(self.current_step_count >= self.Params.update_target_rate):
            self.UpdateAndSave()
