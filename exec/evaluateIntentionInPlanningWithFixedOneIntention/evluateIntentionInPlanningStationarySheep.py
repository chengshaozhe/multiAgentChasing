import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.evaluation import ComputeStatistics

class MeasureIntentionArcheivement:
    def __init__(self, possibleIntentionIds, imaginedWeIds, stateIndex, posIndex, minDistance, judgeSuccessCatchOrEscape):
        self.possibleIntentionIds = possibleIntentionIds
        self.imaginedWeIds = imaginedWeIds
        self.stateIndex = stateIndex
        self.posIndex = posIndex
        self.minDistance = minDistance
        self.judgeSuccessCatchOrEscape = judgeSuccessCatchOrEscape

    def __call__(self, trajectory):
        lastState = np.array(trajectory[-1][self.stateIndex])
        minL2DistancesBetweenImageinedWeAndIntention = [min([np.linalg.norm(lastState[subjectIndividualId][self.posIndex] - lastState[intentionIndividualId][self.posIndex]) 
            for subjectIndividualId, intentionIndividualId in it.product(self.imaginedWeIds, intentionId)]) for intentionId in self.possibleIntentionIds]
        areDistancesInMin = [distance <= self.minDistance for distance in minL2DistancesBetweenImageinedWeAndIntention]
        successArcheivement = [self.judgeSuccessCatchOrEscape(booleanInMinDistance) for booleanInMinDistance in areDistancesInMin]
        return successArcheivement

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, composeIndividualPoliciesByEvaParameters, composeResetPolicy, composeSampleTrajectory, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.composeIndividualPoliciesByEvaParameters = composeIndividualPoliciesByEvaParameters
        self.composeResetPolicy = composeResetPolicy
        self.composeSampleTrajectory = composeSampleTrajectory
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        wolfImaginedWeFixedIntention = parameters['wolfImaginedWeFixedIntention']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(wolfImaginedWeFixedIntention)
        resetPolicy = self.composeResetPolicy(wolfImaginedWeFixedIntention, individualPolicies)
        sampleTrajectory = self.composeSampleTrajectory(resetPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['wolfImaginedWeFixedIntention'] = [(0,), (1,)]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    numOfAgent = 4
    resetState = Reset(xBoundary, yBoundary, numOfAgent)
    
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

    possiblePreyIds = [0, 1]
    possiblePredatorIds = [2, 3]
    posIndexInState = [0, 1]
    getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
    getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
    killzoneRadius = 30
    isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

    # MDP Policy
    lastState = None

    sheepImagindWeIntentionPrior = {(2, 3): 1}
    getImaginedWeIntentionPriors = lambda wolfImaginedWeFixedIntention: [sheepImagindWeIntentionPrior, sheepImagindWeIntentionPrior, 
            {wolfImaginedWeFixedIntention: 1}, {wolfImaginedWeFixedIntention: 1}]

    updateIntentionDistribution = lambda intentionPrior, lastState, state: intentionPrior
    chooseIntention = sampleFromDistribution

    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [2, 3]]
    getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in imaginedWeIdsForAllAgents]

    sheepIndividualActionSpace = [(0, 0)]
    sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
    sheepCentralControlPolicyGivenIntention = RandomPolicy(sheepCentralControlActionSpace)
    	# Wolf Centrol Control NN Policy Given Intention
    numStateSpace = 6
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    predatorPowerRatio = 2
    wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, wolfIndividualActionSpace))
    numWolvesActionSpace = len(wolfCentralControlActionSpace)
    regularizationFactor = 1e-4
    generateWolfCentralControlModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    wolfNNDepth = 9
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    initWolfCentralControlModel = generateWolfCentralControlModel(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    wolfModelPath = os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    wolfCentralControlNNModel = restoreVariables(initWolfCentralControlModel, wolfModelPath)
    wolfCentralControlPolicyGivenIntention = ApproximatePolicy(wolfCentralControlNNModel, wolfCentralControlActionSpace)
    
    softParameter = 1
    softPolicy = SoftPolicy(softParameter)
    softenWolfCentralControlPolicyGivenIntention = lambda state: softPolicy(wolfCentralControlPolicyGivenIntention(state))
    centralControlPoliciesGivenIntentions = [sheepCentralControlPolicyGivenIntention, sheepCentralControlPolicyGivenIntention,
            softenWolfCentralControlPolicyGivenIntention, softenWolfCentralControlPolicyGivenIntention]

    composeIndividualPoliciesByEvaParameters = lambda wolfImaginedWeIntentionPrior: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, policyGivenIntention 
            in zip(getImaginedWeIntentionPriors(wolfImaginedWeIntentionPrior), getStateForPolicyGivenIntentions, centralControlPoliciesGivenIntentions)]

    individualIdsForAllAgents = [0, 1, 2, 3]
    chooseCentrolAction = maxFromDistribution
    assignIndividualActionMethods = [AssignCentralControlToIndividual(imaginedWeId, individualId, chooseCentrolAction) for imaginedWeId, individualId in zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]

    policiesResetAttributes = ['lastState', 'intentionPrior']
    getPoliciesResetAttributeValues = lambda wolfImaginedWeFixedIntention: [dict(zip(policiesResetAttributes, [None, intentionPrior])) for intentionPrior in getImaginedWeIntentionPriors(wolfImaginedWeFixedIntention)]
    composeResetPolicy = lambda wolfImaginedWeFixedIntention, individualPolicies: ResetPolicy(getPoliciesResetAttributeValues(wolfImaginedWeFixedIntention), individualPolicies)
    
    #Sample and Save Trajectory
    maxRunningSteps = 101
    composeSampleTrajectory = lambda resetPolicy: SampleTrajectory(maxRunningSteps, transit, isTerminal, resetState, assignIndividualActionMethods, resetPolicy)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithFixedOneIntention',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'deterministicPrior', 'sheepPolicy':'stationaryPolicy', 'wolfPolicy':'NNPolicy',
            'updateIntention': 'False', 'maxRunningSteps': maxRunningSteps, 'policySoftParameter': softParameter, 'chooseAction': 'max'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 100
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters, 
            composeResetPolicy, composeSampleTrajectory, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    possibleIntentionIds = [[0],[1]]
    wolfImaginedWeId = [2, 3]
    stateIndexInTimestep = 0
    judgeSuccessCatchOrEscape = lambda booleanSign: int(booleanSign)
    measureIntentionArcheivement = MeasureIntentionArcheivement(possibleIntentionIds, wolfImaginedWeId, stateIndexInTimestep, posIndexInState, killzoneRadius, judgeSuccessCatchOrEscape)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['wolfImaginedWeFixedIntention'])
    numRows = 1
    plotCounter = 1

    for wolfImaginedWeFixedIntention, grp in statisticsDf.groupby('wolfImaginedWeFixedIntention'):
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        df = pd.DataFrame(grp.values[0].tolist(), columns = possiblePreyIds, index = ['mean','se']).T
        df.plot.bar(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (0, 1))
        plotCounter = plotCounter + 1

    plt.suptitle('Wolves Intention Archeivement Stationary Sheeps')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
