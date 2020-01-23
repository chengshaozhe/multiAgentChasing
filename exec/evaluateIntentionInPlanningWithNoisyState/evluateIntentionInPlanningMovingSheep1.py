import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import scipy.stats 
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math 

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.inference import CalPolicyLikelihood, CalTransitionLikelihood, InferOneStep, InferOnTrajectory
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
        #successArcheivement = min(1, np.sum([self.judgeSuccessCatchOrEscape(booleanInMinDistance) for booleanInMinDistance in areDistancesInMin]))
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
        perceptNoise = parameters['perceptNoiseForAll']
        maxRunningSteps = parameters['maxRunningSteps']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(perceptNoise)
        resetPolicy = self.composeResetPolicy(individualPolicies)
        sampleTrajectory = self.composeSampleTrajectory(maxRunningSteps, resetPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['perceptNoiseForAll'] = [1e-1, 2e1, 4e1, 8e1, 1e3]
    manipulatedVariables['maxRunningSteps'] = [100]
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
    reset = Reset(xBoundary, yBoundary, numOfAgent)
    
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
    wolfImaginedWeIntentionPrior = {(0, ):0.5, (1,): 0.5}
    imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior, sheepImagindWeIntentionPrior, wolfImaginedWeIntentionPrior, wolfImaginedWeIntentionPrior]
    
    # Inference of Imagined We
    noInferIntention = lambda intentionPrior, lastState, state: intentionPrior
    sheepUpdateIntentionMethod = noInferIntention
    # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
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

    softParameterInInference = 1
    softPolicyInInference = SoftPolicy(softParameterInInference)
    softenWolfCentralControlPolicyGivenIntentionInInference = lambda state: softPolicyInInference(wolfCentralControlPolicyGivenIntention(state))
    
    imaginedWeIdsForInferenceSubjects = [[2, 3], [3, 2]]
    getStateForPolicyGivenIntentionInInferences = [GetStateForPolicyGivenIntention(imaginedWeIds) 
            for imaginedWeIds in imaginedWeIdsForInferenceSubjects] 

    calPoliciesLikelihood = [CalPolicyLikelihood(getState, softenWolfCentralControlPolicyGivenIntentionInInference) 
            for getState in getStateForPolicyGivenIntentionInInferences]

    # Transition Likelihood 
    composeGetOwnState = lambda imaginedWeId: lambda state: np.array(state)[imaginedWeId]
    getOwnStates = [composeGetOwnState(imaginedWeId) for imaginedWeId in imaginedWeIdsForInferenceSubjects]
    composeTransiteLiklihoodFunction = lambda perceptNoise : lambda state, action, nextState: scipy.stats.multivariate_normal.pdf(
            np.array(nextState)[0], transit(state, action)[0], np.diag([perceptNoise**2] * len(nextState[0]))) * scipy.stats.multivariate_normal.pdf(
                    np.array(nextState)[1], transit(state, action)[1], np.diag([perceptNoise**2] * len(nextState[1])))
    getCalTransitionsLikelihood = lambda perceptNoise: [CalTransitionLikelihood(getOwnState, composeTransiteLiklihoodFunction(perceptNoise)) for getOwnState in getOwnStates]

    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calTransitionLikelihood: lambda intention, state, action, nextState: \
        calPolicyLikelihood(intention, state, action) * calTransitionLikelihood(state, action, nextState)
    getCalJointLikelihood = lambda perceptNoise: [composeCalJointLikelihood(calPolicyLikelihood, calTransitionLikelihood) 
        for calPolicyLikelihood, calTransitionLikelihood in zip(calPoliciesLikelihood, getCalTransitionsLikelihood(perceptNoise))]

    # Joint Hypothesis Space
    priorDecayRate = 1
    intentionSpace = [(0,), (1,)]
    actionSpaceInInference = wolfCentralControlActionSpace
    variables = [intentionSpace, actionSpaceInInference]
    jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composePercept = lambda perceptNoise: lambda state: np.array([np.random.multivariate_normal(agentState, np.diag([perceptNoise**2] * len(agentState))) 
        for agentState in state]) 
    composeInferImaginedWe = lambda perceptNoise: [InferOneStep(priorDecayRate, jointHypothesisSpace,
            concernedHypothesisVariable, calJointLikelihood, composePercept(perceptNoise)) for calJointLikelihood in getCalJointLikelihood(perceptNoise)]
    getUpdateIntention = lambda perceptNoise: [sheepUpdateIntentionMethod, sheepUpdateIntentionMethod] + composeInferImaginedWe(perceptNoise)
    chooseIntention = sampleFromDistribution

    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [3, 2]]
    getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in imaginedWeIdsForAllAgents]

    #NN Policy Given Intention
    numStateSpace = 6
    preyPowerRatio = 2.5
    sheepIndividualActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
    numSheepActionSpace = len(sheepCentralControlActionSpace)
    regularizationFactor = 1e-4
    generateSheepCentralControlModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    sheepNNDepth = 5
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    initSheepCentralControlModel = generateSheepCentralControlModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    sheepModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
            'agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
    sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

    softParameterInPlanning = 2.5
    softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
    softenWolfCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfCentralControlPolicyGivenIntention(state))
    centralControlPoliciesGivenIntentions = [sheepCentralControlPolicyGivenIntention, sheepCentralControlPolicyGivenIntention,
            softenWolfCentralControlPolicyGivenIntentionInPlanning, softenWolfCentralControlPolicyGivenIntentionInPlanning]
    composeIndividualPoliciesByEvaParameters = lambda perceptNoise: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
            in zip(imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, getUpdateIntention(perceptNoise), centralControlPoliciesGivenIntentions)]

    individualIdsForAllAgents = [0, 1, 2, 3]
    chooseCentrolAction = [maxFromDistribution]* 2 + [sampleFromDistribution]* 2
    assignIndividualActionMethods = [AssignCentralControlToIndividual(imaginedWeId, individualId, chooseAction) 
            for imaginedWeId, individualId, chooseAction in zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents, chooseCentrolAction)]

    policiesResetAttributes = ['lastState', 'intentionPrior', 'formerIntentionPriors']
    policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [None, intentionPrior, [intentionPrior]])) for intentionPrior in imaginedWeIntentionPriors]
    returnAttributes = ['formerIntentionPriors']
    composeResetPolicy = lambda individualPolicies: ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
    # Sample and Save Trajectory
    composeSampleTrajectory = lambda maxRunningSteps, resetPolicy: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, assignIndividualActionMethods, resetPolicy)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNoisyState',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 200
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
    #measureIntentionArcheivement = lambda df: MeasureIntentionArcheivement(possibleIntentionIds, wolfImaginedWeId, stateIndexInTimestep, posIndexInState, killzoneRadius, judgeSuccessCatchOrEscape)
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < readParametersFromDf(df)['maxRunningSteps']) - 1 / readParametersFromDf(df)['maxRunningSteps'] * len(trajectory) 
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    fig = plt.figure()
    #numColumns = len(manipulatedVariables['perceptNoise'])
    numColumns = 1
    numRows = len(manipulatedVariables['maxRunningSteps'])
    plotCounter = 1
    
    for maxRunningSteps, group in statisticsDf.groupby('maxRunningSteps'):
        group.index = group.index.droplevel('maxRunningSteps')
        
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_ylabel('Accumulated Reward')
        group.index.name = 'Action Perception Noise'
        group.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', xlim = (-5, 1005), ylim = (-1, 1), marker = 'o', rot = 0 )
        #for perceptNoise, grp in group.groupby('perceptNoise'):
            #grp.index = grp.index.droplevel('perceptNoise')
            #if plotCounter <= numColumns:
            #    axForDraw.set_title('perceptNoise = {}'.format(perceptNoise))
            #df = pd.DataFrame(grp.values[0].tolist(), columns = possiblePreyIds, index = ['mean','se']).T
            #df = grp
        plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Accumulated Reward')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
