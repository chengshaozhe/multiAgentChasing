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
        numActionSpaceForOthers = parameters['numActionSpaceForOthers']
        maxRunningSteps = parameters['maxRunningSteps']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(numActionSpaceForOthers)
        resetPolicy = self.composeResetPolicy(individualPolicies)
        sampleTrajectory = self.composeSampleTrajectory(maxRunningSteps, resetPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numActionSpaceForOthers'] = [2, 3, 5]
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
    actionSpace1 = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    possibleActionSpace2 = {2: [(0, 10), (0, -10)], 3: [(0, 10), (0, -10), (0, 0)], 
            5: [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]}
    predatorPowerRatio = 2
    wolf1IndividualActionSpace = list(map(tuple, np.array(actionSpace1) * predatorPowerRatio))
    getWolf2IndividualActionSpace = lambda numActionSpaceForOthers: list(map(tuple, np.array(possibleActionSpace2[numActionSpaceForOthers]) * predatorPowerRatio))
    getWolfCentralControlActionSpace = lambda numActionSpaceForOthers: list(it.product(wolf1IndividualActionSpace, getWolf2IndividualActionSpace(numActionSpaceForOthers)))
    getNumWolvesActionSpace = lambda numActionSpaceForOthers: len(getWolfCentralControlActionSpace(numActionSpaceForOthers))
    regularizationFactor = 1e-4
    composeGenerateWolfCentralControlModel = lambda numActionSpaceForOthers: GenerateModel(numStateSpace, getNumWolvesActionSpace(numActionSpaceForOthers), regularizationFactor)
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    wolfNNDepth = 9
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    getInitWolfCentralControlModel = lambda numActionSpaceForOthers: composeGenerateWolfCentralControlModel(numActionSpaceForOthers)(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    getWolfModelPath = lambda numActionSpaceForOthers: os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=9'+str(numActionSpaceForOthers)+'_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    getWolfCentralControlNNModel = lambda numActionSpaceForOthers: restoreVariables(getInitWolfCentralControlModel(numActionSpaceForOthers), getWolfModelPath(numActionSpaceForOthers))
    wolfCentralControlNNModels = {numActionSpaceForOthers: getWolfCentralControlNNModel(numActionSpaceForOthers) 
            for numActionSpaceForOthers in manipulatedVariables['numActionSpaceForOthers']}
    getWolfCentralControlPolicyGivenIntention = lambda numActionSpaceForOthers: ApproximatePolicy(wolfCentralControlNNModels[numActionSpaceForOthers],
            getWolfCentralControlActionSpace(numActionSpaceForOthers))

    softParameterInInference = 1
    softPolicyInInference = SoftPolicy(softParameterInInference)
    composeSoftenWolfCentralControlPolicyGivenIntentionInInference = lambda numActionSpaceForOthers: lambda state: softPolicyInInference(
            getWolfCentralControlPolicyGivenIntention(numActionSpaceForOthers)(state))
    
    imaginedWeIdsForInferenceSubjects = [[2, 3], [3, 2]]
    getStateForPolicyGivenIntentionInInferences = [GetStateForPolicyGivenIntention(imaginedWeIds) 
            for imaginedWeIds in imaginedWeIdsForInferenceSubjects] 
    getCalPoliciesLikelihood = lambda numActionSpaceForOthers: [CalPolicyLikelihood(getState,
        composeSoftenWolfCentralControlPolicyGivenIntentionInInference(numActionSpaceForOthers)) 
            for getState in getStateForPolicyGivenIntentionInInferences]

    # Transition Likelihood
    composeGetOwnState = lambda imaginedWeId: lambda state: np.array(state)[imaginedWeId]
    getOwnStates = [composeGetOwnState(imaginedWeId) for imaginedWeId in imaginedWeIdsForInferenceSubjects]
    perceptNoise = 4e1
    percept = lambda hypothesisNextState, nextState: scipy.stats.multivariate_normal.pdf(
            hypothesisNextState[0], np.array(nextState)[0], np.diag([1e-1**2] * len(nextState[0]))) * scipy.stats.multivariate_normal.pdf(
                    hypothesisNextState[1], np.array(nextState)[1], np.diag([perceptNoise**2] * len(nextState[1])))
    calTransitionsLikelihood = [CalTransitionLikelihood(getOwnState, transit, percept) for getOwnState in getOwnStates]
    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calTransitionLikelihood: lambda intention, state, action, nextState: \
        calPolicyLikelihood(intention, state, action) * calTransitionLikelihood(state, action, nextState)
    getCalJointsLikelihood = lambda numActionSpaceForOthers: [composeCalJointLikelihood(calPolicyLikelihood, calTransitionLikelihood) 
        for calPolicyLikelihood, calTransitionLikelihood in zip(getCalPoliciesLikelihood(numActionSpaceForOthers), calTransitionsLikelihood)]

    # Joint Hypothesis Space
    priorDecayRate = 1
    intentionSpace = [(0,), (1,)]
    getActionSpaceInInference = lambda numActionSpaceForOthers: getWolfCentralControlActionSpace(numActionSpaceForOthers)
    getVariables = lambda numActionSpaceForOthers: [intentionSpace, getActionSpaceInInference(numActionSpaceForOthers)]
    getJointHypothesisSpace = lambda numActionSpaceForOthers: pd.MultiIndex.from_product(getVariables(numActionSpaceForOthers), names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composeWolvesInferImaginedWe = lambda numActionSpaceForOthers: [InferOneStep(priorDecayRate, getJointHypothesisSpace(numActionSpaceForOthers),
            concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointsLikelihood(numActionSpaceForOthers)]
    getUpdateIntention = lambda numActionSpaceForOthers: [sheepUpdateIntentionMethod, sheepUpdateIntentionMethod] + composeWolvesInferImaginedWe(numActionSpaceForOthers) 
    chooseIntention = sampleFromDistribution

    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [3, 2]]
    getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in imaginedWeIdsForAllAgents]

    #NN Policy Given Intention
    numStateSpace = 6
    preyPowerRatio = 2.5
    sheepIndividualActionSpace = list(map(tuple, np.array(actionSpace1) * preyPowerRatio))
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
    getSoftenWolfCentralControlPolicyGivenIntentionInPlanning = lambda numActionSpaceForOthers: lambda state: softPolicyInPlanning(
            getWolfCentralControlPolicyGivenIntention(numActionSpaceForOthers)(state))
    getCentralControlPoliciesGivenIntentions = lambda numActionSpaceForOthers: [sheepCentralControlPolicyGivenIntention, sheepCentralControlPolicyGivenIntention,
            getSoftenWolfCentralControlPolicyGivenIntentionInPlanning(numActionSpaceForOthers), getSoftenWolfCentralControlPolicyGivenIntentionInPlanning(numActionSpaceForOthers)]
    composeIndividualPoliciesByEvaParameters = lambda numActionSpaceForOthers: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention
            in zip(imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, getUpdateIntention(numActionSpaceForOthers), 
                getCentralControlPoliciesGivenIntentions(numActionSpaceForOthers))]
    
    individualIdsForAllAgents = [0, 1, 2, 3]
    chooseCentrolAction = [sampleFromDistribution]* 2 + [sampleFromDistribution]* 2
    assignIndividualActionMethods = [AssignCentralControlToIndividual(imaginedWeId, individualId, chooseAction) 
            for imaginedWeId, individualId, chooseAction in zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents, chooseCentrolAction)]

    policiesResetAttributes = ['lastState', 'intentionPrior', 'formerIntentionPriors']
    policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [None, intentionPrior, [intentionPrior]])) for intentionPrior in imaginedWeIntentionPriors]
    returnAttributes = ['formerIntentionPriors']
    composeResetPolicy = lambda individualPolicies: ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
    # Sample and Save Trajectory
    composeSampleTrajectory = lambda maxRunningSteps, resetPolicy: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, assignIndividualActionMethods, resetPolicy)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithSmallActionSpaceForOthers',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'sampleNNPolicy', 'wolfPolicy':'NNPolicy',
        'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample', 'perceptNoise': perceptNoise}
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
    numColumns = 1#len(manipulatedVariables['numActionSpaceForOthers'])
    numRows = len(manipulatedVariables['maxRunningSteps'])
    plotCounter = 1

    for maxRunningSteps, group in statisticsDf.groupby('maxRunningSteps'):
        group.index = group.index.droplevel('maxRunningSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        if plotCounter % numColumns == 0 :
            axForDraw.set_ylabel('maxRunningSteps = {}'.format(maxRunningSteps))
            group.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (0, 0.5), marker = 'o')
        #for numActionSpaceForOthers, grp in group.groupby('numActionSpaceForOthers'):
            #if plotCounter <= numColumns:
            #    axForDraw.set_title('numActionSpaceForOthers = {}'.format(numActionSpaceForOthers))
            #df = pd.DataFrame(grp.values[0].tolist(), columns = possiblePreyIds, index = ['mean','se']).T
            #df = grp
            #__import__('ipdb').set_trace()
            df.plot.bar(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (-0.5, 1))
            plotCounter = plotCounter + 1

    plt.suptitle('Wolves Intention Archeivement Moving Sheeps')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
