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

class TransitionLikelihoodFunction:
    def __init__(self, selfIndex, otherIndex, selfActionSpace, otherActionSpace, transite):
        self.selfIndex = selfIndex
        self.otherIndex = otherIndex
        self.selfActionSpace = selfActionSpace
        self.otherActionSpace = otherActionSpace
        self.transite = transite

    def __call__(self, state, action, nextState):
        hypothesisNextState = self.transite(state, action)
        hypothesisSelfNextState, hypothesisOtherNextState = hypothesisNextState[self.selfIndex], hypothesisNextState[self.otherIndex] 
        selfState = state[self.selfIndex]
        otherState = state[self.otherIndex]
        possibleSelfNextStates = [self.transite(selfState, selfAction) for selfAction in self.selfActionSpace]
        possibleOtherNextStates = [self.transite(otherState, otherAction) for otherAction in self.otherActionSpace]
        minDistanceOfPossibleSelfNextState = min([np.linalg.norm(np.array(possibleSelfNextState).flatten() - nextState[self.selfIndex])
                for possibleSelfNextState in possibleSelfNextStates])
        minDistanceOfPossibleOtherNextState = min([np.linalg.norm(np.array(possibleOtherNextState).flatten() - nextState[self.otherIndex])
                for possibleOtherNextState in possibleOtherNextStates])
        realDistanceOfSelfNextState = np.linalg.norm(hypothesisSelfNextState - nextState[self.selfIndex])
        realDistanceOfOtherNextState = np.linalg.norm(hypothesisOtherNextState - nextState[self.otherIndex])
        if np.allclose(minDistanceOfPossibleSelfNextState, realDistanceOfSelfNextState) and np.allclose(minDistanceOfPossibleOtherNextState, realDistanceOfOtherNextState):
            return 1
        else:
            return 0

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
        numActionSpaceForAll = parameters['numActionSpaceForAll']
        maxRunningSteps = parameters['maxRunningSteps']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(numActionSpaceForAll)
        resetPolicy = self.composeResetPolicy(individualPolicies)
        sampleTrajectory = self.composeSampleTrajectory(maxRunningSteps, resetPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numActionSpaceForAll'] = [2, 3, 5, 9]
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
    possibleActionSpace = {2: [(10, 0), (-10, 0)], 3: [(10, 0), (-10, 0), (0, 0)], 
            5: [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)], 9: [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]}
    predatorPowerRatio = 2
    getWolfIndividualActionSpace = lambda numActionSpaceForAll: list(map(tuple, np.array(possibleActionSpace[numActionSpaceForAll]) * predatorPowerRatio))
    getWolfCentralControlActionSpace = lambda numActionSpaceForAll: list(it.product(getWolfIndividualActionSpace(numActionSpaceForAll), getWolfIndividualActionSpace(numActionSpaceForAll)))
    getNumWolvesActionSpace = lambda numActionSpaceForAll: len(getWolfCentralControlActionSpace(numActionSpaceForAll))
    regularizationFactor = 1e-4
    composeGenerateWolfCentralControlModel = lambda numActionSpaceForAll: GenerateModel(numStateSpace, getNumWolvesActionSpace(numActionSpaceForAll), regularizationFactor)
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    wolfNNDepth = 9
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    getInitWolfCentralControlModel = lambda numActionSpaceForAll: composeGenerateWolfCentralControlModel(numActionSpaceForAll)(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    getWolfModelPath = lambda numActionSpaceForAll: os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=' + str(numActionSpaceForAll) + str(numActionSpaceForAll)+'_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    getWolfCentralControlNNModel = lambda numActionSpaceForAll: restoreVariables(getInitWolfCentralControlModel(numActionSpaceForAll), getWolfModelPath(numActionSpaceForAll))
    wolfCentralControlNNModels = {numActionSpaceForAll: getWolfCentralControlNNModel(numActionSpaceForAll) 
            for numActionSpaceForAll in manipulatedVariables['numActionSpaceForAll']}
    getWolfCentralControlPolicyGivenIntention = lambda numActionSpaceForAll: ApproximatePolicy(wolfCentralControlNNModels[numActionSpaceForAll],
            getWolfCentralControlActionSpace(numActionSpaceForAll))

    softParameterInInference = 1
    softPolicyInInference = SoftPolicy(softParameterInInference)
    composeSoftenWolfCentralControlPolicyGivenIntentionInInference = lambda numActionSpaceForAll: lambda state: softPolicyInInference(
            getWolfCentralControlPolicyGivenIntention(numActionSpaceForAll)(state))
    
    imaginedWeIdsForInferenceSubjects = [[2, 3], [3, 2]]
    getStateForPolicyGivenIntentionInInferences = [GetStateForPolicyGivenIntention(imaginedWeIds) 
            for imaginedWeIds in imaginedWeIdsForInferenceSubjects] 
    getCalPoliciesLikelihood = lambda numActionSpaceForAll: [CalPolicyLikelihood(getState,
        composeSoftenWolfCentralControlPolicyGivenIntentionInInference(numActionSpaceForAll)) 
            for getState in getStateForPolicyGivenIntentionInInferences]

    # Transition Likelihood
    composeGetOwnState = lambda imaginedWeId: lambda state: np.array(state)[imaginedWeId]
    getOwnStates = [composeGetOwnState(imaginedWeId) for imaginedWeId in imaginedWeIdsForInferenceSubjects]
    perceptNoise = 1e-1
    selfIndex = 0
    otherIndex = 1
    composeTransitionLikelihoodFunction = lambda numActionSpaceForAll: \
        TransitionLikelihoodFunction(selfIndex, otherIndex, list(it.product(getWolfIndividualActionSpace(numActionSpaceForAll))), 
                list(it.product(getWolfIndividualActionSpace(numActionSpaceForAll))), transit)

    composeCalTransitionsLikelihood = lambda numActionSpaceForAll: [CalTransitionLikelihood(getOwnState, 
        composeTransitionLikelihoodFunction(numActionSpaceForAll)) for getOwnState in getOwnStates]
    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calTransitionLikelihood: lambda intention, state, action, nextState: \
        calPolicyLikelihood(intention, state, action) * calTransitionLikelihood(state, action, nextState)
    getCalJointsLikelihood = lambda numActionSpaceForAll: [composeCalJointLikelihood(calPolicyLikelihood, calTransitionLikelihood) 
        for calPolicyLikelihood, calTransitionLikelihood in zip(getCalPoliciesLikelihood(numActionSpaceForAll), composeCalTransitionsLikelihood(numActionSpaceForAll))]

    # Joint Hypothesis Space
    priorDecayRate = 1
    intentionSpace = [(0,), (1,)]
    getActionSpaceInInference = lambda numActionSpaceForAll: getWolfCentralControlActionSpace(numActionSpaceForAll)
    getVariables = lambda numActionSpaceForAll: [intentionSpace, getActionSpaceInInference(numActionSpaceForAll)]
    getJointHypothesisSpace = lambda numActionSpaceForAll: pd.MultiIndex.from_product(getVariables(numActionSpaceForAll), names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composeWolvesInferImaginedWe = lambda numActionSpaceForAll: [InferOneStep(priorDecayRate, getJointHypothesisSpace(numActionSpaceForAll),
            concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointsLikelihood(numActionSpaceForAll)]
    getUpdateIntention = lambda numActionSpaceForAll: [sheepUpdateIntentionMethod, sheepUpdateIntentionMethod] + composeWolvesInferImaginedWe(numActionSpaceForAll) 
    chooseIntention = sampleFromDistribution

    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [3, 2]]
    getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in imaginedWeIdsForAllAgents]

    #NN Policy Given Intention
    numStateSpace = 6
    preyPowerRatio = 2.5
    actionSpacePlanning = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    sheepIndividualActionSpace = list(map(tuple, np.array(actionSpacePlanning) * preyPowerRatio))
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

    #planning
    
    #wolfIndividualActionSpacePlanning = list(map(tuple, np.array(actionSpacePlanning) * predatorPowerRatio))
    #wolfCentralControlActionSpacePlanning = list(it.product(wolfIndividualActionSpacePlanning, wolfIndividualActionSpacePlanning))
    #numWolvesActionSpacePlanning = len(wolfCentralControlActionSpacePlanning)
    #generateWolfCentralControlModelPlanning = GenerateModel(numStateSpace, numWolvesActionSpacePlanning, regularizationFactor)
    #initWolfCentralControlModelPlanning = generateWolfCentralControlModelPlanning(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
    #        resBlockSize, initializationMethod, dropoutRate)
    #wolfModelPathPlanning = os.path.join('..', '..', 'data', 'preTrainModel', 
    #        'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    #wolfCentralControlNNModelPlanning = restoreVariables(initWolfCentralControlModelPlanning, wolfModelPathPlanning)
    #wolfCentralControlPolicyGivenIntentionPlanning = ApproximatePolicy(wolfCentralControlNNModelPlanning, wolfCentralControlActionSpacePlanning) 
    
    wolf1IndividualActionSpace = list(map(tuple, np.array(actionSpacePlanning) * predatorPowerRatio))
    getWolf2IndividualActionSpace = lambda numActionSpaceForAll: list(map(tuple, np.array(possibleActionSpace[numActionSpaceForAll]) * predatorPowerRatio))
    getWolfCentralControlActionSpace = lambda numActionSpaceForAll: list(it.product(wolf1IndividualActionSpace, getWolf2IndividualActionSpace(numActionSpaceForAll)))
    getNumWolvesActionSpace = lambda numActionSpaceForAll: len(getWolfCentralControlActionSpace(numActionSpaceForAll))
    regularizationFactor = 1e-4
    composeGenerateWolfCentralControlModel = lambda numActionSpaceForAll: GenerateModel(numStateSpace, getNumWolvesActionSpace(numActionSpaceForAll), regularizationFactor)
    getInitWolfCentralControlModel = lambda numActionSpaceForAll: composeGenerateWolfCentralControlModel(numActionSpaceForAll)(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths, 
            resBlockSize, initializationMethod, dropoutRate)
    getWolfModelPath = lambda numActionSpaceForAll: os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=9'+str(numActionSpaceForAll)+'_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    getWolfCentralControlNNModel = lambda numActionSpaceForAll: restoreVariables(getInitWolfCentralControlModel(numActionSpaceForAll), getWolfModelPath(numActionSpaceForAll))
    wolfCentralControlNNModels = {numActionSpaceForAll: getWolfCentralControlNNModel(numActionSpaceForAll) 
            for numActionSpaceForAll in manipulatedVariables['numActionSpaceForAll']}
    getWolfCentralControlPolicyGivenIntention = lambda numActionSpaceForAll: ApproximatePolicy(wolfCentralControlNNModels[numActionSpaceForAll],
            getWolfCentralControlActionSpace(numActionSpaceForAll))

    
    #imagined we infer and plan
    softParameterInPlanning = 2.5
    softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
    #getSoftenWolfCentralControlPolicyGivenIntentionInPlanning = lambda numActionSpaceForAll: lambda state: softPolicyInPlanning(
    #        wolfCentralControlPolicyGivenIntentionPlanning(state))
    getSoftenWolfCentralControlPolicyGivenIntentionInPlanning = lambda numActionSpaceForAll: lambda state: softPolicyInPlanning(
            getWolfCentralControlPolicyGivenIntention(numActionSpaceForAll)(state))
    getCentralControlPoliciesGivenIntentions = lambda numActionSpaceForAll: [sheepCentralControlPolicyGivenIntention, sheepCentralControlPolicyGivenIntention,
            getSoftenWolfCentralControlPolicyGivenIntentionInPlanning(numActionSpaceForAll), getSoftenWolfCentralControlPolicyGivenIntentionInPlanning(numActionSpaceForAll)]
    composeIndividualPoliciesByEvaParameters = lambda numActionSpaceForAll: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention
            in zip(imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, getUpdateIntention(numActionSpaceForAll), 
                getCentralControlPoliciesGivenIntentions(numActionSpaceForAll))]
    
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
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithSmallActionSpaceForAll',
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
    numColumns = 1#len(manipulatedVariables['numActionSpaceForAll'])
    numRows = len(manipulatedVariables['maxRunningSteps'])
    plotCounter = 1

    for maxRunningSteps, group in statisticsDf.groupby('maxRunningSteps'):
        group.index = group.index.droplevel('maxRunningSteps')
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        group.index.name = 'Set Size of Other\'s Action Space'
        #if plotCounter % numColumns == 0 :
        axForDraw.set_ylabel('Accumulated Reward')
        group.plot.line(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (-0.5, 0.5), xlim = (1.8, 9.2), marker = 'o', rot = 0)
        #for numActionSpaceForAll, grp in group.groupby('numActionSpaceForAll'):
            #if plotCounter <= numColumns:
            #    axForDraw.set_title('numActionSpaceForAll = {}'.format(numActionSpaceForAll))
            #df = pd.DataFrame(grp.values[0].tolist(), columns = possiblePreyIds, index = ['mean','se']).T
            #df = grp
            #__import__('ipdb').set_trace()
            #df.plot.bar(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (-0.5, 1))
        plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Accumulated Reward')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
