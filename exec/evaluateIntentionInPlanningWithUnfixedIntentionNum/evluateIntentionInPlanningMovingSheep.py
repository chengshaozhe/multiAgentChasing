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
        numIntentions = parameters['numIntentions']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(numIntentions)
        sampleTrajectory = self.composeSampleTrajectory(numIntentions, individualPolicies)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numIntentions'] = [2, 4, 8]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    getNumOfAgent = lambda numIntentions: numIntentions + 2
    composeReset = lambda numIntentions: Reset(xBoundary, yBoundary, getNumOfAgent(numIntentions))
    
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

    getPossiblePreyIds = lambda numIntentions : list(range(numIntentions))
    getPossiblePredatorIds = lambda numIntentions: [numIntentions, numIntentions + 1]
    posIndexInState = [0, 1]
    getPreyPos = lambda numIntentions: GetAgentPosFromState(getPossiblePreyIds(numIntentions), posIndexInState)
    getPredatorPos = lambda numIntentions: GetAgentPosFromState(getPossiblePredatorIds(numIntentions), posIndexInState)
    killzoneRadius = 30
    getIsTerminal = lambda numIntentions: IsTerminal(killzoneRadius, getPreyPos(numIntentions), getPredatorPos(numIntentions))

    # MDP Policy
    lastState = None
    getSheepImagindWeIntentionPrior = lambda numIntentions: {(numIntentions, numIntentions + 1): 1}
    getWolfImaginedWeIntentionPrior = lambda numIntentions: {(sheepId, ): 1/numIntentions for sheepId in range(numIntentions)}
    getImaginedWeIntentionPriors = lambda numIntentions: [getSheepImagindWeIntentionPrior(numIntentions)]* numIntentions + [getWolfImaginedWeIntentionPrior(numIntentions)] * 2
    
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
    
    getImaginedWeIdsForInferenceSubject = lambda numIntentions : [[numIntentions, numIntentions + 1], [numIntentions + 1, numIntentions]]
    composeGetStateForPolicyGivenIntentionInInference = lambda numIntentions: [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in
            getImaginedWeIdsForInferenceSubject(numIntentions)]

    composeCalPolicyLikelihood = lambda numIntentions: [CalPolicyLikelihood(getStateForPolicyGivenIntentionInInference,
            softenWolfCentralControlPolicyGivenIntentionInInference) for getStateForPolicyGivenIntentionInInference 
            in composeGetStateForPolicyGivenIntentionInInference(numIntentions)]

    # Transition Likelihood 
    composeGetOwnState = lambda imaginedWeId: lambda state: np.array(state)[imaginedWeId]
    getGetOwnStates = lambda numIntentions: [composeGetOwnState(imaginedWeId) for imaginedWeId in getImaginedWeIdsForInferenceSubject(numIntentions)]
    perceptNoise = 1e-1
    transitionLikelihoodFunction = lambda state, action, nextState: scipy.stats.multivariate_normal.pdf(
            transit(state, action)[0], np.array(nextState)[0], np.diag([1e-1**2] * len(nextState[0]))) * scipy.stats.multivariate_normal.pdf(
                    transit(state, action)[1], np.array(nextState)[1], np.diag([perceptNoise**2] * len(nextState[1])))
    composeCalTransitionLikelihood = lambda numIntentions: [CalTransitionLikelihood(getOwnState, transitionLikelihoodFunction) 
            for getOwnState in getGetOwnStates(numIntentions)]

    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calTransitionLikelihood: lambda intention, state, action, nextState: \
        calPolicyLikelihood(intention, state, action) * calTransitionLikelihood(state, action, nextState)
    getCalJointsLikelihood = lambda numIntentions: [composeCalJointLikelihood(calPolicyLikelihood, calTransitionLikelihood) 
        for calPolicyLikelihood, calTransitionLikelihood in zip(composeCalPolicyLikelihood(numIntentions), composeCalTransitionLikelihood(numIntentions))]

    # Joint Hypothesis Space
    priorDecayRate = 1
    getIntentionSpace = lambda numIntentions: [(id,) for id in range(numIntentions)]
    actionSpaceInInference = wolfCentralControlActionSpace
    getVariables = lambda numIntentions: [getIntentionSpace(numIntentions), actionSpaceInInference]
    getJointHypothesisSpace = lambda numIntentions: pd.MultiIndex.from_product(getVariables(numIntentions), names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composeInferImaginedWe = lambda numIntentions: [InferOneStep(priorDecayRate, getJointHypothesisSpace(numIntentions),
            concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointsLikelihood(numIntentions)]
    composeUpdateIntention = lambda numIntentions: [sheepUpdateIntentionMethod] * numIntentions + composeInferImaginedWe(numIntentions)
    chooseIntention = sampleFromDistribution

    getImaginedWeIdsForAllAgents = lambda numIntentions: [[id] for id in range(numIntentions)] + [[numIntentions, numIntentions + 1], [numIntentions + 1, numIntentions]]
    composeGetStateForPolicyGivenIntentions = lambda numIntentions: [GetStateForPolicyGivenIntention(imaginedWeId) 
            for imaginedWeId in getImaginedWeIdsForAllAgents(numIntentions)]

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
    getCentralControlPoliciesGivenIntentions = lambda numIntentions: [sheepCentralControlPolicyGivenIntention] * numIntentions + [softenWolfCentralControlPolicyGivenIntentionInPlanning, softenWolfCentralControlPolicyGivenIntentionInPlanning]
    composeIndividualPoliciesByEvaParameters = lambda numIntentions: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
            in zip(getImaginedWeIntentionPriors(numIntentions), composeGetStateForPolicyGivenIntentions(numIntentions), 
                composeUpdateIntention(numIntentions), getCentralControlPoliciesGivenIntentions(numIntentions))]

    getIndividualIdsForAllAgents = lambda numIntentions : list(range(numIntentions + 2))
    composeChooseCentrolAction = lambda numIntentions: [maxFromDistribution]* numIntentions + [sampleFromDistribution]* 2
    composeAssignIndividualActionMethods = lambda numIntentions: [AssignCentralControlToIndividual(imaginedWeId, individualId, chooseAction) for imaginedWeId, individualId, chooseAction in
            zip(getImaginedWeIdsForAllAgents(numIntentions), getIndividualIdsForAllAgents(numIntentions), composeChooseCentrolAction(numIntentions))]

    policiesResetAttributes = ['lastState', 'intentionPrior', 'formerIntentionPriors']
    getPoliciesResetAttributeValues = lambda numIntentions: [dict(zip(policiesResetAttributes, [None, intentionPrior, [intentionPrior]])) for intentionPrior in
            getImaginedWeIntentionPriors(numIntentions)]
    returnAttributes = ['formerIntentionPriors']
    composeResetPolicy = lambda numIntentions, individualPolicies: ResetPolicy(getPoliciesResetAttributeValues(numIntentions), individualPolicies, returnAttributes)
    # Sample and Save Trajectory
    maxRunningSteps = 100
    composeSampleTrajectory = lambda numIntentions, individualPolicies: SampleTrajectory(maxRunningSteps, transit, getIsTerminal(numIntentions),
            composeReset(numIntentions), composeAssignIndividualActionMethods(numIntentions), composeResetPolicy(numIntentions, individualPolicies))

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNumIntentions',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample', 'maxRunningSteps': maxRunningSteps, 'perceptNoise':perceptNoise}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 180
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters,
            composeResetPolicy, composeSampleTrajectory, saveTrajectoryByParameters)
    #[sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxRunningSteps) - 1 / maxRunningSteps * len(trajectory)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    #__import__('ipdb').set_trace()
    fig = plt.figure()
    statisticsDf.index.name = 'Set Size of Intentions'
    ax = statisticsDf.plot(y = 'mean', yerr = 'se', ylim = (0, 0.5), xlim = (1.95, 8.05), rot = 0)
    ax.set_ylabel('Accumulated Reward')
    #plt.suptitle('Wolves Accumulated Rewards')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
