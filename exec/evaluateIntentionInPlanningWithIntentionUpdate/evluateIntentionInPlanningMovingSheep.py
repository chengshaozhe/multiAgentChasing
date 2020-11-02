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
        wolfUpdateIntention = parameters['updateIntention']
        print(wolfUpdateIntention)
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(wolfUpdateIntention)
        resetPolicy = self.composeResetPolicy(individualPolicies)
        sampleTrajectory = self.composeSampleTrajectory(resetPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['updateIntention'] = ['False', 'inferImaginedWe']
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
    wolf1ImaginedWeIntentionPrior = {(0, ):1, (1,): 0}
    wolf2ImaginedWeIntentionPrior = {(0, ):0, (1,): 1}
    imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior, sheepImagindWeIntentionPrior, wolf1ImaginedWeIntentionPrior, wolf2ImaginedWeIntentionPrior]
    
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
    
    imaginedWeIdsForInferenceSubject = [2, 3]
    getStateForPolicyGivenIntentionInInference = GetStateForPolicyGivenIntention(imaginedWeIdsForInferenceSubject)

    calPolicyLikelihood = CalPolicyLikelihood(getStateForPolicyGivenIntentionInInference, softenWolfCentralControlPolicyGivenIntentionInInference)

    # Transition Likelihood 
    getOwnState = lambda state: np.array(state)[imaginedWeIdsForInferenceSubject]

    calTransitionLikelihood = CalTransitionLikelihood(getOwnState, transit)

    # Joint Likelihood
    calJointLikelihood = lambda intention, state, action, nextState: \
        calPolicyLikelihood(intention, state, action) * calTransitionLikelihood(state, action, nextState)

    # Joint Hypothesis Space
    priorDecayRate = 1
    intentionSpace = [(0,), (1,)]
    actionSpaceInInference = wolfCentralControlActionSpace
    variables = [intentionSpace, actionSpaceInInference]
    jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    inferImaginedWe = InferOneStep(priorDecayRate, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood)
    updateIntentionMethod = {'inferImaginedWe': inferImaginedWe, 'False': noInferIntention}
    getUpdateIntention = lambda wolfUpdateIntention: [sheepUpdateIntentionMethod, sheepUpdateIntentionMethod, 
            updateIntentionMethod[wolfUpdateIntention], updateIntentionMethod[wolfUpdateIntention]]
    chooseIntention = sampleFromDistribution

    imaginedWeIdsForAllAgents = [[0], [1], [2, 3], [2, 3]]
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
    composeIndividualPoliciesByEvaParameters = lambda wolfUpdateIntention: [PolicyOnChangableIntention(lastState, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
            in zip(imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, getUpdateIntention(wolfUpdateIntention), centralControlPoliciesGivenIntentions)]

    individualIdsForAllAgents = [0, 1, 2, 3]
    chooseCentrolAction = sampleFromDistribution
    assignIndividualActionMethods = [AssignCentralControlToIndividual(imaginedWeId, individualId, chooseCentrolAction) for imaginedWeId, individualId in zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]

    policiesResetAttributes = ['lastState', 'intentionPrior']
    policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [None, intentionPrior])) for intentionPrior in imaginedWeIntentionPriors]
    composeResetPolicy = lambda individualPolicies: ResetPolicy(policiesResetAttributeValues, individualPolicies)
    # Sample and Save Trajectory
    maxRunningSteps = 100
    composeSampleTrajectory = lambda resetPolicy: SampleTrajectory(maxRunningSteps, transit, isTerminal, reset, assignIndividualActionMethods, resetPolicy)

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithIntentionUpdate',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'maxRunningSteps': maxRunningSteps, 'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 50
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters,
            composeResetPolicy, composeSampleTrajectory, saveTrajectoryByParameters)
    #[sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    possibleIntentionIds = [[0],[1]]
    wolfImaginedWeId = [2, 3]
    stateIndexInTimestep = 0
    judgeSuccessCatchOrEscape = lambda booleanSign: int(booleanSign)
    #measureIntentionArcheivement = MeasureIntentionArcheivement(possibleIntentionIds, wolfImaginedWeId, stateIndexInTimestep, posIndexInState, killzoneRadius, judgeSuccessCatchOrEscape)
    measurementFunction = lambda df: lambda trajectory: int(len(trajectory) < maxRunningSteps) - 1 / maxRunningSteps * len(trajectory)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['updateIntention'])
    numRows = 1
    plotCounter = 1
    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
 
    statisticsDf.index = ['No Infer Imagined We', 'Infer Imagined We']
    ax = statisticsDf.plot.bar(y = 'mean', yerr = 'se', ylim = (-0.6, 0.6), rot = 0)

    ax.set_ylabel('Accumulated Reward')
    #for wolfUpdateIntention, grp in statisticsDf.groupby('updateIntention'):
    #    axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        #axForDraw.set_title('wolf Update Intention = {}'.format(wolfUpdateIntention))
        #df = pd.DataFrame(grp.values[0].tolist(), columns = possiblePreyIds, index = ['mean','se']).T
        #df = grp
        #__import__('ipdb').set_trace()
    #    df.plot.bar(ax = axForDraw, y = 'mean', yerr = 'se', ylim = (-0.5, 0.5))
    #    plotCounter = plotCounter + 1

    #plt.suptitle('Wolves Intention Archeivement Moving Sheeps')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
