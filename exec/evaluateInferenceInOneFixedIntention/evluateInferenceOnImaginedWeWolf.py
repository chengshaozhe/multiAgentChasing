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
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.evaluation import ComputeStatistics
from src.inference.inference import CalPolicyLikelihood, CalTransitionLikelihood, InferOneStep, InferOnTrajectory
from scipy.interpolate import interp1d

class Interpolate1dData:
    def __init__(self, numToInterpolate):
        self.numToInterpolate = numToInterpolate
    def __call__(self, data):
        x = np.divide(np.arange(len(data)), len(data) - 1)
        y = np.array(data)
        f = interp1d(x, y, kind='nearest')
        xnew = np.linspace(0., 1., self.numToInterpolate)
        interpolatedData = f(xnew)
        return interpolatedData

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['wolfImaginedWeFixedIntention'] = [(0,), (1,)]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    #Load Trajectory
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithFixedOneIntention',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    maxRunningSteps = 100
    softParameterInPlanning = 1
    trajectoryFixedParameters = {'priorType': 'deterministicPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'updateIntention': 'False', 'maxRunningSteps': maxRunningSteps, 'policySoftParameter': softParameterInPlanning, 'chooseAction': 'max'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))

    # Inference
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

    imaginedWeIdsForInferenceSubject = [2, 3]
    getStateForPolicyGivenIntention = GetStateForPolicyGivenIntention(imaginedWeIdsForInferenceSubject) 

    calPolicyLikelihood = CalPolicyLikelihood(getStateForPolicyGivenIntention, wolfCentralControlPolicyGivenIntention)

    # Transition Likelihood
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)
 
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
    inferOneStep = InferOneStep(priorDecayRate, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood)

    prior = {intention: 1/len(intentionSpace) for intention in intentionSpace}
    stateIndexInTimestep = 0
    observe = lambda timeStep: timeStep[stateIndexInTimestep]
    visualize = None
    inferOnTrajectory = InferOnTrajectory(prior, observe, inferOneStep, visualize)

    specificIntention = intentionSpace[0]
    interpolate1dData = Interpolate1dData(maxRunningSteps)
    getInterpolatedPosteriorOnSpecificIntention = lambda wholeTrajectoryPosterior: interpolate1dData(
            [posterior[specificIntention] for posterior in wholeTrajectoryPosterior])

    # Online Inference
    measureOnlineInference = lambda trajectory: getInterpolatedPosteriorOnSpecificIntention(inferOnTrajectory(trajectory))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureOnlineInference)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    
    fig = plt.figure()
    numColumns = len(manipulatedVariables['wolfImaginedWeFixedIntention'])
    numRows = 1
    plotCounter = 1

    for wolfImaginedWeFixedIntention, grp in statisticsDf.groupby('wolfImaginedWeFixedIntention'):
        axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
        axForDraw.set_title('intention In Planning = {}'.format(wolfImaginedWeFixedIntention))
        df = pd.DataFrame(grp.values[0].tolist(), columns = list(range(maxRunningSteps)), index = ['mean','se']).T
        df.plot(ax = axForDraw, label = 'P(intention ={})'.format(specificIntention), y = 'mean', yerr = 'se', ylim = (0, 1))
        
        plotCounter = plotCounter + 1

    plt.suptitle('Wolves Intention Archeivement Stationary Sheeps')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()
