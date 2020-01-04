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
import scipy.stats as stats

from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.inference import CalPolicyLikelihood, CalTransitionLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics
from scipy.interpolate import interp1d

class MeasureCrossEntropy:
    def __init__(self, imaginedWeIds, priorIndex):
        self.baseId, self.nonBaseId = imaginedWeIds
        self.priorIndex = priorIndex

    def __call__(self, trajectory):

        priors = [timeStep[self.priorIndex] for timeStep in trajectory]
        baseDistributions = [list(prior[self.baseId].values())
                for prior in priors] 
        nonBaseDistributions = [list(prior[self.nonBaseId].values())
                for prior in priors] 
        crossEntropies = [stats.entropy(baseDistribution) + stats.entropy(baseDistribution, nonBaseDistribution) 
                for baseDistribution, nonBaseDistribution in zip(baseDistributions, nonBaseDistributions)] 
        return crossEntropies

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
    manipulatedVariables['perceptNoise'] = [1e3]#[1e1, 2e1, 4e1, 8e1, 1.6e2]
    manipulatedVariables['maxRunningSteps'] = [50]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    #trajectory dir
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNoisePerception',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    softParameterInPlanning = 2.5
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    
    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    wolfImaginedWeId = [2, 3]
    priorIndexinTimestep = 3
    measureCrossEntropy = MeasureCrossEntropy(wolfImaginedWeId, priorIndexinTimestep)
    composeInterpolateFunction = lambda df: Interpolate1dData(readParametersFromDf(df)['maxRunningSteps'])
    measureFunction = lambda df: lambda trajectory: composeInterpolateFunction(df)(measureCrossEntropy(trajectory))
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    fig = plt.figure()
    #numColumns = len(manipulatedVariables['perceptNoise'])
    numColumns = 1
    numRows = len(manipulatedVariables['maxRunningSteps'])
    plotCounter = 1
    
    for maxRunningSteps, group in statisticsDf.groupby('maxRunningSteps'):
        group.index = group.index.droplevel('maxRunningSteps')
        for perceptNoise, grp in group.groupby('perceptNoise'):
            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel('maxRunningSteps = {}'.format(maxRunningSteps))
            df = pd.DataFrame(group.values[0].tolist(), columns = list(range(maxRunningSteps)), index = ['mean','se']).T
            df.plot(ax = axForDraw, label = '(perceptNoise ={})'.format(perceptNoise), y = 'mean', yerr = 'se', ylim = (0, 5))
            plotCounter = plotCounter + 1

    plt.suptitle('Wolves Cross Entropy')
    plt.legend(loc='best')
    plt.show()
if __name__ == '__main__':
    main()