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
import pygame as pg
from pygame.color import THECOLORS

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.visualization.drawDemo import DrawBackground, DrawState, ChaseTrialWithTraj
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.inference.inference import CalPolicyLikelihood, CalTransitionLikelihood, InferOneStep, InferOnTrajectory

def updateColorSpace(colorSpace, posterior, intentionSpace, imaginedWeIds):
    colorRepresentProbability = np.array([colorSpace[individualId] * 2 * (1 - max(list(posterior.values()))) +
        np.sum([colorSpace[intention[0]] * max(0, 2 * (posterior[intention] - 1/len(intentionSpace))) 
        for intention in intentionSpace], axis = 0) for individualId in imaginedWeIds])
    updatedColorSpace = colorSpace.copy()
    updatedColorSpace[imaginedWeIds] = colorRepresentProbability
    return updatedColorSpace

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithFixedOneIntention',
                                    'trajectories')
    
    maxRunningSteps = 100
    softParameterInPlanning = 2.5
    trajectoryFixedParameters = {'priorType': 'deterministicPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'updateIntention': 'False', 'maxRunningSteps': maxRunningSteps, 'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectoryParameters = {'wolfImaginedWeFixedIntention':(0,)}
    trajectories = loadTrajectories(trajectoryParameters) 
    
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
    softParameterInInference = 0.15
    softPolicy = SoftPolicy(softParameterInInference)
    softenWolfCentralControlPolicyGivenIntention = lambda state: softPolicy(wolfCentralControlPolicyGivenIntention(state))

    imaginedWeIdsForInferenceSubject = [2, 3]
    getStateForPolicyGivenIntention = GetStateForPolicyGivenIntention(imaginedWeIdsForInferenceSubject) 

    calPolicyLikelihood = CalPolicyLikelihood(getStateForPolicyGivenIntention, softenWolfCentralControlPolicyGivenIntention)

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
    priorDecayRate = 0.9
    intentionSpace = [(0,), (1,)]
    actionSpaceInInference = wolfCentralControlActionSpace
    variables = [intentionSpace, actionSpaceInInference]
    jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    inferOneStep = InferOneStep(priorDecayRate, jointHypothesisSpace, concernedHypothesisVariable, calJointLikelihood)

    prior = {intention: 1/len(intentionSpace) for intention in intentionSpace}
    stateIndexInTimestep = 0
    observe = lambda timeStep: timeStep[stateIndexInTimestep]
    
    # generate demo image
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 20
    circleColorSpace = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255]])
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(4))
    saveImage = True
    
    inferenceDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateInferenceInOneFixIntention',
                                    'trajectories')
    if not os.path.exists(inferenceDirectory):
        os.makedirs(inferenceDirectory) 
    imageSavePath = os.path.join(inferenceDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str(trajectoryParameters)
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    updateColorSpaceByPosterior = lambda colorSpace, posterior : updateColorSpace(
            colorSpace, posterior, intentionSpace, imaginedWeIdsForInferenceSubject)
            
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, 
            positionIndex, saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior)
    
    inferOnTrajectory = InferOnTrajectory(prior, observe, inferOneStep, visualize = drawState)
    posteriorsOnWholeTra = [inferOnTrajectory(trajectory) for trajectory in trajectories[0:10]]

if __name__ == '__main__':
    main()
