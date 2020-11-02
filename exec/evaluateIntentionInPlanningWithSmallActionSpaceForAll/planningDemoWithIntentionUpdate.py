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

from src.visualization.drawDemo import DrawBackground, DrawState, ChaseTrialWithTraj, InterpolateState
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.MDPChasing.envNoPhysics import TransitForNoPhysics, StayInBoundaryByReflectVelocity

def main():
    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithIntentionUpdate',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    
    maxRunningSteps = 100
    softParameterInPlanning = 2.5
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy':'NNPolicy', 'wolfPolicy':'NNPolicy',
            'maxRunningSteps': maxRunningSteps, 'policySoftParameter': softParameterInPlanning, 'chooseAction': 'sample'}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectoryParameters = {'updateIntention': 'inferImaginedWe'}
    #trajectoryParameters = {'updateIntention': 'False'}
    trajectories = loadTrajectories(trajectoryParameters) 
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
    circleColorSpace = [THECOLORS['green'], THECOLORS['green'], THECOLORS['red'], THECOLORS['red']]
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(4))
    saveImage = True
    imageSavePath = os.path.join(trajectoryDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str(trajectoryParameters)
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior)
    
    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 3
    interpolateState = InterpolateState(numFramesToInterpolate, transit)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep)
   
    print(len(trajectories))
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[0:10]]

if __name__ == '__main__':
    main()
