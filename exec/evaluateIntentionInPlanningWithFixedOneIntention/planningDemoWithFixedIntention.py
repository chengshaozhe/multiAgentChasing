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

from src.visualization.drawDemo import DrawBackground, DrawState, ChaseTrialWithTraj
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle


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

def main():
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

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    trajectoryParameters = {'wolfImaginedWeFixedIntention':(0,)}
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
    circleColorSpace = [THECOLORS['red'], THECOLORS['green'], THECOLORS['blue'], THECOLORS['blue']]
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
    
    stateIndexInTimeStep = 0
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState)
   
    print(len(trajectories))
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[0:30]]

if __name__ == '__main__':
    main()
