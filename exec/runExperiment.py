import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import collections as co
import itertools as it
import numpy as np
import pickle
import pygame as pg
from pygame.color import THECOLORS
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback, InitializeScreen
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld, UpdateWorld, StayInBoundary
from src.writer import WriteDataFrameToCSV
from src.trial import Trial
from src.experiment import Experiment
from src.sheepPolicy import GenerateModel, restoreVariables, ApproximatePolicy, chooseGreedyAction, sampleAction, SoftmaxAction


def main():
    gridSize = 15
    bounds = [0, 0, gridSize - 1, gridSize - 1]
    minDistanceForReborn = 30
    condition = [-5, -3, -1, 0, 1, 3, 5]
    counter = [0] * len(condition)
    numPlayers = 4
    initialWorld = InitialWorld(bounds, numPlayers, minDistanceForReborn)
    updateWorld = UpdateWorld(bounds, condition, counter, minDistanceForReborn)

    screenWidth = 800
    screenHeight = 800

    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()

    leaveEdgeSpace = 10
    lineWidth = 1
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [THECOLORS['blue'], (0, 168, 107)]  # [255, 50, 50]
    playerColors = [THECOLORS['orange'], THECOLORS['red']]
    targetRadius = 10
    playerRadius = 10
    stopwatchUnit = 100
    finishTime = 1000 * 60 * 3
    block = 1
    softmaxBeita = -1
    textColorTuple = THECOLORS['green']
    stopwatchEvent = pg.USEREVENT + 1

    saveImage = False
    killzone = 3
    wolfSpeedRatio = 1

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'pictures'))
    resultsPath = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'results'))
    experimentValues = co.OrderedDict()
    # experimentValues["name"] = input("Please enter your name:").capitalize()
    experimentValues["name"] = 'test'
    experimentValues["condition"] = 'all'
    writerPath = os.path.join(resultsPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    introductionImage = pg.image.load(os.path.join(picturePath, 'introduction.png'))
    restImage = pg.image.load(os.path.join(picturePath, 'rest.png'))
    finishImage = pg.image.load(os.path.join(picturePath, 'finish.png'))
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    xBoundary = [bounds[0], bounds[2]]
    yBoundary = [bounds[1], bounds[3]]
    stayInBoundary = StayInBoundary(xBoundary, yBoundary)
#########         
    sheepActionSpace = [(1, 0), (0, 1),(-1, 0),(0, -1),(0, 0)]
    #grid policy
    # sheepPolicySingle =pickle.load(open("SingleWolfTwoSheepsGrid15.pkl","rb"))

    multiPath=os.path.join(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), 'data/policy')))
    sheepPolicyMulti = pickle.load(open(os.path.join(multiPath,"noise0SheepToTwoWolfGird15_policy.pkl"),"rb"))

    sheepPolicy = sheepPolicyMulti
    # sheepPolicy=lambda a:{[0,0]:1}


    softMaxBeta = 30
    softmaxAction = SoftmaxAction(softMaxBeta)
    humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction)
    # modelController = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)

    actionSpace = list(it.product([0, 1, -1], repeat=2))
    actionSpace = [(1, 0), (0, 1),(-1, 0),(0, -1),(0, 0)]
    trial = Trial(humanController, actionSpace, killzone, drawNewState, stopwatchEvent, finishTime)
    experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath)
    giveExperimentFeedback = GiveExperimentFeedback(screen, textColorTuple, screenWidth, screenHeight)

    # drawImage(introductionImage)
    score = [0] * block
    for i in range(block):
        score[i] = experiment(finishTime)
        giveExperimentFeedback(i, score)
        if i == block - 1:
            drawImage(finishImage)
        # else:
            # drawImage(restImage)

    participantsScore = np.sum(np.array(score))
    print(participantsScore)


if __name__ == "__main__":
    main()
