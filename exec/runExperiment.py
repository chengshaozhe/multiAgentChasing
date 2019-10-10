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
from src.visualization import DrawBackground, DrawNewState, DrawImage, GiveExperimentFeedback
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld, UpdateWorld, StayInBoundary
from src.writer import WriteDataFrameToCSV
from src.trial import Trial
from src.experiment import Experiment


def main():
    gridSize = 60
    bounds = [0, 0, gridSize - 1, gridSize - 1]
    minDistanceForReborn = 20
    condition = [-5, -3, -1, 0, 1, 3, 5]
    counter = [0] * len(condition)
    numPlayers = 4
    initialWorld = InitialWorld(bounds, numPlayers)
    updateWorld = UpdateWorld(bounds, condition, counter, minDistanceForReborn)

    pg.init()
    screenWidth = 800
    screenHeight = 800
    screen = pg.display.set_mode((screenWidth, screenHeight))
    leaveEdgeSpace = 10
    lineWidth = 1
    backgroundColor = THECOLORS['grey']  # [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = THECOLORS['yellow']  # [255, 50, 50]
    playerColors = [THECOLORS['blue'], THECOLORS['red']]
    targetRadius = 10
    playerRadius = 10
    stopwatchUnit = 100
    finishTime = 1000 * 60 * 1
    block = 1
    softmaxBeita = -1
    textColorTuple = THECOLORS['green']
    stopwatchEvent = pg.USEREVENT + 1

    saveImage = False

    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    pg.key.set_repeat(120, 120)
    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/pictures/'
    resultsPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/results/'
    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter your name:").capitalize()
    # experimentValues["name"] = 'test'
    experimentValues["condition"] = 'None'
    writerPath = resultsPath + experimentValues["name"] + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    introductionImage = pg.image.load(picturePath + 'introduction.png')
    restImage = pg.image.load(picturePath + 'rest.png')
    finishImage = pg.image.load(picturePath + 'finish.png')
    introductionImage = pg.transform.scale(introductionImage, (screenWidth, screenHeight))
    finishImage = pg.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))

    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    saveImageDir = os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'data'), experimentValues["name"])

    xBoundary = [bounds[0], bounds[2]]
    yBoundary = [bounds[1], bounds[3]]
    stayInBoundary = StayInBoundary(xBoundary, yBoundary)
    humanController = HumanController(writer, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir)

    # policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl","rb"))
    # modelController = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)

    killzone = 3
    actionSpace = list(it.product([0, 1, -1], repeat=2))
    trial = Trial(humanController, actionSpace, killzone, drawNewState, stopwatchEvent, finishTime)
    experiment = Experiment(trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceForReborn)
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
