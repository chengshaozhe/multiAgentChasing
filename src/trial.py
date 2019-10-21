import numpy as np
import pygame as pg
from pygame.color import THECOLORS
from pygame import time
import collections as co
import pickle
from src.visualization import DrawBackground, DrawNewState, DrawImage, drawText
from src.controller import HumanController, ModelController
from src.updateWorld import InitialWorld
import random


class Trial():
    def __init__(self, humanController, actionSpace, killzone, drawNewState, stopwatchEvent, finishTime):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.finishTime = finishTime

    def checkEaten(self, sheep1Grid, sheep2Grid,bean1Grid,bean2Grid, humanGrids):
        if np.any(np.array([np.linalg.norm(np.array(humanGrid) - np.array(sheep1Grid)) for humanGrid in humanGrids]) < self.killzone):
            eatenFlag = [True, False,False,False]
        elif np.any(np.array([np.linalg.norm(np.array(humanGrid) - np.array(sheep2Grid)) for humanGrid in humanGrids]) < self.killzone):
            eatenFlag = [False, True,False,False]
        elif np.any(np.array([np.linalg.norm(np.array(humanGrid) - np.array(bean1Grid)) for humanGrid in humanGrids]) < self.killzone):
            eatenFlag = [False,False, True,False]
        elif np.any(np.array([np.linalg.norm(np.array(humanGrid) - np.array(bean2Grid)) for humanGrid in humanGrids]) < self.killzone):
            eatenFlag = [False,False,False, True]
        else:
            eatenFlag = [False, False,False,False]
        return eatenFlag

    def checkTerminationOfTrial(self, actionList, eatenFlag, currentStopwatch):
        for action in actionList:
            if np.any(eatenFlag) == True or action == pg.QUIT or currentStopwatch >= self.finishTime:
                pause = False
            else:
                pause = True
        return pause

    def __call__(self, sheep1Grid, sheep2Grid,bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent，pg.K_UP, pg.K_DOWN，pg.K_LEFT, pg.K_RIGHT,pg.K_w, pg.K_s, pg.K_a, pg.K_d])

        sheep1Grid, sheep2Grid, playerGrid, action, currentStopwatch, screen = self.humanController(sheep1Grid, sheep2Grid, playerGrid, score, currentStopwatch, trialIndex)

        eatenFlag = self.checkEaten( sheep1Grid,sheep2Grid,bean1Grid, bean2Grid, playerGrid)
        firstResponseTime = time.get_ticks() - initialTime
        score = np.add(score, np.sum(eatenFlag))
        pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
        while pause:
            sheep1Grid, sheep2Grid, playerGrid, action, currentStopwatch, screen = self.humanController(sheep1Grid, sheep2Grid, playerGrid, score, currentStopwatch, trialIndex)
            eatenFlag = self.checkEaten(bean1Grid, bean2Grid, playerGrid)
            score = np.add(score, np.sum(eatenFlag))
            pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)
        wholeResponseTime = time.get_ticks() - initialTime
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP])

        results = co.OrderedDict()

        # results["bean1GridX"] = bean1Grid[0]
        # results["bean1GridY"] = bean1Grid[1]
        # results["bean2GridX"] = bean2Grid[0]
        # results["bean2GridY"] = bean2Grid[1]
        # results["player1GridX"] = initialPlayerGrid[0][0]
        # results["player1GridY"] = initialPlayerGrid[0][1]
        # results["player2GridX"] = initialPlayerGrid[1][0]
        # results["player2GridY"] = initialPlayerGrid[1][1]

        if True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            oldGrid = eval('bean' + str(eatenFlag.index(False) + 1) + 'Grid')
            # drawText(screen, 'caught!', THECOLORS['red'], (screen.get_width() / 2, screen.get_height() / 2))
            # pg.display.update()
            # pg.time.wait(2000)
        else:
            results["beanEaten"] = 0
            oldGrid = None
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime

        return results, [sheep1Grid, sheep2Grid,bean1Grid, bean2Grid], playerGrid, score, currentStopwatch, eatenFlag


def main():
    dimension = 21
    bounds = [0, 0, dimension - 1, dimension - 1]
    minDistanceBetweenGrids = 5
    condition = [-5, -3, -1, 0, 1, 3, 5]
    initialWorld = InitialWorld(bounds)
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 21
    leaveEdgeSpace = 2
    lineWidth = 1
    backgroundColor = [205, 255, 204]
    lineColor = [0, 0, 0]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    stopwatchUnit = 10
    textColorTuple = (255, 50, 50)
    stopwatchEvent = pg.USEREVENT + 1
    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    pg.event.set_allowed([pg.KEYDOWN, pg.QUIT, stopwatchEvent])
    finishTime = 90000
    currentStopwatch = 32888
    score = 0
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    humanController = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl", "rb"))
    modelController = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    trial = Trial(modelController, drawNewState, stopwatchEvent, finishTime)
    bean1Grid, bean2Grid, playerGrid = initialWorld(minDistanceBetweenGrids)
    bean1Grid = (3, 13)
    bean2Grid = (5, 0)
    playerGrid = (0, 8)
    results = trial(bean1Grid, bean2Grid, playerGrid, score, currentStopwatch)


if __name__ == "__main__":
    main()
