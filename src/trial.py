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
from collections import deque
import os

def calculateGridDistance(gridA, gridB):
    return np.linalg.norm(np.array(gridA) - np.array(gridB), ord=1)


def isAnyKilled(humanGrids, targetGrid, killzone):
    return np.any(np.array([calculateGridDistance(humanGrid, targetGrid) for humanGrid in humanGrids]) < killzone)

class AttributionTrail:
    def __init__(self,totalScore,drawAttributionTrail,saveImage,saveImageDir ):
        self.totalScore = totalScore
        self.actionDict = [{ pg.K_LEFT: -1, pg.K_RIGHT: 1}, {pg.K_a: -1, pg.K_d: 1}]
        self.comfirmDict=[pg.K_RETURN,pg.K_SPACE]
        self.distributeUnit=0.1
        self.drawAttributionTrail=drawAttributionTrail
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self,eatenFlag, hunterFlag,currentStopwatch, screen):
        hunterid=hunterFlag.index(True)
        attributionScore=[0,0]
        attributorPercent=0.5#
        pause=True
        self.drawAttributionTrail(hunterid,attributorPercent)
        pg.event.set_allowed([pg.KEYDOWN])

        attributionDelta=0
        stayAttributionBoudray=lambda attributorPercent:max(min(attributorPercent,1),0)
        t = 0
        while  pause:
            t += 1
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    if event.key in self.actionDict[hunterid].keys():
                        attributionDelta = self.actionDict[hunterid][event.key]*self.distributeUnit

                        attributorPercent=stayAttributionBoudray(attributorPercent+attributionDelta)

                        self.drawAttributionTrail(hunterid,attributorPercent)

                        # if self.saveImage == True:
                        #     if not os.path.exists(self.saveImageDir):
                        #         os.makedirs(self.saveImageDir)
                        #     pg.image.save(screen, self.saveImageDir + '/' + format(round(currentStopwatch/100)+t, '04') + ".png")

                    elif event.key == self.comfirmDict[hunterid]:
                        pause=False
            pg.time.wait(10)
            #!
        recipentPercent=1-attributorPercent
        if hunterid==0:
            attributionScore=[self.totalScore*attributorPercent,self.totalScore*recipentPercent]
        else:#hunterid=1
            attributionScore=[self.totalScore*recipentPercent,self.totalScore*attributorPercent]

        return attributionScore

class Trial():
    def __init__(self, humanController, actionSpace, killzone, drawNewState, stopwatchEvent, finishTime, attributionTrail):
        self.humanController = humanController
        self.actionSpace = actionSpace
        self.killzone = killzone
        self.drawNewState = drawNewState
        self.stopwatchEvent = stopwatchEvent
        self.finishTime = finishTime
        self.beanReward=1
        self.attributionTrail=attributionTrail

    def checkEaten(self, sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, humanGrids):
        if isAnyKilled(humanGrids, sheep1Grid, self.killzone):
            eatenFlag = [True, False, False, False]
        elif isAnyKilled(humanGrids, sheep2Grid, self.killzone):
            eatenFlag = [False, True, False, False]
        elif isAnyKilled(humanGrids, bean1Grid, self.killzone):
            eatenFlag = [False, False, True, False]
        elif isAnyKilled(humanGrids, bean2Grid, self.killzone):
            eatenFlag = [False, False, False, True]
        else:
            eatenFlag = [False, False, False, False]

        if isAnyKilled([sheep1Grid, sheep2Grid, bean1Grid, bean2Grid], humanGrids[0], self.killzone):
            hunterFlag = [True, False]
        elif isAnyKilled([sheep1Grid, sheep2Grid, bean1Grid, bean2Grid], humanGrids[1], self.killzone):
            hunterFlag = [False, True]
        else:
            hunterFlag = [False, False]
        return eatenFlag, hunterFlag

    def checkTerminationOfTrial(self, actionList, eatenFlag, currentStopwatch):
        for action in actionList:
            if np.any(eatenFlag) == True or action == pg.QUIT or currentStopwatch >= self.finishTime:
                pause = False
            else:
                pause = True
        return pause

    def __call__(self, sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex):
        initialPlayerGrid = playerGrid
        initialTime = time.get_ticks()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT, self.stopwatchEvent])

        memorySize=5
        stateMemory=deque(maxlen=memorySize)
        stateMemory.append((sheep1Grid, sheep2Grid, playerGrid))

        sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid, action, currentStopwatch, screen = self.humanController(sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex,stateMemory)

        eatenFlag, hunterFlag = self.checkEaten(sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid)
        firstResponseTime = time.get_ticks() - initialTime
        # score = np.add(score, np.sum(eatenFlag))

        pause = self.checkTerminationOfTrial(action, eatenFlag, currentStopwatch)

        while pause:
            stateMemory.append((sheep1Grid, sheep2Grid, playerGrid))
            sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid, action, currentStopwatch, screen = self.humanController(sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex,stateMemory)
            eatenFlag, hunterFlag = self.checkEaten(sheep1Grid, sheep2Grid, bean1Grid, bean2Grid, playerGrid)
            # score = np.add(score, np.sum(eatenFlag))
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
        addSocre=[0,0]
        if True in eatenFlag[:2]:
            addSocre=self.attributionTrail(eatenFlag, hunterFlag,currentStopwatch,screen)
            results["beanEaten"] = eatenFlag.index(True) + 1
        if True in eatenFlag:
            results["beanEaten"] = eatenFlag.index(True) + 1
            addSocre[hunterFlag.index(True)] = self.beanReward
            # oldGrid = eval('bean' + str(eatenFlag.index(False) + 1) + 'Grid')
            # drawText(screen, 'caught!', THECOLORS['red'], (screen.get_width() / 2, screen.get_height() / 2))
            # pg.display.update()
            # pg.time.wait(2000)
        else:
            results["beanEaten"] = 0
            # oldGrid = None
        # results["firstResponseTime"] = firstResponseTime
        results["trialTime"] = wholeResponseTime
        score=np.add(score, addSocre)
        return results, [sheep1Grid, sheep2Grid, bean1Grid, bean2Grid], playerGrid, score, currentStopwatch, eatenFlag


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
