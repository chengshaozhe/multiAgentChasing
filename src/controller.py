import numpy as np
import pygame as pg
import random
import itertools as it
import math
import os
import collections as co
class HumanControllerWithStrait():
    def __init__(self, gridSize):
        self.actionDict = [{pg.K_UP: [0, -1], pg.K_DOWN: [0, 1], pg.K_LEFT: [-1, 0], pg.K_RIGHT: [1, 0]},
                           {pg.K_w: [0, -1], pg.K_s: [0, 1], pg.K_a: [-1, 0], pg.K_d: [1, 0]}]
        self.gridSize = gridSize

    def __call__(self, playerGrid, straitGrid):

        return newGrid, action, actPlayer

class HumanController():
    def __init__(self, writer, gridSize, stopwatchEvent, stopwatchUnit, wolfSpeedRatio, drawNewState, finishTime, stayInBoundary, saveImage, saveImageDir, sheepPolicy, chooseGreedyAction):
        self.writer = writer
        self.gridSize = gridSize
        self.stopwatchEvent = stopwatchEvent
        self.stopwatchUnit = stopwatchUnit
        self.stopwatch = 0
        self.wolfSpeedRatio = wolfSpeedRatio
        self.finishTime = finishTime
        self.drawNewState = drawNewState
        self.stayInBoundary = stayInBoundary
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir
        self.sheepPolicy = sheepPolicy
        self.chooseGreedyAction = chooseGreedyAction
        self.actionDict = [{pg.K_UP: [0, -1], pg.K_DOWN: [0, 1], pg.K_LEFT: [-1, 0], pg.K_RIGHT: [1, 0]}, {pg.K_w: [0, -1], pg.K_s: [0, 1], pg.K_a: [-1, 0], pg.K_d: [1, 0]}]

    def __call__(self, targetPositionA, targetPositionB,targetPositionC,targetPositionD,playerPositions, currentScore, currentStopwatch, trialIndex):
        newStopwatch = currentStopwatch
        remainningTime = max(0, self.finishTime - currentStopwatch)

        screen = self.drawNewState(targetPositionA, targetPositionB,targetPositionC,targetPositionD, playerPositions, remainningTime, currentScore)

        results = co.OrderedDict()
        results["trialIndex"] = trialIndex
        results["timeStep"] = self.stopwatch
        results["sheep1GridX"] = targetPositionA[0]
        results["bean1GridY"] = targetPositionA[1]
        results["bean2GridX"] = targetPositionB[0]
        results["bean2GridY"] = targetPositionB[1]
        results["bean1GridX"] = targetPositionC[0]
        results["sheep1GridY"] = targetPositionC[1]
        results["sheep2GridX"] = targetPositionD[0]
        results["sheep2GridY"] = targetPositionD[1]

        results["player1GridX"] = playerPositions[0][0]
        results["player1GridY"] = playerPositions[0][1]
        results["player2GridX"] = playerPositions[1][0]
        results["player2GridY"] = playerPositions[1][1]
        results["beanEaten"] = 0
        results["trialTime"] = ''
        self.writer(results, self.stopwatch)

        if self.saveImage == True:
            if not os.path.exists(self.saveImageDir):
                os.makedirs(self.saveImageDir)
            pg.image.save(screen, self.saveImageDir + '/' + format(self.stopwatch, '04') + ".png")
        self.stopwatch += 1

        action1 = [0, 0]
        action2 = [0, 0]
        action3 = [0, 0]
        action4 = [0, 0]

        wolfStates =(tuple(playerPositions[0]), tuple(playerPositions[1]))
        wolfStatesReverse = (tuple(playerPositions[1]), tuple(playerPositions[0]))
        try:
            policyForCurrentStateDict1 = self.sheepPolicy[0][tuple(targetPositionA),wolfStates]
        except KeyError as e:
            policyForCurrentStateDict1 = self.sheepPolicy[0][tuple(targetPositionA),wolfStatesReverse]

        try:
            policyForCurrentStateDict2 = self.sheepPolicy[1][tuple(targetPositionB),wolfStates]
        except KeyError as e:
            policyForCurrentStateDict2 = self.sheepPolicy[1][tuple(targetPositionB),wolfStatesReverse]

        actionMaxList1 = [action for action in policyForCurrentStateDict1.keys() if policyForCurrentStateDict1[action] == np.max(list(policyForCurrentStateDict1.values()))]

        actionMaxList2 = [action for action in policyForCurrentStateDict2.keys() if policyForCurrentStateDict2[action] == np.max(list(policyForCurrentStateDict2.values()))]

        pause = True

        if currentStopwatch % 300 == 0:
            action3 = random.choice(actionMaxList1)
            action4 = random.choice(actionMaxList2)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pause = True
                pg.quit()
            elif event.type == self.stopwatchEvent:
                newStopwatch = newStopwatch + self.stopwatchUnit
            if event.type == pg.KEYDOWN:
                if event.key in self.actionDict[0].keys():
                    action1 = self.actionDict[0][event.key]
                    pause = False

                elif event.key in self.actionDict[1].keys():
                    action2 = self.actionDict[1][event.key]
                    pause = False

        # action3 = (0,0)
            # action4 = (0,0)
            # action3 = action[2]
            # action4 = action[3]
            playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]

            targetPositionA = self.stayInBoundary(np.add(targetPositionA, action3))
            targetPositionB = self.stayInBoundary(np.add(targetPositionB, action4))

            remainningTime = max(0, self.finishTime - newStopwatch)
            screen = self.drawNewState(targetPositionA, targetPositionB,targetPositionC,targetPositionD, playerPositions, remainningTime, currentScore)

            pg.display.update()
        return targetPositionA, targetPositionB,targetPositionC,targetPositionD, playerPositions, [action1, action2], newStopwatch, screen


def calculateSoftmaxProbability(probabilityList, beita):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beita, probabilityList)), np.sum(np.exp(np.multiply(beita, probabilityList)))))
    return newProbabilityList


class ModelController():
    def __init__(self, policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita):
        self.policy = policy
        self.gridSize = gridSize
        self.stopwatchEvent = stopwatchEvent
        self.stopwatchUnit = stopwatchUnit
        self.stopwatch = 0
        self.drawNewState = drawNewState
        self.finishTime = finishTime
        self.softmaxBeita = softmaxBeita

    def __call__(self, targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch):
        pause = True
        newStopwatch = currentStopwatch
        remainningTime = max(0, self.finishTime - currentStopwatch)
        self.drawNewState(targetPositionA, targetPositionB, playerPosition, remainningTime, currentScore)
        while pause:
            targetStates = (tuple(targetPositionA), tuple(targetPositionB))
            if targetStates not in self.policy.keys():
                targetStates = (tuple(targetPositionB), tuple(targetPositionA))
            policyForCurrentStateDict = self.policy[targetStates][tuple(playerPosition)]
            if self.softmaxBeita < 0:
                actionMaxList = [action for action in policyForCurrentStateDict.keys() if policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
                action = random.choice(actionMaxList)
            else:
                actionProbability = np.divide(list(policyForCurrentStateDict.values()), np.sum(list(policyForCurrentStateDict.values())))
                softmaxProbabilityList = calculateSoftmaxProbability(list(actionProbability), self.softmaxBeita)
                action = list(policyForCurrentStateDict.keys())[list(np.random.multinomial(1, softmaxProbabilityList)).index(1)]
            playerNextPosition = np.add(playerwaPosition, action)
            if np.any(playerNextPosition < 0) or np.any(playerNextPosition >= self.gridSize):
                playerNextPosition = playerPosition
            pause = False
            for event in pg.event.get():
                if event.type == self.stopwatchEvent:
                    newStopwatch = newStopwatch + self.stopwatchUnit
                    remainningTime = max(0, self.finishTime - newStopwatch)
            self.drawNewState(targetPositionA, targetPositionB, playerNextPosition, remainningTime, currentScore)
            pg.display.flip()
        return playerNextPosition, action, newStopwatch


if __name__ == "__main__":
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 20
    leaveEdgeSpace = 2
    lineWidth = 2
    backgroundColor = [188, 188, 0]
    lineColor = [255, 255, 255]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    targetPositionA = [5, 5]
    targetPositionB = [15, 5]
    playerPosition = [10, 15]
    currentScore = 5
    textColorTuple = (255, 50, 50)
    stopwatchEvent = pg.USEREVENT + 1
    stopwatchUnit = 10
    pg.time.set_timer(stopwatchEvent, stopwatchUnit)
    finishTime = 90000
    currentStopwatch = 32000
    softmaxBeita = 20

    drawBackground = Visualization.DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = Visualization.DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)

    getHumanAction = HumanController(gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime)
    # newProbabilityList=calculateSoftmaxProbability([0.5,0.3,0.2],20)
    # print(newProbabilityList)
    import pickle
    policy = pickle.load(open("SingleWolfTwoSheepsGrid15.pkl", "rb"))
    getModelAction = ModelController(policy, gridSize, stopwatchEvent, stopwatchUnit, drawNewState, finishTime, softmaxBeita)

    # [playerNextPosition,action,newStopwatch]=getHumanAction(targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch)
    [playerNextPosition, action, newStopwatch] = getModelAction(targetPositionA, targetPositionB, playerPosition, currentScore, currentStopwatch)
    print(playerNextPosition, action, newStopwatch)

    pg.quit()
