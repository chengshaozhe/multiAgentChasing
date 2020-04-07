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

    def __call__(self, targetPositionA, targetPositionB, targetPositionC, targetPositionD, playerPositions, currentScore, currentStopwatch, trialIndex, stateMemory):
        newStopwatch = currentStopwatch
        remainningTime = max(0, self.finishTime - currentStopwatch)

        screen = self.drawNewState(targetPositionA, targetPositionB, targetPositionC, targetPositionD, playerPositions, remainningTime, currentScore)

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

        if currentStopwatch % 200 == 0:
            def sortstate(state):
                temp = state.copy()
                temp.sort()
                return temp
            sheep1Memory = [[state[0], sortstate(state[2])] for state in stateMemory]
            sheep2Memory = [[state[1], sortstate(state[2])] for state in stateMemory]
            sheep1Memory = [(state[0], state[1][0], state[1][1]) for state in sheep1Memory]
            sheep2Memory = [(state[0], state[1][0], state[1][1]) for state in sheep2Memory]
            action3 = self.sheepPolicy(sheep1Memory)
            action4 = self.sheepPolicy(sheep2Memory)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            elif event.type == self.stopwatchEvent:
                newStopwatch = newStopwatch + self.stopwatchUnit
            if event.type == pg.KEYDOWN:
                if event.key in self.actionDict[0].keys():
                    action1 = self.actionDict[0][event.key]
                elif event.key in self.actionDict[1].keys():
                    action2 = self.actionDict[1][event.key]

            playerPositions = [self.stayInBoundary(np.add(playerPosition, action)) for playerPosition, action in zip(playerPositions, [action1, action2])]

            targetPositionA = self.stayInBoundary(np.add(targetPositionA, action3))
            targetPositionB = self.stayInBoundary(np.add(targetPositionB, action4))

            remainningTime = max(0, self.finishTime - newStopwatch)
            screen = self.drawNewState(targetPositionA, targetPositionB, targetPositionC, targetPositionD, playerPositions, remainningTime, currentScore)

            pg.display.update()
        return targetPositionA, targetPositionB, targetPositionC, targetPositionD, playerPositions, [action1, action2], newStopwatch, screen


def calculateSoftmaxProbability(probabilityList, beta):
    newProbabilityList = list(np.divide(np.exp(np.multiply(beta, probabilityList)), np.sum(np.exp(np.multiply(beta, probabilityList)))))
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
