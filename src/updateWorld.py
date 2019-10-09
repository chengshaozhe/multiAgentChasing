import random
import numpy as np
import copy


def computeAngleBetweenTwoVectors(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    lenthOfVector1 = np.sqrt(vector1.dot(vector1))
    lenthOfVector2 = np.sqrt(vector2.dot(vector2))
    cosAngle = vector1.dot(vector2) / (lenthOfVector1 * lenthOfVector2)
    angle = np.arccos(cosAngle)
    return angle


def indexCertainNumberInList(list, number):
    indexList = [i for i in range(len(list)) if list[i] == number]
    return indexList


def samplePosition(bounds):
    positionX = np.random.uniform(bounds[0], bounds[2])
    positionY = np.random.uniform(bounds[1], bounds[3])
    position = [positionX, positionY]
    return position


class InitialWorld():
    def __init__(self, bounds, numPlayers):
        self.bounds = bounds
        self.numPlayers = numPlayers

    def __call__(self, minDistance):
        initPlayerGrids = [samplePosition(self.bounds) for i in range(self.numPlayers)]
        target1Grid = samplePosition(self.bounds)
        target2Grid = samplePosition(self.bounds)

        return target1Grid, target2Grid, initPlayerGrids


class UpdateWorld():
    def __init__(self, bounds, conditon, counter):
        self.condition = conditon
        self.bounds = bounds
        self.counter = counter
        self.correctionFactors = 0.0001

    def __call__(self, oldTargetGrid, playerGrid):
        counter = copy.deepcopy(self.counter)
        condition = copy.deepcopy(self.condition)
        counterCorrection = [c + self.correctionFactors if c == 0 else c for c in counter]
        sampleProbability = 1 / np.array(counterCorrection)
        normalizeSampleProbability = sampleProbability / np.sum(sampleProbability)
        nextCondition = np.random.choice(condition, 1, p=list(normalizeSampleProbability))[0]

        newTargetGrid = samplePosition(self.bounds)
        return newTargetGrid, nextCondition


class StayInBoundary:
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        adjustedX, adjustedY = position
        if position[0] >= self.xMax:
            adjustedX = self.xMax
        if position[0] <= self.xMin:
            adjustedX = self.xMin
        if position[1] >= self.yMax:
            adjustedY = self.yMax
        if position[1] <= self.yMin:
            adjustedY = self.yMin
        checkedPosition = (adjustedX, adjustedY)
        return checkedPosition


def main():
    dimension = 15
    bounds = [0, 0, dimension - 1, dimension - 1]
    condition = [-5, -3, -1, 0, 1, 3, 5]
    counter = [0] * len(condition)
    minDistanceBetweenGrids = 1
    initialWorld = InitialWorld(bounds)
    target1Grid, target2Grid, playerGrid = initialWorld(minDistanceBetweenGrids)
    updateWorld = UpdateWorld(bounds, condition, counter)
    target2Grid, nextCondition = updateWorld(target1Grid, playerGrid)
    print(playerGrid, target2Grid, nextCondition)


if __name__ == "__main__":
    main()
