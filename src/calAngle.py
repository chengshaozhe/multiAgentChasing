import numpy as np
from scipy import stats


def calculatePdf(x, assumePrecision):
    return stats.vonmises.pdf(x, assumePrecision) * 2


def vecToAngle(vector):
    return np.angle(complex(vector[0], vector[1]))


class CalHeatSeekingActionProb():
    def __init__(self, assumePrecision, actionSpace):
        self.assumePrecision = assumePrecision
        self.actionSpace = actionSpace
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]

    def __call__(self, sheepState, wolfState):
        continuousAction = np.array(sheepState) - np.array(wolfState)
        discreteAction = self.vecToAngle(continuousAction)
        pdf = np.array([calculatePdf(discreteAction - degree, self.assumePrecision) for degree in self.degreeList])
        normProb = pdf / pdf.sum()
        actionDict = {action: prob for action, prob in zip(actionSpace, normProb)}
        return actionDict


if __name__ == '__main__':
    assumePrecision = 5
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7)]
    calHeatSeekingActionProb = CalHeatSeekingActionProb(assumePrecision, actionSpace)
    sheepState = [10, 18]
    wolfState = [20, 20]
    print(calHeatSeekingActionProb(sheepState, wolfState))
