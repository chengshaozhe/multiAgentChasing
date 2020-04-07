
import numpy as np
import random
import os


class ExpSheepPolicy:
    def __init__(self, passerbyPolicy, singlePolicy, multiPolicy, inferCurrentWolf, chooseAction):
        self.passerbyPolicy = passerbyPolicy
        self.singlePolicy = singlePolicy
        self.multiPolicy = multiPolicy
        self.inferCurrentWolf = inferCurrentWolf
        self.chooseAction = chooseAction

    def __call__(self, dequeState):
        initialState = dequeState[0]
        currentState = dequeState[-1]
        initialState = (tuple(initialState[0]), (tuple(initialState[1]), tuple(initialState[2])))

        currentState = (tuple(currentState[0]), (tuple(currentState[1]), tuple(currentState[2])))

        goal = self.inferCurrentWolf(initialState, currentState)
        if goal == [True, True]:
            policyForCurrentStateDict = self.multiPolicy[currentState]
            print('ggggg')
        elif goal == [True, False]:
            # policyForCurrentStateDict= self.singlePolicy[(currentState[0],currentState[1][0])]
            policyForCurrentStateDict = self.passerbyPolicy[currentState]
            print('orange!!!')

        elif goal == [False, True]:
            # policyForCurrentStateDict= self.singlePolicy[(currentState[0],currentState[1][1])]
            policyForCurrentStateDict = self.passerbyPolicy[currentState]
            print('red!!!')

        elif goal == [False, False]:
            policyForCurrentStateDict = self.passerbyPolicy[currentState]

        actionMaxList = [action for action in policyForCurrentStateDict.keys() if policyForCurrentStateDict[action] == np.max(list(policyForCurrentStateDict.values()))]
        action = random.choice(actionMaxList)
        # action = self.chooseAction(policyForCurrentStateDict)
        return action


def inferGoalGridEnv(initialState, finalState):
    sheepIndex = 0
    goal = [False, False]
    if calculateGridDistance(finalState[1][0], initialState[sheepIndex]) < calculateGridDistance(initialState[1][0], initialState[sheepIndex]):
        goal[0] = True
    if calculateGridDistance(finalState[1][1], initialState[sheepIndex]) < calculateGridDistance(initialState[1][1], initialState[sheepIndex]):
        goal[1] = True
    return goal


def calculateGridDistance(gridA, gridB):
    return np.linalg.norm(np.array(gridA) - np.array(gridB), ord=1)


def chooseGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def sampleAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedAction = actions[selectedIndex]
    return selectedAction


class SoftmaxAction:
    def __init__(self, softMaxBeta):
        self.softMaxBeta = softMaxBeta

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        probs = list(actionDist.values())
        normlizedProbs = list(np.divide(np.exp(np.multiply(self.softMaxBeta, probs)), np.sum(np.exp(np.multiply(self.softMaxBeta, probs)))))
        selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
        selectedAction = actions[selectedIndex]
        return selectedAction
