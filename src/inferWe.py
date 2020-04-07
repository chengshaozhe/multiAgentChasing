import numpy as np
import itertools as it
from scipy import stats
import os
import sys
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
from src.sheepPolicy import GenerateModel, ApproximatePolicy, restoreVariables


class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity


class UnpackCenterControlAction:
    def __init__(self, centerControlIndexList):
        self.centerControlIndexList = centerControlIndexList

    def __call__(self, centerControlAction):
        upackedAction = []
        for index, action in enumerate(centerControlAction):
            if index in self.centerControlIndexList:
                [upackedAction.append(subAction) for subAction in action]
            else:
                upackedAction.append(action)
        return np.array(upackedAction)


class TransiteForNoPhysicsWithCenterControlAction():
    def __init__(self, stayInBoundaryByReflectVelocity, unpackCenterControlAction):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.unpackCenterControlAction = unpackCenterControlAction

    def __call__(self, state, action):
        actionFortansit = self.unpackCenterControlAction(action)
        newState = state + np.array(actionFortansit)
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, actionFortansit)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState


def chooseGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def calculatePdf(x, assumePrecision):
    return stats.vonmises.pdf(x, assumePrecision) * 2


def vecToAngle(vector):
    return np.angle(complex(vector[0], vector[1]))


class TansferContinuousnActionToDiscreteAction():
    def __init__(self, assumePrecision, actionSpace, chooseAction):
        self.assumePrecision = assumePrecision
        self.actionSpace = actionSpace
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]
        self.chooseAction = chooseAction

    def __call__(self, continuousAction):
        actionDict = {}
        if continuousAction != (0, 0):
            discreteAction = self.vecToAngle(continuousAction)
            pdf = np.array([calculatePdf(discreteAction - degree, self.assumePrecision) for degree in self.degreeList])
            normProb = pdf / pdf.sum()
            [actionDict.update({action: prob}) for action, prob in zip(actionSpace, normProb)]
            actionDict.update({(0, 0): 0})
        else:
            [actionDict.update({action: 0}) for action in actionSpace]
            actionDict.update({(0, 0): 1})
        action = self.chooseAction(actionDict)
        return action


class InferGoalWithAction:
    def __init__(self, getPolicyLikelihoodList, tansferContinuousnActionToDiscreteAction):
        self.getPolicyLikelihoodList = getPolicyLikelihoodList
        self.tansferContinuousnActionToDiscreteAction = tansferContinuousnActionToDiscreteAction

    def __call__(self, priorList, state, action):
        sheepAction, wolvesAction = action
        centrolCentrolAction = (self.tansferContinuousnActionToDiscreteAction(action) for action in wolvesAction)
        likelihoodList = self.getPolicyLikelihoodList(state, action)
        evidence = sum([prior * likelihood for (prior, likelihood) in zip(priorList, likelihoodList)])
        posteriorList = [prior * likelihood / evidence for (prior, likelihood) in zip(priorList, likelihoodList)]
        return posteriorList


def calTargetFromPosterior(posteriorList):
    target = np.max(posteriorList)
    return target


class GetContinualTransitionLikelihood:
    def __init__(self, transitAgents, tansferContinuousnActionToDiscreteAction):
        self.transitAgents = transitAgents
        self.tansferContinuousnActionToDiscreteAction = tansferContinuousnActionToDiscreteAction

    def __call__(self, state, allAgentsActions, nextState):
        actions = [self.tansferContinuousnActionToDiscreteAction(action) for action in allAgentsActions]
        agentsNextIntendedState = self.transitAgents(state, actions)
        transitionLikelihood = 1 if np.all(agentsNextIntendedState == nextState) else 0
        return transitionLikelihood


class GetTransitionLikelihood:
    def __init__(self, transitAgents):
        self.transitAgents = transitAgents

    def __call__(self, state, allAgentsActions, nextState):
        agentsNextIntendedState = self.transitAgents(state, allAgentsActions)
        transitionLikelihood = 1 if np.all(agentsNextIntendedState == nextState) else 0
        return transitionLikelihood


class GetCenterControlPolicyLikelihood:
    def __init__(self, centerControlPolicy):
        self.centerControlPolicy = centerControlPolicy

    def __call__(self, state, action):
        sheepStates, wolvesState = state
        wolfState1, wolfState2 = wolvesState

        likelihoodList = [self.centerControlPolicy[sheepState, wolfState1, wolfState2].get(weAction) for sheepState in sheepStates]
        return likelihoodList


class InferWeWithoutAction:
    def __init__(self, getTransitionLikelihood, getPolicyLikelihoodList, weActionSpace):
        self.getTransitionLikelihood = getTransitionLikelihood
        self.getPolicyLikelihoodList = getPolicyLikelihoodList
        self.weActionSpace = weActionSpace

    def __call__(self, state, nextState, prior):
        likelihoodList = [[self.getTransitionLikelihood(state, action, nextState) * policyLikelihood for policyLikelihood in self.getPolicyLikelihoodList(state, action)] for action in self.weActionSpace]
        likelihoodActionsIntegratedOut = np.sum(np.array(likelihoodList), axis=0)

        priorLikelihoodPair = zip(prior, likelihoodActionsIntegratedOut)
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in priorLikelihoodPair]
        evidence = sum(posteriorUnnormalized)

        posterior = [posterior / evidence for posterior in posteriorUnnormalized]
        return posterior


class InferCommitmentAndDraw:
    def __init__(self,):
        self.inferOneStep = inferOneStep

    def __call__(self,):

        while True:
            plt.plot(x, goalPosteriori, label=lables[i])


if __name__ == '__main__':
    sheepId = 0
    wolvesId = 2
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 2
    wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))

    numStateSpace = 6
    numSheepActionSpace = len(sheepActionSpace)
    numWolvesActionSpace = len(wolvesActionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
    generateModelList = [generateSheepModel, generateSheepModel, generateWolvesModel]

    sheepDepth = 5
    wolfDepth = 9
    depthList = [sheepDepth, sheepDepth, wolfDepth]
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'

    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    wolfModelPath = os.path.join('..', 'preTrainModel', 'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    restoredNNModel = restoreVariables(multiAgentNNmodel[wolvesId], wolfModelPath)
    multiAgentNNmodel[wolvesId] = restoredNNModel
    centerControlPolicy = ApproximatePolicy(multiAgentNNmodel[wolvesId], wolvesActionSpace)

    sheepModelPath = os.path.join('..', 'preTrainModel', 'agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    sheepTrainedModel = restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)
    sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

    def policy(state): return [sheepPolicy(state), centerControlPolicy(state)]

    chooseActionList = [chooseGreedyAction, chooseGreedyAction]

    xBoundary = [0, 800]
    yBoundary = [0, 800]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    centerControlIndexList = 1
    unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)
    transitAgents = TransiteForNoPhysicsWithCenterControlAction(stayInBoundaryByReflectVelocity, unpackCenterControlAction)
    getTransitionLikelihood = GetTransitionLikelihood(transitAgents)
    getPolicyLikelihoodList = GetCenterControlPolicyLikelihood(centerControlPolicy)
    inferGoalWithoutAction = InferWeWithoutAction(getTransitionLikelihood, getPolicyLikelihoodList, wolvesActionSpace)

    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()

    prior = [0.5, 0.5]
    x = np.arange(2)
    y = np.array(prior).T

    lables = ['goalA']
    for i in range(len(lables)):
        line, = plt.plot(x, y, label=lables[i])

    # line, = plt.plot(x, y)
    ax = plt.gca()
    initState = ((0, 0), (10, 10), (20, 20))
    state = initState,
    nextState = initState
    while True:
        actionDists = policy(state)
        action = [choose(action) for choose, action in zip(chooseActionList, actionDists)]

        goalPosteriori = inferGoalWithoutAction(state, nextState, prior)
        goalPosteriorList.append(goalPosteriori)

        newNextState = transition(nextState, action)
        nextState = newNextState
        state = nextState
        prior = goalPosteriori

        x_new = np.arange(len(goalPosteriorList))
        y_new = np.array(goalPosteriorList).T

        x = np.append(x, new_x)
        y = np.append(y, new_y)
        line.set_xdata(x)
        line.set_ydata(y)
        ax.relim()
        ax.autoscale_view(True, True, True)  # rescale plot view
        plt.draw()  # plot new figure
        plt.pause(1e-17)  # pause to show the figure
