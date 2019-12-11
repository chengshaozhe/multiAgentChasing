import numpy as np
from scipy import stats

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


def calculatePdf(direction, degeree):
    x = direction - degeree
    return stats.vonmises.pdf(x, assumePrecision) * 2


def vecToAngle(vector):
    return np.angle(complex(vector[0], vector[1]))


class TansferHumanActionToDiscreteDirection():
    def __init__(self, assumePrecision):
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.actionSpace = actionSpace
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]
        self.chooseAction = chooseAction

    def __call__(self, humanAction):
        actionDict = {}
        if humanAction != (0, 0):
            humanDirection = self.vecToAngle(humanAction)
            pdf = np.array([calculatePdf(humanDirection, degeree) for degree in self.degreeList])
            normProb = pdf / pdf.sum()
            [actionDict.update({action: prob}) for action, prob in zip(actionSpace, normProb)]
            actionDict.update({(0, 0): 0})
        else:
            [actionDict.update({action: 0}) for action in actionSpace]
            actionDict.update({(0, 0): 1})
        action = self.chooseAction(actionDict)
        return action


# class ComposeCenterControl:
#     def __init__(self,):

#     def __call__(self, humanAction, ):

#         return centerControlActionDict

class CalTransitionLikelihood:
    def __init__(self, transitAgents):
        self.transitAgents = transitAgents

    def __call__(self, state, allAgentsActions, nextState):
        agentsNextIntendedState = self.transitAgents(state, allAgentsActions)
        transitionLikelihood = 1 if np.all(agentsNextIntendedState == nextState) else 0
        return transitionLikelihood


class CenterControlPolicyLikelihood:
    def __init__(self, NNPolicy):
        self.NNPolicy = NNPolicy

    def __call__(self, state, action):
        sheepStates, wolvesState = state
        wolfState1, wolfState2 = wolvesState
        likelihoodList = [self.NNPolicy[sheepState, wolfState1, wolfState2].get(action) for sheepState in sheepStates]
        return likelihoodList


class InferGoalWithoutAction:
    def __init__(self, calLikelihood):
        self.transiteLikelihood = transiteLikelihood
        self.getPolicyLikelihoodList = getPolicyLikelihoodList
        self.actionSpace = actionSpace

    def __call__(self, state, nextState, prior):

        likelihoodList = [[self.transiteLikelihood(state, action, nextState) * policyLikelihood for policyLikelihood in self.getPolicyLikelihoodList(state, action)] for action in self.actionSpace]

        likelihoodActionsIntegratedOut = np.sum(np.array(likelihoodList), axis=0)

        priorLikelihoodPair = zip(prior, likelihoodActionsIntegratedOut)
        posteriorUnnormalized = [prior * likelihood for prior, likelihood in priorLikelihoodPair]
        evidence = sum(posteriorUnnormalized)

        posterior = [posterior / evidence for posterior in posteriorUnnormalized]

        return posterior


class InferGoalWithAction:
    def __init__(self, partenerPolicy):
        self.calLikelihood = partenerPolicy

    def __call__(self, priorList, wolvesState, wolvesAction, sheepStates):
        centrolCentrolAction = (self.tansferHumanActionToDiscreteDirection(action) for action in wolvesAction)
        likelihoodList = [self.calLikelihood(sheepState, wolvesState, centrolCentrolAction) for sheepState in sheepStates]
        evidence = sum([prior * likelihood for (prior, likelihood) in zip(priorList, likelihoodList)])
        posteriorList = [prior * likelihood / evidence for (prior, likelihood) in zip(priorList, likelihoodList)]
        return posteriorList


def calTargetFromPosterior(posteriorList):
    target = np.max(posteriorList)
    return target


class InferCommitmentAndDraw:
    def __init__(self,):
        self.inferOneStep = inferOneStep

    def __call__(self,):

        while True:
            plt.plot(x, goalPosteriori, label=lables[i])


if __name__ == '__main__':

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 3
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 2
    wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))

    numStateSpace = 6
    numSheepActionSpace = len(sheepActionSpace)
    numWolvesActionSpace = len(wolvesActionSpace)
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
    generateModelList = [generateSheepModel, generateWolvesModel]

    sheepDepth = 5
    wolfDepth = 9
    depthList = [sheepDepth, wolfDepth]
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    trainableAgentIds = [sheepId, wolvesId]

    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    # load Model save dir
    NNModelSaveExtension = ''
    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'multiAgentTrain', 'multiMCTSAgentResNetNoPhysicsCenterControlWithPreTrain', 'NNModelRes')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)
    wolfModelPath=os.path.join(dirName,'preTrainModel','agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=100_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    restoredNNModel = restoreVariables(multiAgentNNmodel[wolvesId], wolfModelPath)
    multiAgentNNmodel[wolvesId] = restoredNNModel
    wolfPolicy = ApproximatePolicy(multiAgentNNmodel[wolvesId], wolvesActionSpace)

    sheepModelPath=os.path.join(dirName,'preTrainModel','agentId=0_depth=5_learningRate=0.0001_maxRunningSteps=150_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    sheepTrainedModel = restoreVariables(multiAgentNNmodel[sheepId], sheepModelPath)
    sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)

    def policy(state): return [sheepPolicy(state), wolfPolicy(state)]

    chooseActionList = [chooseGreedyAction, chooseGreedyAction]

    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()

    prior =  [0.5, 0.5]
    x = np.arange(2)
    y = np.array(prior).T

    lables = ['goalA']
    for i in range(len(lables)):
        line, = plt.plot(x, prior[i], label=lables[i])

    # line, = plt.plot(x, y)
    ax = plt.gca()

    state = initState
    nextState = initState
    while True:
        actionDists = policy(state)
        action = [choose(action) for choose, action in zip(chooseActionList, actionDists)]

        goalPosteriori = inferGoalWithAction(state, nextState, prior)
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
