import numpy as np
from scipy import stats


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
    return targetIndex


class InferCommitmentAndDraw:
    def __init__(self,):
        self.inferOneStep = inferOneStep

    def __call__(self,):

        while True:
            plt.plot(x, goalPosteriori, label=lables[i])


if __name__ == '__main__':
    centerControlPolicy =

    actionSpace =

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(2)
    y = np.array(goalPosteriorList).T

    # lables = ['goalA','goalB']
    # for i in range(len(lables)):
    #     plt.plot(x, y, label=lables[i])

    plt.ion()
    line, = plt.plot(x, y)
    ax = plt.gca()

    state = initState
    nextState = initState
    while True:
        action = policy(state)

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
