import numpy as np
import random
import copy 

def stationaryAgentPolicy(state):
    return {(0, 0): 1}

class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionDist = {action: 1 / len(self.actionSpace) for action in self.actionSpace}
        return actionDist

class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors):
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.computeAngleBetweenVectors = computeAngleBetweenVectors

    def __call__(self, state):
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        heatSeekingVector = np.array(preyPosition) - np.array(predatorPosition)
        anglesBetweenHeatSeekingAndActions = np.array([self.computeAngleBetweenVectors(heatSeekingVector, np.array(action)) for action in self.actionSpace]).flatten()
        minIndex = np.argwhere(anglesBetweenHeatSeekingAndActions == np.min(anglesBetweenHeatSeekingAndActions)).flatten()
        actionsShareProbability = [tuple(self.actionSpace[index]) for index in minIndex]
        actionDist = {action: 1 / len(actionsShareProbability) if action in actionsShareProbability else 0 for action in self.actionSpace}
        return actionDist


class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        assert actionL2Norm != 0
        action = action / actionL2Norm
        action *= self.actionMagnitude

        actionTuple = tuple(action)
        actionDist = {actionTuple: 1}
        return actionDist

class PolicyOnChangableIntention:
    def __init__(self, lastState, intentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention):
        self.lastState = lastState
        self.intentionPrior = intentionPrior
        self.updateIntentionDistribution = updateIntentionDistribution
        self.chooseIntention = chooseIntention
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention
        self.formerIntentionPriors = [self.intentionPrior.copy()]

    def __call__(self, state):
        if not isinstance(self.lastState, type(None)):
            intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, state)
        else:
            intentionPosterior = self.intentionPrior.copy()
        intentionId = self.chooseIntention(intentionPosterior)
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intentionId)
        centralControlActionDist = self.policyGivenIntention(stateRelativeToIntention)
        self.lastState = state.copy()
        self.intentionPrior = intentionPosterior.copy()
        self.formerIntentionPriors.append(intentionPosterior.copy())
        return centralControlActionDist

class SoftPolicy:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter) for probability in list(actionDist.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenActionDist = dict(zip(actions, softenNormalizedProbabilities))
        return softenActionDist

class ResetPolicy:
    def __init__(self, attributeValues, policyObjects, returnAttributes = None):
        self.attributeValues = attributeValues
        self.policyObjects = policyObjects
        self.returnAttributes = returnAttributes
    
    def __call__(self):
        returnAttributeValues = None
        if self.returnAttributes:
            returnAttributeValues = list(zip(*[list(zip(*[getattr(individualPolicy, attribute).copy() for individualPolicy in self.policyObjects])) 
                for attribute in self.returnAttributes]))
        [[setattr(policy, attribute, value) for attribute, value in zip(list(attributeValue.keys()), copy.deepcopy(list(attributeValue.values())))] 
                for policy, attributeValue in zip(self.policyObjects, self.attributeValues)]
        return returnAttributeValues
