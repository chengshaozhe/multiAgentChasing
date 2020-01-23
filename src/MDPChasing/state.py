import numpy as np
import bisect
import functools as ft

class GetAgentPosFromState:
    def __init__(self, agentId, posIndex):
        self.agentId = agentId
        self.posIndex = posIndex

    def __call__(self, state):
        state = np.asarray(state)
        agentStates = state[self.agentId]
        agentPos = np.asarray([state[self.posIndex] for state in agentStates])
        return agentPos

class GetStateForPolicyGivenIntention:
    def __init__(self, agentSelfId):
        self.agentSelfId = agentSelfId

    def __call__(self, state, intentionId):
        IdsRelativeToIntention = list(self.agentSelfId.copy())
        for Id in list(intentionId):
            bisect.insort(IdsRelativeToIntention, Id)
        sortedIds = np.array(IdsRelativeToIntention)
        stateRelativeToIntention = np.array(state)[sortedIds]
        return stateRelativeToIntention
