import numpy as np
from random import randint

class TransiteForNoPhysics():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, state, action):
        newState = state + np.array(action)
        checkedNewStateAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newState, action)]
        newState, newAction = list(zip(*checkedNewStateAndVelocities))
        return newState

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

class Reset:
    def __init__(self, gridSize, lowerBound, agentCount):
        self.gridX, self.gridY = gridSize
        self.lowerBound = lowerBound
        self.agentCount = agentCount

    def __call__(self):
        startState = [(randint(self.lowerBound, self.gridX), randint(self.lowerBound, self.gridY)) for _ in range(self.agentCount)]
        return startState


class StayWithinBoundary:
    def __init__(self, gridSize, lowerBoundary):
        self.gridX, self.gridY = gridSize
        self.lowerBoundary = lowerBoundary

    def __call__(self, nextIntendedState):
        nextX, nextY = nextIntendedState
        if nextX < self.lowerBoundary:
            nextX = self.lowerBoundary
        if nextX > self.gridX:
            nextX = self.gridX
        if nextY < self.lowerBoundary:
            nextY = self.lowerBoundary
        if nextY > self.gridY:
            nextY = self.gridY
        return nextX, nextY

class GetPullingForceValue:
    def __init__(self, distanceForceRatio):
        self.distanceForceRatio = distanceForceRatio

    def __call__(self, pullersRelativeLocation):
        relativeLocationArray = np.array(pullersRelativeLocation)
        distance = np.sqrt(relativeLocationArray.dot(relativeLocationArray))
        force = int(np.floor(distance / self.distanceForceRatio + 1))
        return force


class SamplePulledForceDirection:
    def __init__(self, calculateAngle, forceSpace, lowerBoundAngle, upperBoundAngle):
        self.calculateAngle = calculateAngle
        self.forceSpace = forceSpace
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle

    def __call__(self, pullersRelativeLocation):
        if np.all(np.array(pullersRelativeLocation) == np.array((0, 0))):
            return 0, 0

        forceAndRelativeLocAngle = [self.calculateAngle(pullersRelativeLocation, forceDirection) for forceDirection in
                                    self.forceSpace]

        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        angleFilter = [angleWithinRange(angle) for angle in forceAndRelativeLocAngle]

        forceDirections = [force for force, index in zip(self.forceSpace, angleFilter) if index]

        forceDirectionsLikelihood = [1 / len(forceDirections)] * len(forceDirections)
        forceDirectionSampleIndex = list(np.random.multinomial(1, forceDirectionsLikelihood)).index(1)
        sampledForceDirection = forceDirections[forceDirectionSampleIndex]

        return sampledForceDirection


class GetPulledAgentForce:
    def __init__(self, getPullingAgentPosition, getPulledAgentPosition, samplePulledForceDirection, getPullingForceValue):
        self.getPullingAgentPosition = getPullingAgentPosition
        self.getPulledAgentPosition = getPulledAgentPosition
        self.samplePulledForceDirection = samplePulledForceDirection
        self.getPullingForceValue = getPullingForceValue

    def __call__(self, state):
        pullingAgentState = self.getPullingAgentPosition(state)
        pulledAgentState = self.getPulledAgentPosition(state)

        pullersRelativeLocation = np.array(pullingAgentState) - np.array(pulledAgentState)

        pulledDirection = self.samplePulledForceDirection(pullersRelativeLocation)
        pullingForceValue = self.getPullingForceValue(pullersRelativeLocation)

        pulledAgentForce = np.array(pulledDirection) * pullingForceValue

        return pulledAgentForce

class GetAgentsForce:
    def __init__(self, getPulledAgentForce, pulledAgentIndex, noPullingAgentIndex, pullingAgentIndex):
        self.getPulledAgentForce = getPulledAgentForce
        self.pulledAgentIndex = pulledAgentIndex
        self.noPullingAgentIndex = noPullingAgentIndex
        self.pullingAgentIndex = pullingAgentIndex

    def __call__(self, state):
        pulledAgentForce = np.array(self.getPulledAgentForce(state))
        pullingAgentForce = -pulledAgentForce
        noPullAgentForce = (0,0)

        unorderedAgentsForce = [pulledAgentForce, noPullAgentForce, pullingAgentForce]
        agentsIDOrder = [self.pulledAgentIndex, self.noPullingAgentIndex, self.pullingAgentIndex]

        rearrangeList = lambda unorderedList, order: list(np.array(unorderedList)[np.array(order).argsort()])
        agentsForce = rearrangeList(unorderedAgentsForce, agentsIDOrder)
        return agentsForce

class Transition:
    def __init__(self, stayWithinBoundary, getAgentsForce):
        self.stayWithinBoundary = stayWithinBoundary
        self.getAgentsForce = getAgentsForce

    def __call__(self, actionList, state):
        agentsForce = self.getAgentsForce(state)
        agentsIntendedState = np.array(state) + np.array(agentsForce) + np.array(actionList)
        agentsNextState = [self.stayWithinBoundary(intendedState) for intendedState in agentsIntendedState]
        return agentsNextState


class IsTerminal:
    def __init__(self, locatePredator, locatePrey):
        self.locatePredator = locatePredator
        self.locatePrey = locatePrey

    def __call__(self, state):
        predatorPosition = self.locatePredator(state)
        preyPosition = self.locatePrey(state)

        if np.all(np.array(predatorPosition) == np.array(preyPosition)):
            return True
        else:
            return False

