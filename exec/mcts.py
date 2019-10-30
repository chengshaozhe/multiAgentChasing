import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import pickle
import random
import json
from collections import OrderedDict

from model.algorithms.mcts import MCTS, ScoreChild, establishSoftmaxActionDist, SelectChild, Expand, RollOut, backup, InitializeChildren
import model.constrainedChasingEscapingEnv.envNoPhysics as env
import model.constrainedChasingEscapingEnv.reward as reward
from model.constrainedChasingEscapingEnv.policies import HeatSeekingContinuesDeterministicPolicy, HeatSeekingDiscreteDeterministicPolicy, stationaryAgentPolicy
from model.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from model.constrainedChasingEscapingEnv.analyticGeometryFunctions import computeAngleBetweenVectors

from model.episode import chooseGreedyAction,SampleTrajectory
from model.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal, TransiteForNoPhysics, Reset

import time
from model.trajectoriesSaveLoad import GetSavePath, saveToPickle

from model.inferChasing.discreteGridPolicy import ActHeatSeeking, \
    HeatSeekingPolicy, WolfPolicy

def sheepMCTS():

    xBoundary = [0,30]
    yBoundary = [0,30]
    numSimulations = 50

    killzoneRadius = 1
    maxRolloutSteps = 10

    actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1),(1,1), (1,-1), (-1,1), (-1,-1)]
    # actionSpace = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    numActionSpace = len(actionSpace)

    preyPowerRatio = 1.5
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
    predatorPowerRatio = 1
    wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

    numOfAgent = 2
    sheepId = 0
    wolfOneId = 1
    wolfTwoId = 2
    positionIndex = [0, 1]


    getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
    getPredatorOnePos = GetAgentPosFromState(wolfOneId, positionIndex)
    # getPredatorTwoPos = GetAgentPosFromState(wolfTwoId, positionIndex)
    stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)



    isTerminalOne = env.IsTerminal(getPredatorOnePos, getPreyPos, killzoneRadius)
    # isTerminalTwo =env.IsTerminal(getPredatorTwoPos, getPreyPos, killzoneRadius)
    # isTerminal=lambda state:isTerminalOne(state) or isTerminalTwo(state)

    isTerminal = isTerminalOne
    transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)
    reset = env.Reset(xBoundary, yBoundary, numOfAgent)


    wolfOnePolicy = HeatSeekingDiscreteDeterministicPolicy(
        wolfActionSpace, getPredatorOnePos, getPreyPos, computeAngleBetweenVectors)

    # wolfTwoPolicy=HeatSeekingDiscreteDeterministicPolicy(
    #     wolfActionSpace, getPredatorTwoPos, getPreyPos, computeAngleBetweenVectors)

    cInit = 1
    cBase = 100
    calculateScore = ScoreChild(cInit, cBase)
    selectChild = SelectChild(calculateScore)
    getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    def sheepTransit(state, action): return transitionFunction(
        state, [action, chooseGreedyAction(wolfOnePolicy(state))])

    # def sheepTransit(state, action): return transitionFunction(state, [action, chooseGreedyAction(wolfOnePolicy(state)), chooseGreedyAction(wolfTwoPolicy(state))])

    maxRunningSteps = 10
    aliveBonus = 1 / maxRunningSteps
    deathPenalty = -1
    rewardFunction = reward.RewardFunctionCompete(
        aliveBonus, deathPenalty, isTerminal)


    initializeChildren = InitializeChildren(
        sheepActionSpace, sheepTransit, getActionPrior)
    expand = Expand(isTerminal, initializeChildren)

    def rolloutPolicy(
        state): return sheepActionSpace[np.random.choice(range(numActionSpace))]
    rolloutHeuristicWeight = 1
    rolloutHeuristic = reward.HeuristicDistanceToTarget(
        rolloutHeuristicWeight, getPredatorOnePos, getPreyPos)


    rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                      rewardFunction, isTerminal, rolloutHeuristic)

    sheepPolicy = MCTS(numSimulations, selectChild, expand,
                rollout, backup, establishSoftmaxActionDist)

    return sheepPolicy

def main():

    while True:
        killzoneRadius = 30
        numSimulations = 200
        maxRunningSteps = 100
        numOfAgent = 2
        sheepId = 0
        wolfId = 1
        # wolf2Id = 2
        positionIndex = [0, 1]

        xBoundary = [0,600]
        yBoundary = [0,600]

        #prepare render
        from model.evaluateNoPhysicsEnvWithRender import Render, SampleTrajectoryWithRender
        import pygame as pg
        renderOn = True
        from pygame.color import THECOLORS
        screenColor = THECOLORS['black']
        circleColorList = [THECOLORS['green'], THECOLORS['red'],THECOLORS['orange']]
        circleSize = 10
        screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
        render = Render(numOfAgent, positionIndex,
                        screen, screenColor, circleColorList, circleSize)

        getPreyPos = GetAgentPosFromState(sheepId, positionIndex)
        getPredatorPos = GetAgentPosFromState(wolfId, positionIndex)
        # getPredator2Pos=GetAgentPosFromState(wolf2Id, positionIndex)
        stayInBoundaryByReflectVelocity = env.StayInBoundaryByReflectVelocity(xBoundary, yBoundary)


        # isTerminal1 = env.IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)
        # isTerminal2 =env.IsTerminal(getPredator2Pos, getPreyPos, killzoneRadius)

        # isTerminal=lambda state:isTerminal1(state) or isTerminal1(state)
        isTerminal = env.IsTerminal(getPredatorPos, getPreyPos, killzoneRadius)
        transitionFunction = env.TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

        reset = env.Reset(xBoundary, yBoundary, numOfAgent)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7)]
        numActionSpace = len(actionSpace)


        preyPowerRatio = 3
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        predatorPowerRatio = 2
        wolfActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))


        wolf1Policy = HeatSeekingDiscreteDeterministicPolicy(
            wolfActionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors)

        # wolf2Policy=HeatSeekingDiscreteDeterministicPolicy(
            # wolfActionSpace, getPredator2Pos, getPreyPos, computeAngleBetweenVectors)
        # select child
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(sheepActionSpace) for action in sheepActionSpace}

    # load chase nn policy
        def sheepTransit(state, action): return transitionFunction(
            state, [action, chooseGreedyAction(wolf1Policy(state))])
        # def sheepTransit(state, action): return transitionFunction(
        #     state, [action, chooseGreedyAction(wolf1Policy(state)), chooseGreedyAction(wolf2Policy(state))])

        # reward function

        aliveBonus = 1 / maxRunningSteps
        deathPenalty = -1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # reward function with wall
        # safeBound = 80
        # wallDisToCenter = xBoundary[-1]/2
        # wallPunishRatio = 3
        # rewardFunction = reward.RewardFunctionWithWall(aliveBonus, deathPenalty, safeBound, wallDisToCenter, wallPunishRatio, isTerminal,getPreyPos)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            sheepActionSpace, sheepTransit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return sheepActionSpace[np.random.choice(range(numActionSpace))]

        # rollout
        rolloutHeuristicWeight = 1
        rolloutHeuristic = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getPredatorPos, getPreyPos)
        maxRolloutSteps = 10

        rollout = RollOut(rolloutPolicy, maxRolloutSteps, sheepTransit,
                          rewardFunction, isTerminal, rolloutHeuristic)

        sheepPolicy = MCTS(numSimulations, selectChild, expand,
                    rollout, backup, establishSoftmaxActionDist)

        # sheepPolicy = mctsPolicy()

        # All agents' policies

        policy = lambda state:[sheepPolicy(state),wolf1Policy(state)]
        # policy = lambda state:[sheepPolicy(state),wolf1Policy(state),wolf2Policy(state)]


        # sampleTrajectory=SampleTrajectory(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction)

        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitionFunction, isTerminal, reset, chooseGreedyAction,render,renderOn)

        startTime = time.time()
        trajectories = [sampleTrajectory(policy)]

        finshedTime = time.time() - startTime

        print('lenght:',len(trajectories[0]))

        print('time:',finshedTime)

if __name__ == "__main__":
    main()