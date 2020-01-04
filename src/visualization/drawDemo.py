import pygame as pg
import numpy as np
import os

class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        return


class DrawState:
    def __init__(self, fps, screen, colorSpace, circleSize, agentIdsToDraw, positionIndex, saveImage, imagePath, drawBackGround, updateColorByPosterior):
        self.fps = fps
        self.screen = screen
        self.colorSpace = colorSpace
        self.circleSize = circleSize
        self.agentIdsToDraw = agentIdsToDraw
        self.xIndex, self.yIndex = positionIndex
        self.saveImage = saveImage
        self.imagePath = imagePath
        self.drawBackGround = drawBackGround
        self.updateColorByPosterior = updateColorByPosterior

    def __call__(self, state, posterior = None):
        fpsClock = pg.time.Clock()
        
        self.drawBackGround()
        circleColors = self.updateColorByPosterior(self.colorSpace, posterior)
        for agentIndex in self.agentIdsToDraw:
            agentPos = [np.int(state[agentIndex][self.xIndex]), np.int(state[agentIndex][self.yIndex])]
            agentColor = tuple(circleColors[agentIndex])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)
        pg.display.flip()
        
        if self.saveImage == True:
            filenameList = os.listdir(self.imagePath)
            pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        
        fpsClock.tick(self.fps)
        return self.screen


class ChaseTrialWithTraj:
    def __init__(self, stateIndex, drawState):
        self.stateIndex = stateIndex
        self.drawState = drawState

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory)):
            state = trajectory[timeStepIndex][self.stateIndex]
            screen = self.drawState(state)
        return 
