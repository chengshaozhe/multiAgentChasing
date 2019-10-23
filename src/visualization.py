import pygame as pg
import numpy as np
import os
from pygame.color import THECOLORS


def calculateIncludedAngle(vector1, vector2):
    includedAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
    return includedAngle


def findQuadrant(vector):
    quadrant = 0
    if vector[0] > 0 and vector[1] > 0:
        quadrant = 0


class InitializeScreen:
    def __init__(self, screenWidth, screenHeight, fullScreen):
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.fullScreen = fullScreen

    def __call__(self):
        pg.init()
        if self.fullScreen:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight), pg.FULLSCREEN)
        else:
            screen = pg.display.set_mode((self.screenWidth, self.screenHeight))
        pg.display.init()
        pg.fastevent.init()
        return screen


def drawText(screen, text, textColorTuple, textPositionTuple):
    font = pg.font.Font(None, 50)
    textObj = font.render(text, 1, textColorTuple)
    screen.blit(textObj, textPositionTuple)
    return


class GiveExperimentFeedback():
    def __init__(self, screen, textColorTuple, screenWidth, screenHeight):
        self.screen = screen
        self.textColorTuple = textColorTuple
        self.screenHeight = screenHeight
        self.screenWidth = screenWidth

    def __call__(self, trialIndex, score):
        self.screen.fill((0, 0, 0))
        for j in range(trialIndex + 1):
            drawText(self.screen, "No. " + str(j + 1) + " experiment" + "  score: " + str(score[j]), self.textColorTuple,
                     (self.screenWidth / 5, self.screenHeight * (j + 3) / 12))
        pg.display.flip()
        pg.time.wait(3000)


class DrawBackground():
    def __init__(self, screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple):
        self.screen = screen
        self.gridSize = gridSize
        self.leaveEdgeSpace = leaveEdgeSpace
        self.widthLineStepSpace = np.int(screen.get_width() / (gridSize + 2 * self.leaveEdgeSpace))
        self.heightLineStepSpace = np.int(screen.get_height() / (gridSize + 2 * self.leaveEdgeSpace))
        self.backgroundColor = backgroundColor
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.textColorTuple = textColorTuple

    def __call__(self, currentTime, currentScore):
        self.screen.fill((0, 0, 0))
        pg.draw.rect(self.screen, self.backgroundColor, pg.Rect(np.int(self.leaveEdgeSpace * self.widthLineStepSpace), np.int(self.leaveEdgeSpace * self.heightLineStepSpace), np.int(self.gridSize * self.widthLineStepSpace), np.int(self.gridSize * self.heightLineStepSpace)))

        # seconds = currentTime / 1000
        # drawText(self.screen, 'Time: ' + str("%4.1f" % seconds) + 's', THECOLORS['white'], (self.widthLineStepSpace * 15, self.leaveEdgeSpace * 3))
        # drawText(self.screen, 'Score: ' + str(currentScore), self.textColorTuple, (self.widthLineStepSpace * 50, self.leaveEdgeSpace * 3))
        # drawText(self.screen, 'Score: ' + str(currentScore), self.textColorTuple, (self.widthLineStepSpace * 50, self.leaveEdgeSpace * 3))
        return


class DrawNewState():
    def __init__(self, screen, drawBackground, targetColor, playerColors, targetRadius, playerRadius):
        self.screen = screen
        self.drawBackground = drawBackground
        self.targetColor = targetColor
        self.playerColors = playerColors
        self.targetRadius = targetRadius
        self.playerRadius = playerRadius
        self.leaveEdgeSpace = drawBackground.leaveEdgeSpace
        self.widthLineStepSpace = drawBackground.widthLineStepSpace
        self.heightLineStepSpace = drawBackground.heightLineStepSpace

    def __call__(self, targetPositionA, targetPositionB, targetPositionC, targetPositionD, playerPositions, currentTime, currentScore):
        self.drawBackground(currentTime, currentScore)
        pg.draw.circle(self.screen, self.targetColor[0], [np.int((targetPositionA[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPositionA[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius + 2)
        pg.draw.circle(self.screen, self.targetColor[1], [np.int((targetPositionB[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPositionB[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius + 2)
        pg.draw.circle(self.screen, self.targetColor[2], [np.int((targetPositionC[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPositionC[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius)
        pg.draw.circle(self.screen, self.targetColor[3], [np.int((targetPositionD[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((targetPositionD[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.targetRadius)

        for playerPosition, playerColor in zip(playerPositions, self.playerColors):
            pg.draw.circle(self.screen, playerColor, [np.int((playerPosition[0] + self.leaveEdgeSpace + 0.5) * self.widthLineStepSpace), np.int((playerPosition[1] + self.leaveEdgeSpace + 0.5) * self.heightLineStepSpace)], self.playerRadius)
        return self.screen


class DrawImage():
    def __init__(self, screen):
        self.screen = screen
        self.screenCenter = (self.screen.get_width() / 2, self.screen.get_height() / 2)

    def __call__(self, image):
        imageRect = image.get_rect()
        imageRect.center = self.screenCenter
        pause = True
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
        self.screen.fill((0, 0, 0))
        self.screen.blit(image, imageRect)
        pg.display.flip()
        while pause:
            pg.time.wait(10)
            for event in pg.event.get():
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    pause = False
                elif event.type == pg.QUIT:
                    pg.quit()
            pg.time.wait(10)
        pg.event.set_blocked([pg.KEYDOWN, pg.KEYUP, pg.QUIT])

class AttributionTrail:
    def __init__(self,totalScore,drawAttributionTrail):
        self.totalScore = totalScore
        self.actionDict = [{ pg.K_LEFT: -1, pg.K_RIGHT: 1}, {pg.K_a: -1, pg.K_d: 1}]
        self.comfirmDict=[pg.K_RETURN,pg.K_SPACE]
        self.distributeUnit=0.1
        self.drawAttributionTrail=drawAttributionTrail
    def __call__(self,eatenFlag, hunterFlag):
        hunterid=hunterFlag.index(True)
        attributionScore=[0,0]
        attributorPercent=0.5#
        pause=True
        self.drawAttributionTrail(hunterid,attributorPercent)
        pg.event.set_allowed([pg.KEYDOWN])

        attributionDelta=0
        stayAttributionBoudray=lambda attributorPercent:max(min(attributorPercent,1),0)
        while  pause:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                if event.type == pg.KEYDOWN:
                    print(event.key)
                    if event.key in self.actionDict[hunterid].keys():
                        attributionDelta = self.actionDict[hunterid][event.key]*self.distributeUnit
                        print(attributionDelta)

                        attributorPercent=stayAttributionBoudray(attributorPercent+attributionDelta)

                        self.drawAttributionTrail(hunterid,attributorPercent)
                    elif event.key == self.comfirmDict[hunterid]:
                        pause=False
            pg.time.wait(10)
            #!
        recipentPercent=1-attributorPercent
        if hunterid==0:
            attributionScore=[self.totalScore*attributorPercent,self.totalScore*recipentPercent]
        else:#hunterid=1
            attributionScore=[self.totalScore*recipentPercent,self.totalScore*attributorPercent]

        return attributionScore

class DrawAttributionTrail:
    def __init__(self, screen,playerColors,totalBarLength,barHeight):
        self.screen = screen
        self.playerColors=playerColors
        self.screenCenter =[300,300]# (self.screen.get_width() / 2, self.screen.get_height() / 2)
        self.totalBarLength=totalBarLength
        self.barHeight=barHeight

    def __call__(self, attributorId,attributorPercent):
        print(attributorId)
        recipentId=1-attributorId
        attributorLen=int(self.totalBarLength*attributorPercent)
        # attributorRect=pg.Rect((self.screenCenter[0]-self.totalBarLength/2),(self.screenCenter[1]-self.barHeight/2),(self.screenCenter[0]-self.totalBarLength/2+attributorLen),(self.screenCenter[1]+self.barHeight/2))
        # recipentRect=pg.Rect((self.screenCenter[0]-self.totalBarLength/2+attributorLen),(self.screenCenter[1]-self.barHeight/2),(self.screenCenter[0]+self.totalBarLength/2),(self.screenCenter[1]+self.barHeight/2),)
        attributorRect=((self.screenCenter[0]-self.totalBarLength/2,self.screenCenter[1]-self.barHeight/2),(attributorLen,self.barHeight))
        recipentRect=((self.screenCenter[0]-self.totalBarLength/2+attributorLen,self.screenCenter[1]-self.barHeight/2),(self.totalBarLength-attributorLen,self.barHeight))
        # pg.draw.rect(self.screen,(255, 0, 0, 255),((250.0, 290.0), (30, 20)))
        # pg.draw.rect(self.screen,self.playerColors[attributorId], pg.Rect(250, 290, 30,20))
        # print(self.playerColors[attributorId], attributorRect)
        # pg.draw.rect(self.screen, self.self.playerColors[attributorId], pg.Rect(250, 290, 30,20))
        pg.draw.rect(self.screen, self.playerColors[attributorId], attributorRect)
        pg.draw.rect(self.screen, self.playerColors[recipentId], recipentRect)

        pg.display.flip()
        return self.screen

if __name__ == "__main__":
    pg.init()
    screenWidth = 720
    screenHeight = 720
    screen = pg.display.set_mode((screenWidth, screenHeight))
    gridSize = 20
    leaveEdgeSpace = 2
    lineWidth = 2
    backgroundColor = [188, 188, 0]
    lineColor = [255, 255, 255]
    targetColor = [255, 50, 50]
    playerColor = [50, 50, 255]
    targetRadius = 10
    playerRadius = 10
    targetPositionA = [5, 5]
    targetPositionB = [15, 5]
    playerPosition = [10, 15]
    picturePath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/Pictures/'
    restImage = pg.image.load(picturePath + 'rest.png')
    currentTime = 138456
    currentScore = 5
    textColorTuple = (255, 50, 50)
    drawBackground = DrawBackground(screen, gridSize, leaveEdgeSpace, backgroundColor, lineColor, lineWidth, textColorTuple)
    drawNewState = DrawNewState(screen, drawBackground, targetColor, playerColor, targetRadius, playerRadius)
    drawImage = DrawImage(screen)
    drawBackground(currentTime, currentScore)
    pg.time.wait(5000)
    pg.quit()
