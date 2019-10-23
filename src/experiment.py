import numpy as np
class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath

    def __call__(self, finishTime):
        sheep1Grid, sheep2Grid,bean1Grid, bean2Grid,playerGrid = self.initialWorld()
        trialIndex = 0
        score =np.array ([0,0])
        currentStopwatch = 0
        while True:
            print('trialIndex', trialIndex)
            response = self.experimentValues.copy()
            results, [sheep1Grid, sheep2Grid,bean1Grid, bean2Grid], playerGrid, score, currentStopwatch, eatenFlag = self.trial(sheep1Grid, sheep2Grid,bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex)
            response.update(results)
            self.writer(response, trialIndex)
            if currentStopwatch >= finishTime:
                break
            [sheep1Grid, sheep2Grid,bean1Grid, bean2Grid] = self.updateWorld([sheep1Grid, sheep2Grid,bean1Grid, bean2Grid], playerGrid, eatenFlag)
            trialIndex += 1
        return score
