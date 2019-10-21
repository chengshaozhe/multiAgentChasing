
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
        bean1Grid, bean2Grid, playerGrid = self.initialWorld()
        trialIndex = 0
        score = 0
        currentStopwatch = 0
        while True:
            print('trialIndex', trialIndex)
            response = self.experimentValues.copy()
            results, [bean1Grid, bean2Grid], playerGrid, score, currentStopwatch, eatenFlag = self.trial(bean1Grid, bean2Grid, playerGrid, score, currentStopwatch, trialIndex)
            response.update(results)
            self.writer(response, trialIndex)
            if currentStopwatch >= finishTime:
                break
            [bean1Grid, bean2Grid] = self.updateWorld([bean1Grid, bean2Grid], playerGrid, eatenFlag)
            trialIndex += 1
        return score
