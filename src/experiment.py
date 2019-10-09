
class Experiment():
    def __init__(self, trial, writer, experimentValues, initialWorld, updateWorld, drawImage, resultsPath, minDistanceBetweenGrids):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.initialWorld = initialWorld
        self.updateWorld = updateWorld
        self.drawImage = drawImage
        self.resultsPath = resultsPath
        self.minDistanceBetweenGrids = minDistanceBetweenGrids

    def __call__(self, finishTime):
        bean1Grid, bean2Grid, playerGrid = self.initialWorld(self.minDistanceBetweenGrids)
        trialIndex = 0
        score = 0
        currentStopwatch = 0
        while True:
            results, bean1Grid, playerGrid, score, currentStopwatch = self.trial(bean1Grid, bean2Grid, playerGrid, score, currentStopwatch)
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
            if currentStopwatch >= finishTime:
                break
            bean2Grid, self.experimentValues["condition"] = self.updateWorld(bean1Grid, playerGrid)
            trialIndex += 1
        return score
