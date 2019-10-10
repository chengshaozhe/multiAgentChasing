
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
            print('trialIndex', trialIndex)
            results, eatenGird, playerGrid, score, currentStopwatch, eatenFlag = self.trial(bean1Grid, bean2Grid, playerGrid, score, currentStopwatch)
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
            if currentStopwatch >= finishTime:
                break
            newGrid, self.experimentValues["condition"] = self.updateWorld(eatenGird, playerGrid)
            if eatenFlag.index(True) == 0:
                bean1Grid = newGrid
            elif eatenFlag.index(True) == 1:
                bean2Grid = newGrid
            trialIndex += 1
        return score
