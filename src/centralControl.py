import numpy as np

class AssignCentralControlToIndividual:
    def __init__(self, imaginedWeId, individualId, chooseCentralControlAction):
        self.imaginedWeId = imaginedWeId
        self.individualId = individualId
        self.individualIndexInWe = list(self.imaginedWeId).index(self.individualId)
        self.chooseCentralControlAction = chooseCentralControlAction 

    def __call__(self, centralControlActionDist):
        centralControlAction = np.array(self.chooseCentralControlAction(centralControlActionDist))
        individualAction = centralControlAction[self.individualIndexInWe]
        return individualAction
