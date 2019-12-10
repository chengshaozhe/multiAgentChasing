class tansferHumanActionTo8Direction():

    def __init__(self, assumePrecision):
        self.vecToAngle=lambda vector:np.angle(complex(vector[0], vector[1]))
        self.calcullatePdf=lambda humanDirection,degeree: stats.vonmises.pdf(humanDirection-degeree,assumePrecision)*2

    def __call__(self,humanAction):
        if humanAction != (0,0):
            humanDirection=self.vecToAngle(humanAction)
            pdf=np.array([self.calcullatePdf(humanDirection,degeree) for degeree in eightDegereList])
            normProb=pdf/pdf.sum()
            actionDict={}
            [actionDict.update({action:prob} ) fro action,prob in zip (8actionSpace,normProb)]
            actionDict.update({(0,0):0})
        else:
            [actionDict.update({action:0} ) fro action in 8actionSpace]
            actionDict.update({(0,0):1})
        return actionDict


class stateFileter:
    def __init__(self, stateIdlist):

    def __call__(self,sheepId):
        return filteredState


class simulatePolicy(object):

    def __init__(self,NNModelPolicy):
        self.partenerPolicy=NNModelPolicy


    def __call__(self,state):


        return self.partenerPolicy(state)

class inferPartenerGoal:
    """docstring for inferPartenerGoal"""
    def __init__(self, arg):
        self.arg = arg

    def __call__(self,dequeState)
        sheep1state=[stateFileter(state) for state in dequeState]
        sheep2state=


        return goalLikelyHood