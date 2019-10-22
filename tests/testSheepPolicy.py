import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from ddt import ddt, data, unpack

from src.sheepPolicy import ExpSheepPolicy,inferGoalGridEnv
@ddt
class TestSheepPolicy(unittest.TestCase):
	
	@data(([[0,0],[2,2],[-3,-4]],[[0,0],[2,0],[-1,-4]],[1,1])
		,([[0,0],[2,2],[-3,-4]],[[0,0],[2,0],[-7,-4]],[1,0]),
		([[0,0],[2,2],[-3,-4]],[[0,0],[8,2],[-3,-1]],[0,1]),
		([[0,0],[2,2],[-3,-4]],[[0,0],[8,9],[-8,-7]],[0,0])
		)
	@unpack
	def testInferGoalGridEnv(self,initialState,finalState, groundTruthGoal):
		goal=inferGoal(initialState,finalState)
		truthValue = np.array_equal(goal, groundTruthGoal)
		self.assertTrue(truthValue)

if __name__ == "__main__":
    unittest.main()