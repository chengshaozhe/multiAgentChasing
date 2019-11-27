# from src.neuralNetwork.policyValueNet import GenerateModel, restoreVariables, ApproximatePolicy

import tensorflow as tf
import numpy as np
import scipy.stats as stats
import random
import os

class ComputeLikelihoodByHeatSeeking:

    def __init__(self,baseProb,assumePrecision):

        self.baseProb=baseProb
        self.assumePrecision=assumePrecision
    def __call__(self,vector1, vector2):
        if np.linalg.norm(vector1, ord=2)==0 or np.linalg.norm(vector2, ord=2)==0 :
            return self.baseProb
        else:
            deviationAngle = abs(np.angle(complex(vector1[0], vector1[1]) / complex(vector2[0], vector2[1])))
            pLikelihood = (1-stats.vonmises.cdf(deviationAngle, self.assumePrecision))*2
            return pLikelihood
class InferCurrentWolf:
    def __init__(self, computeLikelihoodByHeatSeeking):
        self.computeLikelihoodByHeatSeeking = computeLikelihoodByHeatSeeking
        self.wolvesID=[2,3]

    def __call__(self,sheepId,dequeState):

        goal =[False]*len(self.wolvesID)
        for (i,wolfId) in enumerate(self.wolvesID):
            heatSeekingVectorList= [dequeState[i - 1][sheepId] - dequeState[i - 1][wolfId] for i in range(1, len(dequeState))]
            wolfVectorList = [dequeState[i][wolfId] - dequeState[i - 1][wolfId]for i in range(1, len(dequeState))]
            pLikelihoodList=np.array([self.computeLikelihoodByHeatSeeking(v1,v2) for v1 ,v2 in zip(heatSeekingVectorList,wolfVectorList)])
            pLikeliMean=np.mean(pLikelihoodList)
            print(sheepId,wolfId,pLikeliMean)
            goal[i]=0<pLikeliMean
        return goal

class BeliefPolicy:
    def __init__(self,passerbyPolicy,singlePolicy,multiPolicy,inferCurrentWolf):
        self.passerbyPolicy=passerbyPolicy
        self.singlePolicy=singlePolicy
        self.multiPolicy=multiPolicy
        self.inferCurrentWolf=inferCurrentWolf

    def __call__(self,sheepId,dequeState):
        goal=self.inferCurrentWolf(sheepId,dequeState)

        currentState=dequeState[-1]
        print(sheepId,goal)
        if goal==[True,True]:
            policyForCurrentStateDict= self.multiPolicy((currentState[sheepId],currentState[2],currentState[3]))
        elif goal==[True,False]:
            policyForCurrentStateDict= self.singlePolicy((currentState[sheepId],currentState[2]))
        elif goal==[False,True]:
            policyForCurrentStateDict= self.singlePolicy((currentState[sheepId],currentState[3]))
        elif goal==[False,False]:
            policyForCurrentStateDict= self.passerbyPolicy(currentState[sheepId],currentState[2:])

        return policyForCurrentStateDict


class SingleChasingPolicy:
    def __init__(self,singlePolicy,inferNearestWolf):
        self.singlePolicy=singlePolicy
        self.inferNearestWolf=inferNearestWolf

    def __call__(self,sheepPos,wolvesPos):

        nearestWolfPos=self.inferNearestWolf(sheepPos,wolvesPos)
        actionDict=self.singlePolicy((sheepPos,nearestWolfPos))

        return actionDict

def inferNearestWolf(sheepPos,wolvesPos):
    getDistance=lambda wolf,sheep:np.linalg.norm(wolf-sheep)
    distancelist=[getDistance(wolfPos,sheepPos) for wolfPos in wolvesPos]

    return wolvesPos[np.argmin(distancelist)]
class GenerateModel:
    def __init__(self, numStateSpace, numActionSpace, regularizationFactor=0.0, valueRelativeErrBound=0.01, seed=128):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.regularizationFactor = regularizationFactor
        self.valueRelativeErrBound = valueRelativeErrBound
        self.seed = seed

    def __call__(self, sharedWidths, actionLayerWidths, valueLayerWidths, resBlockSize=0, initialization='uniform',
                 dropoutRate=0.0, summaryPath="./tbdata"):
        print(f'Generating NN with shared layers: {sharedWidths}, action layers: {actionLayerWidths}, '
              f'value layers: {valueLayerWidths}\nresBlockSize={resBlockSize}, init={initialization}, '
              f'dropoutRate={dropoutRate}')
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope("inputs"):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                tf.add_to_collection("inputs", states_)

            with tf.name_scope("groundTruths"):
                groundTruthAction_ = tf.placeholder(tf.int32, [None, self.numActionSpace], name="action")
                groundTruthValue_ = tf.placeholder(tf.float32, [None, 1], name="value")
                tf.add_to_collection("groundTruths", groundTruthAction_)
                tf.add_to_collection("groundTruths", groundTruthValue_)

            with tf.name_scope("trainingParams"):
                learningRate_ = tf.constant(0, dtype=tf.float32)
                actionLossCoef_ = tf.constant(1, dtype=tf.float32)
                valueLossCoef_ = tf.constant(1, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)
                tf.add_to_collection("lossCoefs", actionLossCoef_)
                tf.add_to_collection("lossCoefs", valueLossCoef_)

            initWeightDict = {'uniform': tf.random_uniform_initializer(-0.03, 0.03),
                              'normal': tf.random_normal_initializer(mean=0.0, stddev=1.0),
                              'glorot': tf.glorot_normal_initializer(),
                              'he': tf.initializers.he_normal()}
            initWeight = initWeightDict[initialization]
            initBias = tf.constant_initializer(0.001)

            with tf.variable_scope("shared"):
                fcLayer = tf.layers.Dense(units=sharedWidths[0], activation=tf.nn.relu, kernel_initializer=initWeight,
                                          bias_initializer=initBias, name="fc1")
                activation_ = tf.nn.dropout(fcLayer(states_), rate=dropoutRate)
                tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                if resBlockSize > 0:
                    blockHeads = set([i for i in range(2, len(sharedWidths) + 1) if (i - 2) % resBlockSize == 0])
                    blockTails = set([i + resBlockSize - 1 for i in blockHeads])
                else:
                    blockHeads, blockTails = set(), set()
                for i in range(2, len(sharedWidths) + 1):
                    if i in blockHeads:
                        resBlockInput_ = activation_
                    fcLayer = tf.layers.Dense(units=sharedWidths[i-1], activation=None, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i))
                    preActivation_ = tf.nn.dropout(fcLayer(activation_), rate=dropoutRate)
                    activation_ = tf.nn.relu(preActivation_ + resBlockInput_) if i in blockTails \
                        else tf.nn.relu(preActivation_)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                sharedOutput_ = tf.identity(activation_, name="output")

            with tf.variable_scope("action"):
                activation_ = sharedOutput_
                for i in range(len(actionLayerWidths)):
                    fcLayer = tf.layers.Dense(units=actionLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.nn.dropout(fcLayer(activation_), rate=dropoutRate)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                outputFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(actionLayerWidths) + 1))
                actionOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{actionOutputLayerActivation_.name}"],
                                      actionOutputLayerActivation_)

            with tf.name_scope("actionOutputs"):
                actionDistributions_ = tf.nn.softmax(actionOutputLayerActivation_, name="distributions")
                actionIndices_ = tf.argmax(actionDistributions_, axis=1, name="indices")
                tf.add_to_collection("actionDistributions", actionDistributions_)
                tf.add_to_collection("actionIndices", actionIndices_)

            with tf.variable_scope("value"):
                activation_ = sharedOutput_
                for i in range(len(valueLayerWidths)):
                    fcLayer = tf.layers.Dense(units=valueLayerWidths[i], activation=tf.nn.relu, kernel_initializer=initWeight,
                                              bias_initializer=initBias, name="fc{}".format(i+1))
                    activation_ = tf.nn.dropout(fcLayer(activation_), rate=dropoutRate)
                    tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                    tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)

                outputFCLayer = tf.layers.Dense(units=1, activation=None, kernel_initializer=initWeight,
                                                bias_initializer=initBias, name="fc{}".format(len(valueLayerWidths) + 1))
                valueOutputLayerActivation_ = outputFCLayer(activation_)
                tf.add_to_collections(["weights", f"weight/{outputFCLayer.kernel.name}"], outputFCLayer.kernel)
                tf.add_to_collections(["biases", f"bias/{outputFCLayer.bias.name}"], outputFCLayer.bias)
                tf.add_to_collections(["activations", f"activation/{valueOutputLayerActivation_.name}"],
                                      valueOutputLayerActivation_)

            with tf.name_scope("valueOutputs"):
                values_ = tf.identity(valueOutputLayerActivation_, name="values")
                tf.add_to_collection("values", values_)

            with tf.name_scope("evaluate"):
                with tf.name_scope("action"):
                    crossEntropy_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=actionOutputLayerActivation_,
                                                                               labels=groundTruthAction_)
                    actionLoss_ = tf.reduce_mean(crossEntropy_, name='loss')
                    tf.add_to_collection("actionLoss", actionLoss_)
                    actionLossSummary = tf.summary.scalar("actionLoss", actionLoss_)

                    groundTruthActionIndices_ = tf.argmax(groundTruthAction_, axis=1)
                    actionAccuracy_ = tf.reduce_mean(tf.cast(tf.equal(actionIndices_, groundTruthActionIndices_), tf.float32), name="accuracy")
                    tf.add_to_collection("actionAccuracy", actionAccuracy_)
                    actionAccuracySummary = tf.summary.scalar("actionAccuracy", actionAccuracy_)

                with tf.name_scope("value"):
                    valueLoss_ = tf.losses.mean_squared_error(groundTruthValue_, values_)
                    tf.add_to_collection("valueLoss", valueLoss_)
                    valueLossSummary = tf.summary.scalar("valueLoss", valueLoss_)

                    relativeErrorBound_ = tf.constant(self.valueRelativeErrBound)
                    relativeValueError_ = tf.cast((tf.abs((values_ - groundTruthValue_) / groundTruthValue_)), relativeErrorBound_.dtype)
                    valueAccuracy_ = tf.reduce_mean(tf.cast(tf.less(relativeValueError_, relativeErrorBound_), tf.float64), name="accuracy")
                    tf.add_to_collection("valueAccuracy", valueAccuracy_)
                    valueAccuracySummary = tf.summary.scalar("valueAccuracy", valueAccuracy_)

                with tf.name_scope("regularization"):
                    batchSize_ = tf.shape(states_)[0]
                    halfWeightSquaredSum_ = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                                      if 'bias' not in v.name])
                    l2RegularizationLoss_ = tf.multiply(
                        halfWeightSquaredSum_,
                        tf.truediv(self.regularizationFactor, tf.cast(batchSize_, tf.float32)),
                        name="l2RegLoss")
                    tf.summary.scalar("l2RegLoss", l2RegularizationLoss_)

                loss_ = tf.add_n([actionLossCoef_*actionLoss_, valueLossCoef_*valueLoss_, l2RegularizationLoss_], name="loss")
                tf.add_to_collection("loss", loss_)
                lossSummary = tf.summary.scalar("loss", loss_)

            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer')
                gradVarPairs_ = optimizer.compute_gradients(loss_)
                trainOp = optimizer.apply_gradients(gradVarPairs_)
                tf.add_to_collection("trainOp", trainOp)

                with tf.name_scope("inspectGrad"):
                    for grad_, var_ in gradVarPairs_:
                        keyPrefix = "weightGradient" if "kernel" in var_.name else "biasGradient"
                        tf.add_to_collection(f"{keyPrefix}/{var_.name}", grad_)
                    gradients_ = [tf.reshape(grad_, [1, -1]) for (grad_, _) in gradVarPairs_]
                    allGradTensor_ = tf.concat(gradients_, 1)
                    allGradNorm_ = tf.norm(allGradTensor_)
                    tf.add_to_collection("allGradNorm", allGradNorm_)
                    tf.summary.histogram("allGradients", allGradTensor_)
                    tf.summary.scalar("allGradNorm", allGradNorm_)
            fullSummary = tf.summary.merge_all()
            evalSummary = tf.summary.merge([lossSummary, actionLossSummary, valueLossSummary,
                                            actionAccuracySummary, valueAccuracySummary])
            tf.add_to_collection("summaryOps", fullSummary)
            tf.add_to_collection("summaryOps", evalSummary)

            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)

            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model

class ApproximatePolicy:
    def __init__ (self, policyValueNet, actionSpace):
        self.policyValueNet = policyValueNet
        self.actionSpace = actionSpace

    def __call__(self, stateBatch):
        if np.array(stateBatch).ndim == 3:
            stateBatch = [np.concatenate(state) for state in stateBatch]
        if np.array(stateBatch).ndim == 2:
            stateBatch = np.concatenate(stateBatch)
        if np.array(stateBatch).ndim == 1:
            stateBatch = np.array([stateBatch])
        graph = self.policyValueNet.graph
        state_ = graph.get_collection_ref("inputs")[0]
        actionDist_ = graph.get_collection_ref("actionDistributions")[0]
        actionProbs = self.policyValueNet.run(actionDist_, feed_dict={state_: stateBatch})[0]
        actionDist = {action: prob for action, prob in zip(self.actionSpace, actionProbs)}
        return actionDist


def chooseGreedyAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    maxIndices = np.argwhere(probs == np.max(probs)).flatten()
    selectedIndex = np.random.choice(maxIndices)
    selectedAction = actions[selectedIndex]
    return selectedAction


def sampleAction(actionDist):
    actions = list(actionDist.keys())
    probs = list(actionDist.values())
    normlizedProbs = [prob / sum(probs) for prob in probs]
    selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
    selectedAction = actions[selectedIndex]
    return selectedAction


class SoftmaxAction:
    def __init__(self, softMaxBeta):
        self.softMaxBeta = softMaxBeta

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        probs = list(actionDist.values())
        normlizedProbs = list(np.divide(np.exp(np.multiply(self.softMaxBeta, probs)), np.sum(np.exp(np.multiply(self.softMaxBeta, probs)))))
        selectedIndex = list(np.random.multinomial(1, normlizedProbs)).index(1)
        selectedAction = actions[selectedIndex]
        return selectedAction


def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model


if __name__ == '__main__':
    sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    numActionSpace = len(sheepActionSpace)

    numStateSpace = 6
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]
    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    generateSheepModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    initSheepNNModel = generateSheepModel(sharedWidths * 4, actionLayerWidths, valueLayerWidths)
    sheepPreTrainModelPath = os.path.join('..', 'trainedModels', 'agentId=0_depth=4_learningRate=0.0001_maxRunningSteps=60_miniBatchSize=256_numSimulations=100_trainSteps=10000')
    print(sheepPreTrainModelPath)
    sheepPreTrainModel = restoreVariables(initSheepNNModel, sheepPreTrainModelPath)
    sheepPolicy = ApproximatePolicy(sheepPreTrainModel, sheepActionSpace)
    state = ([10, 10], (8, 9), (20, 2))

    print(chooseGreedyAction(sheepPolicy(state)))
