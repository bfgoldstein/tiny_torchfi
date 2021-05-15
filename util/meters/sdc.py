class SDCMeter(object):
    """Stores the SDCs probabilities"""
    def __init__(self):
        self.reset()

    def updateAcc(self, acc1, acc5):
        self.acc1 = acc1
        self.acc5 = acc5

    def updateGoldenData(self, scoreTensors):
        for scores in scoreTensors.cpu().numpy():
            self.goldenScoresAll.append(scores)

    def updateFaultyData(self, scoreTensors):
        for scores in scoreTensors.cpu().numpy():
            self.faultyScoresAll.append(scores)
    
    def updateGoldenBatchPred(self, predTensors):
        self.goldenPred.append(predTensors)

    def updateFaultyBatchPred(self, predTensors):
        self.faultyPred.append(predTensors)

    def updateGoldenBatchScore(self, scoreTensors):
        self.goldenScores.append(scoreTensors)

    def updateFaultyBatchScore(self, scoreTensors):
        self.faultyScores.append(scoreTensors)

    def calculteSDCs(self):
        top1Sum = 0
        top5Sum = 0
        for goldenTensor, faultyTensor in zip(self.goldenPred, self.faultyPred):
            correct = goldenTensor.ne(faultyTensor)
            top1Sum += correct[:1].view(-1).int().sum(0, keepdim=True)
            for goldenRow, faultyRow in zip(goldenTensor.t(), faultyTensor.t()):
                if goldenRow[0] not in faultyRow:
                    top5Sum += 1
        # calculate top1 and top5 SDCs by dividing sum to numBatches * batchSize
        self.top1SDC = float(top1Sum[0]) / float(len(self.goldenPred) * len(self.goldenPred[0][0]))
        self.top5SDC = float(top5Sum) / float(len(self.goldenPred) * len(self.goldenPred[0][0]))
        self.top1SDC *= 100
        self.top5SDC *= 100

    def reset(self):
        self.acc1 = 0
        self.acc5 = 0
        self.top1SDC = 0.0
        self.top5SDC = 0.0
        self.goldenPred = []
        self.faultyPred = []
        self.goldenScores = []
        self.faultyScores = []
        self.goldenScoresAll = []
        self.faultyScoresAll = []