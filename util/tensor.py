import torch
import numpy as np


def countZeroWeights(model):
    zeros = 0
    nnz = 0
    for param in model.parameters():
        if param is not None:
            zeros += param.numel() - param.nonzero().size(0)
            nnz += param.nonzero().size(0)
    sparsity = zeros / float(zeros + nnz)
    return zeros, nnz, sparsity


def getSparsity(model, norm=False):
    _, _, sparsity = countZeroWeights(model)
    if norm:
        sparsity = np.floor(sparsity *  100)
    return sparsity


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # t() == transpose tensor
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def correctPred(output, target):
    """Return the accuracy of the expected label"""
    with torch.no_grad():
        res = []
        for out, label in zip(output, target):
            acc = out[label]
            res.append([int(label.cpu()), float(acc.cpu())])
    return res


def topN(output, target, topk=(1,)):
    """Return label prediction from top 5 classes"""
    with torch.no_grad():
        maxk = max(topk)
        scores, pred = output.topk(maxk, 1, True, True)
    return scores.t(), pred.t()

        
def loadStateDictModel(model, state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    

types_factory = {torch.float32: 32,
                torch.float: 32,
                torch.float64: 64,
                torch.double: 64,
                torch.float16: 16,
                torch.half: 16,
                torch.int64: 64,
                torch.long: 64,
                torch.int32: 32,
                torch.int: 32,
                torch.int16: 16,
                torch.short: 16
                }


def getNumBits(dtype: torch.dtype):
    return types_factory[dtype]