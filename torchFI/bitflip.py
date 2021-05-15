import sys
import numpy as np
from bitstring import BitArray

from util.log import *


def flipFloat(val, bit=None, log=False, random_state=None):
    # Cast float to BitArray and flip (invert) random bit 0-31

    faultValue = BitArray(float=val, length=32)
    if bit == None:
        if random_state is not None:
            bit = random_state.randint(0, faultValue.len)
        else:
            bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)

    if log:
        logInjectionBit("\tFlipping bit ", bit)
        logInjectionVal("\tOriginal: ", float(val), " Corrupted: ", faultValue.float)

    return faultValue.float, bit

def bitFlip(value, bit=None, log=False, random_state=None):
    return flipFloat(value, bit, log, random_state)