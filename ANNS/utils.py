import numpy as np

class ActivationFunc:

    def __init__(self) -> None:
        pass

    def getVal (self, x):
        return 0
    def getDerrivative (self, x):
        return 0

class Sigmoid (ActivationFunc):

    def __init__(self) -> None:
        super().__init__()

    def getVal (self, x):
        return 1 / (1 + np.exp(-x))

    def getDerrivative (self, x):
        return self.getVal(x) * (1 - self.getVal(x))

class ReLu (ActivationFunc):

    def __init__(self, a = 1) -> None:
        super().__init__()
        self.a = a

    def getVal (self, x):
        return max(self.a * x, x)

    def getDerrivative (self, x):
        return max(self.a, 1) if x >= 0 else min(self.a, 1)

class ELU (ActivationFunc):

    def __init__(self, a) -> None:
        super().__init__()
        self.a = a

    def getVal(self, x) -> np.float32:
        return np.float32(x if x >= 0 else self.a * (np.exp(x) - 1))

    def getDerrivative(self, x) -> np.float32:
        return np.float32(1 if x >= 0 else self.a * np.exp(x))

class TanH (ActivationFunc):

    def __init__(self) -> None:
        super().__init__()

    def getVal (self, x):
        return np.tanh(x)

    def getDerrivative (self, x):
        return 4 * np.power(np.exp(x) + np.exp(-x), -2)

class TransformFunc :

    def __init__(self) -> None:
        pass

    def tranform (self, last_layer : np.ndarray) -> np.ndarray:
        return last_layer

class SoftMax (TransformFunc):

    def __init__(self, T = 1) -> None:
        super().__init__()
        self.t = T

    def tranform(self, last_layer: np.ndarray) -> np.ndarray:
        last_layer -= np.max(last_layer)
        exps = np.exp(last_layer / self.t)
        return exps / np.sum(exps)

class Linear (ActivationFunc, TransformFunc):

    def __init__(self) -> None:
        super(ActivationFunc).__init__()
        super(TransformFunc).__init__()

    def getVal(self, x):
        return x

    def getDerrivative(self, x):
        return 1
    
    def tranform(self, last_layer: np.ndarray) -> np.ndarray:
        return last_layer

class Initializer:

    def __init__(self) -> None:
        pass

    def getNext (self, shape = np.shape((1, 1))) -> np.ndarray:
        return np.empty(0)
    
class Uniform (Initializer):

    def __init__(self, min = 0, max = 1) -> None:
        super().__init__()
        self.min = min
        self.max = max
    
    def getNext(self, shape=np.shape((1, 1))):
        return np.random.uniform(self.min, self.max, size = shape)

class Normal (Initializer):

    def __init__(self, mean = 0, sd = 1) -> None:
        super().__init__()
        self.mean = mean
        self.sd = sd
    
    def getNext(self, shape=np.shape((1, 1))):
        return np.random.normal(self.mean, self.sd, size = shape)
    
class GlorotUniform (Initializer):

    def __init__(self) -> None:
        super().__init__()

    def getNext(self, shape=np.shape((1, 1))):
        fan_in, fan_out = shape
        sig = np.sqrt(6 / (fan_in + fan_out))
        a = np.sqrt(sig / (fan_in + fan_out))
        return np.random.uniform(low = -a, high = a, size = shape)

class GlorotNormal (Initializer):

    def __init__(self) -> None:
        super().__init__()

    def getNext(self, shape=np.shape((1, 1))):
        fan_in, fan_out = shape
        sig = np.sqrt(6 / (fan_in + fan_out))
        return np.random.normal(0, sig, size = shape)
    
class HeUniform (Initializer):
    def __init__(self) -> None:
        super().__init__()

    def getNext(self, shape=np.shape((1, 1))):
        fan_in, fan_out = shape
        a = np.sqrt(6 / fan_in)
        b = np.sqrt(6 / fan_out)
        return np.random.uniform(-a, b, size = shape)
    
class HeNormal (Initializer):

    def __init__(self) -> None:
        super().__init__()

    def getNext(self, shape=np.shape((1, 1))):
        fan_in, fan_out = shape
        sig = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sig, size = shape)

### to think 
class Results:

    def __init__(self) -> None:
        pass

class Tester:

    def __init__(self) -> None:
        pass

class LossFunc:

    def __init__(self) -> None:
        pass

    def computeLoss (self, predicted : np.ndarray, expected : np.ndarray) -> np.ndarray:
        return np.zeros(0)
    
    def computeLossDerivative (self, predicted : np.ndarray, expected : np.ndarray) -> np.ndarray:
        return np.zeros(0)
    
    def computeTotalCost (self, predicted : np.ndarray, expected : np.ndarray):
        return np.sum(self.computeLoss(predicted, expected))
    
class MSE (LossFunc):

    def __init__(self) -> None:
        super().__init__()

    def computeLoss(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return ((predicted - expected)**2)
    
    def computeLossDerivative(self, predicted: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return 2 * (predicted - expected)

class Optimizer:

    def __init__(self) -> None:
        pass
    def updateWeight (self, dw : np.ndarray) -> np.ndarray:
        return dw
    
class SGD (Optimizer):

    def __init__(self, lrate, batch_size) -> None:
        super().__init__()
        self.lrate = lrate
        self.batch_size = batch_size

    def updateWeight(self, dw: np.ndarray) -> np.ndarray:
        return dw * self.lrate / self.batch_size


if __name__ == "__main__":
    a = SoftMax()
    print(a.tranform(np.array([3.0, 1.0, 0.2])))

    b = MSE()

