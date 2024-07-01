import numpy as np

class KNN:

    def __init__(self, data : np.ndarray, k : int) -> None:
        self.data = data
        self.k = k

        if len(data) == 0: return
        if k > len(self.data) or k < 0 : raise(Exception("Wrong k: should be > 0 and lower than the number of points"))

        self.data_shape = len(self.data[0])

    def search (self, x : np.ndarray, dist_fun = lambda p: np.sqrt(np.dot(p, p))):
        if (len(x) != self.data_shape): raise(Exception(f"Incorrect number of dimenstions: point: {len(x)} | data: {self.data_shape}"))

        arr_flattened = np.apply_along_axis(lambda p: p - x, 1, self.data)
        arr_flattened = np.apply_along_axis(dist_fun, 1, self.data)

        sorted_distance = self.data[arr_flattened.argsort()]

        knn = sorted_distance[0:self.k]

        return knn

if __name__ == "__main__":
    # test
    pass
