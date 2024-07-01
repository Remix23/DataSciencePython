import numpy as np
from time import sleep

class Clusteroid:

    def __init__(self, pos : np.ndarray, id : int, data_shape : int, distfunc) -> None:
        self.dim = data_shape
        self.points = []

        self.pos = pos

        self.distance = distfunc if distfunc else self.distA

        self.id = id

    def update(self):

        if len(self.points) == 0: return

        avgs = np.zeros(self.dim)
        for point in self.points:
            for dim in range(self.dim):
                avgs[dim] += point[dim]

        avgs = avgs / len(self.points)
        self.pos = avgs

    def addPoints (self, point):
        self.points.append(point)

    def distA (self, point : np.ndarray):
        vec = point - self.pos
        return np.sqrt(np.dot(vec, vec))

    def __repr__(self) -> str:
        return f"ID: {self.id} P: {self.pos} N: {len(self.points)}"

    def clean (self):
        self.points.clear()

class KMeans :

    def __init__(self, data : np.ndarray, k : int, distFun, ranges : list = []) -> None:
        self.data = data
        self.numOfClusteroids = k

        if len(data) == 0: return

        self.data_shape = len(self.data[0])
        self.ranges = ranges

        if not ranges:
            self.genRangesMinMax()

        if len(self.ranges) < self.data_shape:
            for i in range(len(self.ranges), self.data_shape):
                index = i % len(self.ranges)
                self.ranges.append(self.ranges[index])

        self.clusteroids : list[Clusteroid] = []
        self.genClusteroids(distFun) # update the positions of the clusteroids

    def genRangesMinMax (self):
        self.ranges = [[self.data[0][i], self.data[0][i]] for i in range(self.data_shape)] ## get min and max for each dim
        for obj in self.data:
            for i in range (self.data_shape):
                if (obj[i] < self.ranges[i][0]): self.ranges[i][0] = obj[i]
                if (obj[i] > self.ranges[i][1]): self.ranges[i][1] = obj[i]

    def genClusteroids (self, distfun):
        for i in range(self.numOfClusteroids):
            starting_pos = np.zeros(self.data_shape)
            for dim in range(self.data_shape):
                starting_pos[dim] = np.random.uniform(self.ranges[dim][0], self.ranges[dim][1])

            self.clusteroids.append(Clusteroid(starting_pos, i, self.data_shape, distfunc = distfun))

    def updateClusters (self):
        for cluster in self.clusteroids:
            cluster.update()
            cluster.clean()

    def updatePoints (self):
        for point in self.data:
            min_dics = np.inf
            closest = self.clusteroids[0]
            for cluster in self.clusteroids:
                dist = cluster.distance(point)
                if dist < min_dics:
                    closest = cluster
                    min_dics = dist

            closest.addPoints(point)

    def getClusteroidsPos (self):
        return np.array([x.pos for x in self.clusteroids])

    def performIterations (self, n) -> np.ndarray:
        for i in range (n):
            self.updatePoints()
            self.updateClusters()

        return self.getClusteroidsPos()

    def performUntilConvergence (self, max_iter = 100, atol = 1e-5, rtol = 1e-8, wait_time = 0.5) -> tuple[np.ndarray, int]:
        previous = self.getClusteroidsPos()

        iter = 0

        while (iter < max_iter):
            ### update
            self.updatePoints()
            self.updateClusters()

            iter += 1

            curr = self.getClusteroidsPos()

            if np.allclose(previous, curr, atol, rtol): break

            previous = curr

        return previous, iter


if __name__ == "__main__":
    data = np.array([np.random.randint(-100, 100, 5) for _ in range (10000)])

    k = KMeans(data = data, k = 10, distFun = None)

    after10 = k.performIterations(10)

    print(after10)
