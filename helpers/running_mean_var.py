class RunningMeanVar:
    # https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values

    def __init__(self):
        self.m = 0
        self.v = 0
        self.k = 0

    def add(self, x):
        self.k += 1
        prev_m = self.m
        self.m += (x - prev_m) / self.k
        self.v += (x - prev_m) * (x - self.m)

    def mean(self):
        return self.m

    def variance(self):
        return self.v / self.k


if __name__ == '__main__':
    import numpy as np

    v = np.array([1.0, 2.0, 3.0, 4.0])
    print(np.mean(v))
    print(np.var(v))

    rmv = RunningMeanVar()
    rmv.add(1.0)
    rmv.add(2.0)
    rmv.add(3.0)
    rmv.add(4.0)
    print(rmv.mean())
    print(rmv.variance())
