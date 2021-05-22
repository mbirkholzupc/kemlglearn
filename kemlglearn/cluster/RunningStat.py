# Computationally efficient (constant-time) class for calculating running mean, variance
# and standard deviation
# Originally in C++, ported to python
# Reference: https://www.johndcook.com/blog/standard_deviation/

from math import sqrt


class RunningStat:
    def __init__(self):
        self.m_n = 0
        # The following aren't strictly necessary, but good to set to 0 anyways
        self.m_oldM = 0
        self.m_newM = 0
        self.m_oldS = 0
        self.m_newS = 0

    def clear(self):
        self.m_n = 0

    def push(self, x):
        self.m_n = self.m_n+1

        # See Knuth TAOCP vol 2, 3rd edition, page 232
        if self.m_n == 1:
            self.m_newM = x
            self.m_oldM = self.m_newM
            self.m_oldS = 0
        else:
            self.m_newM = self.m_oldM + (x - self.m_oldM) / self.m_n
            self.m_newS = self.m_oldS + (x - self.m_oldM) * (x - self.m_newM)

            # Set up for next iteration
            self.m_oldM = self.m_newM
            self.m_oldS = self.m_newS

    def num_data_values(self):
        return self.m_n

    def mean(self):
        mean = 0
        if self.m_n > 0:
            mean = self.m_newM
        return mean

    def variance(self):
        var = 0
        if self.m_n > 1:
            var = self.m_newS/(self.m_n - 1)
        return var

    def standard_deviation(self):
        return sqrt(self.variance())
