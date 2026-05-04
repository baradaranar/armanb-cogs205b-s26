from math import comb
from scipy import integrate

class BayesFactor:
    def __init__(self, n, k, a=0.4999, b=0.5001):
        if type(n) != int:
            raise TypeError("n must be an integer.")
        if n <= 0:
            raise ValueError("n must be positive.")

        if type(k) != int:
            raise TypeError("k must be an integer.")
        if k < 0:
            raise ValueError("k cannot be negative.")

        if not (type(a) == int or type(a) == float):
            raise TypeError("a must be numerical.")
        if not (type(b) == int or type(b) == float):
            raise TypeError("b must be numerical.")

        if k > n:
            raise ValueError("k cannot exceed n.")
        if a >= b:
            raise ValueError("a must be less than b.")
        if a < 0:
            raise ValueError("a cannot be negative.")
        if b > 1:
            raise ValueError("b cannot be greater than 1.")

        self.n = n
        self.k = k
        self.a = a
        self.b = b

    def likelihood(self, theta):
        if not (type(theta) == int or type(theta) == float):
            raise TypeError("theta must be numerical.")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be in [0, 1]")
        return comb(self.n, self.k) * (theta ** self.k) * ((1 - theta) ** (self.n - self.k))

    def evidence_slab(self):
        result = integrate.quad(self.likelihood, 0, 1)[0]
        return result

    def evidence_spike(self):
        c = self.b - self.a
        result = integrate.quad(self.likelihood, self.a, self.b)[0]
        return result / c

    def bayes_factor(self):
        if self.evidence_slab() == 0:
            raise ValueError("evidence_slab is zero, so Bayes factor is undefined.")
        return self.evidence_spike() / self.evidence_slab()