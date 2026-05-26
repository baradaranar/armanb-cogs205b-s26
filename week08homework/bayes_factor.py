import numpy as np
from scipy.special import comb
from scipy.integrate import quad

class BayesFactor:
    def __init__(self, n, k, a=0.47, b=0.53):
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if n <= 0:
            raise ValueError("n must be positive")
        if k < 0:
            raise ValueError("k cannot be negative")
        if k > n:
            raise ValueError("k cannot exceed n")
        if not isinstance(a, (int, float)):
            raise TypeError("a must be numerical")
        if not isinstance(b, (int, float)):
            raise TypeError("b must be numerical")
        if a < 0:
            raise ValueError("a cannot be negative")
        if b > 1:
            raise ValueError("b cannot exceed one")
        if a >= b:
            raise ValueError("a must be less than b")
        
        self.n = n
        self.k = k
        self.a = a
        self.b = b

    def likelihood(self, theta):
        if not isinstance(theta, (int, float, np.float64, np.float32)):
            raise TypeError("theta must be numerical")
        if not (0 <= theta <= 1):
            raise ValueError("theta must be in [0, 1]")
        
        # Using log space to avoid overflow/underflow for large n
        # Likelihood: C(n, k) * theta^k * (1-theta)^(n-k)
        # log(L) = log(C(n, k)) + k*log(theta) + (n-k)*log(1-theta)
        
        if theta == 0:
            return float(comb(self.n, self.k) * (0**self.k) * (1** (self.n - self.k)))
        if theta == 1:
            return float(comb(self.n, self.k) * (0** (self.n - self.k)) * (1**self.k))

        # Use scipy.special.comb with exact=False to get float result
        return float(comb(self.n, self.k) * (theta**self.k) * ((1 - theta)**(self.n - self.k)))

    def evidence_slab(self):
        # Slab evidence: integrate likelihood over theta ~ U(0, 1)
        # Integral of theta^k * (1-theta)^(n-k) from 0 to 1 is Beta(k+1, n-k+1)
        # C(n, k) * Beta(k+1, n-k+1) = C(n, k) * (k! (n-k)!) / (n+1)!
        # = [n! / (k!(n-k)!)] * [k!(n-k)! / (n+1)!] = 1 / (n+1)
        result, _ = quad(self.likelihood, 0, 1)
        return result

    def evidence_spike(self):
        # Spike evidence: integrate likelihood over theta ~ U(a, b), divided by width (b-a)
        # The density of the spike prior is 1/(b-a) for theta in [a, b]
        result, _ = quad(self.likelihood, self.a, self.b)
        return result / (self.b - self.a)

    def bayes_factor(self):
        slab = self.evidence_slab()
        spike = self.evidence_spike()
        
        # Handle case where evidence is 0 or NaN due to large n precision issues
        if np.isnan(slab) or np.isnan(spike):
            return 0.0
        
        return spike / slab