import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(self, r, S, K, T, sigma):
        self.r = r
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma

    def calculate_ds(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * (self.sigma**2)) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def black_scholes(self, type):
        d1, d2 = self.calculate_ds()
        try:
            if type == "Call":
                price = self.S * norm.cdf(d1, 0, 1) - self.K * np.exp(
                    -self.r * self.T
                ) * norm.cdf(d2, 0, 1)
            elif type == "Put":
                price = self.K * np.exp(-self.r * self.T) * norm.cdf(
                    -d2, 0, 1
                ) - self.S * norm.cdf(-d1, 0, 1)
            return round(price, 3)
        except:
            return 0.0

    def greeks(self, type):
        d1, d2 = self.calculate_ds()
        try:
            if type == "Call":
                delta = norm.cdf(d1, 0, 1)
                gamma = (norm.pdf(d1, 0, 1)) / (self.S * self.sigma * np.sqrt(self.T))
                vega = self.S * norm.pdf(d1, 0, 1) * np.sqrt(self.T)
                theta = -self.S * norm.pdf(d1, 0, 1) * self.sigma / (
                    2 * np.sqrt(self.T)
                ) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0, 1)
                rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2, 0, 1)
            elif type == "Put":
                delta = -norm.cdf(-d1, 0, 1)
                gamma = (norm.pdf(d1, 0, 1)) / (self.S * self.sigma * np.sqrt(self.T))
                vega = self.S * norm.pdf(d1, 0, 1) * np.sqrt(self.T)
                theta = -self.S * norm.pdf(d1, 0, 1) * self.sigma / (
                    2 * np.sqrt(self.T)
                ) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2, 0, 1)
                rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2, 0, 1)

            return {
                "delta": round(delta, 3),
                "gamma": gamma,
                "theta": round(theta / 365, 4),
                "vega": round(vega * 0.01, 3),
                "rho": round(rho * 0.01, 3),
            }
        except:
            return 0
