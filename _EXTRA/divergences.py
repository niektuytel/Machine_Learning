# https://machinelearningmastery.com/divergence-between-probability-distributions/
# https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
import numpy as np

class Kullback_Leibler_Divergence:
    def __call__(self, P, Q):
        return sum(P[i] * np.log(P[i]/Q[i]) for i in range(len(P)))

class Jensen_Shannon_Divergence:
    def __call__(self, P, Q):
        kl_divergence = Kullback_Leibler_Divergence()

        m = 0.5 * (P + Q)
        return 0.5 * kl_divergence(P, m) + 0.5 * kl_divergence(Q, m)

