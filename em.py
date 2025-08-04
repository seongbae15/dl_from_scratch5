import os
import numpy as np

path = os.path.join("old_faithful.txt")
xs = np.loadtxt(path)
print(xs.shape)
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)])

K = len(phis)
N = len(xs)
MAX_ITERS = 100
TRESHOLD = 1e-4


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp(-0.5 * (x - mu).T @ inv @ (x - mu))
    return y


def gmm(x, phis, mus, covs):
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


def likelihood(xs, phis, mus, covs):
    eps = 1e-8
    L = 0
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N


current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS):
    # E-Step
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)

    # M-Step
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        phis[k] = qs_sum[k] / N

        c = 0
        for n in range(N):
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]

    next_likelihood = likelihood(xs, phis, mus, covs)
    print(
        f"Current likelihood: {current_likelihood:.4f}, Next likelihood: {next_likelihood:.4f}"
    )
    diff = np.abs(next_likelihood - current_likelihood)
    if diff < TRESHOLD:
        break
    current_likelihood = next_likelihood
