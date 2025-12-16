import numpy as np
from utility import *
from gerve import *

import random

random.seed(1)
np.random.seed(1)

d = 2

# Example: Triangle Mixture
KMM = 3
sig2 = 0.25
piMM = np.array([1/3, 1/3, 1/3])
meanMM = np.array([[0., 1.], [np.cos(np.pi/6), -0.5], [-np.cos(np.pi/6), -0.5]])
covMM = np.repeat(np.diag([sig2] * d)[np.newaxis, :, :], KMM, axis=0) 
precMM = np.repeat(np.linalg.inv(np.diag([sig2] * d))[np.newaxis, :, :], KMM, axis=0) 

# Generate samples
n = 10000
samples = rMVNmixture(n, piMM, meanMM, covMM)
print(f"Generated {samples.shape[0]} samples from triangle mixture!")

# GERVE hyperparameters
Kvar = 7
covfix = 2.
piInit = np.array([1/Kvar for _ in range(Kvar)])
meanInit = np.random.uniform(-2, 0, (Kvar, d))
precInit = np.repeat(np.linalg.inv(np.diag([covfix] * d))[np.newaxis, :, :], Kvar, axis=0)
covInit = np.repeat(np.diag([covfix] * d)[np.newaxis, :, :], Kvar, axis=0)

N = 30000
Nt = 1000
w0 = 50.
w_decrease_power = 1.1
w_offset = 0.004
wKL = w0 / np.arange(1, N+1)**w_decrease_power + w_offset
lrate_0 = 3.e-4
lr_w_power = 0.7
learning = lrate_0 * (w0 / wKL)**lr_w_power
epsilon = 0.01
lower = np.array([-10.0, -10.0], dtype=float)
upper = np.array([ 10.0,  10.0], dtype=float)

hyperparameters = { 
    "N_iter": N,
    "B_drive": Nt,
    "B_entropy": Nt,
    "burn": 0,
    "sig2_min": epsilon,
    "sig2_max": covfix,
    "init_mixture_param": [piInit, meanInit, covInit, precInit]
}
schedules = { 
    "wKL": wKL,
    "learning": learning
}
box_args = { 
    "lower": lower,
    "upper": upper
}
es_args = {
    "early_stopping" : False, 
    "eps_mu" : 1e-2, 
    "eps_S" : 1e-1, 
    "eps_pi" : 1,
    "check_every" : 10, 
    "patience" : 3, 
    "min_iters" : 500
}

print(f"Fitting with {N} iterations...")
res = gerve_fit(samples=samples, **hyperparameters, **schedules, **box_args, **es_args)
pivect, meanvect, invprec, precvect = res

print("Plotting component trajectories, final covariances and weights...")
# Plot only a fraction of data points
subsample_fraction = 0.5
n_total = samples.shape[0]
n_plot = max(1, int(subsample_fraction * n_total))
idx = np.random.choice(n_total, size=n_plot, replace=False)
samples_plot = samples[idx]
Nstop = N - 1
plot_gmm_clusters(samples_plot, meanvect[:,:,Nstop], invprec[:,:,:,Nstop], pivect[:,Nstop], trajectories=meanvect[:,:,:Nstop], 
                  point_size=10, low_traj=True, plot_ellipses=True, alpha=0.2)

print("Done!")