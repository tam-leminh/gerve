import numpy as np
from scipy import stats

def make_box_support(lower, upper):
    """Create and validate box support for bounded entropy computation."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    vol = np.prod(upper - lower)
    return lower, upper, vol

def in_box(X, lower, upper):
    """Check which samples fall inside the box support."""
    return np.all((X >= lower) & (X <= upper), axis=1)

def project_mean_to_margin(mean, cov, lower, upper, m=3.0):
    """Project mean to stay at least m standard deviations away from box walls."""
    smax = np.sqrt(np.max(np.linalg.eigvalsh(cov)))
    lo = lower + m * smax
    hi = upper - m * smax
    return np.minimum(np.maximum(mean, lo), hi)

def G_fun(S, epsi=0):
    """Compute pseudo inverse of matrix S."""
    if epsi == 0:
        try:
            inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(S)
    else:
        S_T = S.T
        di = S.shape[0]
        StS = S_T @ S
        StS_reg = StS + epsi * np.eye(di)
        inv = np.linalg.solve(StS_reg, S_T)
    return inv

def blackbox_prefixes(mean, prec, xik, pre_cov=True, n_parents=0):
    """Compute gradient prefixes."""
    if n_parents == 0:
        n_parents = xik.shape[0]

    diff = xik[:n_parents] - mean[None, :]
    all_pre_mean = diff @ prec.T
    if pre_cov:
        all_pre_cov = all_pre_mean[:, :, None] * all_pre_mean[:, None, :] - prec[None, :, :]
        return all_pre_mean, all_pre_cov

    return all_pre_mean

def blackbox_gradient_mu(all_pre_mean, f_values, is_weights=None, ret_indiv_grad=False):
    """Compute gradient wrt mean."""
    n_parents = all_pre_mean.shape[0]
    f_vals_flat = f_values[:n_parents].flatten()
    if is_weights is None:
        grad_mu_mc = all_pre_mean * f_vals_flat[:, None]
    else:
        is_weights_flat = is_weights[:n_parents].flatten()
        combined_weights = f_vals_flat * is_weights_flat
        grad_mu_mc = all_pre_mean * combined_weights[:, None]

    if ret_indiv_grad:
        return np.mean(grad_mu_mc, axis=0), grad_mu_mc
    return np.mean(grad_mu_mc, axis=0)

def blackbox_gradient_cov(all_pre_cov, f_values, is_weights=None):
    """Compute gradient wrt covariance."""
    n_parents = all_pre_cov.shape[0]
    f_vals_flat = f_values[:n_parents].flatten()

    if is_weights is None:
        grad_s1_mc = all_pre_cov * f_vals_flat[:, None, None]
    else:
        is_weights_flat = is_weights[:n_parents].flatten()
        combined_weights = f_vals_flat * is_weights_flat
        grad_s1_mc = all_pre_cov * combined_weights[:, None, None]

    return np.mean(grad_s1_mc, axis=0)

def rMVNmixture(nb, pis, means, covs):
    """Generate samples from multivariate normal mixture."""
    Kvar = pis.shape[0]
    select_comp = np.random.choice(Kvar, nb, p=pis)
    d = means.shape[1]
    samples = np.zeros((nb, d))

    for k in range(Kvar):
        mask = (select_comp == k)
        n_k = np.sum(mask)
        if n_k > 0:
            samples[mask] = stats.multivariate_normal.rvs(
                mean=means[k, :], cov=covs[k, :, :], size=n_k
            ).reshape(n_k, d)

    return samples

def parse_init_param(N_iter, init_mixture_param):
    """Parse initialization parameters."""
    piInit, meanInit, covInit, precInit = init_mixture_param[:4]
    Kvar = len(piInit)
    meanvect = np.tile(meanInit[:, :, np.newaxis], (1, 1, N_iter))
    precvect = np.tile(precInit[:, :, :, np.newaxis], (1, 1, 1, N_iter))
    invprec = np.tile(covInit[:, :, :, np.newaxis], (1, 1, 1, N_iter))
    pivect = np.tile(piInit[:, np.newaxis], (1, N_iter))

    log_ratios = np.log(piInit[:-1]) - np.log(piInit[-1])
    v_init = np.append(log_ratios, 0)
    vvect = np.tile(v_init[:, np.newaxis], (1, N_iter))

    return Kvar, meanvect, precvect, invprec, pivect, vvect

def entropy_only(pis, means, covs, xik):
    """Compute entropy values for samples."""
    n_samples = xik.shape[0]
    n_components = len(pis)
    batch_threshold = 5000 

    if n_samples <= batch_threshold:
        densities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            densities[:, k] = stats.multivariate_normal.pdf(
                xik, mean=means[k, :], cov=covs[k, :, :], allow_singular=True
            )
        mixture_densities = densities @ pis
        log_densities = np.where(mixture_densities > 0,
                               np.log(mixture_densities),
                               -743)
    else:
        batch_size = 3000  
        log_densities = np.zeros(n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_xik = xik[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx
            densities = np.zeros((batch_size_actual, n_components))
            for k in range(n_components):
                densities[:, k] = stats.multivariate_normal.pdf(
                    batch_xik, mean=means[k, :], cov=covs[k, :, :], allow_singular=True
                )
            mixture_densities = densities @ pis
            log_densities[start_idx:end_idx] = np.where(mixture_densities > 0,
                                                       np.log(mixture_densities),
                                                       -743)

    return log_densities.reshape(-1, 1)

def mean_update(mean, cov, grad_mu, lr, update_cap_mean=0):
    """Update mean parameter using natural gradient."""
    delta_mu = lr * (cov @ grad_mu)

    if update_cap_mean > 0:
        norm_update = np.linalg.norm(delta_mu)
        if norm_update > update_cap_mean:
            delta_mu = update_cap_mean / norm_update * delta_mu

    return mean + delta_mu

def precision_update(cov, prec, grad_s1, lr, damping, sig2_min, sig2_max, update_cap_prec=0):
    """Update precision matrix using natural gradient."""
    temp_product = grad_s1 @ cov
    triple_product = temp_product @ grad_s1

    lr_sq_half = (lr * lr) * 0.5 
    delta_s = - lr * grad_s1 + lr_sq_half * triple_product
    if update_cap_prec > 0:
        norm_update = np.linalg.norm(delta_s)
        if norm_update > update_cap_prec:
            delta_s = update_cap_prec / norm_update * delta_s
    new_prec = prec + delta_s
    new_prec = (new_prec + new_prec.T) * 0.5
    new_cov = G_fun(new_prec, 0)
    new_cov = 0.5 * (new_cov + new_cov.T)
    d = new_cov.shape[0]
    if damping > 0:
        new_cov.flat[::d+1] += damping

    try:
        np.linalg.cholesky(new_cov)
    except np.linalg.LinAlgError:
        lmin = np.linalg.eigvalsh(new_cov)[0] 
        new_cov.flat[::d+1] += -lmin + 1e-12
        print(new_cov.flat[::d+1])

    if sig2_min <= sig2_max:
        eigenvals = np.linalg.eigvalsh(new_cov)
        lmin = eigenvals[0]
        lmax = eigenvals[-1]
        if lmin < sig2_min or lmax > sig2_max:
            eigenvals, eigenvecs = np.linalg.eigh(new_cov)
            eigenvals_clipped = np.clip(eigenvals, sig2_min, sig2_max)
            new_cov = eigenvecs @ np.diag(eigenvals_clipped) @ eigenvecs.T

    new_prec = G_fun(new_cov, 0)
    return new_cov, new_prec

def weight_update(pis, vs, fs, lr, update_cap_weight=0, random_ref=True):
    """Update mixture weights using natural gradient."""
    Kvar = pis.shape[0]
    new_vs = np.empty(Kvar, dtype=vs.dtype)
    if random_ref:
        ref_idx = np.random.randint(Kvar)
    else:
        ref_idx = Kvar - 1

    vs_normalized = vs - vs[ref_idx]
    ref_f = fs[ref_idx]
    grad_update = lr * (fs - ref_f)

    if update_cap_weight > 0:
        norm_update = np.linalg.norm(grad_update)
        if norm_update > update_cap_weight:
            grad_update = update_cap_weight / norm_update * grad_update

    new_vs = np.clip(vs_normalized + grad_update, -10, 10)
    new_vs[ref_idx] = 0 
    max_vs = np.max(new_vs)
    exp_vs = np.exp(new_vs - max_vs)
    sum_exp = np.sum(exp_vs)
    new_pis = exp_vs / sum_exp

    return new_pis, new_vs

def blackbox_updates(xik_drive, xik_entropy, mean, cov, prec, p_values, e_values, wKL, lr, damping, sig2_min, sig2_max, update_cap_mean, update_cap_prec, update_prec, is_weights=None, use_box=False, lower=None, upper=None, margin_m=3.0):
    """Update parameters using blackbox gradients."""

    if update_prec:
        all_pre_mean_drive, all_pre_cov_drive = blackbox_prefixes(mean, prec, xik_drive, pre_cov=True)
        all_pre_mean_entropy, all_pre_cov_entropy = blackbox_prefixes(mean, prec, xik_entropy, pre_cov=True)
        grad_s1_drive = blackbox_gradient_cov(all_pre_cov_drive, p_values, is_weights=None)
        grad_s1_entropy = blackbox_gradient_cov(all_pre_cov_entropy, e_values, is_weights)

        grad_s1 = grad_s1_drive - wKL * grad_s1_entropy

        new_cov, new_prec = precision_update(cov, prec, grad_s1, lr, damping, sig2_min, sig2_max, update_cap_prec)
    else:
        all_pre_mean_drive = blackbox_prefixes(mean, prec, xik_drive, pre_cov=False)
        all_pre_mean_entropy = blackbox_prefixes(mean, prec, xik_entropy, pre_cov=False)
        new_cov, new_prec = cov, prec

    grad_mu_drive = blackbox_gradient_mu(all_pre_mean_drive, p_values, is_weights=None, ret_indiv_grad=False)

    grad_mu_entropy = blackbox_gradient_mu(all_pre_mean_entropy, e_values, is_weights, ret_indiv_grad=False)
    grad_mu = grad_mu_drive - wKL * grad_mu_entropy
    new_mean = mean_update(mean, new_cov, grad_mu, lr, update_cap_mean)

    if use_box and lower is not None and upper is not None:
        new_mean = project_mean_to_margin(new_mean, new_cov, lower, upper, m=margin_m)

    return new_mean, new_cov, new_prec

def gerve_fit(N_iter, B_drive, B_entropy, learning, wKL, samples, init_mixture_param,
                   burn=0, damping=0, sig2_min=1e-12, sig2_max=0,
                   update_cap_mean=0, update_cap_prec=0, update_cap_weight=0, update_weights=True,
                   lower=None, upper=None, margin_m=3.0,
                   early_stopping=False, eps_mu=1e-3, eps_S=1e-3, eps_pi=1e-4,
                   check_every=10, patience=3, min_iters=1000):
    """
    GERVE algorithm

    Parameters
    ----------
    N_iter : int
        Maximum number of iterations
    B_drive : int
        Number of samples for driving objective
    B_entropy : int
        Number of samples for entropy estimation
    learning : array-like
        Learning rate schedule
    wKL : array-like
        Entropy weight schedule
    samples : ndarray
        Data samples
    init_mixture_param : tuple
        Initial mixture parameters (pis, means, covs, precs)
    burn : int, optional
        Burn-in period before updating covariances
    damping : float, optional
        Damping for covariance updates
    sig2_min : float, optional
        Minimum eigenvalue for covariance
    sig2_max : float, optional
        Maximum eigenvalue for covariance
    update_cap_mean : float, optional
        Cap on mean update magnitude
    update_cap_prec : float, optional
        Cap on precision update magnitude
    update_cap_weight : float, optional
        Cap on weight update magnitude
    update_weights : bool, optional
        Whether to update mixture weights
    lower : array-like or None, optional
        Lower bounds for box support S. If None, no bounded support is used.
    upper : array-like or None, optional
        Upper bounds for box support S. If None, no bounded support is used.
    margin_m : float, optional
        Number of standard deviations to maintain as margin from box walls.
        Only used when both lower and upper are provided.
    early_stopping : bool, optional
        Master switch for early stopping. If False, all stopping logic is disabled.
    eps_mu : float or None, optional
        Early stopping tolerance for mean change (Euclidean norm).
    eps_S : float or None, optional
        Early stopping tolerance for covariance change (relative Frobenius norm).
    eps_pi : float or None, optional
        Early stopping tolerance for weight change (L1 norm). Set to None to disable weight check.
    check_every : int, optional
        Check early stopping criteria every N iterations.
    patience : int, optional
        Number of consecutive checks that must pass all thresholds before early stopping.
    min_iters : int, optional
        Minimum number of iterations before early stopping is allowed.

    Returns
    -------
    pivect : ndarray
        History of mixture weights
    meanvect : ndarray
        History of component means
    invprec : ndarray
        History of component covariances
    precvect : ndarray
        History of component precisions
    """

    Kvar, meanvect, precvect, invprec, pivect, vvect = parse_init_param(N_iter, init_mixture_param)

    if B_drive > len(samples):
        B_drive = len(samples)
    use_box = (lower is not None) and (upper is not None)
    if use_box:
        lower, upper, volS = make_box_support(lower, upper)
    if early_stopping:
        pass_streak = 0
        tiny = 1e-12  
        _eps_mu = np.inf if eps_mu is None else eps_mu
        _eps_S = np.inf if eps_S is None else eps_S
        _eps_pi = np.inf if eps_pi is None else eps_pi
    phi = np.zeros(Kvar)
    fs = np.zeros(Kvar)
    all_p_values = np.zeros((Kvar, B_drive))

    # ITERATION LOOP
    for i in range(1, N_iter):
        means_cur, covs_cur, precs_cur, pis_cur, vs_cur = meanvect[:, :, i-1], invprec[:, :, :, i-1], precvect[:, :, :, i-1], pivect[:, i-1], vvect[:, i-1]
        w_cur, lr_cur = wKL[i], learning[i]

        # MINI-BATCH
        if B_drive < len(samples):
            idx_chosen = np.random.choice(len(samples), size=B_drive, replace=True)
            samples_chosen = samples[idx_chosen, :]
        else:
            samples_chosen = samples

        # COMPONENT DENSITIES AT SAMPLES
        for k in range(Kvar):
            all_p_values[k, :] = stats.multivariate_normal.pdf(
                samples_chosen, mean=means_cur[k, :], cov=covs_cur[k, :, :], allow_singular=True
            )
            fs[k] = np.mean(all_p_values[k, :])

        # MEAN AND COVARIANCE UPDATES
        for k in range(Kvar):
            xik = stats.multivariate_normal.rvs(mean=means_cur[k, :], cov=covs_cur[k, :, :], size=B_entropy)
            if B_entropy == 1:
                xik = xik.reshape(1, -1)

            if use_box:
                mask = in_box(xik, lower, upper)
                if not np.any(mask):
                    dens_k = stats.multivariate_normal.pdf(
                        xik, mean=means_cur[k,:], cov=covs_cur[k,:,:], allow_singular=True)
                    keep = np.argmax(dens_k)
                    mask = np.zeros(xik.shape[0], dtype=bool)
                    mask[keep] = True
                xik_e = xik[mask, :]
            else:
                xik_e = xik 

            e_values = entropy_only(pis_cur, means_cur, covs_cur, xik_e)
            is_weights = None
            phi[k] = np.mean(e_values)
            xik_for_update = xik_e 

            p_values = all_p_values[k, :].reshape(-1, 1)

            meanvect[k, :, i], invprec[k, :, :, i], precvect[k, :, :, i] = blackbox_updates(
                samples_chosen, xik_for_update,
                means_cur[k, :], covs_cur[k, :, :], precs_cur[k, :, :],
                p_values, e_values, w_cur, lr_cur, damping, sig2_min, sig2_max, update_cap_mean, update_cap_prec,
                update_prec=(i > burn), is_weights=is_weights,
                use_box=use_box, lower=lower, upper=upper, margin_m=margin_m
            )

        # WEIGHT UPDATE
        if update_weights:
            grad_pi = fs - w_cur*phi
            pivect[:, i], vvect[:, i] = weight_update(pis_cur, vs_cur, grad_pi, lr_cur, update_cap_weight)

        # EARLY STOPPING CHECK
        if early_stopping and (i % check_every == 0) and (i > max(burn, min_iters)):
            means_prev = meanvect[:, :, i-1]
            means_curr = meanvect[:, :, i]
            covs_prev = invprec[:, :, :, i-1]
            covs_curr = invprec[:, :, :, i]
            pis_prev = pivect[:, i-1]
            pis_curr = pivect[:, i]

            means_prev = means_prev.astype(np.float64)
            means_curr = means_curr.astype(np.float64)
            covs_prev = covs_prev.astype(np.float64)
            covs_curr = covs_curr.astype(np.float64)
            pis_prev = pis_prev.astype(np.float64)
            pis_curr = pis_curr.astype(np.float64)

            d_mu = np.max([np.linalg.norm(means_curr[k] - means_prev[k]) for k in range(Kvar)])

            d_S_vals = []
            for k in range(Kvar):
                cov_diff_norm = np.linalg.norm(covs_curr[k] - covs_prev[k], 'fro')
                cov_prev_norm = np.linalg.norm(covs_prev[k], 'fro')
                d_S_k = cov_diff_norm / (cov_prev_norm + tiny)
                d_S_vals.append(d_S_k)
            d_S = np.max(d_S_vals)

            d_pi = np.sum(np.abs(pis_curr - pis_prev))

            passed = (d_mu <= _eps_mu) and (d_S <= _eps_S) and (d_pi <= _eps_pi)

            if passed:
                pass_streak += 1
            else:
                pass_streak = 0

            if pass_streak >= patience:
                print(f"\nEarly stopping at iteration {i}")
                print(f"  d_mu = {d_mu:.6e} (threshold: {_eps_mu:.6e})")
                print(f"  d_S  = {d_S:.6e} (threshold: {_eps_S:.6e})")
                print(f"  d_pi = {d_pi:.6e} (threshold: {_eps_pi:.6e})")

                used = i + 1
                meanvect = meanvect[:, :, :used]
                invprec = invprec[:, :, :, :used]
                precvect = precvect[:, :, :, :used]
                pivect = pivect[:, :used]
                vvect = vvect[:, :used]

                return pivect, meanvect, invprec, precvect

    return pivect, meanvect, invprec, precvect