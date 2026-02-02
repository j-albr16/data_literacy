import jax.numpy as jnp
from jax import jit
import jax
from tqdm import tqdm

Q_FACTOR = 2e1
R_FACTOR = 2e3

def KalmanFilter(A, Q, H, R):
    @jit
    def predict(m, P, t, std):
        m_ = A @ m
        P_ = A @ P @ A.T + Q(t, std)
        return m_, P_

    @jit
    def update(m_, P_, z, t, std):
        S = H @ P_ @ H.T + R(t, std)
        K = jnp.linalg.solve(S.T, H @ P_).T
        m = m_ + K @ (z - H @ m_)
        P = P_ - K @ H @ P_
        P = P + 1e-8 * jnp.eye(P.shape[0])
        return m, P

    @jit
    def smooth(ms, Ps, m, P, t, std):
        m_ = A @ m
        P_ = A @ P @ A.T + Q(t, std)
        P_ = P_ + 1e-6 * jnp.eye(P_.shape[0])
        G = jnp.linalg.solve(P_, A @ P).T
        ms = m + G @ (ms - m_)
        Ps = P + G @ (Ps - P) @ G.T
        return ms, Ps

    return predict, update, smooth

def smooth(emotion_logits, std_logits, days):
    dim = emotion_logits.shape[1]
    A = jnp.eye(dim)

    Q = lambda delta_t, std: (jnp.clip(std, min=5e-2, max=5e1)) * Q_FACTOR * delta_t * jnp.eye(dim)
    R = lambda delta_t, _: delta_t * R_FACTOR * jnp.eye(dim)

    H = jnp.eye(dim) 
    predict, update, smooth = KalmanFilter(A, Q, H, R)
    m0 = jnp.ones(dim) / dim 
    P0 = 1 * jnp.eye(dim)
    m = m0
    P = P0

    means = []
    covs = []
    days_list = []
    last_day = 0

    # filter
    for i, day in enumerate(tqdm(days)):
        delta_t = day - last_day
        m_, P_ = predict(m, P, delta_t, jnp.diag(std_logits[i, :]))
        x = emotion_logits[i, :]

        m, P = m_, P_
        if not jnp.isnan(x).all():
            m, P = update(m_, P_, x, delta_t, jnp.diag(std_logits[i, :]))
        
        means.append(m)
        covs.append(P)
        days_list.append(day)
        last_day = day

    # smooth
    means = jnp.array(means)
    covs = jnp.array(covs)

    smooth_means = [means[-1]]
    smooth_covs = [covs[-1]]

    for i in tqdm(range(len(means) - 2, -1, -1)):
        delta_t = days_list[i+1] - days_list[i]
        ms, Ps = smooth(smooth_means[0], smooth_covs[0], means[i], covs[i], delta_t, jnp.diag(std_logits[i, :]))

        smooth_means.insert(0, ms)
        smooth_covs.insert(0, Ps)

    smooth_means = jnp.array(smooth_means)
    smooth_means = jax.nn.softmax(smooth_means)
    smooth_covs = jnp.diagonal(jnp.array(smooth_covs), axis1=-1, axis2=-2)
    smooth_covs = jax.nn.softmax(smooth_covs)
    return smooth_means, smooth_covs


