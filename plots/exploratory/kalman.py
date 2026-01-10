import jax.numpy as jnp
from jax import jit
import jax

Q_FACTOR = 1e3
R_FACTOR = 1e4

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
        # Ensure P doesn't collapse to zero (numerical stability)
        P = P + 1e-8 * jnp.eye(P.shape[0])
        return m, P

    @jit
    def smooth(ms, Ps, m, P, t, std):
        m_ = A @ m
        P_ = A @ P @ A.T + Q(t, std)
        # Add regularization to prevent singular matrix when P collapses to zero
        P_ = P_ + 1e-6 * jnp.eye(P_.shape[0])
        G = jnp.linalg.solve(P_, A @ P).T
        ms = m + G @ (ms - m_)
        Ps = P + G @ (Ps - P) @ G.T
        return ms, Ps

    return predict, update, smooth

def smooth(emotion_logits, std_logits, days):
    # std_logits = jnp.sqrt(jnp.clip(jnp.log(std_logits), min=0))
    dim = emotion_logits.shape[1]
    A = jnp.eye(dim)
    Q = lambda delta_t, std: jnp.clip(std, min=1e-3, max=100) * Q_FACTOR * delta_t * jnp.eye(dim)
    H = jnp.eye(dim) 
    R = lambda delta_t, std: jnp.clip(std, min=1e-3, max=100) * delta_t * R_FACTOR * jnp.eye(dim)

    predict, update, smooth = KalmanFilter(A, Q, H, R)
    m0 = jnp.ones(dim) / dim      # uniform emotions prior
    P0 = 1 * jnp.eye(dim)
    m = m0
    P = P0

    means = []
    covs = []
    days_list = []
    last_day = 0

    # Forward pass (filtering)
    for i, day in enumerate(days):
        delta_t = day - last_day
        m_, P_ = predict(m, P, delta_t, jnp.diag(std_logits[i, :]))
        x = emotion_logits[i, :]

        m, P = m_, P_
        if not jnp.isnan(x).all():
            m, P = update(m_, P_, x, delta_t, jnp.diag(std_logits[i, :]))

        if jnp.isnan(m).any():
            print(f"NaN detected at iteration {i}, day {day}")
            print(f"  m_: {m_}, has NaN: {jnp.isnan(m_).any()}")
            print(f"  m: {m}, has NaN: {jnp.isnan(m).any()}")
            print(f"  P diagonal: {jnp.diag(P)}")
            break

        means.append(m)
        covs.append(P)
        days_list.append(day)
        last_day = day

    # Backward pass (smoothing)
    means = jnp.array(means)
    covs = jnp.array(covs)
    print(f"Forward pass complete. means has NaN: {jnp.isnan(means).any()}, covs has NaN: {jnp.isnan(covs).any()}")
    smooth_means = [means[-1]]
    smooth_covs = [covs[-1]]

    for i in range(len(means) - 2, -1, -1):
        delta_t = days_list[i+1] - days_list[i]
        ms, Ps = smooth(smooth_means[0], smooth_covs[0], means[i], covs[i], delta_t, jnp.diag(std_logits[i, :]))

        if jnp.isnan(ms).any():
            print(f"NaN in backward pass at i={i}")
            print(f"  delta_t: {delta_t}")
            print(f"  covs[i] diagonal: {jnp.diag(covs[i])}")

        smooth_means.insert(0, ms)
        smooth_covs.insert(0, Ps)

    smooth_means = jnp.array(smooth_means)
    print("smooth_means before softmax:", smooth_means.shape, "has NaN:", jnp.isnan(smooth_means).any(), "has inf:", jnp.isinf(smooth_means).any())
    smooth_means = jax.nn.softmax(smooth_means)
    print("smooth_means after softmax:", smooth_means.shape, "has NaN:", jnp.isnan(smooth_means).any())
    smooth_covs = jnp.diagonal(jnp.array(smooth_covs), axis1=-1, axis2=-2)
    smooth_covs = jax.nn.softmax(smooth_covs)
    return smooth_means, smooth_covs


