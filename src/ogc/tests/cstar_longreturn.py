import numpy as np

def cstar_return_indicator(count_series, max_lag=200, rng=0):
    x = np.array(count_series, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    acf = np.array([np.dot(x[:-lag], x[lag:]) / (len(x)-lag) for lag in range(1, max_lag+1)])
    stat_obs = float(np.mean(acf[int(max_lag*0.5):]))

    rnd = np.random.default_rng(rng)
    B, block = 200, max(5, max_lag//10)
    stats_null = []
    for _ in range(B):
        blocks = [x[i:i+block] for i in range(0, len(x), block)]
        rnd.shuffle(blocks)
        xs = np.concatenate(blocks)[:len(x)]
        acf_s = np.array([np.dot(xs[:-lag], xs[lag:]) / (len(xs)-lag) for lag in range(1, max_lag+1)])
        stats_null.append(float(np.mean(acf_s[int(max_lag*0.5):])))
    stats_null = np.array(stats_null)
    p_right = float((stats_null >= stat_obs).mean())
    return {"stat": stat_obs, "p_value": p_right, "tail_mean_acf": float(acf.mean())}

