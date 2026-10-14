"""
Shifted Beta-Geometric (sBG) model for contractual, discrete renewals
(Fader & Hardie, 2007, "How to Project Customer Retention").

Each member has a churn probability p at every renewal; p varies across members
following Beta(a, b). Because the most churn-prone members leave first, the
aggregate retention rate rises with tenure, which a constant-churn model misses.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln


def log_likelihood(params, renewals, churned):
    a, b = np.exp(params)  # keep a, b positive
    base = betaln(a, b)
    # churned at renewal k = renewals + 1:  P = B(a+1, b+k-1) / B(a, b)
    ll_churn = betaln(a + 1, b + renewals) - base
    # still active after `renewals` renewals (right-censored): S = B(a, b+T) / B(a, b)
    ll_alive = betaln(a, b + renewals) - base
    return np.where(churned, ll_churn, ll_alive).sum()


def fit(renewals, churned):
    renewals = np.asarray(renewals, dtype=float)
    churned = np.asarray(churned, dtype=bool)
    res = minimize(lambda x: -log_likelihood(x, renewals, churned),
                   x0=np.log([1.0, 1.0]), method="Nelder-Mead")
    a, b = np.exp(res.x)
    return a, b


def survival(a, b, t):
    """Share of an acquired cohort still active after t renewals (t = 0 -> 1.0)."""
    t = np.asarray(t, dtype=float)
    return np.exp(betaln(a, b + t) - betaln(a, b))


def ltv(a, b, annual_margin, years=5, discount_rate=0.10):
    """Expected discounted margin per ACQUIRED member over `years` membership years.
    The year-0 fee and margin arrive at sign-up; later years only if the member renews."""
    t = np.arange(years)
    return float(np.sum(annual_margin * survival(a, b, t) / (1 + discount_rate) ** t))
