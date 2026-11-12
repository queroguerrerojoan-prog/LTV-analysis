"""
Generate a SYNTHETIC dataset of travel-subscription members.

The structure mirrors an annual travel membership (like a Prime-style programme):
members pay an annual fee, book travel while active, and decide at each yearly
renewal date whether to stay. All parameters below are invented assumptions for
illustration; they are not company data.
"""
import numpy as np
import pandas as pd

SEED = 42
OBSERVATION_END = pd.Timestamp("2026-03-31")

# Channel assumptions (synthetic).
# beta_a / beta_b: heterogeneity in the per-renewal churn probability, p ~ Beta(a, b)
#   (mean churn = a / (a + b)); annual_margin: mean contribution per active year in EUR
#   (membership fee + booking margin - servicing costs); cac: cost to acquire one member.
CHANNELS = {
    "Organic / SEO":      dict(beta_a=1.2, beta_b=2.2, annual_margin=62, cac=22, monthly_joins=320, start="2021-04"),
    "Paid search":        dict(beta_a=1.6, beta_b=1.7, annual_margin=58, cac=46, monthly_joins=420, start="2021-04"),
    "Metasearch":         dict(beta_a=2.1, beta_b=1.4, annual_margin=49, cac=36, monthly_joins=380, start="2021-04"),
    "App":                dict(beta_a=1.0, beta_b=2.6, annual_margin=74, cac=41, monthly_joins=220, start="2021-04"),
    "Conversational AI":  dict(beta_a=1.3, beta_b=2.1, annual_margin=66, cac=30, monthly_joins=90,  start="2024-10"),
}
LAST_COHORT = "2025-03"  # joiners after this have not reached a first renewal yet


def simulate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for channel, c in CHANNELS.items():
        for cohort in pd.period_range(c["start"], LAST_COHORT, freq="M"):
            n = rng.poisson(c["monthly_joins"])
            # join day spread over the month
            days = rng.integers(0, cohort.days_in_month, size=n)
            join_dates = cohort.start_time + pd.to_timedelta(days, unit="D")
            p_churn = rng.beta(c["beta_a"], c["beta_b"], size=n)
            # member-level value per active year (skewed: some members travel a lot)
            margin = c["annual_margin"] * rng.lognormal(mean=-0.18, sigma=0.6, size=n)
            # number of yearly renewal dates already passed by the observation end
            jd = pd.DatetimeIndex(join_dates)
            passed_anniversary = (jd.month < OBSERVATION_END.month) | (
                (jd.month == OBSERVATION_END.month) & (jd.day <= OBSERVATION_END.day))
            opportunities = OBSERVATION_END.year - jd.year - (~passed_anniversary).astype(int)
            # renewal number at which the member would cancel: geometric with prob p
            churn_at = rng.geometric(p_churn)
            churned = churn_at <= opportunities
            renewals = np.where(churned, churn_at - 1, opportunities)
            rows.append(pd.DataFrame(dict(
                channel=channel,
                join_date=jd,
                renewal_opportunities=np.asarray(opportunities),
                renewals=renewals,
                churned=churned,
                margin_per_active_year=np.round(margin, 2),
            )))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "member_id", np.arange(len(df)))
    df.insert(3, "cohort_fy", df["join_date"].map(fiscal_year))
    return df


def fiscal_year(d: pd.Timestamp) -> str:
    """Fiscal year runs April-March, e.g. Apr 2021-Mar 2022 = FY22."""
    fy = d.year + 1 if d.month >= 4 else d.year
    return f"FY{str(fy)[-2:]}"


if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/members.csv", index=False)
    print(df.shape)
    print(df.groupby("channel").size())
