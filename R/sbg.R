# Shifted Beta-Geometric (sBG) model for contractual, discrete renewals
# (Fader & Hardie, 2007, "How to Project Customer Retention").
#
# Each member has a churn probability p at every renewal; p ~ Beta(a, b) across
# members. The most churn-prone members leave first, so aggregate retention rises
# with tenure, which a constant-churn model misses.

sbg_loglik <- function(log_params, renewals, churned) {
  a <- exp(log_params[1]); b <- exp(log_params[2])
  base <- lbeta(a, b)
  ll_churn <- lbeta(a + 1, b + renewals) - base  # cancelled at renewal k = renewals + 1
  ll_alive <- lbeta(a, b + renewals) - base      # still active after `renewals` renewals
  sum(ifelse(churned, ll_churn, ll_alive))
}

sbg_fit <- function(renewals, churned) {
  opt <- optim(c(0, 0), function(x) -sbg_loglik(x, renewals, churned),
               method = "Nelder-Mead", control = list(reltol = 1e-12, maxit = 5000))
  setNames(exp(opt$par), c("a", "b"))
}

# Share of an acquired cohort still active after t renewals (t = 0 -> 1)
sbg_survival <- function(a, b, t) exp(lbeta(a, b + t) - lbeta(a, b))

# Expected discounted margin per ACQUIRED member over `years` membership years.
# Year-0 margin arrives at sign-up; later years only if the member renews.
sbg_ltv <- function(a, b, annual_margin, years = 5, discount_rate = 0.10) {
  t <- 0:(years - 1)
  sum(annual_margin * sbg_survival(a, b, t) / (1 + discount_rate)^t)
}
