# Generate a SYNTHETIC dataset of travel-subscription members (R version).
# Same assumptions as src/simulate.py. R and Python use different random number
# generators, so the members differ, but the channel-level results are close.
# All parameters are invented for illustration; they are not company data.

OBSERVATION_END <- as.Date("2026-03-31")
LAST_COHORT <- as.Date("2025-03-01")

CHANNELS <- data.frame(
  channel       = c("Organic / SEO", "Paid search", "Metasearch", "App", "Conversational AI"),
  beta_a        = c(1.2, 1.6, 2.1, 1.0, 1.3),   # p(churn per renewal) ~ Beta(a, b)
  beta_b        = c(2.2, 1.7, 1.4, 2.6, 2.1),
  annual_margin = c(62, 58, 49, 74, 66),        # mean margin per active year (EUR)
  cac           = c(22, 46, 36, 41, 30),        # acquisition cost (EUR)
  monthly_joins = c(320, 420, 380, 220, 90),
  start         = as.Date(c("2021-04-01", "2021-04-01", "2021-04-01", "2021-04-01", "2024-10-01")),
  stringsAsFactors = FALSE
)

fiscal_year <- function(d) {
  y <- as.integer(format(d, "%Y")); m <- as.integer(format(d, "%m"))
  sprintf("FY%02d", (ifelse(m >= 4, y + 1, y)) %% 100)
}

# Anniversaries passed by `as_of` (renewal opportunities)
anniversaries_passed <- function(join_date, as_of) {
  jy <- as.integer(format(join_date, "%Y")); jmd <- format(join_date, "%m%d")
  ay <- as.integer(format(as_of, "%Y"));     amd <- format(as_of, "%m%d")
  pmax(ay - jy - as.integer(jmd > amd), 0L)
}

simulate_members <- function(seed = 42) {
  set.seed(seed)
  out <- list()
  for (i in seq_len(nrow(CHANNELS))) {
    ch <- CHANNELS[i, ]
    for (m in seq(ch$start, LAST_COHORT, by = "month")) {
      m <- as.Date(m, origin = "1970-01-01")
      n <- rpois(1, ch$monthly_joins)
      if (n == 0) next
      days_in_month <- as.integer(format(seq(m, by = "month", length.out = 2)[2] - 1, "%d"))
      join <- m + sample.int(days_in_month, n, replace = TRUE) - 1
      p <- rbeta(n, ch$beta_a, ch$beta_b)
      margin <- ch$annual_margin * rlnorm(n, meanlog = -0.18, sdlog = 0.6)
      opp <- anniversaries_passed(join, OBSERVATION_END)
      churn_at <- rgeom(n, p) + 1            # renewal at which the member would cancel
      churned <- churn_at <= opp
      out[[length(out) + 1]] <- data.frame(
        channel = ch$channel, join_date = join, renewal_opportunities = opp,
        renewals = ifelse(churned, churn_at - 1, opp), churned = churned,
        margin_per_active_year = round(margin, 2), stringsAsFactors = FALSE)
    }
  }
  df <- do.call(rbind, out)
  df$member_id <- seq_len(nrow(df)) - 1L
  df$cohort_fy <- fiscal_year(df$join_date)
  df[, c("member_id", "channel", "join_date", "cohort_fy",
         "renewal_opportunities", "renewals", "churned", "margin_per_active_year")]
}
