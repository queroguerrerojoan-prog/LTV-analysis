# End-to-end LTV analysis for a travel subscription programme (R version).
#
#   Rscript R/run_analysis.R              # uses data/members.csv (same data as Python)
#   Rscript R/run_analysis.R --simulate   # generates its own synthetic data in R
#
# Steps: observed retention -> sBG fit per channel -> 5-year LTV, LTV:CAC, payback
#        -> back-test as of 31 Mar 2024 -> young-channel scenarios -> value concentration

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(ggplot2)
  library(scales); library(patchwork)
})
source("R/sbg.R")
source("R/simulate.R")

YEARS <- 5; DISCOUNT <- 0.10
CUTOFF <- as.Date("2024-03-31")
MATURE <- c("Organic / SEO", "Paid search", "Metasearch", "App")
FIG <- "figures/r"; OUT <- "results/r"
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------- 1. data
use_sim <- "--simulate" %in% commandArgs(trailingOnly = TRUE) || !file.exists("data/members.csv")
members <- if (use_sim) simulate_members() else
  read_csv("data/members.csv", show_col_types = FALSE) %>%
    mutate(churned = as.logical(churned), join_date = as.Date(join_date))
cat(sprintf("%s members (%s)\n", format(nrow(members), big.mark = ","),
            if (use_sim) "simulated in R" else "data/members.csv"))

# ---------------------------------------------------------------- helpers
# Share of an acquired cohort still active after t renewals.
# Only members who have reached renewal t count (handles right-censoring).
observed_retention <- function(d, max_t = 4) {
  s <- 1; surv <- 1
  for (t in seq_len(max_t)) {
    at_risk <- d %>% filter(renewal_opportunities >= t, renewals >= t - 1)
    if (nrow(at_risk) < 50) break
    s <- s * mean(at_risk$renewals >= t)
    surv <- c(surv, s)
  }
  surv
}

# The common shortcut: take the first-renewal churn rate and assume it every year
constant_churn_fit <- function(d) {
  reached <- d %>% filter(renewal_opportunities >= 1)
  mean(reached$renewals == 0)
}

# What the data looked like at `cutoff` (later renewals and cancellations hidden)
as_of <- function(d, cutoff) {
  d %>%
    filter(join_date <= cutoff) %>%
    mutate(opp = anniversaries_passed(join_date, cutoff),
           churned = churned & (renewals + 1 <= opp),
           renewals = pmin(renewals, opp),
           renewal_opportunities = opp) %>%
    select(-opp)
}

eur <- function(x) paste0("€", format(round(x), big.mark = ",", trim = TRUE))
pct <- function(x, acc = 1) percent(x, accuracy = acc)

# ---------------------------------------------------------------- 2-3. sBG and LTV by channel
fits <- list()
summary_tbl <- CHANNELS %>% rowwise() %>% do({
  ch <- .
  d <- members %>% filter(channel == ch$channel)
  ab <- sbg_fit(d$renewals, d$churned); fits[[ch$channel]] <<- ab
  margin <- mean(d$margin_per_active_year)
  obs <- observed_retention(d)
  c1 <- constant_churn_fit(d)
  data.frame(
    channel = ch$channel, members = nrow(d),
    first_renewal_rate = obs[2],
    retention_y4 = if (length(obs) >= 5) obs[5] else NA_real_,
    sbg_a = ab[["a"]], sbg_b = ab[["b"]],
    avg_margin_per_year = margin,
    ltv_5y = sbg_ltv(ab[["a"]], ab[["b"]], margin, YEARS, DISCOUNT),
    ltv_5y_constant_churn = sum(margin * (1 - c1)^(0:(YEARS - 1)) / (1 + DISCOUNT)^(0:(YEARS - 1))),
    cac = ch$cac, stringsAsFactors = FALSE)
}) %>% ungroup() %>%
  mutate(ltv_cac = ltv_5y / cac, payback_months = cac / (avg_margin_per_year / 12))

# ---------------------------------------------------------------- 4. back-test
calib <- as_of(members, CUTOFF)
backtest <- bind_rows(lapply(MATURE, function(ch) {
  dc <- calib %>% filter(channel == ch)
  ab <- sbg_fit(dc$renewals, dc$churned)
  c1 <- constant_churn_fit(dc)
  actual <- observed_retention(members %>% filter(channel == ch, cohort_fy == "FY22"))
  data.frame(channel = ch, renewal = c(3, 4), actual = actual[c(4, 5)],
             sbg = sbg_survival(ab[["a"]], ab[["b"]], c(3, 4)),
             constant = (1 - c1)^c(3, 4))
})) %>%
  mutate(sbg_error_pp = (sbg - actual) * 100, constant_error_pp = (constant - actual) * 100)
write_csv(backtest %>% mutate(across(where(is.numeric), ~ round(.x, 4))), file.path(OUT, "backtest.csv"))

# ---------------------------------------------------------------- 5. young channel
# One renewal observed: the data pins down year-1 retention but not later years.
# (a) bootstrap the year-1 rate, (b) borrow each mature channel's later-year shape.
ai <- members %>% filter(channel == "Conversational AI")
ai_margin <- mean(ai$margin_per_active_year)
ai_r1 <- observed_retention(ai)[2]
scenarios <- sapply(MATURE, function(ch) {
  ab <- fits[[ch]]; t <- 0:(YEARS - 1)
  shape <- sbg_survival(ab[["a"]], ab[["b"]], t) / sbg_survival(ab[["a"]], ab[["b"]], 1)
  surv <- c(1, ai_r1 * shape[-1])
  sum(ai_margin * surv / (1 + DISCOUNT)^t)
})
set.seed(7)
boot_r1 <- replicate(500, observed_retention(ai[sample.int(nrow(ai), replace = TRUE), ])[2])
ai_r1_ci <- quantile(boot_r1, c(0.05, 0.95))

ai_mid <- mean(range(scenarios))
summary_tbl <- summary_tbl %>%
  mutate(ltv_5y = ifelse(channel == "Conversational AI", ai_mid, ltv_5y),
         ltv_cac = ltv_5y / cac)
write_csv(summary_tbl %>% mutate(across(where(is.numeric), ~ round(.x, 3))), file.path(OUT, "ltv_by_channel.csv"))

# ---------------------------------------------------------------- 6. value concentration
# A member who has renewed n times is lower-risk: churn probability ~ Beta(a, b + n)
active <- members %>% filter(!churned)
active$value_3y <- mapply(function(ch, n, m) {
  ab <- fits[[ch]]; k <- 1:3
  sum(m * exp(lbeta(ab[["a"]], ab[["b"]] + n + k) - lbeta(ab[["a"]], ab[["b"]] + n)) / (1 + DISCOUNT)^k)
}, active$channel, active$renewals, active$margin_per_active_year)
v <- sort(active$value_3y, decreasing = TRUE)
top10 <- sum(head(v, floor(length(v) / 10))) / sum(v)
by_tenure <- active %>% group_by(renewals) %>% summarise(v = mean(value_3y), .groups = "drop")

# ---------------------------------------------------------------- figures
BLUE <- "#2a78d6"; ORANGE <- "#eb6834"; INK <- "#0b0b0b"; INK2 <- "#52514e"
GRID <- "#e1e0d9"; SURFACE <- "#fcfcfb"
theme_ltv <- theme_minimal(base_size = 11) + theme(
  plot.background = element_rect(fill = SURFACE, colour = NA),
  panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
  panel.grid.major.y = element_line(colour = GRID, linewidth = 0.4),
  axis.text = element_text(colour = INK2), axis.title = element_text(colour = INK2),
  plot.title = element_text(colour = INK, size = 13), plot.title.position = "plot",
  strip.text = element_text(colour = INK, hjust = 0, size = 11),
  legend.position = "top", legend.justification = "right", legend.title = element_blank())

# Fig 1: retention curves by channel
t <- 0:4
curves <- bind_rows(lapply(MATURE, function(ch) {
  ab <- fits[[ch]]; c1 <- constant_churn_fit(members %>% filter(channel == ch))
  bind_rows(
    data.frame(channel = ch, t = t, value = sbg_survival(ab[["a"]], ab[["b"]], t), series = "sBG model"),
    data.frame(channel = ch, t = t, value = (1 - c1)^t, series = "Constant churn (year-1 rate every year)"))
})) %>% mutate(channel = factor(channel, MATURE))
observed <- bind_rows(lapply(MATURE, function(ch) {
  o <- observed_retention(members %>% filter(channel == ch))
  data.frame(channel = ch, t = seq_along(o) - 1, value = o)
})) %>% mutate(channel = factor(channel, MATURE))

p1 <- ggplot(curves, aes(t, value)) +
  geom_line(aes(colour = series, linetype = series), linewidth = 0.9) +
  geom_point(data = observed, aes(shape = "Observed"), colour = INK, fill = INK, size = 2.2) +
  facet_wrap(~channel, nrow = 1) +
  scale_colour_manual(values = c("sBG model" = BLUE, "Constant churn (year-1 rate every year)" = ORANGE),
                      breaks = c("sBG model", "Constant churn (year-1 rate every year)")) +
  scale_linetype_manual(values = c("sBG model" = "solid", "Constant churn (year-1 rate every year)" = "dashed"),
                        breaks = c("sBG model", "Constant churn (year-1 rate every year)")) +
  scale_shape_manual(values = c(Observed = 21)) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1.02)) +
  labs(title = "Retention by acquisition channel", x = "Renewals", y = "% of acquired members still active") +
  theme_ltv
ggsave(file.path(FIG, "01_retention_by_channel.png"), p1, width = 13, height = 3.9, dpi = 160, bg = SURFACE)

# Fig 2: LTV vs CAC
s2 <- summary_tbl %>% arrange(ltv_cac) %>% mutate(channel = factor(channel, channel))
bars <- s2 %>% select(channel, ltv_5y, cac) %>%
  pivot_longer(-channel, names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(recode(metric, ltv_5y = "5-year LTV (discounted margin)", cac = "Acquisition cost (CAC)"),
                         c("Acquisition cost (CAC)", "5-year LTV (discounted margin)")),
         label = ifelse(grepl("LTV", metric),
                        paste0(eur(value), "  ·  LTV:CAC ", sprintf("%.1f", value / s2$cac[match(channel, s2$channel)])),
                        eur(value)))
p2 <- ggplot(bars, aes(value, channel, fill = metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.72) +
  geom_text(aes(x = value + 2, label = label, colour = metric), position = position_dodge(width = 0.8),
            hjust = 0, size = 3.2, show.legend = FALSE) +
  scale_fill_manual(values = c("5-year LTV (discounted margin)" = BLUE, "Acquisition cost (CAC)" = ORANGE),
                    breaks = c("5-year LTV (discounted margin)", "Acquisition cost (CAC)")) +
  scale_colour_manual(values = c("5-year LTV (discounted margin)" = INK, "Acquisition cost (CAC)" = INK2)) +
  scale_x_continuous(labels = label_dollar(prefix = "€"), limits = c(0, max(s2$ltv_5y) * 1.45),
                     expand = expansion(mult = c(0, 0))) +
  labs(title = "Value per acquired member vs. cost to acquire", x = NULL, y = NULL) +
  theme_ltv + theme(panel.grid.major.x = element_line(colour = GRID, linewidth = 0.4),
                    panel.grid.major.y = element_blank())
ggsave(file.path(FIG, "02_ltv_vs_cac.png"), p2, width = 8.8, height = 4.4, dpi = 160, bg = SURFACE)

# Fig 3: back-test (renewal 4)
bt4 <- backtest %>% filter(renewal == 4) %>% mutate(channel = factor(channel, rev(MATURE)))
pts <- bt4 %>% select(channel, Actual = actual, `sBG forecast` = sbg,
                      `Constant-churn forecast (year-1 rate)` = constant) %>%
  pivot_longer(-channel, names_to = "series", values_to = "value") %>%
  mutate(series = factor(series, c("Actual", "sBG forecast", "Constant-churn forecast (year-1 rate)")))
p3 <- ggplot(pts, aes(value, channel)) +
  geom_segment(data = bt4, aes(x = constant, xend = actual, y = channel, yend = channel),
               colour = GRID, linewidth = 1.2) +
  geom_point(aes(colour = series, shape = series, size = series), stroke = 1.3) +
  scale_colour_manual(values = c(INK, BLUE, ORANGE)) +
  scale_shape_manual(values = c(1, 18, 15)) +
  scale_size_manual(values = c(5.5, 3.6, 3)) +
  scale_x_continuous(labels = label_percent(), limits = c(0, 0.45)) +
  labs(title = "Back-test: forecasts made with data to Mar 2024 vs. what happened",
       x = "FY22 cohort still active after 4 renewals", y = NULL) +
  theme_ltv + theme(legend.position = "bottom", legend.justification = "left",
                    panel.grid.major.x = element_line(colour = GRID, linewidth = 0.4),
                    panel.grid.major.y = element_blank())
ggsave(file.path(FIG, "03_backtest.png"), p3, width = 8.8, height = 4.2, dpi = 160, bg = SURFACE)

# ---------------------------------------------------------------- printed summary
cat("\nLTV by channel (5-year horizon, 10% discount rate)\n")
print(as.data.frame(summary_tbl %>% arrange(desc(ltv_cac)) %>%
  transmute(channel, members, first_renewal = pct(first_renewal_rate),
            active_after_4 = ifelse(is.na(retention_y4), "n/a", pct(retention_y4)),
            margin_year = eur(avg_margin_per_year),
            ltv_5y = paste0(eur(ltv_5y), ifelse(channel == "Conversational AI", "*", "")),
            cac = eur(cac), ltv_cac = sprintf("%.1f", ltv_cac),
            payback = sprintf("%.0f mo", payback_months))), row.names = FALSE)
cat("\nBack-test MAE: sBG", sprintf("%.1f pp", mean(abs(backtest$sbg_error_pp))),
    "| constant churn", sprintf("%.1f pp", mean(abs(backtest$constant_error_pp))), "\n")
cat("Conversational AI: year-1 renewal", pct(ai_r1), "(90% CI", pct(ai_r1_ci[1]), "-", pct(ai_r1_ci[2]),
    "); 5-yr LTV", eur(min(scenarios)), "to", eur(max(scenarios)), "\n")
cat("Top 10% of active members hold", pct(top10), "of expected 3-year margin\n")
