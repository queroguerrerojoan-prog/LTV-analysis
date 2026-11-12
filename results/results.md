### Data

64,525 synthetic members across 5 channels, cohorts April 2021 to March 2025, observed to 31 Mar 2026.

### LTV by channel (5-year horizon, 10% annual discount rate)

| Channel | Members | 1st renewal | Active after 4 renewals | Margin / year | 5-yr LTV | CAC | LTV:CAC | Payback |
|---|---|---|---|---|---|---|---|---|
| Organic / SEO | 15,356 | 65% | 30% | €62 | €152 | €22 | 6.9 | 4 mo |
| Conversational AI | 554 | 66% | n/a | €63 | €150* | €30 | 5.0 | 6 mo |
| App | 10,450 | 72% | 40% | €74 | €205 | €41 | 5.0 | 7 mo |
| Paid search | 20,082 | 52% | 17% | €58 | €118 | €46 | 2.6 | 10 mo |
| Metasearch | 18,083 | 40% | 9% | €49 | €83 | €36 | 2.3 | 9 mo |

\* Midpoint of a scenario range; see "Uncertainty for a young channel".

### Back-test (model fitted on data available at 31 Mar 2024; FY22 cohort)

| Channel | Renewal | Actual | sBG | Constant churn | sBG error | Constant error |
|---|---|---|---|---|---|---|
| Organic / SEO | 3 | 35.2% | 35.9% | 26.8% | +0.7 pp | -8.4 pp |
| Organic / SEO | 4 | 28.6% | 29.0% | 17.3% | +0.3 pp | -11.4 pp |
| Paid search | 3 | 23.1% | 24.2% | 13.8% | +1.1 pp | -9.3 pp |
| Paid search | 4 | 17.4% | 18.7% | 7.1% | +1.4 pp | -10.2 pp |
| Metasearch | 3 | 13.4% | 13.4% | 6.5% | -0.1 pp | -6.9 pp |
| Metasearch | 4 | 8.8% | 9.1% | 2.6% | +0.3 pp | -6.1 pp |
| App | 3 | 46.7% | 47.3% | 37.9% | +0.6 pp | -8.8 pp |
| App | 4 | 40.3% | 40.5% | 27.4% | +0.3 pp | -12.9 pp |

Mean absolute error: **sBG 0.6 pp** vs. **constant churn 9.3 pp**. Applying first-year churn to every year understates 5-year LTV by **8%** on average across mature channels (the gap grows with longer horizons).

### Uncertainty for a young channel

Conversational AI members (554) have reached one renewal at most. Their first-renewal rate is **66%** (bootstrapped 90% interval 62% to 69%), but the data says nothing yet about later years, so the sBG fit for this channel is not reliable on its own. Borrowing the later-year retention shape of each mature channel gives a 5-year LTV between **€136 and €164** (Metasearch €136, Paid search €146, Organic / SEO €157, App €164). Even the low end gives LTV:CAC of **4.5**.

### Value concentration among active members

The top 10% of currently active members hold **27%** of expected margin over the next three years. Expected 3-year value rises with tenure: €83 after 1 renewal, €96 after 2 renewals, €105 after 3 renewals, €114 after 4 renewals.
