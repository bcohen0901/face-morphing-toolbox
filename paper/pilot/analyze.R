# Analysis of the familiarity pilot reported in "A Pilot Study of Perceived Familiarity."
#
# pilot_trials.csv holds one row per test trial for the 50 participants who finished.
#   participant  random label
#   list         counterbalancing list (1-4)
#   identity     CFD identity of the source face
#   source_cat, shown_cat  source and apparent category (A, B, L, W)
#   role         old = studied face; dopp = cross-category doppelganger of a studied
#                face; new_sc, new_xc = new face, same- or cross-category
#   rating       familiarity, 1 (not at all familiar) to 7 (extremely familiar)
#   rt           response time in ms
#   flagged      1 = missed an attention check or performed poorly on the filler task
#
# The data are de-identified: no platform IDs, demographics, or free-text answers.
# The N = 51 check in the paper includes a participant who is not in this file.
#
#   Rscript analyze.R

suppressPackageStartupMessages({
  library(dplyr)
  library(lmerTest)
  library(emmeans)
})

here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
roles <- c("old", "dopp", "new_sc", "new_xc")
trials <- read.csv(file.path(here, "pilot_trials.csv")) |>
  mutate(role = factor(role, levels = roles))

means <- trials |>
  group_by(participant, role) |>
  summarise(rating = mean(rating), .groups = "drop") |>
  group_by(role) |>
  summarise(M = round(mean(rating), 2), SD = round(sd(rating), 2))

contrasts_of <- function(df) {
  fit <- lmer(rating ~ role + (1 | participant) + (1 | identity), data = df)
  emm <- emmeans(fit, ~ role, lmer.df = "satterthwaite", lmerTest.limit = 5000)
  con <- contrast(emm, list("old - new_sc" = c(1, 0, -1, 0), "new_sc - new_xc" = c(0, 0, 1, -1),
                            "dopp - new_xc" = c(0, 1, 0, -1), "old - dopp" = c(1, -1, 0, 0)))
  ci <- confint(con)
  sm <- summary(con)
  sprintf("  %-15s b = %.2f, SE = %.2f, 95%% CI [%.2f, %.2f], t(%.0f) = %.2f, p = %.4f",
          sm$contrast, sm$estimate, sm$SE, ci$lower.CL, ci$upper.CL, sm$df, sm$t.ratio, sm$p.value)
}

unflagged <- filter(trials, flagged == 0)
report <- c(
  sprintf("Participants: %d; trials: %d", n_distinct(trials$participant), nrow(trials)), "",
  capture.output(print(as.data.frame(means), row.names = FALSE)), "",
  "Mixed model: rating ~ condition + (1 | participant) + (1 | identity)",
  contrasts_of(trials), "",
  sprintf("Sensitivity, unflagged participants (N = %d):", n_distinct(unflagged$participant)),
  contrasts_of(unflagged)[3]
)
writeLines(report)
writeLines(report, file.path(here, "results_summary.txt"))
