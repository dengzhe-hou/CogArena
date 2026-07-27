# Exploratory Bayesian Multilevel Bifactor Analysis

Status: exploratory and outside the frozen manuscript chain.

## Question

Does adding five theory-aligned grouping factors improve on a single broad
competence factor once model-family clustering is represented?

## Design

- Input: the frozen strict 55 by 13 accuracy matrix.
- Families: 24 merged model families from the frozen family map.
- General model: a checkpoint-level general factor plus a family-level
  general factor.
- Bifactor model: the general model plus five orthogonal grouping factors at
  checkpoint and family levels.
- Inference: Pyro AutoNormal mean-field variational inference with three
  random restarts.
- Prior sensitivity: half-normal grouping-loading scales of 0.15, 0.30, and
  0.60.
- Model comparison: six-fold family-held-out joint Gaussian scoring. The
  predictive covariance integrates new-family and new-checkpoint latent
  factors, while fitted global parameters use posterior medians.

## Main result

The bifactor model does not improve prediction for held-out model families.

| Quantity | Result |
|---|---:|
| Family-equal delta log score per cell, bifactor minus general | -0.0144 |
| Family-bootstrap 95% CI | [-0.0420, 0.0095] |
| Cell-weighted delta log score per cell | -0.0064 |
| Families favoring bifactor | 12 of 24 |
| Families favoring general factor | 12 of 24 |

The general model's in-sample correlation RMSE is 0.1007. The bifactor model
reduces this to 0.0949, 0.0903, and 0.0888 as the grouping-loading prior is
widened. This small training-sample improvement does not transport to held-out
families.

## Prior sensitivity

The posterior median share of variance attributed to grouping-specific
factors is strongly prior-sensitive.

| Grouping-loading prior scale | Mean specific variance share |
|---:|---:|
| 0.15 | 3.7% |
| 0.30 | 7.5% |
| 0.60 | 11.1% |

Under the primary 0.30 prior, the largest specific shares occur in n-back
(27.6%) and operation span (18.6%). The average working-memory-specific share
is 16.2%. Metacognition is near zero at 0.4%. This pattern suggests localized
residual covariance rather than five stable and comparably identified
dimensions.

## Interpretation

The exploratory bifactor result is consistent with the frozen paper's
boundary conclusion. Additional grouping factors can absorb a small amount of
sample covariance, especially for working memory, but their magnitude depends
on the prior and they do not improve held-out-family prediction.

This result should not be added to the current AAAI draft. It is post-hoc,
mean-field uncertainty can be too narrow, the Gaussian likelihood is an
approximation for bounded accuracies, and theory of mind and metacognition
have only two indicators each. Retain it as a rebuttal diagnostic or as the
starting point for a larger-family extension.
