---
title: Ensembles
summary: Combine many models (usually trees) into one that is more accurate than any of them.
order: 6
algorithms:
  - { id: bagging, name: Bagging }
  - { id: random_forest, name: Random Forest }
  - { id: extra_trees, name: Extra Trees }
  - { id: adaboost, name: AdaBoost }
  - { id: gradient_boosting, name: Gradient Boosting }
  - { id: hist_gradient_boosting, name: HistGradientBoosting }
  - { id: external_boosting, name: XGBoost / LightGBM / CatBoost }
  - { id: voting_stacking, name: Voting & Stacking }
---

Ensembles average away the mistakes of individual models: bagging reduces variance by
training on resampled data, and boosting reduces bias by fitting each model to the errors of
the previous ones.
