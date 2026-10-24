---
title: Linear Models
summary: Predict with a weighted sum of the features. The simplest models, and the foundation for the rest.
order: 1
algorithms:
  - { id: linear_regression, name: Linear Regression }
  - { id: polynomial_regression, name: Polynomial Regression }
  - { id: ridge, name: Ridge }
  - { id: lasso, name: Lasso }
  - { id: elastic_net, name: Elastic Net }
  - { id: logistic_regression, name: Logistic Regression }
  - { id: sgd_regressor, name: SGD Regressor }
  - { id: sgd_classifier, name: SGD Classifier }
---

Linear models assume the target is (roughly) a weighted sum of the features. They train in
seconds, their coefficients are readable, and the ideas they introduce (loss functions,
gradient descent, regularisation) reappear in almost every other model.
