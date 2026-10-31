---
title: Instance-based
summary: Predict from the most similar training examples instead of learning parameters.
order: 2
algorithms:
  - { id: knn_regressor, name: KNN Regressor }
  - { id: knn_classifier, name: KNN Classifier }
---

Instance-based ("lazy") learners keep the training data and answer each query by looking at
its nearest neighbours. They introduce distance metrics and the curse of dimensionality.
