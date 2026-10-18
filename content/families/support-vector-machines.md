---
title: Support Vector Machines
summary: Find the boundary with the widest margin between classes; kernels make it non-linear.
order: 4
algorithms:
  - { id: linear_svm, name: Linear SVM }
  - { id: kernel_svm, name: Kernel SVM (SVC) }
  - { id: svr, name: SVR }
---

SVMs pick the decision boundary that stays as far as possible from the closest training
points (the support vectors). The kernel trick lets the same idea draw curved boundaries.
