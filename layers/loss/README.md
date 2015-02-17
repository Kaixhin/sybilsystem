Losses
======

Loss functions, of the form l(h, y), measure the difference between the output of the network (i.e. the hypothesis (h) or prediction) and the target data (y).
3 loss functions and their derivatives are provided. The proper statistical treatment of loss functions are not discussed.

| Loss                     | Function                     | Usage          |
| ------------------------ | ---------------------------- | -------------- |
| Squared / Euclidean      | (1/2)(h - y)^2               | Regression     |
| Logistic / Cross-entropy | y.log(h) + (1 - y)log(1 - h) | Classification |
| Hinge / SVM              | max(0, 1 - t.y)              | Classification |