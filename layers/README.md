Layers
======

Layers apply element-wise (often non-linear) functions f(z) to the data z where z is a K dimensional vector.
Several layers and their derivatives are provided.

| Layer                       | Function                    |
| --------------------------- | --------------------------- |
| Linear                      | z                           |
| Sigmoid                     | 1 / (1 + e^-z)              |
| Tanh                        | (e^z - e^-z) / (e^z + e^-z) |
| Radial Basis                |                             | TODO
| Rectified Linear            | max(0, z)                   | TODO
| Parametric Rectified Linear | max(0, z) + a\*min(0, z)    | TODO
| Softplus                    |                             | TODO
| Maxout                      |                             | TODO
| Softmax                     | e^z_j / sum(e^z_k)          |

NB: tanh is inbuilt, but its derivative is not.
