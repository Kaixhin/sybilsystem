Layers
======

Layers apply element-wise (often non-linear) functions f(z) to the data z where z is a K dimensional vector.
3* layers and their derivatives are provided.

| Layer   | Function                    |
| ------- | --------------------------- |
| Sigmoid | 1 / (1 + e^-z)              |
| Tanh    | (e^z - e^-z) / (e^z + e^-z) |
| Softmax | e^z_j / sum(e^z_k)          |

\* tanh is inbuilt, but its derivative is not.