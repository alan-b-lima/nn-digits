# Recognizing Handwritten Digits

This project is a simple implementation of a Neural Network to recognize handwritten digits using the MNIST dataset[^lecun][^siramdasu]. The neural network is built from scratch in Go.

[![image](./ui/web/dist/assets/promote.png)](https://alan-b-lima.github.io/nn-digits/ui/web/dist)

The canvas for test your own digits is on [GitHub Pages](https://alan-b-lima.github.io/nn-digits/ui/web/dist).

# The Project

The goal of this project is to learn about one of the most basic forms of neural network, as well as to train a neural network to recognize handwritten digits. To achive the latter, shall we get into some math.

## The Neural Network

The model of neural network we'll be taking a look at is my interpretation of the _Multilayer Perceptron_[^sanderson]. The Multilayer Perceptron is defined as a tuple of the so called _layers_. We shall define those first.

A layer, informally, is a group of neurons that takes an vector input and produces an vector output. The way they operate is through a variation of the Pitts-McCulloch neuron[^mcculloch;pitts]. The Pitts-McCulloch neuron follow an "all-or-none" activation principle, our neurons have _activation values_ ranging over the real numbers.

### Neuron

A neuron is the smallest concernable unit of the neural network. A neuron possesses an _activation value_ $`a \in \mathbb{R}`$ _weights_ $`\mathbf{w} = (w_i)`$, $`\mathbf{w} \in \mathbb{R}^n`$, a _bias_ $`b \in \mathbb{R}`$ and an _activation function_ $`\sigma : \mathbb{R} \to \mathbb{R}`$. The activation value is computed, given input $`\mathbf{u} = (u_i)`$, $`\mathbf{u} \in \mathbb{R}^n`$:

```math
a(\mathbf{u}) = \sigma \left( b + \sum_{i = 1}^{n} {w_i u_i} \right)
```

or simply:

```math
a(\mathbf{u}) = \sigma \left( \mathbf{w} \cdot \mathbf{u} + b \right)
```

### Layer

A layer is a collection of neurons, since the activation of a single neuron can be thought of as a dot product added to a bias and passed through a activation function, the natural way to extend this idea is via matrices. A layer is a triple $`L = (\mathbf{W}, \mathbf{b}, \sigma)`$, where, for a layer with $`m`$ _input activation values_ and $`n`$ neurons:

- $`\mathbf{W} = [w_{i,j}]`$, $`\mathbf{W} \in \mathcal{M}_{n \times m} (\mathbb{R})`$, is a matrix of weights, whereas $`w_{p,q}`$ is the connection between the $`q`$-th input activation value and the $`p`$-th neuron;

- $`\mathbf{b} = [b_i]`$, $`\mathbf{b} \in \mathcal{M}_{n \times 1} (\mathbb{R})`$, is column vector of biases, whereas $`b_p`$ is the bias for the $`p`$-th neuron;

- $`\sigma : \mathbb{R} \to \mathbb{R}`$ is an activation function, we allow $`\sigma([v_{i,j}]) = [\sigma(v_{i,j})]`$.

Given an input activation vector $`\mathbf{u} = [u_i]`$, $`\mathbf{u} \in \mathcal{M}_{m \times 1} (\mathbb{R})`$, the activation of each neuron of the layer, $`\mathbf{a} = [a_i]`$, $`\mathbf{a} \in \mathcal{M}_{n \times 1} (\mathbb{R})`$, is defined:

```math
a_i(\mathbf{u}) = \sigma \left( b_{i} + \sum_{j = 1}^{m} {w_{i,j} u_j} \right)
```

or:

```math
\mathbf{a}(\mathbf{u}) = \sigma(\mathbf{W}\mathbf{u} + \mathbf{b})
```

### Neural Network

A neural network is a collection of finite ordered layers $`\mathcal{N} = (L_i)`$. Notation-wise, to reference, for example, the weight matrix of of the $`L`$-th layer, we write $`\mathbf{W}^{L}`$. Note that $`m_L = n_{L - 1}`$. The _output activation vector_ of the network is $\mathbf{y} = \mathbf{a}^{|\mathcal{N}|}$, $`\mathbf{y} \in \mathcal{M}_{n_{|\mathcal{N}|} \times 1} (\mathbb{R})`$, given an input vector $`\mathbf{u} = [u_i]`$, $`\mathbf{u} \in \mathcal{M}_{m_1 \times 1} (\mathbb{R})`$, is achieved through:

```math
\begin{align*}
    a^{0}_i(\mathbf{u}) &= u_i \\
    a^L_i(\mathbf{u}) &= \sigma_L \left( b^L_{i} + \sum_{j = 1}^{m_L} {w^L_{i,j} a^{L - 1}_j(\mathbf{u})} \right) \\
\end{align*}
```

or:

```math
\begin{align*}
    \mathbf{a}^{0}(\mathbf{u}) &= \mathbf{u} \\
    \mathbf{a}^L(\mathbf{u}) &= \sigma_L(\mathbf{W}^L\mathbf{a}^{L - 1}(\mathbf{u}) + \mathbf{b}^L) \\
\end{align*}
```

We will introduce a variable $`\mathbf{z}^L = [z^L_i]`$ that represents the middle step, before the application of $`\sigma_L`$, that is:

```math
\begin{align*}
    z^L_i(\mathbf{u}) &= b^L_{i} + \sum_{j = 1}^{m_L} {w^L_{i,j} a^{L - 1}_j(\mathbf{u})} \\
    \mathbf{z}^L(\mathbf{u}) &= \mathbf{W}^L\mathbf{a}^{L - 1}(\mathbf{u}) + \mathbf{b}^L \\
\end{align*}
```

Now we may proceed to training the network.

In order to train a network, we must have an explicit way to tell how good is the network doing. We will measure it through a cost function.

### Cost Function

For a neural network $`\mathcal{N}`$, a labeled dataset $`X \subset \left\{(\mathbf{u}, \mathbf{v}) \mid \mathbf{u} \in \mathcal{M}_{m_1 \times 1} (\mathbb{R}), \mathbf{v} \in \mathcal{M}_{n_{|\mathcal{N}|} \times 1} (\mathbb{R}) \right\}`$ is given, where $`\mathbf{u}`$ is the _input_ and $`\mathbf{v} = [v_i]`$ is the _label_, i.e, the expected output, the cost function will put the input through the network and analyze it against the label.

The cost is simply the average of the error of all sample. The single sample error is given by:

```math
E(\mathcal{N}, (\mathbf{u}, \mathbf{v})) = \sum_{i = 1}^{n_{|\mathcal{N}|}} \left( y_i(\mathbf{u}) - v_i \right)^2
```

or:

```math
E(\mathcal{N}, (\mathbf{u}, \mathbf{v})) = {\left\Vert \mathbf{y}(\mathbf{u}) - \mathbf{v} \right\Vert}^2
```

Finally, the cost is calculated by:

```math
C(\mathcal{N}, X) = \frac{1}{|X|} \sum_{\mathbf{x} \in X} E(\mathcal{N}, \mathbf{x})
```

### Gradient Descent

Given a cost function, we know what to minimize. The gradient is a multidimensional derivative, if we consider a dataset $`X`$ as fixed and the weights and biases as variables (each one of them as a dimension), we can compute a vector that points in the direction of steepest ascent, negating that vector, we'll get the direction of steepest decent.

Moving in the direction of steepest decent, with certain consideration, will likely cause the cost to lower, and the network to perform better. The learning rate, $`\eta \in \mathbb{R}^{+}`$, is a parameter that tries to prevent the changes to overshoot an minumal (either local or global). If $`\eta`$ is too large, the network might take too big of a step and miss a minimal point. This might be desirable to encounter better solutions (see Simulated Anneling), but, for now, will stick to simple descent.

At every episode, the dataset $`X`$ will be put though a network $`\mathcal{N}_t`$ and the gradient of the cost function, alongside a learning rate $`\eta \in \mathbb{R}^{+}`$, will compute $`\mathcal{N}_{t+1}`$[^wiki]. That is:

```math
\mathcal{N}_{t+1} = \mathcal{N}_t - \eta\nabla{C(\mathcal{N}_t, X)}
```

The starting network $`\mathcal{N}_0`$ will simply be initialized with random real numbers. Note that we abuse notation defining the episode serie, the gradient should also range over the dataset, but here we admit the dataset is not only contant, but its dimensions do not appear in the gradient.

#### Computing the gradient

Computing the gradient is relatively easy, consider a simple network with two a layers, each one with a single neuron, $`\mathcal{N} = (([w^1], [b^1], \sigma_1), ([w^2], [b^2], \sigma_2))`$, Note that the superscripts are still identifiers of layer, not exponents. For $`\mathbf{u} = [u]`$:

```math
\begin{align*}
    \mathbf{y}
    &= \mathbf{a}^2(\mathbf{u}) \\
    &= \sigma_2(\mathbf{z}^2(\mathbf{u})) \\
    &= \sigma_2(w^2\mathbf{a}^1(\mathbf{u}) + b^2) \\
    &= \sigma_2(w^2\sigma_1(\mathbf{z}^1(\mathbf{u})) + b^2) \\
    &= \sigma_2(w^2\sigma_1(w^1\mathbf{a}^0(\mathbf{u}) + b^1) + b^2) \\
    &= \sigma_2(w^2\sigma_1(w^1\mathbf{u} + b^1) + b^2) \\
    y &= \sigma_2(w^2\sigma_1(w^1u + b^1) + b^2) \\
\end{align*}
```

using the chain rule, we get:

```math
\frac{\partial C}{\partial u} =
\frac{\partial C}{\partial a^2}
\frac{\partial a^2}{\partial z^2}
\frac{\partial z^2}{\partial a^1}
\frac{\partial a^1}{\partial z^1}
\frac{\partial z^1}{\partial u}
```

The derivative of the cost in respect to the output, $`\frac{\partial C}{\partial a^2}`$, is $`\frac{1}{|X|} \sum_{\mathbf{x} \in X} \frac{\partial E}{\partial a^2}(\mathcal{N}, \mathbf{x})`$, which all evaluated to $`\frac{1}{|X|} \sum_{\mathbf{x} \in X} 2(y(u) - v)`$. $`\frac{\partial a^2}{\partial z^2}`$ is simply $`\sigma_2'(z^2(u))`$. And $`\frac{\partial z^2}{\partial a^1}`$ is $`w^2`$. The rest can be easily derived with the same logic.

Now, only the inner parts, weights and biases, have their derivatives unknown. Most of what we have done already covers what they are, the last step is given: $`\frac{\partial z^2}{\partial w^2} = a^1`$, and $`\frac{\partial z^2}{\partial b^2} = 1`$.

Now, to obtain, for example, $`\frac{\partial C}{\partial w^1}`$, we just attach the parts we know:

```math
\frac{\partial C}{\partial a^2}
\frac{\partial a^2}{\partial z^2}
\frac{\partial z^2}{\partial a^1}
\frac{\partial a^1}{\partial z^1}
\frac{\partial z^1}{\partial w^1}
\\ =
\left(\frac{1}{|X|} \sum_{\mathbf{x} \in X} \frac{\partial E}{\partial a^2}(\mathcal{N}, \mathbf{x})\right) \cdot
\sigma_2'(z^2(u)) \cdot
w^2 \cdot
\sigma_1'(z^1(u)) \cdot
u
```

Most computations can be reused throughout the computation of the gradient, coming from the output back to the start, as if we are... backpropagating.

### Backpropagation

Computing the gradient through a numerical approximation will cause every sample to be run through the network for every time we nud a weight or bias, this is really inefficient. As it was hint in the last subsection, we'll get around this through a method called backpropagation.

...

### References

[^lecun]: LeCUN, Yann; CORTES, Corinna; BURGES; Christopher J. C. THE MNIST DATABASE of handwritten digits. **Yann LeCun ExDB MNIST**. Available in: http://yann.lecun.com/exdb/mnist/ (through the [Wayback Machine](https://web.archive.org/web/20250114200757/http://yann.lecun.com/exdb/mnist/)). Access in May 11th, 2025.

[^siramdasu]: SIRAMDASU, Vardhan. KAGGLE-DIGIT-RECOGNIZER. **Github Repository Kaggle-Digit-Recognizer**. Avaliable in: https://github.com/vardhan-siramdasu/Kaggle-Digit-Recognizer/blob/main/data/. Access in May 11th, 2025.

[^sanderson]: SANDERSON, Grant. NEURAL NETWORKS. **Youtube Channel 3Blue1Brown**. Available in: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi. Access in May 11th, 2025.

[^mcculloch;pitts]: McCULLOCH, Warren S.; PITTS, Walter. A LOGICAL CALCULUS OF THE IDEAS IMMANENT IN NERVOUS ACTIVITY*. **Bulletin of Mathematical Biology**. Vol. 52, No. 1/2, pp. 99-115, 1990. Available in: https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf. Access in May 11th, 2025.

[^wiki] WIKIPEDIA. Gradient descent. **Wikipedia - The Free Encyclopedia**. Avaliable in: https://en.wikipedia.org/wiki/Gradient_descent. Access in December 19th, 2025.