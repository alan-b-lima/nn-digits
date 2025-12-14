# Recognizing Handwritten Digits

This project is a simple implementation of a Neural Network to recognize handwritten digits using the MNIST dataset[^siramdasu]. The neural network is built from scratch in Go.

# The Project

The goal of this project is to learn about one of the most basic forms of neural network, as well as to train a neural network to recognize handwritten digits. To achive this last goal, shall we get into some math.

## The Neural Network

The model of neural network we'll be my interpretation of the _Multilayer Perceptron_[^sanderson]. The Multilayer Perceptron is defined as a tuple of the so called _layers_. We shall define those first.

A layer, informally, is a group of neurons that takes an vector input and produces an vector output. The way they operate is through a variation of the Pitts-McCulloch neuron[^mcculloch;pitts], which follow an "all-or-none" activation principle, our neurons have _activation values_ on the domain of the real numbers.

### Definition 1. Neuron

A neuron is a thing that possesses a _activation value_ $`a \in \mathbb{R}`$, connections called _weights_ $`\mathbf{w} = (w_i)`$, $`\mathbf{w} \in \mathbb{R}^n`$, a _bias_ $`b \in \mathbb{R}`$ and an _activation function_ $`\sigma : \mathbb{R} \to \mathbb{R}`$. The activation of a neuron is determined by its input $`\mathbf{u} = (u_i)`$, $`\mathbf{u} \in \mathbb{R}^n`$:

```math
a(\mathbf{u}) = \sigma \left( b + \sum_{i = 1}^{n} {w_i u_i} \right)
```

Or simply:

```math
a(\mathbf{u}) = \sigma \left( \mathbf{w} \cdot \mathbf{u} + b \right)
```

### Definition 2. Layer

A layer is a collection of neurons, since the activation of a single neuron can be thought of as a dot product added to a bias and passed through a activation function, the natural way to extend this idea is via matrices. A layer is a triple $`L = (\mathbf{W}, \mathbf{b}, \sigma)`$, where, for a layer with $`m_L`$ _input activation values_ and $`n_L`$ neurons:

- $`\mathbf{W}^L = [w^L_{i,j}]`$, $`\mathbf{W}^L \in \mathcal{M}_{n_L \times m_L} (\mathbb{R})`$, is a matrix of weights, whereas $`w^L_{p,q}`$ is the connection between the $`q`$-th input activation value and the $`p`$-th neuron;

- $`\mathbf{b}^L = [b^L_i]`$, $`\mathbf{b}^L \in \mathcal{M}_{n_L \times 1} (\mathbb{R})`$, is column vector of biases, whereas $`b^L_p`$ is the bias for the $`p`$-th neuron;

- $`\sigma_L : \mathbb{R} \to \mathbb{R}`$ is an activation function, we allow $`\sigma_L([v_{i,j}]) = [\sigma_L(v_{i,j})]`$.

Given an input activation vector $`\mathbf{u} = [u_i]`$, the activation of each neuron of the layer, $`\mathbf{a}^L = [a^L_i]`$ is defined:

```math
a^L_i(\mathbf{u}) = \sigma_L \left( b^L_{i} + \sum_{j = 1}^{m_L} {w^L_{i,j} u_j} \right)
```

Or:

```math
\mathbf{a}^L(\mathbf{u}) = \sigma_L(\mathbf{W}^L\mathbf{u} + \mathbf{b}^L)
```

### Definition 3. Neural Network

A neural network is a collection of finite ordered layers $`\mathcal{N} = (L_i)`$. In notation, to reference, for example, the weight matrix of $`L_i`$, instead of $`\mathbf{W}^{L_i}`$, we can do simply $`\mathbf{W}^{i}`$. Note that $`m_L = n_{L - 1}`$. The _output activation vector_ of the network is $\mathbf{y} = \mathbf{a}^{|\mathcal{N}|}$, given an input vector $`\mathbf{u} = [u_i]`$, is achieved through:

```math
\begin{align*}
    a^{0}_i(\mathbf{u}) &= u_i \\
    a^L_i(\mathbf{u}) &= \sigma_L \left( b^L_{i} + \sum_{j = 1}^{m_L} {w^L_{i,j} a^{L - 1}_j(\mathbf{u})} \right) \\
\end{align*}
```

Or:

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

### Definition 4. Cost Function

Given a labeled dataset $`X \subset \left\{(\mathbf{u}, \mathbf{y}) | \mathbf{u} \in \mathcal{M}_{m_1 \times 1} (\mathbb{R}), \mathbf{y} \in \mathcal{M}_{n_{|\mathcal{N}|} \times 1} (\mathbb{R}) \right\}`$, where $`\mathbf{u}`$ is the _input_ and $`\mathbf{y} = [y_i]`$ is the _label_, the cost function will put the input through the network and analyze it against the label. It is done the following way:

```math
C(X, \mathcal{N}) = \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{i = 1}^{n_{|\mathcal{N}|}} \left( a^{|\mathcal{N}|}_i(\mathbf{u}) - y_i \right)^2
```

Or:

```math
C(X, \mathcal{N}) = \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} { {\left\Vert \mathbf{a}^{|\mathcal{N}|}(\mathbf{u}) - \mathbf{y} \right\Vert}^2 }
```

### Example 1

Before calculating the gradient of $`C`$ in relation $`\mathcal{N}`$, we will differentiate the parameters of the last layer. Consider $`L = |\mathcal{N}|`$, we'll differentiate $`a^L_i`$, $`z^L_i`$, $`b^L_i`$ and $`w^L_{j,i}`$.

```math
\begin{align*}
    \frac{\partial C}{\partial a^L_i}(X, \mathcal{N})
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} {2 (a^L_i(\mathbf{u}) - y_i)} \\
    
    \frac{\partial C}{\partial z^L_i} (X, \mathcal{N})
        &= \frac{\partial C}{\partial a^L_i} \frac{\partial a^L_i}{\partial z^L_i} (X, \mathcal{N})
        = \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} {2 (a^L_i(\mathbf{u}) - y_i) \cdot \sigma_L'(z^L_i(\mathbf{u}))} \\
    
    \frac{\partial C}{\partial b^L_i} (X, \mathcal{N})
        &= \frac{\partial C}{\partial a^L_i} \frac{\partial a^L_i}{\partial z^L_i} \frac{\partial z^L_i}{\partial b^L_i} (X, \mathcal{N})
        = \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} {2 (a^L_i(\mathbf{u}) - y_i) \cdot \sigma_L'(z^L_i(\mathbf{u})) \cdot 1} \\
    
    \frac{\partial C}{\partial w^L_{j,i}} (X, \mathcal{N})
        &= \frac{\partial C}{\partial a^L_i} \frac{\partial a^L_i}{\partial z^L_i} \frac{\partial z^L_i}{\partial w^L_{j,i}} (X, \mathcal{N})
        = \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} {2 (a^L_i(\mathbf{u}) - y_i) \cdot \sigma_L'(z^L_i(\mathbf{u})) \cdot a^{L - 1}_j(\mathbf{u})} \\
\end{align*}
```

Although cluttered, it's quite simple, and a lot of reuse of work is possible. For the last layer, everything goes on without problem. For the other layeys, things get a little more complex, take the derivative of $`C`$ in relation to $`a^{L - 1}_i`$:

```math
\begin{align*}
    \frac{\partial C}{\partial a^{L - 1}_i} (X, \mathcal{N})
        &= \frac{\partial}{\partial a^{L - 1}_i} \left( \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} (a^L_j(\mathbf{u}) - y_j)^2 \right) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} \frac{\partial}{\partial a^{L - 1}_i} (a^L_j(\mathbf{u}) - y_j)^2 \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \frac{\partial}{\partial a^{L - 1}_i} a^L_j(\mathbf{u}) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \frac{\partial}{\partial a^{L - 1}_i} \sigma_L(z^L_j(\mathbf{u})) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \sigma_L'(z^L_j(\mathbf{u})) \cdot \frac{\partial}{\partial a^{L - 1}_i} z^L_j(\mathbf{u}) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \sigma_L'(z^L_j(\mathbf{u})) \cdot \frac{\partial}{\partial a^{L - 1}_i} \left( b^L_j + \sum_{k = 1}^{m_L} w^L_{j,k}a^{L - 1}_k(\mathbf{u}) \right) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \sigma_L'(z^L_j(\mathbf{u})) \cdot \frac{\partial}{\partial a^{L - 1}_i} \left(w^L_{j,i}a^{L - 1}_i(\mathbf{u}) \right) \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} 2(a^L_j(\mathbf{u}) - y_j) \cdot \sigma_L'(z^L_j(\mathbf{u})) \cdot w^L_{j,i} \\
        &= \frac{1}{|X|} \sum_{(\mathbf{u}, \mathbf{y}) \in X} \sum_{j = 1}^{n_L} \frac{\partial C}{\partial a^L_j} \frac{\partial a^L_j}{\partial z^L_j} \frac{\partial z^L_j}{\partial a^{L - 1}_i} \\
\end{align*}
```

And to derivate, say $`\partial C / \partial a^{L - 2}_i`$, we'd go this way:

```math
\begin{align*}
    \frac{\partial C}{\partial a^{L - 2}}
\end{align*}
```

## Backpropagation

We had some fun playing with the chain rule to differentiate some individual values in the network. Now we'll give a precise (and efficient) way to compute $`\nabla_\mathcal{N} C(X, \mathcal{N})`$.

Notation-wise, let $`\mathbf{v} = [v_i]`$, $`\nabla_\mathbf{v} f(\dots, \mathbf{v}, \dots) = [\frac{\partial f}{\partial v_i} (\dots, \mathbf{v}, \dots)]`$. Now we're ready.

There are a few schema that will show up. 

```math
\nabla_{\mathbf{a}^l} C(X, \mathcal{N}) = \frac{1}{|X|} \sum_{(\mathbf{u},\mathbf{v}) \in X} 2 \left\Vert \mathbf{a}^l(\mathbf{u}) - \mathbf{y} \right\Vert
```

```math
\nabla_{\mathbf{z}^l} \mathbf{a}^l (\mathbf{u}) = \sigma_L'(\mathbf{z}^l(\mathbf{u}))
```

```math
\nabla_{\mathbf{b}^l} \mathbf{z}^l (\mathbf{u}) = [1]^{m_l} 
```

```math
\nabla_{\mathbf{W}^l} \mathbf{z}^l (\mathbf{u}) = [\mathbf{a}^{l - 1}]^{m_l}
```

```math
\nabla_{\mathbf{a}^{l - 1}} \mathbf{a}^l (\mathbf{u}) = [\mathbf{a}^{l - 1}]^{m_l}
```

```math
\def\pd#1#2{\frac{\partial#1}{\partial#2}}

\begin{align*}
    \pd{a^l_i}{a^{l - 1}_j}
        &= \pd{}{a^{l - 1}_j} \sigma_l(z^l_i) \\
        &= \sigma_l'(z^l_i) \pd{}{a^{l - 1}_j} z^l_i \\
        &= \sigma_l'(z^l_i) \pd{}{a^{l - 1}_j} \left( b^l_i + \sum_{k = 1}^{m_l} w^l_{i,k}a^{l - 1}_k \right) \\
        &= \sigma_l'(z^l_i) \sum_{k = 1}^{m_l} \pd{}{a^{l - 1}_j} w^l_{i,k}a^{l - 1}_k \\
\end{align*}
```

<!-- ```math

\pd{C}{a^{L-1}_j} = 
``` -->

<!-- - https://www.cis.jhu.edu/~sachin/digit/digit.html -->

[^siramdasu]: SIRAMDASU, Vardhan. KAGGLE-DIGIT-RECOGNIZER. **Github Repository Kaggle-Digit-Recognizer**. Avaliable in: https://github.com/vardhan-siramdasu/Kaggle-Digit-Recognizer/blob/main/data/. Access in 11 of May, 2025.

[^sanderson]: SANDERSON, Grant. NEURAL NETWORKS. **Youtube Channel 3Blue1Brown**. Available in: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi. Access in 11 of May, 2025.

[^mcculloch;pitts]: McCULLOCH, Warren S.; PITTS, Walter. A LOGICAL CALCULUS OF THE IDEAS IMMANENT IN NERVOUS ACTIVITY\*. **Bulletin of Mathematical Biology**. Vol. 52, No. 1/2, pp. 99-115, 1990. Available in: https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf. Access in 11 of May, 2025.
