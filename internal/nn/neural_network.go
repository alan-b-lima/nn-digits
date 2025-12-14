package nn

import (
	"math"

	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

type NeuralNetwork struct {
	layers []Layer
}

type Layer struct {
	Weights    nnmath.Matrix
	Biases     nnmath.Vector
	Activation nnmath.Vector
}

func New(dims ...int) NeuralNetwork {
	nn := make([]Layer, len(dims)-1)

	for i := range len(dims) - 1 {
		nn[i] = Layer{
			Weights: nnmath.MakeMatRandom(dims[i+1], dims[i]),
			Biases:  nnmath.MakeVecRandom(dims[i+1]),
		}
	}

	return NeuralNetwork{layers: nn}
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	sig := Sigmoid(x)
	return sig * (1 - sig)
}

func SigmoidDerivativeFromActivation(x float64) float64 {
	return x * (1 - x)
}

func (nn *NeuralNetwork) FeedForward(input nnmath.Vector) nnmath.Vector {
	if len(nn.layers) == 0 {
		return nnmath.Vector{}
	}

	for i := range nn.layers[:len(nn.layers)-1] {
		layer := &nn.layers[i]
		input = nnmath.Apply(nnmath.Add(nnmath.Mul(layer.Weights, input), layer.Biases), Sigmoid)
		// layer.Activation = input
	}

	last := nn.layers[len(nn.layers)-1]

	softmax := nnmath.Softmax(nnmath.Add(nnmath.Mul(last.Weights, input), last.Biases).Data, math.Exp)
	last.Activation = nnmath.MakeVecData(len(softmax), softmax)

	return last.Activation
}
