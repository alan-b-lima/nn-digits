package nn

import (
	"math"
	"math/rand/v2"

	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

// NeuralNetwork is a composition of layers that holds weights and biases for
// a Multilayer Perceptron.
//
// NeuralNetwork is not concurrent safe, due to pre-allocation of buffers for
// computations.
type NeuralNetwork struct {
	layers []Layer
}

type Layer struct {
	Weights nnmath.Matrix
	Biases  nnmath.Vector

	activation      nnmath.Vector
	weight_gradient nnmath.Matrix
	bias_gradient   nnmath.Vector
}

func New(dims ...int) NeuralNetwork {
	if len(dims) < 2 {
		panic("there must be at least two layers")
	}

	layers := make([]Layer, len(dims)-1)

	var size int
	for i := range len(dims) - 1 {
		size += dims[i+1]*dims[i] + dims[i+1]
		size += dims[i+1] + dims[i+1]*dims[i] + dims[i+1]
	}

	buf := make([]float64, size)
	for i := range len(buf) {
		buf[i] = rand.NormFloat64()
	}

	for i := range len(dims) - 1 {
		layers[i] = Layer{
			Weights:         nnmath.MakeMatData(dims[i+1], dims[i], take(&buf, dims[i+1]*dims[i])),
			Biases:          nnmath.MakeVecData(dims[i+1], take(&buf, dims[i+1])),
			activation:      nnmath.MakeVecData(dims[i+1], take(&buf, dims[i+1])),
			weight_gradient: nnmath.MakeMatData(dims[i+1], dims[i], take(&buf, dims[i+1]*dims[i])),
			bias_gradient:   nnmath.MakeVecData(dims[i+1], take(&buf, dims[i+1])),
		}
	}

	return NeuralNetwork{layers: layers}
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

		nnmath.MulP(layer.activation, layer.Weights, input)
		nnmath.AddP(layer.activation, layer.activation, layer.Biases)
		nnmath.ApplyP(layer.activation, layer.activation, Sigmoid)

		input = nn.layers[i].activation
	}

	last := nn.layers[len(nn.layers)-1]

	nnmath.MulP(last.activation, last.Weights, input)
	nnmath.AddP(last.activation, last.activation, last.Biases)
	input = last.activation

	softmax := nnmath.Softmax(input.Data(), math.Exp)
	last.activation = nnmath.MakeVecData(len(softmax), softmax)

	return last.activation
}

func take[E any](s *[]E, len int) []E {
	res := (*s)[:len]
	*s = (*s)[len:]
	return res
}
