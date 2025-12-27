package nn

import (
	"math"
	"math/rand/v2"

	"github.com/alan-b-lima/nn-digits/pkg/mem"
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

// NeuralNetwork is a composition of layers that holds weights and biases for
// a Multilayer Perceptron.
//
// NeuralNetwork is not concurrent safe, due to pre-allocation of buffers for
// computations.
type NeuralNetwork struct {
	Layers []Layer
	buf    []float64
}

type Layer struct {
	Weights nnmath.Matrix
	Biases  nnmath.Vector

	Computation
	Learning
}

type Computation struct {
	Activation nnmath.Vector
}

type Learning struct {
	WeightGradient   nnmath.Matrix
	BiasGradient     nnmath.Vector
	ErrorPropagation nnmath.Vector
}

func New(dims ...int) NeuralNetwork {
	if len(dims) < 2 {
		panic("there must be at least two layers")
	}

	nn := NeuralNetwork{buf: make([]float64, size_nn(dims...))}
	for i := range len(nn.buf) {
		nn.buf[i] = rand.NormFloat64()
	}

	nn.Layers = slice_nn(nn.buf, dims...)
	return nn
}

func (nn *NeuralNetwork) FeedForward(input nnmath.Vector) nnmath.Vector {
	if len(nn.Layers) == 0 {
		return nnmath.Vector{}
	}

	for i := range nn.Layers[:len(nn.Layers)-1] {
		layer := &nn.Layers[i]

		nnmath.MulP(layer.Activation, layer.Weights, input)
		nnmath.AddP(layer.Activation, layer.Activation, layer.Biases)
		nnmath.ApplyP(layer.Activation, layer.Activation, Sigmoid)

		input = nn.Layers[i].Activation
	}

	last := nn.Layers[len(nn.Layers)-1]

	nnmath.MulP(last.Activation, last.Weights, input)
	nnmath.AddP(last.Activation, last.Activation, last.Biases)
	input = last.Activation

	nnmath.Softmax(input.Data(), math.Exp)
	return last.Activation
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

func size_nn(dims ...int) int {
	var size int
	for i := range len(dims) - 1 {
		size += dims[i+1]*dims[i] + dims[i+1]
		size += dims[i+1] + dims[i+1]*dims[i] + dims[i+1] + dims[i+1]
	}

	return size
}

func slice_nn(buf []float64, dims ...int) []Layer {
	layers := make([]Layer, 0, len(dims)-1)
	for i := range len(dims) - 1 {
		layer := Layer{
			Weights: nnmath.MakeMatData(dims[i+1], dims[i], mem.Take(&buf, dims[i+1]*dims[i])),
			Biases:  nnmath.MakeVecData(dims[i+1], mem.Take(&buf, dims[i+1])),
			Computation: Computation{
				Activation: nnmath.MakeVecData(dims[i+1], mem.Take(&buf, dims[i+1])),
			},
			Learning: Learning{
				WeightGradient:   nnmath.MakeMatData(dims[i+1], dims[i], mem.Take(&buf, dims[i+1]*dims[i])),
				BiasGradient:     nnmath.MakeVecData(dims[i+1], mem.Take(&buf, dims[i+1])),
				ErrorPropagation: nnmath.MakeVecData(dims[i+1], mem.Take(&buf, dims[i+1])),
			},
		}

		layers = append(layers, layer)
	}

	return layers
}
