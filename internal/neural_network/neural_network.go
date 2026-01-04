package nn

import (
	"math"
	"math/rand/v2"
	"sync"

	"github.com/alan-b-lima/nn-digits/pkg/mem"
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

// NeuralNetwork is a composition of layers that holds weights and biases for
// a Multilayer Perceptron.
//
// The zero value is valid, i.e., will not cause runtime panics.
//
// NeuralNetwork is safe for concurrent usage by multiple gorotines.
type NeuralNetwork struct {
	// layers are the layers of the network.
	//
	// no need to lock for len(layers) or any of its
	// content's lengths, but you do for the content
	// itself.
	layers []layer

	// buf is here for ease of marshal and unmarshal, but
	// also useful for cache locality.
	buf []float64

	comp  mem.Pool[*[]computation] // no need to lock for comp
	learn mem.Pool[*[]learning]    // no need to lock for learn

	mu sync.RWMutex
}

type layer struct {
	Weights nnmath.Matrix
	Biases  nnmath.Vector
}

func New(dims ...int) *NeuralNetwork {
	if len(dims) < 2 {
		panic("there must be at least two layers")
	}

	nn := NeuralNetwork{
		buf: make([]float64, size_nn(dims...)),
	}

	nn.comp = mem.NewPool(nn.new_comp)
	nn.learn = mem.NewPool(nn.new_learn)

	for i := range len(nn.buf) {
		nn.buf[i] = rand.NormFloat64()
	}

	nn.layers = slice_nn(nn.buf, dims...)
	return &nn
}

// Len returns the number of layers in the network, counting the input layer as
// a layer.
func (nn *NeuralNetwork) Len() int {
	return len(nn.layers) + 1
}

// InLen returns the number of input neurons in the network.
func (nn *NeuralNetwork) InLen() int {
	return nn.layers[0].Weights.Cols()
}

// OutLen returns the number of output neurons in the network.
func (nn *NeuralNetwork) OutLen() int {
	return nn.layers[len(nn.layers)-1].Weights.Rows()
}

// Dims returns an slice containing the dimensions of the network, i.e., how
// many neurons are in each layer, counting the input layer as a layer.
func (nn *NeuralNetwork) Dims() []int {
	dims := make([]int, 0, len(nn.layers)+1)

	dims = append(dims, nn.layers[0].Weights.Cols())
	for _, layer := range nn.layers {
		dims = append(dims, layer.Weights.Rows())
	}

	return dims
}

// FeedForward computes the output of the neural network given an input vector.
// 
// FeedForward panics if the input is not a matrix [n x 1] (a vector of length
// n), where n = [NeuralNetwork.InLen]().
func (nn *NeuralNetwork) FeedForward(input nnmath.Vector) nnmath.Vector {
	comp := nn.get_comp()
	defer nn.free_comp(comp)

	return nn.feed_forward(comp, input)
}

func (nn *NeuralNetwork) feed_forward(comp *[]computation, input nnmath.Vector) nnmath.Vector {
	nn.mu.RLock()
	defer nn.mu.RUnlock()

	if len(nn.layers) == 0 {
		return nnmath.Vector{}
	}

	for i := range nn.layers[:len(nn.layers)-1] {
		layer := &nn.layers[i]
		activation := (*comp)[i].Activation

		nnmath.AddMulP(activation, layer.Biases, layer.Weights, input)
		nnmath.ApplyP(activation, activation, Sigmoid)

		input = activation
	}

	last := nn.layers[len(nn.layers)-1]
	activation := (*comp)[len(nn.layers)-1].Activation

	nnmath.AddMulP(activation, last.Biases, last.Weights, input)
	nnmath.Softmax(activation.Data())

	input = activation
	return activation
}

type computation struct {
	Activation nnmath.Vector
}

type learning struct {
	Activation       nnmath.Vector
	WeightGradient   nnmath.Matrix
	BiasGradient     nnmath.Vector
	ErrorPropagation nnmath.Vector
}

func (nn *NeuralNetwork) new_comp() *[]computation {
	nn.mu.RLock()
	defer nn.mu.RUnlock()

	var size int
	for _, layer := range nn.layers {
		next, _ := layer.Weights.Dim()
		size += next
	}

	buf := make([]float64, size)
	comp := make([]computation, len(nn.layers))

	for i, layer := range nn.layers {
		len := layer.Weights.Rows()
		comp[i].Activation = nnmath.MakeVecData(len, mem.Take(&buf, len))
	}

	return &comp
}

func (nn *NeuralNetwork) get_comp() *[]computation {
	return nn.comp.Get()
}

func (nn *NeuralNetwork) free_comp(v *[]computation) {
	nn.comp.Put(v)
}

func (nn *NeuralNetwork) new_learn() *[]learning {
	nn.mu.RLock()
	defer nn.mu.RUnlock()

	var size int
	for _, layer := range nn.layers {
		next, curr := layer.Weights.Dim()
		size += next + next*curr + next + next
	}

	buf := make([]float64, size)
	learn := make([]learning, len(nn.layers))

	for i, layer := range nn.layers {
		next, curr := layer.Weights.Dim()

		learn[i].Activation = nnmath.MakeVecData(next, mem.Take(&buf, next))
		learn[i].WeightGradient = nnmath.MakeMatData(next, curr, mem.Take(&buf, next*curr))
		learn[i].BiasGradient = nnmath.MakeVecData(next, mem.Take(&buf, next))
		learn[i].ErrorPropagation = nnmath.MakeVecData(next, mem.Take(&buf, next))
	}

	return &learn
}

func (nn *NeuralNetwork) get_learn() *[]learning {
	v := nn.learn.Get()

	return v
}

func (nn *NeuralNetwork) free_learn(v *[]learning) {
	nn.learn.Put(v)
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
	}

	return size
}

func slice_nn(buf []float64, dims ...int) []layer {
	layers := make([]layer, 0, len(dims)-1)
	for i := range len(dims) - 1 {
		layer := layer{
			Weights: nnmath.MakeMatData(dims[i+1], dims[i], mem.Take(&buf, dims[i+1]*dims[i])),
			Biases:  nnmath.MakeVecData(dims[i+1], mem.Take(&buf, dims[i+1])),
		}

		layers = append(layers, layer)
	}

	return layers
}
