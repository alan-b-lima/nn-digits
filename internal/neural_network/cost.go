package nn

import (
	"cmp"

	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

func (nn *NeuralNetwork) Cost(dataset []LabeledSample) (int, float64) {
	nn.mu.RLock()
	defer nn.mu.RUnlock()

	var cost float64
	var correct int

	for _, sample := range dataset {
		output := nn.FeedForward(sample.Values).Data()
		expected := sample.Label.Data()

		for i := range len(output) {
			diff := output[i] - expected[i]
			cost += diff * diff
		}

		index := index_of_max(output)
		label := index_of_max(expected)

		if index == label {
			correct++
		}
	}

	return correct, .5 * cost / float64(len(dataset))
}

func (nn *NeuralNetwork) sample_cost_derivative(learn *[]computation, cost nnmath.Vector, sample LabeledSample) {
	output := nn.feed_forward(learn, sample.Values).Data()
	expected := sample.Label.Data()

	vector := cost.Data()
	for i := range len(output) {
		vector[i] = output[i] - expected[i]
	}
}

func index_of_max[T ~[]E, E cmp.Ordered](s T) int {
	if len(s) == 0 {
		return -1
	}

	var index int
	for i, v := range s[1:] {
		if v > s[index] {
			index = i
		}
	}

	return index
}
