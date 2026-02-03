package nn

import (
	"cmp"

	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

func (nn *NeuralNetwork) Performance(dataset []Sample) (correct int, cost float64) {
	comp := nn.get_comp()
	defer nn.free_comp(comp)

	for _, sample := range dataset {
		nn.feed_forward(comp, sample.Values)

		output := (*comp)[len(*comp)-1].Activation.Data()
		expected := sample.Label.Data()

		for i := range len(output) {
			diff := output[i] - expected[i]
			cost += diff * diff
		}

		class := index_of_max(output)
		label := index_of_max(expected)

		if class == label {
			correct++
		}
	}

	return correct, .5 * cost / float64(len(dataset))
}

func (nn *NeuralNetwork) sample_cost_derivative(comp *[]computation, cost nnmath.Vector, sample Sample) {
	nn.feed_forward(comp, sample.Values)

	output := (*comp)[len(*comp)-1].Activation
	expected := sample.Label

	nnmath.Sub(cost, output, expected)
}

func index_of_max[T ~[]E, E cmp.Ordered](s T) int {
	if len(s) == 0 {
		return -1
	}

	var index int
	for i, v := range s {
		if v > s[index] {
			index = i
		}
	}

	return index
}
