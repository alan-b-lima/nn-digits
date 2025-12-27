package nn

import "github.com/alan-b-lima/nn-digits/pkg/nnmath"

func (nn *NeuralNetwork) Cost(dataset []LabeledSample) float64 {
	var cost float64
	for _, sample := range dataset {
		output := nn.FeedForward(sample.Values).Data()
		expected := sample.Label.Data()

		for i := range len(output) {
			diff := output[i] - expected[i]
			cost += diff * diff
		}
	}

	return .5 * cost / float64(len(dataset))
}

func (nn *NeuralNetwork) Status(dataset []LabeledSample) (int, float64) {
	var cost float64
	var correct int

	for _, sample := range dataset {
		output := nn.FeedForward(sample.Values).Data()
		expected := sample.Label.Data()

		for i := range len(output) {
			diff := output[i] - expected[i]
			cost += diff * diff
		}

		index := reduce(output, func(index int, v float64, i int) int {
			if v > output[index] {
				return i
			}
			return index
		})

		label := reduce(expected, func(label int, v float64, i int) int {
			if v > output[label] {
				return i
			}
			return label
		})

		if index == label {
			correct++
		}
	}

	return correct, .5 * cost / float64(len(dataset))
}

func (nn *NeuralNetwork) SampleCostDerivative(cost nnmath.Vector, sample LabeledSample) {
	output := nn.FeedForward(sample.Values).Data()
	expected := sample.Label.Data()

	vector := cost.Data()
	for i := range len(output) {
		vector[i] += output[i] - expected[i]
	}
}

func reduce[E, R any](s []E, fn func(acc R, v E, i int) R) R {
	var acc R
	for i, v := range s {
		acc = fn(acc, v, i)
	}

	return acc
}
