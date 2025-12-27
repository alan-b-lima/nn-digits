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

func (nn *NeuralNetwork) SampleCostDerivative(cost nnmath.Vector, sample LabeledSample) {
	output := nn.FeedForward(sample.Values).Data()
	expected := sample.Label.Data()

	vector := cost.Data()
	for i := range len(output) {
		vector[i] += output[i] - expected[i]
	}
}
