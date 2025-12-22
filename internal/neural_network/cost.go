package nn

func (nn *NeuralNetwork) Cost(dataset []LabeledSample) float64 {
	var cost float64
	for _, sample := range dataset {
		output := nn.FeedForward(sample.Values).Data()
		expected := sample.Label.Data()

		var cost float64
		for i := range len(output) {
			diff := output[i] - expected[i]
			cost += diff * diff
		}
	}

	return cost / float64(len(dataset))
}

func (nn *NeuralNetwork) CostDerivative(dataset []LabeledSample) float64 {
	var cost float64
	for _, sample := range dataset {
		output := nn.FeedForward(sample.Values).Data()
		expected := sample.Label.Data()

		var cost float64
		for i := range len(output) {
			cost += output[i] - expected[i]
		}
	}

	return 2 * cost / float64(len(dataset))
}
