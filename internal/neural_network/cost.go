package nn

func (nn *NeuralNetwork) Cost(dataset []LabeledSample) float64 {
	var cost float64
	for _, sample := range dataset {
		cost += nn.SampleCost(sample)
	}

	return cost / float64(len(dataset))
}

func (nn *NeuralNetwork) SampleCost(sample LabeledSample) float64 {
	output := nn.FeedForward(sample.Values).Data()
	expected := sample.Label.Data()

	var cost float64
	for i := range len(output) {
		cost += NodeCost(output[i], expected[i])
	}

	return cost * .5
}

func NodeCost(output, expected float64) float64 {
	diff := output - expected
	return diff * diff
}
