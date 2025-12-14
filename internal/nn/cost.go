package nn

func Cost(nn NeuralNetwork, dataset []LabeledSample) float64 {
	var cost float64
	for _, sample := range dataset {
		cost += sampleCost(nn, sample)
	}

	return cost / float64(len(dataset))
}

func sampleCost(nn NeuralNetwork, sample LabeledSample) float64 {
	output := nn.FeedForward(sample.Values)

	var cost float64
	for i, v := range output.Data {
		diff := v - sample.Label.Data[i]
		cost += diff * diff
	}

	return cost * .5
}
