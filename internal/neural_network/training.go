package nn

import (
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

func (nn *NeuralNetwork) Learn(dataset []LabeledSample, rate float64) {
	cost := nn.Cost(dataset)
	const h = 1e-3

	for _, layer := range nn.layers {
		weights := layer.Weights
		gweight := layer.weight_gradient

		for r := range weights.Rows() {
			for c := range weights.Cols() {
				weights.Set(r, c, weights.At(r, c)+h)
				diff := nn.Cost(dataset) - cost
				weights.Set(r, c, weights.At(r, c)-h)

				gweight.Set(r, c, diff/h)
			}
		}

		biases := layer.Weights
		gbias := layer.bias_gradient

		for r := range biases.Rows() {
			biases.Set(r, 0, biases.At(r, 0)+h)
			diff := nn.Cost(dataset) - cost
			biases.Set(r, 0, biases.At(r, 0)-h)

			gbias.Set(r, 0, diff/h)
		}
	}

	for _, layer := range nn.layers {
		nnmath.SMulP(layer.weight_gradient, -rate, layer.weight_gradient)
		nnmath.SMulP(layer.bias_gradient, -rate, layer.bias_gradient)

		nnmath.AddP(layer.Weights, layer.Weights, layer.weight_gradient)
		nnmath.AddP(layer.Biases, layer.Biases, layer.bias_gradient)
	}
}
