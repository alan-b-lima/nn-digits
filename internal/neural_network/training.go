package nn

import (
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

const use_backpropagation = true

func (nn *NeuralNetwork) Learn(dataset []LabeledSample, rate float64) {
	nn.compute_gradient(dataset)
	nn.apply_gradient(rate)
}

func (nn *NeuralNetwork) apply_gradient(rate float64) {
	for _, layer := range nn.Layers {
		nnmath.SMulP(layer.WeightGradient, -rate, layer.WeightGradient)
		nnmath.SMulP(layer.BiasGradient, -rate, layer.BiasGradient)

		nnmath.AddP(layer.Weights, layer.Weights, layer.WeightGradient)
		nnmath.AddP(layer.Biases, layer.Biases, layer.BiasGradient)
	}
}

func (nn *NeuralNetwork) compute_gradient(dataset []LabeledSample) {
	if use_backpropagation {
		nn.backpropagate(dataset)
	} else {
		nn.naive_brute_force(dataset)
	}
}

func (nn *NeuralNetwork) naive_brute_force(dataset []LabeledSample) {
	const h = 1e-3
	cost := nn.Cost(dataset)

	for _, layer := range nn.Layers {
		weights := layer.Weights
		biases := layer.Biases

		gweight := layer.WeightGradient
		gbias := layer.BiasGradient

		for r := range weights.Rows() {
			for c := range weights.Cols() {
				weights.Set(r, c, weights.At(r, c)+h)
				diff := nn.Cost(dataset) - cost
				weights.Set(r, c, weights.At(r, c)-h)

				gweight.Set(r, c, diff/h)
			}

			biases.Set(r, 0, biases.At(r, 0)+h)
			diff := nn.Cost(dataset) - cost
			biases.Set(r, 0, biases.At(r, 0)-h)

			gbias.Set(r, 0, diff/h)
		}
	}
}

func (nn *NeuralNetwork) backpropagate(dataset []LabeledSample) {
	if len(nn.Layers) == 0 {
		return
	}

	for _, layer := range nn.Layers {
		nnmath.Zero(layer.WeightGradient)
		nnmath.Zero(layer.BiasGradient)
	}

	for _, sample := range dataset {
		{
			input := sample.Values
			if len(nn.Layers) > 1 {
				input = nn.Layers[len(nn.Layers)-2].Activation
			}

			curr := nn.Layers[len(nn.Layers)-1]

			nn.SampleCostDerivative(curr.ErrorPropagation, sample)

			nnmath.SoftmaxDerivativeFromActivation(curr.Activation.Data())
			nnmath.HMulP(curr.ErrorPropagation, curr.ErrorPropagation, curr.Activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMulP(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.AddP(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}

		for i := len(nn.Layers) - 2; i >= 0; i-- {
			input := sample.Values
			if i > 0 {
				input = nn.Layers[i-1].Activation
			}

			next := nn.Layers[i+1]
			curr := nn.Layers[i]

			t := nnmath.MakeMatData(1, next.ErrorPropagation.Rows(), next.ErrorPropagation.Data())
			r := nnmath.MakeMatData(1, curr.ErrorPropagation.Rows(), curr.ErrorPropagation.Data())
			nnmath.MulP(r, t, next.Weights)

			nnmath.ApplyP(curr.Activation, curr.Activation, SigmoidDerivativeFromActivation)
			nnmath.HMulP(curr.ErrorPropagation, curr.ErrorPropagation, curr.Activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMulP(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.AddP(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}

		factor := 1 / float64(len(dataset))
		for _, layer := range nn.Layers {
			nnmath.SMulP(layer.WeightGradient, factor, layer.WeightGradient)
			nnmath.SMulP(layer.BiasGradient, factor, layer.BiasGradient)
		}
	}
}
