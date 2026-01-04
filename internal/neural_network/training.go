package nn

import "github.com/alan-b-lima/nn-digits/pkg/nnmath"

func (nn *NeuralNetwork) Learn(dataset []LabeledSample, rate float64) {
	learn := nn.get_learn()
	defer nn.free_learn(learn)

	nn.compute_gradient(learn, dataset)
	nn.apply_gradient(learn, rate)
}

func (nn *NeuralNetwork) apply_gradient(learn *[]learning, rate float64) {
	nn.mu.Lock()
	defer nn.mu.Unlock()

	for i, layer := range nn.layers {
		learn := (*learn)[i]

		nnmath.AddSMulP(layer.Weights, layer.Weights, -rate, learn.WeightGradient)
		nnmath.AddSMulP(layer.Biases, layer.Biases, -rate, learn.BiasGradient)
	}
}

func (nn *NeuralNetwork) compute_gradient(learn *[]learning, dataset []LabeledSample) {
	if len(nn.layers) == 0 {
		return
	}

	comp := make([]computation, len(nn.layers))
	for i, buf := range *learn {
		comp[i] = computation{buf.Activation}
	}

	for _, sample := range dataset {
		{
			input := sample.Values
			if len(nn.layers) > 1 {
				input = (*learn)[len(nn.layers)-1].Activation
			}

			curr := (*learn)[len(nn.layers)-1]

			nn.sample_cost_derivative(&comp, curr.ErrorPropagation, sample)

			nnmath.SoftmaxDerivativeFromActivation(curr.Activation.Data())
			nnmath.HMulP(curr.ErrorPropagation, curr.ErrorPropagation, curr.Activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMulP(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.AddP(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}

		nn.mu.RLock()
		for i := len(nn.layers) - 2; i >= 0; i-- {
			input := sample.Values
			if i > 0 {
				input = (*learn)[len(nn.layers)-1].Activation
			}

			next := (*learn)[i+1]
			curr := (*learn)[i]

			t := nnmath.MakeMatData(1, next.ErrorPropagation.Rows(), next.ErrorPropagation.Data())
			r := nnmath.MakeMatData(1, curr.ErrorPropagation.Rows(), curr.ErrorPropagation.Data())
			nnmath.MulP(r, t, nn.layers[i+1].Weights)

			nnmath.ApplyP(curr.Activation, curr.Activation, SigmoidDerivativeFromActivation)
			nnmath.HMulP(curr.ErrorPropagation, curr.ErrorPropagation, curr.Activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMulP(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.AddP(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}
		nn.mu.RUnlock()

		factor := 1 / float64(len(dataset))
		for _, layer := range *learn {
			nnmath.SMulP(layer.WeightGradient, factor, layer.WeightGradient)
			nnmath.SMulP(layer.BiasGradient, factor, layer.BiasGradient)
		}
	}
}
