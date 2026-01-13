package nn

import "github.com/alan-b-lima/nn-digits/pkg/nnmath"

func (nn *NeuralNetwork) Learn(dataset []Sample, rate float64) {
	comp, learn := nn.get_learn()
	defer nn.free_learn(comp, learn)

	nn.compute_gradient(comp, learn, dataset)
	nn.apply_gradient(learn, rate)
}

func (nn *NeuralNetwork) apply_gradient(learn *[]learning, rate float64) {
	nn.mu.Lock()
	defer nn.mu.Unlock()

	for i, layer := range nn.layers {
		learn := (*learn)[i]

		nnmath.AddSMul(layer.Weights, layer.Weights, -rate, learn.WeightGradient)
		nnmath.AddSMul(layer.Biases, layer.Biases, -rate, learn.BiasGradient)
	}
}

func (nn *NeuralNetwork) compute_gradient(comp *[]computation, learn *[]learning, dataset []Sample) {
	if len(nn.layers) == 0 {
		return
	}

	for _, sample := range dataset {
		{
			input := sample.Values
			if len(nn.layers) > 1 {
				input = (*comp)[len(nn.layers)-2].Activation
			}

			activation := (*comp)[len(nn.layers)-1].Activation
			curr := (*learn)[len(nn.layers)-1]

			nn.sample_cost_derivative(comp, curr.ErrorPropagation, sample)

			SoftmaxDerivativeFromActivation(activation.Data())
			nnmath.HMul(curr.ErrorPropagation, curr.ErrorPropagation, activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMul(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.Add(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}

		nn.mu.RLock()
		for i := len(nn.layers) - 2; i >= 0; i-- {
			input := sample.Values
			if i > 0 {
				input = (*comp)[i-1].Activation
			}

			activation := (*comp)[i].Activation
			next := (*learn)[i+1]
			curr := (*learn)[i]

			t := nnmath.MakeMatData(1, next.ErrorPropagation.Rows(), next.ErrorPropagation.Data())
			r := nnmath.MakeMatData(1, curr.ErrorPropagation.Rows(), curr.ErrorPropagation.Data())
			nnmath.Mul(r, t, nn.layers[i+1].Weights)

			nnmath.Apply(activation, activation, SigmoidDerivativeFromActivation)
			nnmath.HMul(curr.ErrorPropagation, curr.ErrorPropagation, activation)

			a := nnmath.MakeMatData(1, input.Rows(), input.Data())
			nnmath.AddMul(curr.WeightGradient, curr.WeightGradient, curr.ErrorPropagation, a)
			nnmath.Add(curr.BiasGradient, curr.BiasGradient, curr.ErrorPropagation)
		}
		nn.mu.RUnlock()

		factor := 1 / float64(len(dataset))
		for _, layer := range *learn {
			nnmath.SMul(layer.WeightGradient, factor, layer.WeightGradient)
			nnmath.SMul(layer.BiasGradient, factor, layer.BiasGradient)
		}
	}
}
