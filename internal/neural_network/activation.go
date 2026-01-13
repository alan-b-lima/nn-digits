package nn

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	sig := Sigmoid(x)
	return sig * (1 - sig)
}

func SigmoidDerivativeFromActivation(x float64) float64 {
	return x * (1 - x)
}

func Softmax(values []float64) []float64 {
	var sum float64
	for i := range len(values) {
		values[i] = math.Exp(values[i])
		sum += values[i]
	}

	for i := range len(values) {
		values[i] /= sum
	}

	return values
}

func SoftmaxDerivativeFromActivation(values []float64) []float64 {
	for i := range len(values) {
		values[i] = values[i] * (1 - values[i])
	}

	return values
}
