package nnmath

import "math"

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
