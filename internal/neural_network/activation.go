package nn

import (
	"math"

	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

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

func ReLU(x float64) float64 {
	return max(x, 0)
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func ReLUDerivativeFromActivation(x float64) float64 {
	return ReLUDerivative(x)
}

func Softmax(vector nnmath.Vector) {
	values := vector.Data()

	var sum float64
	for i := range len(values) {
		values[i] = math.Exp(values[i])
		sum += values[i]
	}

	nnmath.SMul(vector, 1/sum, vector)
}

func SoftmaxDerivativeFromActivation(vector nnmath.Vector) {
	values := vector.Data()

	for i := range len(values) {
		values[i] = values[i] * (1 - values[i])
	}
}
