package nnmath

type SofteningFunc func(x float64) float64

func Softmax(values []float64, fn SofteningFunc) []float64 {
	var sum float64
	for i := range len(values) {
		values[i] = fn(values[i])
		sum += values[i]
	}

	for i := range len(values) {
		values[i] /= sum
	}

	return values
}

func SoftmaxDerivative(values []float64, fn SofteningFunc) []float64 {
	var sum float64
	for i := range len(values) {
		values[i] = fn(values[i])
		sum += values[i]
	}

	for i := range len(values) {
		dem := values[i] + sum
		values[i] *= sum / (dem * dem)
	}

	return values
}
