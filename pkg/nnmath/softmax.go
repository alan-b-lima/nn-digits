package nnmath

type SofteningFunc func(x float64) float64

func Softmax(values []float64, fn SofteningFunc) []float64 {
	result := make([]float64, len(values))

	var sum float64
	for i := range len(values) {
		result[i] = fn(values[i])
		sum += result[i]
	}

	for i := range len(values) {
		result[i] /= sum
	}

	return result
}
