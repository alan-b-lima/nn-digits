package digits

import (
	"errors"

	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

type (
	Classifier interface {
		Classify(Request) (Result, error)
	}
)

type (
	Request [28 * 28]float64
	Result  [10]float64
)

type classifier struct {
	nn *nn.NeuralNetwork
}

var _ Classifier = &classifier{}

var errBadDimensions = errors.New("digits: request dimensions must constitute a 28x28 square")

func NewClassifier(nn *nn.NeuralNetwork) Classifier {
	return &classifier{nn: nn}
}

func (s *classifier) Classify(req Request) (Result, error) {
	mat := nnmath.MakeVecData(len(req), req[:])
	res := s.nn.FeedForward(mat)

	data := res.Data()
	if len(data) != 10 {
		return Result{}, errBadDimensions
	}

	return Result(data), nil
}

