//go:build wasm && js

package wasm

import (
	"errors"

	"github.com/alan-b-lima/nn-digits/internal/nn"
	serve "github.com/alan-b-lima/nn-digits/internal/service"
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

type Service struct {
	nn nn.NeuralNetwork
}

var _ serve.Classifier = &Service{}

var errBadDimensions = errors.New("nn: request dimensions must constitute a 28x28 square")

func New() serve.Classifier {
	return &Service{
		nn: nn.New(28*28, 12, 13, 7, 10),
	}
}

func (s *Service) Classify(req serve.Request) (serve.Result, error) {
	if req.Width != 28 || req.Height != 28 {
		return serve.Result{}, errBadDimensions
	}

	if len(req.Data) != 28*28 {
		return serve.Result{}, errBadDimensions
	}

	mat := nnmath.MakeVecData(28*28, req.Data)
	res := s.nn.FeedForward(mat)

	return serve.Result(res.Data), nil
}
