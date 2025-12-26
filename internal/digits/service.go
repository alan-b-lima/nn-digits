package digits

import (
	"errors"
	"math/rand/v2"
	"os"
	"slices"

	mnist "github.com/alan-b-lima/nn-digits/internal/dataset"
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

type Learner struct {
	nn.NeuralNetwork

	Training []nn.LabeledSample
	Tests    []nn.LabeledSample

	LearningRate float64
}

func NewLearner(training, tests string) (*Learner, error) {
	l := &Learner{
		NeuralNetwork: nn.New(28*28, 8, 16, 10),
	}

	if d, err := load(training); err != nil {
		return nil, err
	} else {
		l.Training = d
	}

	if d, err := load(tests); err != nil {
		return nil, err
	} else {
		l.Tests = d
	}

	return l, nil
}

func (l *Learner) Status() (int, float64) {
	var correct int
	for _, test := range l.Tests {
		result := l.NeuralNetwork.FeedForward(test.Values).Data()

		var index int
		for i, v := range result {
			if v > result[index] {
				index = i
			}
		}

		label := slices.Index(test.Label.Data(), 1)

		if label == index {
			correct++
		}
	}

	return correct, l.NeuralNetwork.Cost(l.Tests)
}

func (l *Learner) LearnBatch(size int) {
	batch := l.Training
	if len(l.Training) > size {
		offset := rand.IntN(len(l.Training) - size)
		batch = batch[offset : offset+size]
	}

	l.Learn(batch, l.LearningRate)
}

func load(path string) ([]nn.LabeledSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return mnist.LoadFromCSV(f)
}
