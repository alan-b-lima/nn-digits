package nn

import "github.com/alan-b-lima/nn-digits/pkg/nnmath"

type Sample struct {
	Values nnmath.Vector
}

type LabeledSample struct {
	Label  nnmath.Vector
	Values nnmath.Vector
}
