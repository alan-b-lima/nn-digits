package mnist

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
	"github.com/alan-b-lima/nn-digits/pkg/nnmath"
)

const (
	side = 28
	pixs = side * side
	cols = 1 + pixs
)

var Labels [10]nnmath.Vector

func init() {
	for i := range len(Labels) {
		Labels[i] = nnmath.MakeVec(10)
		Labels[i].Set(i, 0, 1)
	}
}

func NewLabeledFromCSV(r io.Reader) ([]nn.LabeledSample, error) {
	var dataset []nn.LabeledSample

	reader := csv.NewReader(r)
	reader.ReuseRecord = true

	// discard first row
	if _, err := reader.Read(); err != nil {
		return nil, err
	}

	for i := 0; ; i++ {
		row, err := reader.Read()
		if err != nil {
			if err == io.EOF {
				break
			}

			return nil, fmt.Errorf("mnist: could not read next row: %w", err)
		}

		if len(row) != cols {
			return nil, fmt.Errorf("mnist: expected %d columns at row %d, found %d", cols, i, len(row))
		}

		label, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("mnist: could not parse label: %w", err)
		}
		if label < 0 || 10 <= label {
			return nil, fmt.Errorf("mnist: label out of range")
		}

		sample := nn.LabeledSample{
			Label:  Labels[label],
			Values: nnmath.MakeVec(pixs),
		}

		for i, cell := range row[1:] {
			pixel, err := strconv.Atoi(cell)
			if err != nil {
				return nil, fmt.Errorf("mnist: could not parse pixel value: %w", err)
			}
			if label < 0 || 255 < label {
				return nil, fmt.Errorf("mnist: label out of range")
			}

			sample.Values.Set(i, 0, float64(pixel)/255)
		}

		dataset = append(dataset, sample)
	}

	return dataset, nil
}
