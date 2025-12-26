package dataset

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"strconv"

	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
	"github.com/alan-b-lima/nn-digits/pkg/mem"
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

func LoadFromCSV(r io.Reader) ([]nn.LabeledSample, error) {
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

			return nil, fmt.Errorf("dataset: could not read next row: %w", err)
		}

		if len(row) != cols {
			return nil, fmt.Errorf("dataset: expected %d columns at row %d, found %d", cols, i, len(row))
		}

		label, err := strconv.Atoi(row[0])
		if err != nil {
			return nil, fmt.Errorf("dataset: could not parse label: %w", err)
		}
		if label < 0 || 10 <= label {
			return nil, fmt.Errorf("dataset: label out of range")
		}

		sample := nn.LabeledSample{
			Label:  Labels[label],
			Values: nnmath.MakeVec(pixs),
		}

		for i, cell := range row[1:] {
			pixel, err := strconv.Atoi(cell)
			if err != nil {
				return nil, fmt.Errorf("dataset: could not parse pixel value: %w", err)
			}
			if label < 0 || 255 < label {
				return nil, fmt.Errorf("dataset: label out of range")
			}

			sample.Values.Set(i, 0, float64(pixel)/255)
		}

		dataset = append(dataset, sample)
	}

	return dataset, nil
}

func LoadFromJSON(r io.Reader) ([]nn.LabeledSample, error) {
	var s []sample
	if err := json.NewDecoder(r).Decode(&s); err != nil {
		return nil, fmt.Errorf("dataset: parse JSON file: %w", err)
	}

	if len(s) == 0 {
		return []nn.LabeledSample{}, nil
	}

	label_size := len(s[0].Label)
	values_size := len(s[0].Values)

	samples := make([]nn.LabeledSample, len(s))
	buf := make([]float64, (label_size+values_size)*len(s))

	for i, s := range s {
		if len(s.Label) != label_size {
			return nil, fmt.Errorf("dataset: inconsistant label size at row %d: expected %d, got %d", i+1, label_size, len(s.Label))
		}

		if len(s.Values) != values_size {
			return nil, fmt.Errorf("dataset: inconsistant values size at row %d: expected %d, got %d", i+1, label_size, len(s.Label))
		}

		label := mem.Take(&buf, label_size)
		values := mem.Take(&buf, values_size)

		copy(label, s.Label)
		copy(values, s.Values)

		samples[i] = nn.LabeledSample{
			Label:  nnmath.MakeVecData(label_size, label),
			Values: nnmath.MakeVecData(values_size, values),
		}
	}

	return samples, nil
}

func StoreToJSON(w io.Writer, samples []nn.LabeledSample) error {
	ss := make([]sample, 0, len(samples))
	for _, s := range samples {
		ss = append(ss, sample{
			Label:  s.Label.Data(),
			Values: s.Values.Data(),
		})
	}

	if err := json.NewEncoder(w).Encode(ss); err != nil {
		return fmt.Errorf("dataset: encode JSON: %w", err)
	}

	return nil
}

type sample struct {
	Label  mem.Float64Slice `json:"label"`
	Values mem.Float64Slice `json:"values"`
}
