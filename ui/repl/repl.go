package repl

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"regexp"
	"slices"
	"strconv"

	"github.com/alan-b-lima/nn-digits/internal/dataset"
	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
)

type Directive func(*State, io.Writer, io.Reader, ...string) error

type Context struct {
	NeuralNetwork *nn.NeuralNetwork

	Training []nn.LabeledSample
	Tests    []nn.LabeledSample

	LearningRate float64
	Unsaved      bool
}

type State struct {
	ctxs  map[string]*Context
	focus string
}

func (s *State) Unsaved() bool {
	for _, ctx := range s.ctxs {
		if ctx.Unsaved {
			return true
		}
	}

	return false
}

func (s *State) Focused() *Context {
	return s.ctxs[s.focus]
}

var (
	reArgs = regexp.MustCompile(`\S+`)
	reName = regexp.MustCompile(`[a-z\-]+`)
)

var directives = map[string]Directive{
	"help":   Help,
	"new":    New,
	"list":   List,
	"focus":  Focus,
	"load":   Load,
	"store":  Store,
	"train":  Train,
	"cycle":  Cycle,
	"status": Status,
	"rate":   LearningRate,
	"exit":   Quit,
	"quit":   Quit,
}

func LaunchREPLoop(w io.Writer, r io.Reader) {
	state := State{
		ctxs: make(map[string]*Context),
	}

	reader := bufio.NewReader(r)
	for {
		fmt.Fprintf(w, "%s> ", state.focus)
		bytes, _, err := reader.ReadLine()
		if err != nil {
			fmt.Fprintln(w, err)
			continue
		}

		matches := reArgs.FindAllString(string(bytes), -1)
		if len(matches) == 0 {
			continue
		}

		directive, in := directives[matches[0]]
		if !in {
			fmt.Fprintf(w, "unknown directive %q\n", matches[0])
			continue
		}

		if err := directive(&state, w, r, matches[1:]...); err != nil {
			if err == QuitMessage {
				break
			}

			fmt.Fprintln(w, err)
			continue
		}
	}
}

var (
	QuitMessage = errors.New("quit")

	ErrNilContext      = errors.New("nil context")
	ErrContextNotFound = errors.New("context not found")

	ErrNewMissingArgs       = errors.New("bad args: new <name> { <dims> }")
	ErrNewMissingDimensions = errors.New("bad args: there must be at least two dimensions")
	ErrLoadMissingArgs      = errors.New("bad args: store ( model <name> | training  | tests ) <path>")
	ErrStoreMissingArgs     = errors.New("bad args: store model <path>")
	ErrTrainMissingArgs     = errors.New("bad args: train <size>")
	ErrCycleMissingArgs     = errors.New("bad args: cycle <size> <iterations>")

	ErrBadInput  = func(e, g int) error { return fmt.Errorf("input length: expected %d, got %d", e, g) }
	ErrBadOutput = func(e, g int) error { return fmt.Errorf("output length: expected %d, got %d", e, g) }

	ErrUnknownDirective = func(directive string) error { return fmt.Errorf("unknown directive %q", directive) }
	ErrBadName          = errors.New("bad name: name must only include lowercase latin letters and dashes (-)")
	ErrBadNumber        = func(err error) error { return fmt.Errorf("bad number: %w", err) }
)

func Help(state *State, w io.Writer, _ io.Reader, args ...string) error {
	fmt.Fprintln(w, help)
	return nil
}

func New(state *State, w io.Writer, r io.Reader, args ...string) error {
	if len(args) < 1 {
		return ErrNewMissingArgs
	}
	if len(args[1:]) < 2 {
		return ErrNewMissingDimensions
	}

	if !reName.MatchString(args[0]) {
		return ErrBadName
	}
	name := args[0]

	dims := make([]int, 0, len(args[1:]))
	for _, arg := range args[1:] {
		dim, err := strconv.Atoi(arg)
		if err != nil {
			return ErrBadNumber(err)
		}

		dims = append(dims, dim)
	}

	nn := nn.New(dims...)

	if ctx, in := state.ctxs[name]; in && ctx.Unsaved {
		overwrite, err := overwrite_loop(w, r, name)
		if err != nil || !overwrite {
			return nil
		}
	}

	state.ctxs[name] = &Context{
		NeuralNetwork: &nn,
		Unsaved:       true,
	}

	state.focus = name
	return nil
}

func List(state *State, w io.Writer, _ io.Reader, args ...string) error {
	keys := make([]string, 0, len(state.ctxs))
	for k := range state.ctxs {
		keys = append(keys, k)
	}

	slices.Sort(keys)
	for _, key := range keys {
		if state.ctxs[key].Unsaved {
			fmt.Fprintln(w, key+"*")
		} else {
			fmt.Fprintln(w, key)
		}
	}

	return nil
}

func Focus(state *State, w io.Writer, _ io.Reader, args ...string) error {
	if len(args) < 1 {
		fmt.Fprintln(w, state.focus)
		return nil
	}
	name := args[0]

	if !reName.MatchString(name) {
		return ErrBadName
	}

	if _, in := state.ctxs[name]; !in {
		return ErrContextNotFound
	}

	state.focus = name
	return nil
}

func Load(state *State, w io.Writer, r io.Reader, args ...string) error {
	if len(args) < 2 {
		return ErrLoadMissingArgs
	}

	path := args[1]
	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}

	switch directive := args[0]; directive {
	case "model":
		if len(args) < 3 {
			return ErrLoadMissingArgs
		}
		name := args[2]

		nn, err := load_model(path)
		if err != nil {
			return fmt.Errorf("load model: %w", err)
		}

		if ctx, in := state.ctxs[name]; in && ctx.Unsaved {
			overwrite, err := overwrite_loop(w, r, name)
			if err != nil || !overwrite {
				return nil
			}
		}

		state.ctxs[name] = &Context{
			NeuralNetwork: nn,
			Unsaved:       true,
		}

	case "training":
		data, err := load_data(path)
		if err != nil {
			return fmt.Errorf("load training: %w", err)
		}
		if len(data) == 0 {
			return nil
		}

		if e, i := ctx.NeuralNetwork.Layers[0].Weights.Cols(), data[0].Values.Rows(); e != i {
			return ErrBadInput(e, i)
		}

		if e, o := ctx.NeuralNetwork.Layers[len(ctx.NeuralNetwork.Layers)-1].Weights.Rows(), data[0].Label.Rows(); e != o {
			return ErrBadInput(e, o)
		}

		ctx.Training = append(ctx.Training, data...)

	case "tests":
		data, err := load_data(path)
		if err != nil {
			return fmt.Errorf("load tests: %w", err)
		}

		ctx.Tests = append(ctx.Tests, data...)

	default:
		return ErrUnknownDirective(directive)
	}

	return nil
}

func Store(state *State, w io.Writer, r io.Reader, args ...string) error {
	if len(args) < 2 {
		return ErrStoreMissingArgs
	}

	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}
	path := args[1]

	directive := args[0]
	if directive != "model" {
		return ErrUnknownDirective(directive)
	}

	if err := store_model(path, ctx.NeuralNetwork); err != nil {
		return fmt.Errorf("store model: %w", err)
	}

	return nil
}

func Train(state *State, w io.Writer, _ io.Reader, args ...string) error {
	if len(args) < 1 {
		return ErrTrainMissingArgs
	}

	size, err := strconv.Atoi(args[0])
	if err != nil {
		return ErrBadNumber(err)
	}

	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}

	learn_batch(ctx.NeuralNetwork, ctx.Training, ctx.LearningRate, size)

	ctx.Unsaved = true
	return nil
}

func Cycle(state *State, w io.Writer, _ io.Reader, args ...string) error {
	if len(args) < 2 {
		return ErrCycleMissingArgs
	}

	size, err := strconv.Atoi(args[0])
	if err != nil {
		return ErrBadNumber(err)
	}

	iterations, err := strconv.Atoi(args[1])
	if err != nil {
		return ErrBadNumber(err)
	}

	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}

	for i := range iterations {
		fmt.Fprintf(w, "\r%d/%d", i+1, iterations)
		learn_batch(ctx.NeuralNetwork, ctx.Training, ctx.LearningRate, size)
	}
	fmt.Fprintln(w)

	ctx.Unsaved = true
	return nil
}

func Status(state *State, w io.Writer, _ io.Reader, args ...string) error {
	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}

	total := len(ctx.Tests)
	correct, cost := ctx.NeuralNetwork.Status(ctx.Tests)

	fmt.Fprintf(w,
		"Correct: %d\nIncorrect: %d\nTotal: %d\nPerformance: %.2f%%\n\nCost: %f\n",
		correct, total-correct, total, 100*float64(correct)/float64(total), cost,
	)

	return nil
}

func LearningRate(state *State, w io.Writer, _ io.Reader, args ...string) error {
	ctx := state.Focused()
	if ctx == nil {
		return ErrNilContext
	}

	if len(args) < 1 {
		fmt.Fprintf(w, "Learning rate: %f\n", ctx.LearningRate)
		return nil
	}

	rate, err := strconv.ParseFloat(args[0], 64)
	if err != nil {
		return ErrBadNumber(err)
	}

	ctx.LearningRate = rate
	return nil
}

func Quit(state *State, w io.Writer, r io.Reader, args ...string) error {
	if !state.Unsaved() {
		return QuitMessage
	}

	for {
		var char rune
		fmt.Fprint(w, "There are unsaved changes, do you want to quit ([y] or n)? ")
		n, err := fmt.Fscanf(r, "%c\n", &char)
		if err != nil {
			return err
		}
		if n == 0 {
			return QuitMessage
		}

		switch char {
		case 'Y', 'y':
			return QuitMessage

		case 'N', 'n':
			return nil
		}
	}
}

func overwrite_loop(w io.Writer, r io.Reader, name string) (bool, error) {
	for {
		var char rune
		fmt.Fprintf(w, "There are unsaved changes, do you want to overwrite %q ([y] or n)? ", name)
		n, err := fmt.Fscanf(r, "%c\n", &char)
		if err != nil {
			return false, err
		}
		if n == 0 {
			return true, nil
		}

		switch char {
		case 'Y', 'y':
			return true, nil

		case 'N', 'n':
			return false, nil
		}
	}
}

func learn_batch(nn *nn.NeuralNetwork, training []nn.LabeledSample, rate float64, size int) {
	if size < len(training) {
		offset := rand.IntN(len(training) - size)
		training = training[offset : offset+size]
	}

	nn.Learn(training, rate)
}

func store_model(path string, nn *nn.NeuralNetwork) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	j, err := json.Marshal(nn)
	if err != nil {
		return err
	}

	_, err = f.Write(j)
	return err
}

func load_data(path string) ([]nn.LabeledSample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return dataset.LoadFromJSON(f)
}

func load_model(path string) (*nn.NeuralNetwork, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var nn nn.NeuralNetwork
	if err := json.NewDecoder(f).Decode(&nn); err != nil {
		return nil, err
	}

	return &nn, nil
}

const help = `NN Digits v0.0.2

NN Digits is an interactive shell for training a basic Multilayer Perceptron.

	new <name> { <dims> }
		creates a new neural network with the given dimensions and
		puts it on focus.

	list
		lists all named neural networks currently available.

	focus
		shows the current focused model.

	focus <name>
		changes the focused model.

	load training <path>
		loads training data from the file at <path> onto the focused
		model. If the data does not matches the size of the input and
		output layer sizes, it will be rejected.

	load tests <path>
		loads test data from the file at <path> onto the focused
		model. If the data does not matches the size of the input and
		output layer sizes, it will be rejected.

	load model <name> <path>
		loads a model from the file at <path> and puts it on focus.

	store model <path>
		stores a model on the give file path. This might be
		destructive.

	status
		shows the current performance of the model agaings its test
		data.

	rate
		shows the current learning rate of the focused model.

	rate <rate>
		changes the learning rate of the focused model.

	train <size>
		trains the network on a batch of size <size>. If <size>
		exceeds the number of training samples, all avalaible
		training samples will be used.

	cycle <size> <iterations>
		trains the network <iterations> times on batches of size
		<size>, batches are chosen randomly and contiguously out of
		the training set. If <size> exceeds the number of training
		samples, all avalaible training samples will be used.

	help
		shows this screen.

	quit
	exit
		exits the read-execute-print-loop. If there are any unsaved
		changes, it will ask if you still want to quit.`
