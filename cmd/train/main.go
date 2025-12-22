package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strconv"

	nn "github.com/alan-b-lima/nn-digits/internal/neural_network"
	"github.com/alan-b-lima/nn-digits/internal/service"
)

var pattern = regexp.MustCompile(`\S+`)

func main() {
	l, err := service.NewLearner(".data/mnist/training.csv", ".data/mnist/test_labeled.csv")
	if err != nil {
		fmt.Println(err)
		return
	}

	var unsaved bool

	reader := bufio.NewReader(os.Stdin)

REPLoop:
	for {
		fmt.Print("> ")
		bytes, _, err := reader.ReadLine()
		if err != nil {
			fmt.Println(err)
			continue
		}

		matches := pattern.FindAll(bytes, -1)
		if len(matches) == 0 {
			continue
		}

		args := make([]string, 0, len(matches))
		for _, match := range matches {
			args = append(args, string(match))
		}

		switch args[0] {
		case "train":
			if len(args) < 2 {
				fmt.Println("batch size must be given")
				break
			}

			size, err := strconv.Atoi(args[1])
			if err != nil {
				fmt.Println(err)
				break
			}

			l.LearnBatch(size)
			unsaved = true

		case "cycle":
			if len(args) < 3 {
				fmt.Println("batch size and iterations must be given")
				break
			}

			size, err := strconv.Atoi(args[1])
			if err != nil {
				fmt.Println(err)
				break
			}

			iterations, err := strconv.Atoi(args[2])
			if err != nil {
				fmt.Println(err)
				break
			}

			for i := range iterations {
				fmt.Printf("\r%d/%d", i+1, iterations)
				l.LearnBatch(size)
			}

			fmt.Println()
			unsaved = true

		case "status":
			total := len(l.Tests)
			correct, cost := l.Status()

			fmt.Printf("Correct: %d\n", correct)
			fmt.Printf("Incorrect: %d\n", total-correct)
			fmt.Printf("Total: %d\n", total)
			fmt.Printf("\n")
			fmt.Printf("Cost: %f\n", cost)

		case "rate":
			if len(args) < 2 {
				fmt.Printf("Learning rate: %f\n", l.LearningRate)
				break
			}

			rate, err := strconv.ParseFloat(args[1], 64)
			if err != nil {
				fmt.Println(err)
				break
			}

			l.LearningRate = rate

		case "store":
			if len(args) < 2 {
				fmt.Println("filepath must be given")
				break
			}

			if err := save(args[1], &l.NeuralNetwork); err != nil {
				fmt.Println(err)
				break
			}

			unsaved = false

		case "quit":
			if !unsaved {
				break REPLoop
			}

		QuitLoop:
			for {
				var char rune
				fmt.Print("There are unsaved changes, do you want to quit ([y] or n)? ")
				n, err := fmt.Scanf("%c\n", &char)
				if err != nil {
					fmt.Println(err)
					continue
				}
				if n == 0 {
					break REPLoop
				}

				switch char {
				case 'Y', 'y':
					break REPLoop

				case 'N', 'n':
					break QuitLoop
				}
			}

		default:
			fmt.Printf("unknown option %s\n", args[0])
		}
	}
}

func save(path string, nn *nn.NeuralNetwork) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0o666)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := f.Truncate(0); err != nil {
		return err
	}

	j, err := json.Marshal(nn)
	if err != nil {
		return err
	}

	_, err = f.Write(j)
	return err
}
