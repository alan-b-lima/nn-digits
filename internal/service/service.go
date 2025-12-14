package serve

type Identifier interface {
	Identify(Request) (Result, error)
}

type (
	Request struct {
		Height int
		Width  int
		Data   []float64
	}

	Result [10]float64
)
