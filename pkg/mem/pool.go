package mem

import "sync"

type Pool[T any] interface {
	Get() T
	Put(T)
}

func NewPool[T any](new func() T) Pool[T] {
	pool := &pool[T]{Pool: sync.Pool{
		New: func() any { return new() },
	}}

	return pool
}

type pool[T any] struct {
	sync.Pool
}

func (p *pool[T]) Get() T {
	return p.Pool.Get().(T)
}

func (p *pool[T]) Put(x T) {
	p.Pool.Put(x)
}
