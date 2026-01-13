package work

import "sync"

type Pool struct {
	tasks   chan func()
	stopped bool
	wg      sync.WaitGroup
}

func New(max_workers int) *Pool {
	p := &Pool{
		tasks: make(chan func(), max_workers),
	}

	for range max_workers {
		go p.worker()
	}

	return p
}

func (p *Pool) Stop() {
	p.stopped = true
	close(p.tasks)
}

func (p *Pool) Wait() {
	p.wg.Wait()
}

func (p *Pool) Enqueue(task func()) {
	if p.stopped {
		return
	}

	p.wg.Add(1)
	p.tasks <- task
}

func (p *Pool) worker() {
	for task := range p.tasks {
		task()
		p.wg.Done()
	}
}
