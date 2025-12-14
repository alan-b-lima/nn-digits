export default class Bitmap {
    #width: number
    #heigth: number
    #data: Uint32Array

    constructor(height: number, width: number) {
        this.#width = width
        this.#heigth = height
        this.#data = new Uint32Array(width * height)
    }

    get Width(): number {
        return this.#width
    }

    get Height(): number {
        return this.#heigth
    }

    At(x: number, y: number): number {
        return this.#data[y * this.#width + x]
    }

    Set(x: number, y: number, color: number): void {
        this.#data[y * this.#width + x] = color
    }

    *[Symbol.iterator]() {
        for (const pixel of this.#data) {
            yield pixel            
        }
    }
}