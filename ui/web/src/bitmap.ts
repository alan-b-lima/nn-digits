export default class Bitmap {
    #width: number
    #heigth: number
    #data: Uint32Array

    constructor(width: number, height: number) {
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

    Copy(): Bitmap {
        const copy = new Bitmap(this.#width, this.#heigth)
        for (let i = 0; i < this.#data.length; i++) {
            copy.#data[i] = this.#data[i]
        }

        return copy
    }
}