import Bitmap from "./bitmap.ts"
import jsxmm from "./module/jsxmm/element.ts"

type IdentifierFunc = (data: number[], width: number, height: number) => number[] | null

export const MainBrush = 0xFFFFFF
export const AltBrush = 0x000000

export default class Canvas {
    #ctx: CanvasRenderingContext2D
    #bitmap: AdvanceBuffer<Bitmap>

    #brush: number
    #outputs: HTMLInputElement[]

    #classify: IdentifierFunc

    constructor(canvas: HTMLCanvasElement, classify: IdentifierFunc) {
        const ctx = canvas.getContext("2d")
        if (ctx === null) {
            throw new Error("2D drawing on canvas element not supported")
        }

        this.#ctx = ctx
        this.#bitmap = new AdvanceBuffer<Bitmap>(32)
        this.#bitmap.Append(new Bitmap(28, 28))

        this.#brush = MainBrush
        this.#outputs = this.Outputs()

        this.#classify = classify

        this.setup_listeners(canvas)
        this.classify()
    }

    Outputs(): HTMLInputElement[] {
        if (this.#outputs !== undefined) {
            return this.#outputs
        }

        this.#outputs = new Array(10)
        for (let i = 0; i < 10; i++) {
            const output = jsxmm.Element("input", {
                className: "output", type: "range",
                min: "0", max: "1", step: ".01", value: "0",
                tabIndex: -1,
            })

            this.#outputs[i] = output
        }

        return this.#outputs
    }

    Brush(): number {
        return this.#brush
    }

    SetBrush(brush: number): void {
        this.#brush = brush
    }

    Reset(): void {
        this.do()

        const bitmap = this.#bitmap.Current()
        for (let x = 0; x < bitmap.Width; x++) {
            for (let y = 0; y < bitmap.Height; y++) {
                this.set(x, y, 0)
            }
        }

        this.classify()
    }

    Undo(): boolean {
        if (!this.#bitmap.Revert()) {
            return false
        }

        this.redraw()
        return true
    }

    Redo(): boolean {
        if (!this.#bitmap.Advance()) {
            return false
        }

        this.redraw()
        return true
    }

    redraw(): void {
        const bitmap = this.#bitmap.Current()
        for (let x = 0; x < bitmap.Width; x++) {
            for (let y = 0; y < bitmap.Height; y++) {
                this.#ctx.fillStyle = pixel(bitmap.At(x, y))
                this.#ctx.fillRect(x, y, 1, 1)
            }
        }

        this.classify()
    }

    #buffer?: number[]

    private classify(): number[] {
        const bitmap = this.#bitmap.Current()

        if (this.#buffer === undefined) {
            this.#buffer = new Array(bitmap.Width * bitmap.Height)
        }

        let i = 0
        for (let x = 0; x < bitmap.Width; x++) {
            for (let y = 0; y < bitmap.Height; y++) {
                const color = bitmap.At(x, y)
                this.#buffer[i] = saturation(color)
                i++
            }
        }

        const result = this.#classify(this.#buffer, bitmap.Width, bitmap.Height)
        if (result === null) {
            throw new Error("untracable error occurred")
        }

        const max = Math.max(...result)

        for (let i = 0; i < result.length; i++) {
            if (result[i] === max) {
                this.#outputs[i].classList.add("classification")
            } else {
                this.#outputs[i].classList.remove("classification")
            }

            this.#outputs[i].value = result[i].toString()
        }

        return result
    }

    private set(x: number, y: number, color: number) {
        this.#bitmap.Current().Set(x, y, color)
        this.#ctx.fillStyle = pixel(color)
        this.#ctx.fillRect(x, y, 1, 1)
    }

    private stroke(x: number, y: number, radius: number, brush: number) {
        const bitmap = this.#bitmap.Current()

        const offx = Math.max(0, Math.floor(x - radius))
        const offy = Math.max(0, Math.floor(y - radius))
        const limx = Math.min(Math.ceil(x + radius), bitmap.Width - 1)
        const limy = Math.min(Math.ceil(y + radius), bitmap.Height - 1)

        for (let xi = offx; xi <= limx; xi++) {
            for (let yi = offy; yi <= limy; yi++) {
                const color = lerp(
                    bitmap.At(xi, yi),
                    brush,
                    Math.min(Math.hypot(xi - x, yi - y) / (radius + 1), 1),
                )

                this.set(xi, yi, color)
            }
        }
    }

    private do(): Bitmap {
        const bitmap = this.#bitmap.Current().Copy()
        this.#bitmap.Append(bitmap)
        return bitmap
    }

    private setup_listeners(canvas: HTMLCanvasElement) {
        const BrushRadius = 1.5

        const brush = {
            state: BrushUp,
            color: MainBrush,
        }

        let bitmap = this.#bitmap.Current()

        canvas.tabIndex = 0
        canvas.width = bitmap.Width
        canvas.height = bitmap.Height
        this.Reset()

        canvas.addEventListener("keydown", (evt: KeyboardEvent): void => {
            if (!evt.ctrlKey) {
                return
            }

            switch (evt.key) {
            case "z":
                this.Undo()
                break
            case "y":
                this.Redo()
                break
            default:
                return
            }

            evt.preventDefault()
        })

        canvas.addEventListener("click", (evt: MouseEvent): void => {
            if (brush.state === BrushStroke) {
                return
            }
            bitmap = this.do()

            const x = (evt.offsetX - canvas.clientLeft) * canvas.width / canvas.clientWidth - .5
            const y = (evt.offsetY - canvas.clientTop) * canvas.height / canvas.clientHeight - .5
            this.stroke(x, y, BrushRadius, brush.color)
            this.classify()
        })

        canvas.addEventListener("contextmenu", (evt: MouseEvent): void => {
            evt.preventDefault()
        })

        canvas.addEventListener("mousemove", (evt: MouseEvent): void => {
            if (brush.state === BrushUp) {
                return
            }
            brush.state = BrushStroke

            const x = (evt.offsetX - canvas.clientLeft) * canvas.width / canvas.clientWidth - .5
            const y = (evt.offsetY - canvas.clientTop) * canvas.height / canvas.clientHeight - .5
            this.stroke(x, y, BrushRadius, brush.color)
            this.classify()
        })

        canvas.addEventListener("touchmove", (evt: TouchEvent): void => {
            evt.preventDefault()

            if (brush.state === BrushUp || evt.touches.length === 0) {
                return
            }
            brush.state = BrushStroke

            const touch = evt.touches[evt.touches.length - 1]
            const rect = canvas.getBoundingClientRect()

            const x = (touch.clientX - rect.left) * canvas.width / canvas.clientWidth - .5
            const y = (touch.clientY - rect.top) * canvas.height / canvas.clientHeight - .5
            this.stroke(x, y, BrushRadius, this.#brush)
            this.classify()
        })

        canvas.addEventListener("mousedown", (evt: MouseEvent): void => {
            if (evt.button === 0) {
                brush.color = this.Brush()
            } else if (evt.button === 2) {
                brush.color = 0xFFFFFF - this.Brush()
            }

            if (brush.state === BrushUp) {
                bitmap = this.do()
                brush.state = BrushDown
            }
        })

        canvas.addEventListener("mouseup", (): void => {
            if (brush.state !== BrushStroke) {
                this.#bitmap.Revert()
            }
            setTimeout(() => { brush.state = BrushUp })
        })

        canvas.addEventListener("touchstart", (evt: TouchEvent): void => {
            evt.preventDefault()

            if (brush.state === BrushUp) {
                bitmap = this.do()
                brush.state = BrushDown
            }
        })

        canvas.addEventListener("touchend", (evt: TouchEvent): void => {
            evt.preventDefault()

            if (brush.state !== BrushStroke) {
                this.#bitmap.Revert()
            }
            setTimeout(() => { brush.state = BrushUp })
        })
    }
}

const BrushUp = 0
const BrushDown = 1
const BrushStroke = 2

class AdvanceBuffer<T> {
    #buffer: Array<T>
    #capacity: number

    #cursor: number
    #head: number
    #tail: number

    constructor(capacity: number) {
        this.#buffer = new Array(capacity)
        this.#capacity = capacity

        this.#cursor = -1
        this.#head = 0
        this.#tail = 0
    }

    get Capacity(): number {
        return this.#capacity
    }

    get Length(): number {
        return this.#head - this.#tail
    }

    Append(state: T): void {
        this.#cursor++
        this.set(state)

        this.#head = this.#cursor + 1
        if (this.Length >= this.#capacity) {
            this.#tail = this.#head - this.#capacity
        }
    }

    Current(): T {
        if (this.Length === 0) {
            return undefined as T
        }

        return this.get()
    }

    Advance(): boolean {
        if (this.Length === 0) {
            return false
        }

        const next = this.#cursor + 1
        if (next < this.#head) {
            this.#cursor = next
        } else {
            return false
        }

        return true
    }

    Revert(): boolean {
        if (this.Length === 0) {
            return false
        }

        const prev = this.#cursor - 1
        if (prev >= this.#tail) {
            this.#cursor = prev
        } else {
            return false
        }

        return true
    }

    private get(): T {
        return this.#buffer[this.index(this.#cursor)]
    }

    private set(value: T): void {
        this.#buffer[this.index(this.#cursor)] = value
    }

    private index(index: number): number {
        index = index % this.#capacity
        if (index < 0) {
            return index + this.#capacity
        }
        return index
    }
}

function lerp(color0: number, color1: number, step: number): number {
    const rn = Math.floor(((color0 >> 16) & 0xFF) * step + ((color1 >> 16) & 0xFF) * (1 - step))
    const gn = Math.floor(((color0 >> 8) & 0xFF) * step + ((color1 >> 8) & 0xFF) * (1 - step))
    const bn = Math.floor((color0 & 0xFF) * step + (color1 & 0xFF) * (1 - step))

    return (rn << 16) | (gn << 8) | bn
}

function pixel(num: number): string {
    return "#" + num.toString(16).padStart(6, "0")
}

function saturation(color: number): number {
    const r = ((color >> 16) & 0xFF) / 255
    const g = ((color >> 8) & 0xFF) / 255
    const b = (color & 0xFF) / 255

    return (r + g + b) / 3
}