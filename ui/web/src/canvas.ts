import Bitmap from "./bitmap.ts"
import jsxmm from "./module/jsxmm/element.ts"

type IdentifierFunc = (data: number[], width: number, height: number) => number[] | null

export const MainBrush = 0xFFFFFF
export const AltBrush = 0x000000

export default class Canvas {
    #ctx: CanvasRenderingContext2D
    #bitmap: Bitmap

    #brush: number
    #outputs: HTMLInputElement[]

    #identify: IdentifierFunc
    #buffer: Array<number>

    constructor(canvas: HTMLCanvasElement, identify: IdentifierFunc) {
        const ctx = canvas.getContext("2d")
        if (ctx === null) {
            throw new Error("2D drawing on canvas element not supported")
        }

        this.#ctx = ctx
        this.#bitmap = new Bitmap(28, 28)

        this.#brush = MainBrush
        this.#outputs = this.Outputs()

        this.#identify = identify
        this.#buffer = new Array(28 * 28)

        this.setup_listeners(canvas)
        this.identify()
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

    Reset() {
        for (let x = 0; x < this.#bitmap.Width; x++) {
            for (let y = 0; y < this.#bitmap.Height; y++) {
                this.set(x, y, 0)
            }
        }

        this.identify()
    }

    Brush(): number {
        return this.#brush
    }

    SetBrush(brush: number): void {
        this.#brush = brush
    }

    private identify(): number[] {
        let i = 0
        for (let x = 0; x < this.#bitmap.Width; x++) {
            for (let y = 0; y < this.#bitmap.Height; y++) {
                const color = this.#bitmap.At(x, y)
                this.#buffer[i] = saturation(color)
                i++
            }
        }

        const result = this.#identify(this.#buffer, this.#bitmap.Width, this.#bitmap.Height)
        if (result === null) {
            throw new Error("untracable error occurred")
        }

        for (let i = 0; i < result.length; i++) {
            this.#outputs[i].value = result[i].toString()
        }

        return result
    }

    private set(x: number, y: number, color: number) {
        this.#bitmap.Set(x, y, color)
        this.#ctx.fillStyle = pixel(color)
        this.#ctx.fillRect(x, y, 1, 1)
    }

    private stroke(x: number, y: number, radius: number, brush: number) {
        const offx = Math.max(0, Math.floor(x - radius))
        const offy = Math.max(0, Math.floor(y - radius))
        const limx = Math.min(Math.ceil(x + radius), this.#bitmap.Width - 1)
        const limy = Math.min(Math.ceil(y + radius), this.#bitmap.Height - 1)

        for (let xi = offx; xi <= limx; xi++) {
            for (let yi = offy; yi <= limy; yi++) {
                const color = lerp(
                    this.#bitmap.At(xi, yi),
                    brush,
                    Math.min(Math.hypot(xi - x, yi - y) / (radius + 1), 1),
                )

                this.set(xi, yi, color)
            }
        }
    }

    private setup_listeners(canvas: HTMLCanvasElement) {
        const BrushRadius = 1.5

        const state = {
            down: false,
            brush: MainBrush,
        }

        canvas.tabIndex = 0
        canvas.width = this.#bitmap.Width
        canvas.height = this.#bitmap.Height
        this.Reset()

        canvas.addEventListener("click", (evt: MouseEvent): void => {
            const x = (evt.offsetX - canvas.clientLeft) * canvas.width / canvas.clientWidth - .5
            const y = (evt.offsetY - canvas.clientTop) * canvas.height / canvas.clientHeight - .5
            this.stroke(x, y, BrushRadius, state.brush)
            this.identify()
        })

        canvas.addEventListener("contextmenu", (evt: MouseEvent): void => {
            evt.preventDefault()
        })

        canvas.addEventListener("mousemove", (evt: MouseEvent): void => {
            if (!state.down) {
                return
            }

            const x = (evt.offsetX - canvas.clientLeft) * canvas.width / canvas.clientWidth - .5
            const y = (evt.offsetY - canvas.clientTop) * canvas.height / canvas.clientHeight - .5
            this.stroke(x, y, BrushRadius, state.brush)
            this.identify()
        })

        canvas.addEventListener("mousedown", (evt: MouseEvent): void => {
            if (evt.button === 0) {
                state.brush = this.Brush()
            } else if (evt.button === 2) {
                state.brush = 0xFFFFFF - this.Brush()
            }

            state.down = true
        })

        canvas.addEventListener("mouseup", (): void => {
            state.down = false
        })
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