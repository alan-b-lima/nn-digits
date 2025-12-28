import Canvas, { AltBrush, BrushRadius, MainBrush } from "./canvas.js"
import { AsyncTry } from "./module/errors/try.js"
import jsxmm from "./module/jsxmm/element.js"
import "./wasm/wasm_exec.js"

declare function load(data: string): void | Error
declare function classify(data: number[]): number[] | Error

async function main(): Promise<void> {
    await start_go()

    const Root = root()

    const response = await fetch(new URL("nn/model/digits.json", Root))
    if (!response.ok) {
        throw new Error("retrieve neural network")
    }

    const neural_network = await response.text()
    go(load)(neural_network)

    const canvas = document.querySelector<HTMLCanvasElement>("#canvas")
    if (canvas === null) {
        throw new Error("there must be a #canvas element")
    }

    const draw = new Canvas(canvas, go(classify))

    const undo_input = document.querySelector<HTMLElement>("#undo")
    if (undo_input !== null) {
        undo_input.addEventListener("click", Canvas.prototype.Undo.bind(draw))
    }

    const redo_input = document.querySelector<HTMLElement>("#redo")
    if (redo_input !== null) {
        redo_input.addEventListener("click", Canvas.prototype.Redo.bind(draw))
    }

    const brush_input = document.querySelector<HTMLElement>("#brush")
    if (brush_input !== null) {
        brush_input.addEventListener("click", (evt: Event) => {
            if (evt.target !== evt.currentTarget) {
                return
            }

            if (brush_input.classList.contains("alt")) {
                draw.SetBrush(MainBrush)
            } else {
                draw.SetBrush(AltBrush)
            }

            brush_input.classList.toggle("alt")
        })
    }

    const radius_input = document.querySelector<HTMLInputElement>("#radius")
    if (radius_input !== null) {
        jsxmm.Assign(radius_input, {
            min: ".5", step: ".5", max: "3",
            value: "1",
        })

        radius_input.addEventListener("click", (evt: Event) => {
            let radius = +radius_input.value
            if (Number.isNaN(radius)) {
                radius = BrushRadius
            }

            draw.SetRadius(radius)
        })
    }

    const clear_input = document.querySelector<HTMLElement>("#clear")
    if (clear_input !== null) {
        clear_input.addEventListener("click", Canvas.prototype.Reset.bind(draw))
    }

    const result_input = document.querySelector<HTMLElement>("#result")
    if (result_input === null) {
        throw new Error("there must be a #result element")
    }

    const outputs = draw.Outputs()
    for (let i = 0; i < outputs.length; i++) {
        result_input.append(
            jsxmm.Element("label", { className: "result", textContent: `${i}` }, outputs[i]),
        )
    }
}

function root(): string {
    if (location.origin === "https://alan-b-lima.github.io") {
        return "https://alan-b-lima.github.io/nn-digits/"
    }

    return location.origin
}

async function start_go(): Promise<void> {
    const [response, error] = await AsyncTry(() => fetch("./script/wasm/main.wasm"))
    if (error !== null) {
        throw new Error("Go Wasm module not found")
    }

    const go = new Go()
    const result = await WebAssembly.instantiateStreaming(response, go.importObject)
    go.run(result.instance)
}

function go<T extends (...args: any) => any>(func: T): (...args: Parameters<T>) => Exclude<ReturnType<T>, Error> {
    return function (...args: Parameters<T>): ReturnType<T> {
        const ret = func(...args)
        if (ret instanceof Error) {
            throw ret
        }
        return ret
    } as any
}

window.addEventListener("DOMContentLoaded", main, { once: true })