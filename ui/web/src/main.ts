import Canvas, { AltBrush, BrushRadius, MainBrush } from "./canvas.js"
import { AsyncTry } from "./module/errors/try.js"
import jsxmm from "./module/jsxmm/element.js"
import "./wasm/wasm_exec.js"

declare function load(json: string): boolean
declare function classify(data: number[], width: number, height: number): number[] | null

async function main(): Promise<void> {
    await start_go()

    const response = await fetch("../../../.data/nn/nn-1.json")
    if (!response.ok) {
        throw new Error("failed to retrieve neural network")
    }

    const data = await response.text()
    if (!load(data)) {
        throw new Error("failed to retrieve neural network")
    }

    const canvas = document.querySelector<HTMLCanvasElement>("#canvas")
    if (canvas === null) {
        throw new Error("there must be a #canvas element")
    }

    const draw = new Canvas(canvas, classify)

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

async function start_go(): Promise<void> {
    const [response, error] = await AsyncTry(() => fetch("./script/wasm/main.wasm"))
    if (error !== null) {
        throw new Error("Go Wasm module not found")
    }

    const go = new Go()
    const result = await WebAssembly.instantiateStreaming(response, go.importObject)
    go.run(result.instance)
}

window.addEventListener("DOMContentLoaded", main, { once: true })