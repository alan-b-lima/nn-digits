import Canvas, { AltBrush, MainBrush } from "./canvas.js"
import { AsyncTry } from "./module/errors/try.js"
import jsxmm from "./module/jsxmm/element.js"
import "./wasm/wasm_exec.js"

declare function identify(data: number[], width: number, height: number): number[] | null

async function main(): Promise<void> {
    await start_go()

    const canvas = document.querySelector<HTMLCanvasElement>("#canvas")
    if (canvas === null) {
        throw new Error("there must be a #canvas element")
    }

    const draw = new Canvas(canvas, identify)

    const brush_input = document.querySelector<HTMLElement>("#brush")
    if (brush_input !== null) {
        brush_input.addEventListener("click", () => {
            if (brush_input.classList.contains("alt")) {
                draw.SetBrush(MainBrush)
            } else {
                draw.SetBrush(AltBrush)
            }

            brush_input.classList.toggle("alt")
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
            jsxmm.Element("label", { className: "result", textContent: `${i + 1}` }, outputs[i]),
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