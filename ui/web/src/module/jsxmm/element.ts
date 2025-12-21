namespace jsxmm {
    type Properties<T extends keyof HTMLElementTagNameMap> = Partial2<HTMLElementTagNameMap[T]> & Record<string, any>

    export function Element<T extends keyof HTMLElementTagNameMap>(tag: T, properties: Properties<T> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[T] {
        const element = document.createElement(tag)
        replace(element, properties)

        for (let i = 0; i < children.length; i++) {
            const child = children[i]
            element.append(child)
        }

        return element
    }

    export function Style(element: HTMLElement, style: Partial<CSSStyleDeclaration>): void {
        replace(element.style, style)
    }

    export function Assign<T extends HTMLElement>(element: T, properties: Partial2<T>): void {
        replace(element, properties)
    }

    function replace(base: Record<PropertyKey, any>, replacement: Record<PropertyKey, any>): void {
        for (const key in replacement) {
            if (!(key in base)) {
                console.error(`${key} not present in ${base} element`)
                // continue regardless, that's a problem for the caller
            }

            if (typeof replacement[key] === "object") {
                replace(base[key], replacement[key])
            } else {
                base[key] = replacement[key]
            }
        }
    }

    type Partial2<T> = { [K in keyof T]?: T[K] extends Object ? Partial<T[K]> : T[K] }
}

export default jsxmm