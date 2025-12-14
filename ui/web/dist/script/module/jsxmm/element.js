var jsxmm;
(function (jsxmm) {
    function Element(tag, properties = {}, ...children) {
        const element = document.createElement(tag);
        replace(element, properties);
        for (let i = 0; i < children.length; i++) {
            const child = children[i];
            element.append(child);
        }
        return element;
    }
    jsxmm.Element = Element;
    function Style(element, style) {
        replace(element.style, style);
    }
    jsxmm.Style = Style;
    function replace(base, replacement) {
        for (const key in replacement) {
            if (!(key in base)) {
                console.error(`${key} not present in ${base} element`);
            }
            if (typeof replacement[key] === "object") {
                replace(base[key], replacement[key]);
            }
            else {
                base[key] = replacement[key];
            }
        }
    }
})(jsxmm || (jsxmm = {}));
export default jsxmm;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZWxlbWVudC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uL3NyYy9tb2R1bGUvanN4bW0vZWxlbWVudC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSxJQUFVLEtBQUssQ0FtQ2Q7QUFuQ0QsV0FBVSxLQUFLO0lBR1gsU0FBZ0IsT0FBTyxDQUF3QyxHQUFNLEVBQUUsYUFBNEIsRUFBRSxFQUFFLEdBQUcsUUFBMkI7UUFDakksTUFBTSxPQUFPLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxHQUFHLENBQUMsQ0FBQTtRQUMzQyxPQUFPLENBQUMsT0FBTyxFQUFFLFVBQVUsQ0FBQyxDQUFBO1FBRTVCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxRQUFRLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDdkMsTUFBTSxLQUFLLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFBO1lBQ3pCLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUE7UUFDekIsQ0FBQztRQUVELE9BQU8sT0FBTyxDQUFBO0lBQ2xCLENBQUM7SUFWZSxhQUFPLFVBVXRCLENBQUE7SUFFRCxTQUFnQixLQUFLLENBQUMsT0FBb0IsRUFBRSxLQUFtQztRQUMzRSxPQUFPLENBQUMsT0FBTyxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQTtJQUNqQyxDQUFDO0lBRmUsV0FBSyxRQUVwQixDQUFBO0lBRUQsU0FBUyxPQUFPLENBQUMsSUFBOEIsRUFBRSxXQUFxQztRQUNsRixLQUFLLE1BQU0sR0FBRyxJQUFJLFdBQVcsRUFBRSxDQUFDO1lBQzVCLElBQUksQ0FBQyxDQUFDLEdBQUcsSUFBSSxJQUFJLENBQUMsRUFBRSxDQUFDO2dCQUNqQixPQUFPLENBQUMsS0FBSyxDQUFDLEdBQUcsR0FBRyxtQkFBbUIsSUFBSSxVQUFVLENBQUMsQ0FBQTtZQUUxRCxDQUFDO1lBRUQsSUFBSSxPQUFPLFdBQVcsQ0FBQyxHQUFHLENBQUMsS0FBSyxRQUFRLEVBQUUsQ0FBQztnQkFDdkMsT0FBTyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxXQUFXLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQTtZQUN4QyxDQUFDO2lCQUFNLENBQUM7Z0JBQ0osSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxHQUFHLENBQUMsQ0FBQTtZQUNoQyxDQUFDO1FBQ0wsQ0FBQztJQUNMLENBQUM7QUFHTCxDQUFDLEVBbkNTLEtBQUssS0FBTCxLQUFLLFFBbUNkO0FBRUQsZUFBZSxLQUFLLENBQUEifQ==