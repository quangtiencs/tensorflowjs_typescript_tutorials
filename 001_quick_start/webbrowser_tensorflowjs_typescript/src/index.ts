import { Tensor, tensor2d, Tensor2D } from "@tensorflow/tfjs"

let t1: Tensor2D = tensor2d([[1, 2], [3, 4], [5, 6]])
let t2: Tensor2D = tensor2d([[1, 2, 3], [4, 5, 6]])

console.log("Tensor 1")
t1.print()
let dom_1 = document.querySelector("#place_holder_1")
if (dom_1 != null) {
    dom_1.innerHTML = t1.toString()
}

t1.toString()
console.log("Tensor 2")
t2.print()
let dom_2 = document.querySelector("#place_holder_2")
if (dom_2 != null) {
    dom_2.innerHTML = t2.toString()
}

console.log("Dot product")

let t3: Tensor = t1.dot(t2)
t3.print()
let dom_3 = document.querySelector("#place_holder_3")
if (dom_3 != null) {
    dom_3.innerHTML = "Dot Product: " + t1.toString()
}

t1.dispose()
t2.dispose()
console.log(t1.isDisposed)
console.log(t2.isDisposed)
console.log(t3.isDisposed)