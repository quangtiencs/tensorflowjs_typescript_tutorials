import { Tensor, tensor2d, Tensor2D } from "@tensorflow/tfjs-node"

let t1: Tensor2D = tensor2d([[1, 2], [3, 4], [5, 6]])
let t2: Tensor2D = tensor2d([[1, 2, 3], [4, 5, 6]])

console.log("Tensor 1")
t1.print()

console.log("Tensor 2")
t2.print()

console.log("Dot product")

let t3: Tensor = t1.dot(t2)
t3.print()

t1.dispose()
t2.dispose()
console.log(t1.isDisposed)
console.log(t2.isDisposed)
console.log(t3.isDisposed)