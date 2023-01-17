import { Tensor, tensor2d, Tensor2D } from "@tensorflow/tfjs-node"
import * as tf from '@tensorflow/tfjs-node';


console.log(`Memory before create tensor: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)
var t1: Tensor2D | null = tensor2d([[1, 2], [3, 4], [5, 6]])
console.log(t1)
console.log(`Memory before after create tensor: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)
t1.dispose();
console.log(`Memory before after .dispose tensor: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)
console.log("Cheers!")