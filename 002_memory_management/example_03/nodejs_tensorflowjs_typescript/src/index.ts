import * as tf from '@tensorflow/tfjs-node';

console.log(`Memory init: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)

var keepTensor: tf.Tensor | null = null;

const returnTensor = tf.tidy(() => {
    const one = tf.scalar(1);
    const a = tf.scalar(16);

    keepTensor = tf.keep(a.square());

    console.log(`Memory inside tidy: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)

    return keepTensor.sub(one);
});

console.log(`Memory outside tidy: ${tf.memory().numTensors} tensors, ${tf.memory().numBytes} bytes`)

console.log('Return Tensor:');
returnTensor.print();
console.log('Keep Tensor:');
keepTensor!!.print();