import { Tensor, Tensor2D, tensor2d } from "@tensorflow/tfjs";
import embed from "vega-embed";
import * as tf from "@tensorflow/tfjs";

try {
  // I know. I'm lazy :-D
  tf.setBackend("webgl");
} catch (err) {
  console.log(err);
}

document.querySelector("#device")!!.innerHTML =
  `Backend by: ${tf.getBackend()}`;

function make_synthetic_data(true_w: number, true_b: number) {
  let x: tf.Tensor1D = tf.randomUniform([500], 0.0, 2.0);
  let noise = tf.randomNormal([500], 0, 0.8);
  let y = x.mul(tf.scalar(true_w)).add(tf.scalar(true_b)).add(noise);

  return { "x": x, "y": y };
}

let data = make_synthetic_data(2, 3);

let vegaData: { x: number; y: number }[] = [];
for (let i = 0; i < data.x.size; i++) {
  vegaData.push({ "x": data.x.dataSync()[i], "y": data.y.dataSync()[i] });
}

function quantileLoss90(yTrue: tf.Tensor, yPred: tf.Tensor) {
  let error = yTrue.sub(yPred);
  let alpha = 0.9;
  return tf.maximum(error.mul(alpha), error.mul(alpha - 1.0)).mean();
}

let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
model.add(tf.layers.dense({ units: 1 }));

model.compile({
  optimizer: "sgd",
  loss: quantileLoss90,
});

model.summary();

model.fit(data.x, data.y, { epochs: 100, batchSize: 64 }).then((history) => {
  document.querySelector("#status")!!.innerHTML = "Completed :D";
  document.querySelector("#status")!!.setAttribute("style", "color: green");

  const prediction = (model.predict(data.x) as tf.Tensor)
    .dataSync();
  //   console.log(`data sync ${prediction}`);

  console.log("[Train Model] Completed!");
  console.log("Loss Here :D");
  console.log(history);

  let finalPlotData = [];
  for (let i = 0; i < data.x.size; i++) {
    finalPlotData.push({
      "x": data.x.dataSync()[i],
      "y": data.y.dataSync()[i],
      "y_hat": prediction[i],
    });
  }

  // @ts-ignore
  var specquantile = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 400,
    "height": 200,
    data: {
      values: finalPlotData,
    },
    "layer": [{
      "mark": "circle",
      "encoding": {
        "x": { "field": "x", "type": "quantitative" },
        "y": { "field": "y", "type": "quantitative" },
      },
    }, {
      "mark": "line",
      "encoding": {
        "x": { "field": "x", "type": "quantitative" },
        "y": { "field": "y_hat", "type": "quantitative" },
        "color": { "value": "red" },
      },
    }],
  };

  // @ts-ignore
  embed("#quantileregression", specquantile);
});

var specDataPoint = {
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 400,
  "height": 200,
  data: {
    values: vegaData,
  },
  "mark": "circle",
  "encoding": {
    "x": { "field": "x", "type": "quantitative" },
    "y": { "field": "y", "type": "quantitative" },
  },
};

// @ts-ignore
embed("#vis", specDataPoint);
