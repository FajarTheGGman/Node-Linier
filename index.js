// Copyright © 2019 By Fajar Firdaus

// dependencies
var tf = require("@tensorflow/tfjs-node")
var style = require("colors")
var banner = require('jsome')
var r = require('readline')

// Create empty model
const model = tf.sequential()

var hiddenLayer = tf.layers.dense({
  units: 1,
  inputShape: [1]
})

// Add hidden layer
model.add(hiddenLayer)


// Compile the model
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError'
})

// Train The model
async function trainModel(){
  const trainData1 = tf.tensor2d([-1, 0, 1, 2, 3, 5], [6,1])
  const trainData2 = tf.tensor2d([-6, 5, 4, 9, -9, 1], [6,1])

  const train = await model.fit(trainData1, trainData2, {
    epochs: 100,
    shuffle: true
  })

  const userInput = r.createInterface({
    input: process.stdin,
    output: process.stdout
  })


  console.log("")
  console.log("-----------------------------------\n")
  console.log("[ Node-Linier By FajarTheGGman ]".bgBlue)
  console.log("")
  console.log("[ Linier Regression With Tensorflow.js ]".yellow)

  banner({
    "Coder" : "FajarTheGGman",
    "Github": "github.com/FajarTheGGman",
    "IG" : "FajarTheGGman"
  })

  userInput.question(style.green("[?] Input Values : "), (x) => {
    var data = parseInt(x)

    console.log('[!] Total Loss : '.red + train.history.loss[0] + "\n")
    console.log("[ --- The Results -- ]".rainbow)
    model.predict(tf.tensor2d([data], [1, 1])).print()
    process.exit()
  })
}
trainModel()

console.log(" ")
