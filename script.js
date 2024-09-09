import { MnistData } from './data.js';

async function showExamples(data) {
    // Create a container in the visor
    const surface =
        tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data' });

    // Get the examples
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
        const imageTensor = tf.tidy(() => {
            // Reshape the image to 28x28 px
            return examples.xs
                .slice([i, 0], [1, examples.xs.shape[1]])
                .reshape([28, 28, 1]);
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);

        imageTensor.dispose();
    }
}

async function loadModel() {
    const model = await tf.loadLayersModel('my-model.json');
    return model;
}

let model;
async function run() {
    const data = new MnistData();
    await data.load();
    // await showExamples(data);

    // let model;

    // Load the model if it exists, otherwise train and save a new one.
    try {
        model = await loadModel();
        console.log("Model loaded from storage.");
    } catch (e) {
        console.log("No saved model found. Training a new one.");
        model = getModel();
        await train(model, data);
    }

    // tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);

    // await showAccuracy(model, data);
    // await showConfusion(model, data);

    document.getElementById('predictButton').disabled = false;
}

document.addEventListener('DOMContentLoaded', run);

function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));


    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels];
}


async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = { name: 'Accuracy', tab: 'Evaluation' };
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

    labels.dispose();
}

async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
    tfvis.render.confusionMatrix(container, { values: confusionMatrix, tickLabels: classNames });

    labels.dispose();
}


// -------------------------------------------------------------

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Initialize the canvas background to white
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Stroke width adjuster
const strokeWidthInput = document.getElementById('strokeWidth');
const strokeWidthValue = document.getElementById('strokeWidthValue');

// Update the displayed stroke width value
// strokeWidthInput.addEventListener('input', () => {
//     strokeWidthValue.textContent = strokeWidthInput.value;
// });

// Drawing functions
// canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mousedown', () => {
    drawing = true;
    document.getElementById('predictionResult').innerHTML = '<h3>Draw a digit on the canvas and click the predict button</h3><h2>Predictions (Highest to Lowest):</h2>';
    ctx.beginPath(); // Start a new path when drawing begins
});
// canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.closePath(); // Close the path when drawing ends
});
canvas.addEventListener('mouseout', () => drawing = false);

canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!drawing) return;
    // const strokeWidth = parseInt(strokeWidthInput.value, 10); // Get the current stroke width
    // const strokeWidth = parseInt(50, 10);
    // Define the radial gradient
    // Center of the gradient (x1, y1), radius of the inner circle (r1)
    // Center of the gradient (x2, y2), radius of the outer circle (r2)

    // Set the gradient as the stroke style

    let strokeWidth = 40;
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    // let tempX = event.clientX - canvas.offsetLeft;
    // let tempY = event.clientY - canvas.offsetTop;
    // let gradient = ctx.createRadialGradient(
    //     tempX, tempY, 20, // Inner circle
    //     tempX, tempY, 40 // Outer circle
    // );
    // ctx.strokeStyle = gradient;

    // Add color stops to the gradient with varying opacity
    // gradient.addColorStop(0, 'rgba(0, 0, 0, 1)'); // Fully opaque black in the middle
    // gradient.addColorStop(1, 'rgba(0, 0, 0, 0)'); // Fully transparent black on the outside

    // console.log(`before: ${tempX}, ${tempY}`);
    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    // let tempX2 = event.clientX - canvas.offsetLeft;
    // let tempY2 = event.clientY - canvas.offsetTop;
    // console.log(`after: ${tempX2}, ${tempY2}`);
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);



    // strokeWidth = 40;
    // ctx.lineWidth = strokeWidth;
    // ctx.lineCap = 'round';
    // ctx.strokeStyle = 'grey';

    // ctx.lineTo(tempX, tempY);
    // ctx.stroke();
    // ctx.beginPath();
    // ctx.moveTo(tempX2, tempY2);
}

// Clear the canvas
document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictionResult').innerHTML = '<h3>Draw a digit on the canvas and click the predict button</h3><h2>Predictions (Highest to Lowest):</h2>';
});

document.getElementById('predictButton').addEventListener('click', async () => {
    if (isCanvasBlank(canvas)) {
        alert('Please draw something on the canvas before predicting.');
        return;
    }

    const tensor = preprocessCanvasImage(canvas);
    // const model = await loadModel();
    const prediction = model.predict(tensor);
    const predictionArray = prediction.arraySync()[0]; // Get the array of predictions

    // Create an array of prediction objects with index and value
    const predictionsWithIndex = predictionArray.map((value, index) => ({ index, value }));

    // Sort the array by value in descending order
    predictionsWithIndex.sort((a, b) => b.value - a.value);

    // Create a string to display all predictions
    let resultHtml = '<h3>Draw a digit on the canvas and click the predict button</h3><h2>Predictions (Highest to Lowest):</h2>';
    predictionsWithIndex.forEach((prediction, rank) => {
        const className = classNames[prediction.index];
        // resultHtml += `<p>${rank + 1}. ${className}: ${prediction.value.toFixed(4)}</p>`;
        if (rank == 0)
            resultHtml += `<h3>${rank + 1}. ${className}</h3>`;
        else
            resultHtml += `<p>${rank + 1}. ${className}</p>`;
    });

    document.getElementById('predictionResult').innerHTML = resultHtml;
});


// Function to check if the canvas is blank
function isCanvasBlank(canvas) {
    const blank = document.createElement('canvas');
    blank.width = canvas.width;
    blank.height = canvas.height;

    const blankCtx = blank.getContext('2d');
    blankCtx.fillStyle = "white";
    blankCtx.fillRect(0, 0, blank.width, blank.height);

    return canvas.toDataURL() === blank.toDataURL();
}


function preprocessCanvasImage(canvas) {
    return tf.tidy(() => {
        // Convert the canvas to a tensor and resize to 28x28
        let tensor = tf.browser.fromPixels(canvas)
            .resizeNearestNeighbor([28, 28])
            .mean(2) // Convert to grayscale by averaging along the color channel
            .expandDims(2) // Add a dimension to match the input shape [28, 28, 1]
            .expandDims(0) // Add a batch dimension to match the input shape [1, 28, 28, 1]
            .toFloat();

        tensor = tensor.div(255.0); // Normalize to [0, 1] range

        // Invert the colors
        tensor = tf.sub(1, tensor);

        return tensor;
    });
}
