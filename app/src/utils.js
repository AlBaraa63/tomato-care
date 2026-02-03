import * as ort from 'onnxruntime-web';

// IMPORTANT: Configure WASM paths to look in the root of the public/dist folder
ort.env.wasm.wasmPaths = '/';

export const CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
];

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export async function preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224).data;
    const red = new Float32Array(224 * 224);
    const green = new Float32Array(224 * 224);
    const blue = new Float32Array(224 * 224);

    for (let i = 0; i < imageData.length; i += 4) {
        const pixelIndex = i / 4;
        red[pixelIndex] = (imageData[i] / 255 - MEAN[0]) / STD[0];
        green[pixelIndex] = (imageData[i + 1] / 255 - MEAN[1]) / STD[1];
        blue[pixelIndex] = (imageData[i + 2] / 255 - MEAN[2]) / STD[2];
    }

    const floatData = new Float32Array(3 * 224 * 224);
    floatData.set(red, 0);
    floatData.set(green, 224 * 224);
    floatData.set(blue, 2 * 224 * 224);

    return new ort.Tensor('float32', floatData, [1, 3, 224, 224]);
}

export async function runInference(modelPath, tensor) {
    try {
        console.log("Loading model from:", modelPath);
        // Use 'wasm' as a safer default if WebGL is tricky on some systems
        const session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm'],
        });

        console.log("Running session...");
        const inputs = { input: tensor };
        const outputs = await session.run(inputs);

        // Some models might name the output differently, let's find it if 'output' fails
        const outputKey = Object.keys(outputs)[0];
        const output = outputs[outputKey].data;

        // Softmax
        const expValues = output.map(Math.exp);
        const sumExp = expValues.reduce((a, b) => a + b, 0);
        const probabilities = expValues.map(v => v / sumExp);

        let maxProb = -1;
        let maxIdx = -1;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIdx = i;
            }
        }

        return {
            label: CLASS_NAMES[maxIdx],
            confidence: maxProb,
            allProbabilities: probabilities
        };
    } catch (err) {
        console.error("Inference Detail Error:", err);
        throw err;
    }
}
