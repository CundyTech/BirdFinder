import { Asset } from 'expo-asset';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import { toByteArray } from 'base64-js';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

const TARGET_SIZE = 224;
const TOP_PREDICTIONS = 5;
const labels = require('../../assets/labels.json');

let sessionPromise;

async function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const modelAsset = Asset.fromModule(require('../../assets/bird_classifier_model.onnx'));
      await modelAsset.downloadAsync();
      if (!modelAsset.localUri) throw new Error('The bundled bird model could not be loaded.');

      return InferenceSession.create(modelAsset.localUri, {
        executionProviders: ['cpu'],
        intraOpNumThreads: 2,
      });
    })().catch((error) => {
      sessionPromise = null;
      throw error;
    });
  }
  return sessionPromise;
}

export async function warmLocalInference() {
  await getSession();
  return true;
}

export async function predictBirdOnDevice(imageUri) {
  const session = await getSession();

  const resized = await manipulateAsync(
    imageUri,
    [{ resize: { width: TARGET_SIZE, height: TARGET_SIZE } }],
    { compress: 0.9, format: SaveFormat.JPEG, base64: true }
  );

  if (!resized.base64) throw new Error('The photo could not be prepared for the model.');

  const jpegBytes = toByteArray(resized.base64);
  const pixels = decodeJpeg(jpegBytes, {
    useTArray: true,
    formatAsRGBA: true,
  });

  const floatPixels = new Float32Array(TARGET_SIZE * TARGET_SIZE * 3);
  let outputIndex = 0;
  for (let pixelIndex = 0; pixelIndex < pixels.data.length; pixelIndex += 4) {
    floatPixels[outputIndex++] = (pixels.data[pixelIndex] / 127.5) - 1;
    floatPixels[outputIndex++] = (pixels.data[pixelIndex + 1] / 127.5) - 1;
    floatPixels[outputIndex++] = (pixels.data[pixelIndex + 2] / 127.5) - 1;
  }

  const inputName = session.inputNames[0];
  const inputTensor = new Tensor('float32', floatPixels, [1, TARGET_SIZE, TARGET_SIZE, 3]);
  const output = await session.run({ [inputName]: inputTensor }, session.outputNames);
  const outputName = session.outputNames[0];
  const outputTensor = output[outputName];
  const scores = Array.from(await outputTensor.getData());

  if (scores.length !== labels.length) {
    throw new Error(`Model returned ${scores.length} scores for ${labels.length} bird classes.`);
  }

  const ranked = scores
    .map((score, index) => ({ class: labels[index], score }))
    .sort((a, b) => b.score - a.score);
  const topPredictions = ranked.slice(0, TOP_PREDICTIONS);

  return {
    predicted_class: topPredictions[0].class,
    scores,
    top_predictions: topPredictions,
  };
}
