import os
import argparse
import tensorflow as tf

def write_labels(labels_dir, out_path):
    classes = sorted([d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))])
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in classes:
            f.write(c + '\n')
    print(f'Wrote {len(classes)} labels to {out_path}')

def convert(h5_path, out_tflite, labels_dir, out_labels, quantize=False):
    print('Loading model:', h5_path)
    model = tf.keras.models.load_model(h5_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        print('Applying default optimizations (post-training quantization)')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_tflite, 'wb') as f:
        f.write(tflite_model)
    print('Wrote tflite model to', out_tflite)
    write_labels(labels_dir, out_labels)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5', default='/work/model.h5', help='Path to input Keras .h5 model')
    p.add_argument('--out', default='/work/model.tflite', help='Output tflite path')
    p.add_argument('--labels-dir', default='/work/labels_dir', help='Directory with class subfolders')
    p.add_argument('--out-labels', default='/work/labels.txt', help='Output labels.txt')
    p.add_argument('--quantize', action='store_true', help='Enable default post-training quantization')
    args = p.parse_args()
    convert(args.h5, args.out, args.labels_dir, args.out_labels, quantize=args.quantize)
