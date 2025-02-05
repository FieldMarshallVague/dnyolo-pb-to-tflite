import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path, override=True)
env_vars = dict(os.environ)

CALIBRATION_OUTPUT_DIR = env_vars.get("CALIBRATION_OUTPUT_DIR")
SAVED_MODEL_DIR = env_vars.get("SAVED_MODEL_DIR")

# Define a representative dataset generator
def representative_dataset(prepared_samples):
    for sample in prepared_samples:
        yield [sample]

# inspect model's input signature
model = tf.saved_model.load(SAVED_MODEL_DIR)
concrete_func = model.signatures['serving_default']
for op in concrete_func.graph.get_operations():
    print(op.name, op.type)
input_tensor_spec = list(concrete_func.structured_input_signature[1].values())[0]

print("Expected shape:", input_tensor_spec.shape)
print("Expected dtype:", input_tensor_spec.dtype)

def get_calibration_data():
    calibration_samples = np.load(os.join(CALIBRATION_OUTPUT_DIR, 'calib_data.npy'))
    prepared_samples = []

    print(f"Calibration Sample [0]: Shape {calibration_samples[0].shape}")  # Debugging

    for i, sample in enumerate(calibration_samples):
        sample_with_batch = np.expand_dims(sample, axis=0)  # e.g. NHWC
        sample_with_batch = np.transpose(sample_with_batch, (0, 3, 1, 2))  # e.g. Convert from (1, H, W, C) to (1, C, H, W)
        sample_with_batch = sample_with_batch.astype(np.float32)
        print(f"Calibration Sample {i}: Shape {sample_with_batch.shape}")  # Debugging
        prepared_samples.append(sample_with_batch)

    return prepared_samples

def convert_fp32():
    """
        Float 32 conversion without quantization (for testing/confirming).
        If this model doesn't convert, or doesn't work once converted, you got issues.
        Might be your model's operations aren't supported in TFLITE or the tflite conversion can't handle them gracefully.
        Saw somewhere that it might be possible to re-wire some ops yourself, but this is outside my knowledge.

    """
    converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_fp32.optimizations = []
    tflite_model_fp32 = converter_fp32.convert()
    with open('model_fp32.tflite', 'wb') as f:
        f.write(tflite_model_fp32)

def convert_int8():
    """Full integer quantization (int8)"""

    calib_data = get_calibration_data()
    
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.representative_dataset = lambda: representative_dataset(calib_data)
    converter_int8.inference_input_type = tf.int8   # or tf.uint8 as needed
    converter_int8.inference_output_type = tf.int8
    tflite_model_int8 = converter_int8.convert()
    with open('model_int8.tflite', 'wb') as f:
        f.write(tflite_model_int8)

def convert_int8_with_int16_activations():
    """Integer quantization with int8 weights and int16 activations"""

    calib_data = get_calibration_data()

    converter_int8_int16 = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_int8_int16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8_int16.representative_dataset = lambda: representative_dataset(calib_data)
    # Instruct the converter to allow int16 for activations.
    converter_int8_int16.target_spec.supported_types = [tf.int16]
    tflite_model_int8_int16 = converter_int8_int16.convert()
    with open('model_int8_int16.tflite', 'wb') as f:
        f.write(tflite_model_int8_int16)

def main():
    convert_fp32()
    convert_int8()
    convert_int8_with_int16_activations()

main()