# Darknet Yolo conversions to TFLITE with integer quantization.

This contains scripts to convert a .pb model to a tflite model.  I started with the onnx2tf repo by pinto0309, but after struggling with the outputs, I started over with a minimal implementation, following the google tf lite docs.

https://github.com/PINTO0309/onnx2tf
https://ai.google.dev/edge/litert/models/model_optimization

## How to use:

NOTE: This repo assumes you have a the model already converted from .weights to a saved_model (PB) format, using LordOfKillz yolo4_pytorch repo.

https://github.com/lordofkillz/yolo4_pytorch

### Step 1:

Copy the .env.template to a new .env file.
Replace the values with correct ones for your setup.
I used 2500 example input images as my calibration set.  These were from the training data (which may be a mistake, not sure).
Run ```python calibrate.py``` and wait.

Note: Make a note of the MEAN and STD values that are output at the end of calibration processing, if you want to try using the onnx2tf repo (which uses them as inputs for the conversion command when including calibration data) instead of this one for converting to tflite with quantization. 

### Step 2:

__Option 1:__
Run convert.py as-is (generates 3 models, takes a while)
__Option2:__
comment out the function calls in conert.py's main function (at the bottom).  Run only fp32 test first, then others if that works.

Models are saved to the MODELS_DIR folder set in your .env file.

