import cv2
import glob
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path, override=True)
env_vars = dict(os.environ)

CALIBRATION_INPUT_DATA_GLOB = env_vars.get("CALIBRATION_INPUT_DATA_GLOB")
calibration_files = glob.glob(CALIBRATION_INPUT_DATA_GLOB)
CALIBRATION_OUTPUT_DIR = env_vars.get("CALIBRATION_OUTPUT_DIR")
if not os.path.exists(CALIBRATION_OUTPUT_DIR):
    os.makedirs(CALIBRATION_OUTPUT_DIR)
NETWORK_WIDTH=env_vars.get("NETWORK_WIDTH")
NETWORK_HEIGHT=env_vars.get("NETWORK_HEIGHT")

print(f"num files: {len(calibration_files)}")

img_datas = []
for idx, file in enumerate(calibration_files):
  bgr_img = cv2.imread(file)
  rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
  resized_img = cv2.resize(rgb_img, dsize=(NETWORK_WIDTH,NETWORK_HEIGHT))
  extend_batch_size_img = resized_img[np.newaxis, :]
  normalized_img = extend_batch_size_img / 255 # 0.0 - 1.0
  floats = np.array(normalized_img).astype(np.float32)
  print(
    f'{str(idx+1).zfill(2)}. extend_batch_size_img.shape: {extend_batch_size_img.shape}'
  )
  img_datas.append(floats)

calib_data = np.vstack(img_datas)
print(f'calib_datas.shape: {calib_data.shape}') 

# calc Mean
mean = np.mean(calib_data, axis=(0,1,2))
print(f"mean: {mean}")

# calc Standard Deviation
std = np.std(calib_data, axis=(0,1,2))
print(f"std: {std}")

# Save file to output folder
calib_data_file = os.join(CALIBRATION_OUTPUT_DIR, 'calib_data.npy')
np.save(file=calib_data_file, arr=calib_data)

# confirm saved file loads, show shape
loaded_data = np.load(calib_data_file)
print(f'loaded_data.shape: {loaded_data.shape}')
