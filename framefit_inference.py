import numpy as np
from datasets import load_dataset
from PIL import Image
from setfit import SetFitModel

dataset = load_dataset("mnist")

model = SetFitModel.from_pretrained("setfit-mnist")

# Run inference
x = 0
preds = model.predict([Image.fromarray(np.array(dataset["train"]["image"][x]))])
print(preds == dataset["train"]["label"][x])
