from datasets import load_dataset
from setfit import FrameFitTrainer, SetFitModel

dataset = load_dataset("mnist")
model = SetFitModel.from_pretrained("clip-ViT-B-32")

dataset["train"] = dataset["train"].shuffle().select(list(range(100)))
dataset["test"] = dataset["test"].shuffle().select(list(range(20)))

trainer = FrameFitTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
trainer.train()
trainer.model.save_pretrained("setfit-mnist")

model = SetFitModel.from_pretrained("setfit-mnist")
# Run inference
preds = model.predict(dataset["test"][0])
