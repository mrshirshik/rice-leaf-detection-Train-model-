from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="data.yaml")

print("Validation Results:")
print(metrics)
