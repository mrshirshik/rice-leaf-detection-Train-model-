from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("best.pt")

# Image to predict
image_path = "test.jpg"
results = model(image_path, save=True, conf=0.25)  # default is 0.25, try 0.1 or 0.05 if needed



# ✅ Print detections
print("Detection results:")
for r in results:
    print(r.boxes)

# ✅ Optional: Show saved image
output_path = f"runs/detect/predict/{image_path}"
img = cv2.imread(output_path)

if img is not None:
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load output image.")

