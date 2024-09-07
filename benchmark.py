from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

# Load the models
pretrained_model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8n model
dsar_model = YOLO(os.path.join(ROOT_DIR, "dsar_yolo_v8n.pt"))  # Custom-trained YOLOv8 model

# Validate both models and extract metrics
pretrained_metrics = pretrained_model.val(data=os.path.join(ROOT_DIR, "yolo_v8_benchmark_config.yaml"), imgsz='1280')
dsar_metrics = dsar_model.val(data=os.path.join(ROOT_DIR, "yolo_v8_benchmark_config.yaml"), imgsz='1280')

# Extract the desired metrics for pre-trained YOLOv8n
pretrained_map_50_95 = pretrained_metrics.box.map  # mAP@0.5:0.95
pretrained_map_50 = pretrained_metrics.box.map50   # mAP@0.5
pretrained_map_75 = pretrained_metrics.box.map75   # mAP@0.75
pretrained_class_wise_map = pretrained_metrics.box.maps  # Class-wise mAP

# Extract the desired metrics for custom YOLOv8 model
dsar_map_50_95 = dsar_metrics.box.map  # mAP@0.5:0.95
dsar_map_50 = dsar_metrics.box.map50   # mAP@0.5
dsar_map_75 = dsar_metrics.box.map75   # mAP@0.75
dsar_class_wise_map = dsar_metrics.box.maps  # Class-wise mAP

# Prepare data for plotting
models = ['Pre-trained YOLOv8n', 'DSaR YOLOv8n']
map_50_values = [pretrained_map_50, dsar_map_50]
map_50_95_values = [pretrained_map_50_95, dsar_map_50_95]
map_75_values = [pretrained_map_75, dsar_map_75]

# Create the bar chart to compare mAP@0.5, mAP@0.5:0.95, and mAP@0.75
plt.figure(figsize=(12, 6))

# Plot mAP@0.5 comparison
plt.subplot(1, 3, 1)
plt.bar(models, map_50_values, color='skyblue')
plt.ylabel('mAP@0.5')
plt.title('mAP@0.5 Comparison')

# Plot mAP@0.5:0.95 comparison
plt.subplot(1, 3, 2)
plt.bar(models, map_50_95_values, color='salmon')
plt.ylabel('mAP@0.5:0.95')
plt.title('mAP@0.5:0.95 Comparison')

# Plot mAP@0.75 comparison
plt.subplot(1, 3, 3)
plt.bar(models, map_75_values, color='lightgreen')
plt.ylabel('mAP@0.75')
plt.title('mAP@0.75 Comparison')

plt.tight_layout()

plt.savefig('dsar_yolov8n_mAP_comparison.jpg', format='jpg', dpi=300)

plt.show()
