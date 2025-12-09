from ultralytics import YOLO

# Load a YOLO model
model = YOLO("/home/chhlee28/traffic/traffic-sign-localization/newultralytics/runs/detect/train10/weights/best.pt")

# Validate on separate data
model.val(
    data="/home/chhlee28/traffic/traffic-sign-localization/newultralytics/dfgtest.yaml",
    project="/home/chhlee28/traffic/results",
    name="dfg_val",
)
