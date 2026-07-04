import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


def download_model():
    """Download pre-trained violence detection model from HuggingFace."""
    print("Downloading violence detection model (first run only)...")
    model_path = hf_hub_download(
        repo_id="Musawer14/fight_detection_yolov8",
        filename="yolo_small_weights.pt"
    )
    print(f"Model ready: {model_path}")
    return model_path

class ViolenceDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = download_model()
        self.model = YOLO(model_path)
        os.makedirs("incidents", exist_ok=True)


    def detect(self, frame, confidence=0.5):
        results = self.model(frame, conf=confidence, device="cpu", verbose=False)
        detections = []
        annotated_frame = frame.copy()

        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Skip "no violence" or "no fight" class
                if "no" in class_name.lower():
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class": class_name
                })

                # Draw red bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{class_name}: {conf:.0%}"
                label_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )[0]
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )

        return annotated_frame, detections

    def save_incident(self, frame, detections):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"incidents/incident_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "num_detections": len(detections),
            "max_confidence": max(d["confidence"] for d in detections)
        }