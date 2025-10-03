from fastapi import FastAPI
import uvicorn
import os
from minio import Minio
from minio.error import S3Error
from rfdetr import RFDETRNano
import cv2
import numpy as np
import tempfile
import shutil

app = FastAPI(
    title="Object Detection API",
    description="API for processing images and detecting objects.",
    version="0.1.0"
)

# MinIO Configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "miniodevpassword")

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
except S3Error as e:
    print(f"Error connecting to MinIO: {e}")
    minio_client = None

# Initialize RF-DETR Model
try:
    rfdetr_model = RFDETRNano()
    print("RF-DETR Nano model loaded successfully.")
except Exception as e:
    print(f"Error loading RF-DETR Nano model: {e}")
    rfdetr_model = None

from fastapi import FastAPI, HTTPException
import uvicorn
import os
from minio import Minio
from minio.error import S3Error
from rfdetr import RFDETRNano
import cv2
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any
import json
from datetime import datetime

app = FastAPI(
    title="Object Detection API",
    description="API for processing images and detecting objects.",
    version="0.1.0"
)

# MinIO Configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "miniodevpassword")
RESULTS_BUCKET_NAME = "detection-results"

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
except S3Error as e:
    print(f"Error connecting to MinIO: {e}")
    minio_client = None

# Initialize RF-DETR Model
try:
    rfdetr_model = RFDETRNano()
    print("RF-DETR Nano model loaded successfully.")
except Exception as e:
    print(f"Error loading RF-DETR Nano model: {e}")
    rfdetr_model = None

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Object Detection service is running."}

@app.post("/detect/video_file")
async def detect_objects_in_video_file(
    bucket_name: str,
    object_name: str
) -> Dict[str, Any]:
    """Detects objects in a video file stored in MinIO and saves results to MinIO."""
    if not minio_client:
        raise HTTPException(status_code=500, detail="MinIO client not initialized.")
    if not rfdetr_model:
        raise HTTPException(status_code=500, detail="RF-DETR model not loaded.")

    temp_dir = None
    try:
        # 1. Create a temporary directory for the video file
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, object_name)

        # 2. Download video from MinIO
        try:
            minio_client.fget_object(bucket_name, object_name, video_path)
        except S3Error as e:
            raise HTTPException(status_code=404, detail=f"Error downloading video from MinIO: {e}")

        # 3. Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Error opening video file.")

        results = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RF-DETR expects RGB images, OpenCV reads BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference
            detections = rfdetr_model.predict(images=[rgb_frame])

            frame_results = []
            if detections and len(detections) > 0:
                for i in range(len(detections[0].xyxy)):
                    x1, y1, x2, y2 = detections[0].xyxy[i]
                    class_id = detections[0].class_id[i]
                    confidence = detections[0].confidence[i]
                    frame_results.append({
                        "frame": frame_count,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class_id": int(class_id),
                        "confidence": float(confidence)
                    })
            results.append(frame_results)
            frame_count += 1

        cap.release()

        # 4. Save results to MinIO
        results_json = json.dumps(results, indent=2)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        results_object_name = f"{os.path.splitext(object_name)[0]}_detection_results_{timestamp}.json"

        # Ensure results bucket exists
        found = minio_client.bucket_exists(RESULTS_BUCKET_NAME)
        if not found:
            minio_client.make_bucket(RESULTS_BUCKET_NAME)

        minio_client.put_object(
            RESULTS_BUCKET_NAME,
            results_object_name,
            data=io.BytesIO(results_json.encode('utf-8')),
            length=len(results_json.encode('utf-8')),
            content_type='application/json'
        )

        return {
            "status": "success",
            "message": "Object detection completed and results saved to MinIO.",
            "results_bucket": RESULTS_BUCKET_NAME,
            "results_object": results_object_name
        }

    finally:
        # 5. Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
