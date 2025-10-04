from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
import os
from minio import Minio
from minio.error import S3Error
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES
import cv2
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any
import json
import io

app = FastAPI(
    title="Object Detection API",
    description="API for processing images and detecting objects.",
    version="0.1.0"
)

# --- Constants & Configuration ---
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "miniodevpassword")
USER_DATA_BUCKET_NAME = "user-data"
USER_OUT_BUCKET_NAME = "user-out"
USER_OUT_VISUAL_BUCKET_NAME = "user-out-visual"
BATCH_SIZE = 16 # Process 16 frames at a time for efficiency

# --- Service Initialization ---
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

try:
    rfdetr_model = RFDETRNano()
    print("RF-DETR Nano model loaded successfully.")
except Exception as e:
    print(f"Error loading RF-DETR Nano model: {e}")
    rfdetr_model = None

# --- Core Processing Logic ---
def process_batch_results(detections_batch, frame_batch, writer, all_results, start_frame_count):
    """Helper to process the results of a batch prediction."""
    for i, detections in enumerate(detections_batch):
        frame = frame_batch[i]
        frame_count = start_frame_count + i
        frame_results = []

        if detections and detections.xyxy is not None:
            for j in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[j]
                class_id = detections.class_id[j]
                confidence = detections.confidence[j]
                frame_results.append({
                    "frame": frame_count,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class_id": int(class_id),
                    "confidence": float(confidence)
                })

                # Draw on the frame for the visualized video
                label = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        all_results.append(frame_results)
        writer.write(frame)

def process_and_visualize_video(object_name: str):
    """Processes a video in batches, saves raw data, and creates a visualized video."""
    if not minio_client or not rfdetr_model:
        print("MinIO client or model not initialized. Skipping processing.")
        return

    temp_dir = tempfile.mkdtemp()
    try:
        input_video_path = os.path.join(temp_dir, object_name)
        output_video_path = os.path.join(temp_dir, f"viz_{object_name}")

        minio_client.fget_object(USER_DATA_BUCKET_NAME, object_name, input_video_path)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {object_name}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        all_results = []
        frame_count = 0
        frame_batch = []
        rgb_frame_batch = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_batch.append(frame)
            rgb_frame_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if len(frame_batch) == BATCH_SIZE:
                detections_batch = rfdetr_model.predict(images=rgb_frame_batch, threshold=0.5)
                process_batch_results(detections_batch, frame_batch, writer, all_results, frame_count)
                frame_count += len(frame_batch)
                frame_batch.clear()
                rgb_frame_batch.clear()

        # Process any remaining frames that didn't form a full batch
        if frame_batch:
            detections_batch = rfdetr_model.predict(images=rgb_frame_batch, threshold=0.5)
            process_batch_results(detections_batch, frame_batch, writer, all_results, frame_count)

        cap.release()
        writer.release()

        # Upload JSON results
        results_json = json.dumps(all_results, indent=2)
        results_object_name = f"{os.path.splitext(object_name)[0]}.json"
        minio_client.put_object(
            USER_OUT_BUCKET_NAME,
            results_object_name,
            data=io.BytesIO(results_json.encode('utf-8')),
            length=len(results_json.encode('utf-8')),
            content_type='application/json'
        )

        # Upload visualized video
        minio_client.fput_object(
            USER_OUT_VISUAL_BUCKET_NAME,
            f"viz_{object_name}",
            output_video_path,
            content_type='video/mp4'
        )
        print(f"Successfully processed {object_name}. JSON and visualized video saved.")

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Background Task & API Endpoints ---
def run_batch_detection():
    """The background task for detecting objects in all new videos."""
    if not minio_client:
        print("Cannot run batch detection: MinIO client not initialized.")
        return

    for bucket in [USER_DATA_BUCKET_NAME, USER_OUT_BUCKET_NAME, USER_OUT_VISUAL_BUCKET_NAME]:
        try:
            if not minio_client.bucket_exists(bucket):
                minio_client.make_bucket(bucket)
        except S3Error as e:
            print(f"Error ensuring bucket '{bucket}' exists: {e}")
            return

    try:
        user_data_files = [obj.object_name for obj in minio_client.list_objects(USER_DATA_BUCKET_NAME, recursive=True)]
        user_out_files = [obj.object_name for obj in minio_client.list_objects(USER_OUT_BUCKET_NAME, recursive=True)]
    except S3Error as e:
        print(f"Error listing objects in MinIO for batch processing: {e}")
        return

    processed_files_basenames = {os.path.splitext(f)[0] for f in user_out_files}
    files_to_process = [f for f in user_data_files if os.path.splitext(f)[0] not in processed_files_basenames]

    if not files_to_process:
        print("No new files to process.")
        return

    print(f"Starting batch processing for {len(files_to_process)} file(s).")
    for object_name in files_to_process:
        try:
            process_and_visualize_video(object_name)
        except Exception as e:
            print(f"An error occurred while processing {object_name}: {e}")
    print("Batch processing finished.")

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Object Detection service is running."}

@app.post("/detect/batch")
async def detect_batch(background_tasks: BackgroundTasks):
    """Triggers a background task to detect objects and create visualized videos."""
    background_tasks.add_task(run_batch_detection)
    return {
        "status": "accepted",
        "message": "Batch detection and visualization process started in the background."
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
