from dotenv import load_dotenv
import threading
import time
import logging
import os
from typing import Dict
from queue import Queue, Empty
import cv2
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes
import requests
from io import BytesIO


load_dotenv()
logging.basicConfig(level=logging.INFO)

KEMAL_IMAGE_UPLOAD_URL = 'http://localhost:3000/api/receive_image'
MODEL_ID = os.getenv("MODEL_ID", "svalik/1")
VIDEO_REFERENCE = os.getenv("VIDEO_REFERENCE", 'http://192.168.68.107:8080/video')
ANNOUNCEMENT_DEBOUNCE_TIME = int(os.getenv("ANNOUNCEMENT_DEBOUNCE_TIME", 5))  # seconds
NTFY_URL = "https://ntfy.sh/bocnwwsKuU9nHZUK"

output_size = (640, 480)
video_sink = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25.0, output_size)
# annotator = sv.BoxAnnotator()
fps_monitor = sv.FPSMonitor()
announcement_queue = Queue()
last_announcement_time = time.time() - ANNOUNCEMENT_DEBOUNCE_TIME

def say_label():
    global last_announcement_time
    while True:
        try:
            label, timestamp = announcement_queue.get(timeout=1)
            current_time = time.time()
            time_since_last_announcement = current_time - last_announcement_time

            if time_since_last_announcement >= ANNOUNCEMENT_DEBOUNCE_TIME:
                threading.Thread(target=lambda: os.system(f"say {label}"), daemon=True).start()
                last_announcement_time = current_time

        except Empty:
            continue

def on_prediction(predictions: Dict, video_frame: VideoFrame):
    label_commands = {
        'IED': 'IED',
        'GSW': 'GSW',
        'Eye_Injury': 'Eye Injury',
        'Burn': 'Burn'
    }

    try:
        labels = [p["class"] for p in predictions.get("predictions", [])]
        current_time = time.time()
        if labels and current_time - last_announcement_time >= ANNOUNCEMENT_DEBOUNCE_TIME:
            label_code = label_commands.get(labels[0])
            if label_code:
                logging.info(f"Detected Code: {label_code}, Time: {current_time}")
                response = requests.post(NTFY_URL, data=f"Detected Code: {label_code}, Time: {current_time}")
                announcement_queue.put((label_code, current_time))
                

        # detections = sv.Detections.from_inference(predictions)
        render_boxes(
                predictions=predictions,
                video_frame=video_frame,
                fps_monitor=fps_monitor,
                display_statistics=True,
            )

        # image = annotator.annotate(
        #     scene=video_frame.image.copy(),
        #     detections=detections,
        #     labels=labels
        # )

        # ret, buffer = cv2.imencode('.jpg', image)
        # img_bytes = BytesIO(buffer)
        # files = {'image': ('frame.jpg', img_bytes, 'image/jpeg')}
        # image_response = requests.post(KEMAL_IMAGE_UPLOAD_URL, files=files)
        # logging.info(f"Image sent, response: {image_response.status_code}")
        # cv2.imshow("Predictions", image)
        # cv2.waitKey(1)

    except Exception as e:
        logging.error(f"An error occurred in on_prediction: {e}", exc_info=True)


def main():
    try:
        pipeline = InferencePipeline.init(
            model_id=MODEL_ID,
            video_reference=VIDEO_REFERENCE,
            on_prediction=on_prediction,
            confidence=0.6,
        )

        pipeline.start()
        pipeline.join()

    except Exception as e:
        logging.error(f"An error occurred in main: {e}", exc_info=True)


if __name__ == "__main__":
    announcement_thread = threading.Thread(target=say_label, daemon=True)
    announcement_thread.start()
    main()
