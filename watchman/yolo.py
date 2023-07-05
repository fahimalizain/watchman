import cv2
import time
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from typing import List
import traceback

from .multiprocessing import SyncRunner

YOLO_v8n_LABELS = [
    0,  # 'person',
    1,  # 'bicycle',
    2,  # 'car',
    3,  # 'motorcycle',
    4,  # 'airplane',
    5,  # 'bus',
    6,  # 'train',
    7,  # 'truck',
    8,  # 'boat',
    14,  # 'bird',
    15,  # 'cat',
    16,  # 'dog',
    17,  # 'horse',
    18,  # 'sheep',
    19,  # 'cow',
    20,  # 'elephant',
    21,  # 'bear',
    22,  # 'zebra',
    23,  # 'giraffe',
    67,  # 'cell phone',
]


class YoloUnit(SyncRunner):
    def setup(self):
        self.model = YOLO('yolov8l.pt')

    def run(self):
        self.setup()

        while True:
            try:
                channel_name = 'NO_CHANNEL'
                data = next(self.get_from_queue(), None)
                if not data:
                    continue

                channel_name = data.get('channel_name')
                frames = data.get('frames')

                self.log_info(
                    f"Starting Yolo on {channel_name} with {len(frames)} frames")

                labels = set()
                for frame in frames:
                    frame, _labels = self.apply_yolo(frame)
                    labels.update(_labels)

                if not len(labels):
                    continue

                self.log_info("Sending to Telegram with labels %s", labels)
                self.output_queue.put(
                    dict(channel_name=channel_name, labels=labels, frames=frames))

            except Exception as e:
                # Log traceback
                tb = "\n".join(traceback.format_tb(e.__traceback__))
                self.log_error(f"Yolo Error {channel_name} {str(e)}\n{tb}")
            finally:
                time.sleep(0.5)

    def apply_yolo(self, frame):
        results: List[Results] = self.model(
            frame, device='cuda', verbose=False)

        text_padding = 5
        text_scale = 0.5
        text_thickness = 2

        if not results:
            return frame

        labels = set()

        for result in results:
            boxes = result.cpu().boxes
            class_ids = boxes.cls
            conf = boxes.conf

            for box, class_id, conf in zip(boxes.xyxy.numpy(), class_ids.numpy(), conf):
                class_id = class_id.astype(int)
                if class_id not in YOLO_v8n_LABELS:
                    continue

                # Draw the bounding box and label on the frame
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(128, 0, 0),
                    thickness=2,
                )

                class_name = self.model.model.names[class_id]
                labels.add(class_name)

                text = f"{class_name} {conf:0.2f}"
                text_width, text_height = cv2.getTextSize(
                    text=text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=text_scale,
                    thickness=text_thickness,
                )[0]

                text_x = x1 + text_padding
                text_y = y1 - text_padding

                text_background_x1 = x1
                text_background_y1 = y1 - 2 * text_padding - text_height

                text_background_x2 = x1 + 2 * text_padding + text_width
                text_background_y2 = y1

                cv2.rectangle(
                    img=frame,
                    pt1=(text_background_x1, text_background_y1),
                    pt2=(text_background_x2, text_background_y2),
                    color=(0, 0, 0),
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    img=frame,
                    text=text,
                    org=(text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=text_scale,
                    color=(255, 255, 255),
                    thickness=text_thickness,
                    lineType=cv2.LINE_AA,
                )

        return frame, labels
