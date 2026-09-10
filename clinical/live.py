import threading
import time

import av
import cv2
import numpy as np
import torch

from clinical.analysis import model_input_channel
from clinical.quality import assess_image_quality
from data.preprocessing import MedicalImagePreprocessor


class LiveVesselProcessor:
    """Throttled single-pass preview processor for browser camera frames."""

    def __init__(
        self,
        model,
        device="cpu",
        decision_threshold=0.5,
        process_every=2,
        input_size=128,
    ):
        self.model = model
        self.device = device
        self.decision_threshold = decision_threshold
        self.process_every = max(int(process_every), 1)
        self.input_size = int(input_size)
        self.preprocessor = MedicalImagePreprocessor(
            use_clahe=True, norm_mode="minmax"
        )
        self.lock = threading.Lock()
        self.frame_number = 0
        self.latest_mask = None
        self.latest_quality = "Position a retinal image inside the guide"
        self.latest_latency_ms = 0

    def _predict_mask(self, rgb_image):
        small = cv2.resize(
            rgb_image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        quality = assess_image_quality(small)
        if quality.status == "Retake recommended":
            return None, "Adjust focus, lighting, and retinal framing"

        channel = model_input_channel(small)
        tensor = self.preprocessor.process(channel).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        self.model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        self.latest_latency_ms = int((time.perf_counter() - started) * 1000)
        return probability >= self.decision_threshold, quality.status

    @staticmethod
    def _draw_status(frame, message, color):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (18, 27, 31), -1)
        cv2.putText(
            frame,
            message,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            color,
            2,
            cv2.LINE_AA,
        )

    def process(self, frame):
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with self.lock:
            self.frame_number += 1
            should_process = self.frame_number % self.process_every == 1
            if should_process:
                mask, quality = self._predict_mask(rgb)
                self.latest_mask = mask
                self.latest_quality = quality

            output = bgr.copy()
            if self.latest_mask is not None:
                mask = cv2.resize(
                    self.latest_mask.astype(np.uint8),
                    (output.shape[1], output.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                teal_bgr = np.empty_like(output)
                teal_bgr[:, :] = (166, 184, 20)
                output[mask] = (
                    output[mask].astype(np.float32) * 0.55
                    + teal_bgr[mask].astype(np.float32) * 0.45
                ).astype(np.uint8)
                message = f"LIVE VESSEL PREVIEW | {self.latest_latency_ms} ms"
                color = (190, 245, 225)
            else:
                message = self.latest_quality.upper()
                color = (80, 190, 255)

            h, w = output.shape[:2]
            cv2.circle(
                output,
                (w // 2, h // 2),
                int(min(h, w) * 0.38),
                (215, 225, 220),
                2,
                cv2.LINE_AA,
            )
            self._draw_status(output, message, color)

        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def snapshot(self):
        with self.lock:
            return {
                "quality": self.latest_quality,
                "latency_ms": self.latest_latency_ms,
                "processed_frames": self.frame_number,
            }

