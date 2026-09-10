import av
import cv2
import numpy as np
import torch

from clinical.live import LiveVesselProcessor


class ConstantVesselModel(torch.nn.Module):
    def forward(self, tensor):
        return torch.full_like(tensor, 2.0)


def _fundus_like_frame():
    rng = np.random.default_rng(2)
    gray = np.zeros((240, 320), dtype=np.uint8)
    field = np.clip(rng.normal(115, 35, (190, 190)), 20, 220).astype(np.uint8)
    yy, xx = np.ogrid[:190, :190]
    field[(xx - 95) ** 2 + (yy - 95) ** 2 > 90**2] = 0
    gray[25:215, 65:255] = field
    cv2.line(gray, (80, 120), (240, 120), 25, 3)
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_live_processor_returns_an_annotated_video_frame():
    processor = LiveVesselProcessor(
        ConstantVesselModel(), process_every=2, input_size=128
    )
    input_frame = _fundus_like_frame()

    output = processor.process(av.VideoFrame.from_ndarray(input_frame, format="bgr24"))
    output_array = output.to_ndarray(format="bgr24")

    assert output_array.shape == input_frame.shape
    assert processor.latest_mask is not None
    assert processor.snapshot()["processed_frames"] == 1


def test_live_processor_throttles_model_inference():
    processor = LiveVesselProcessor(
        ConstantVesselModel(), process_every=2, input_size=128
    )
    frame = av.VideoFrame.from_ndarray(_fundus_like_frame(), format="bgr24")

    processor.process(frame)
    first_mask = processor.latest_mask
    processor.process(frame)

    assert processor.latest_mask is first_mask
    assert processor.snapshot()["processed_frames"] == 2

