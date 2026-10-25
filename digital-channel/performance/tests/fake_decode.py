"""CPU-only test process. Not an experimental watermark implementation."""
import sys
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from wmbench.io import read_json
from wmbench_fast.worker import decode


class FakeAdapter:
    def decode(self, image, metadata):
        with Image.open(image) as pixels:
            pixels.load()
            assert pixels.mode == "RGB"
        assert read_json(metadata)["expected_bits"] == [0, 0]
        return {"bits": [0, 1]}


if __name__ == "__main__":
    with patch("wmbench_digital.worker.load_adapter", return_value=FakeAdapter()):
        decode(read_json(Path(sys.argv[1])))
