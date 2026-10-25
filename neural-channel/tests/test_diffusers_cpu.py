"""Actual Diffusers 0.32.2 APIs and small random-weight CPU VAE, no downloads.

This tests library interfaces and operator mathematics, NOT published weights
or GPU model quality. Production construction still requires CUDA.
"""

import inspect
import unittest

import numpy as np
from PIL import Image


class DiffusersAPI(unittest.TestCase):
    def test_pinned_diffusion_call_signatures(self):
        import diffusers
        self.assertEqual(diffusers.__version__, "0.32.2")
        from diffusers import (StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline,
                               StableDiffusionInstructPix2PixPipeline)
        common = {"image", "prompt", "negative_prompt", "num_inference_steps", "guidance_scale", "generator",
                  "output_type", "callback_on_step_end"}
        sdxl = {"strength", "original_size", "target_size", "crops_coords_top_left",
                "negative_original_size", "negative_target_size", "negative_crops_coords_top_left"}
        for klass, required in [(StableDiffusionXLImg2ImgPipeline, common|sdxl),
                                (StableDiffusionXLInpaintPipeline, common|sdxl|{"mask_image", "width", "height"}),
                                (StableDiffusionInstructPix2PixPipeline, common|{"image_guidance_scale"})]:
            self.assertTrue(required <= set(inspect.signature(klass.__call__).parameters), klass.__name__)

    def test_actual_vae_two_pass_quantized_mode(self):
        import torch
        from diffusers import AutoencoderKL
        from wmbench_neural.channel import NeuralChannel, rgb8
        from wmbench_neural.spec import ATTACKS
        torch.set_num_threads(2)
        torch.manual_seed(71)
        model = AutoencoderKL(in_channels=3, out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
            block_out_channels=(4, 4, 4, 4), layers_per_block=1, norm_num_groups=1, latent_channels=2,
            sample_size=512, mid_block_add_attention=False).eval()
        operator = NeuralChannel.__new__(NeuralChannel)
        operator.torch, operator.engine, operator.model, operator.device = torch, "vae", model, "cpu"
        arrays = np.random.default_rng(73).integers(0, 256, (2, 512, 512, 3), dtype=np.uint8)
        images = [Image.fromarray(a) for a in arrays]
        outputs, flags, details = operator.apply(images, ATTACKS[2], [1, 2], ["", ""])
        expected = arrays
        with torch.inference_mode():
            for _ in range(2):
                x = torch.from_numpy(expected.copy()).permute(0, 3, 1, 2).float()/127.5-1
                y = model.decode(model.encode(x).latent_dist.mode()).sample
                expected = rgb8(((y.clamp(-1, 1)+1)/2).permute(0, 2, 3, 1).numpy())
        np.testing.assert_array_equal(np.stack(outputs), expected)
        self.assertEqual(flags, [False, False])
        self.assertEqual(details["posterior"], "mode")
        self.assertEqual(details["actual_denoising_steps"], 0)
        self.assertGreater(details["batch_seconds"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
