# Validation scope

The build checks every delivered Python file and every notebook code cell with
Python's AST parser, validates notebook structure, verifies the bundled digital
source archive checksum and includes the reviewed production sources unchanged.

CPU tests cover the 21-condition matrix, independent seed mapping, fixed-width
padding, caption selection, geometry, quantization, package/result hashes,
restart and deletion safety, safety-blocked results, clean-control drift,
paired statistics and an end-to-end synthetic generation/decode fixture.
They also run an actual small, randomly initialized Diffusers AutoencoderKL on
CPU to check two-pass posterior-mode reconstruction and intermediate 8-bit
rounding, and check all called parameters against Diffusers 0.32.2 signatures.
The real 29-method registry is loaded to verify five-to-100 run identity and
prefix invariance. Random test weights are not substituted for attack models.

The local host has no CUDA GPU. Real SDXL, InstructPix2Pix, inpainting and all
29 isolated Linux model environments cannot be certified here. Those remain the
purpose of the five-image Colab acceptance run. Synthetic test pixels are never
reported as experimental measurements.
