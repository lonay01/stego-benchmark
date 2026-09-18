"""Terminal channel outcomes, distinct from detector scores and runtime failures."""


class GenerationBlocked(RuntimeError):
    """The fixed editor rejected this sample; do not resample until it passes."""
