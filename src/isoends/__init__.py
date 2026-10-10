"""IsoEnds: poly(A) tail length, alternative polyadenylation and transcription start sites from IsoForge runs,
tested over biological replicates.

    import isoends as ie
    res = ie.load("isoends_out")
    res.volcano("tandem")
"""
__version__ = "1.0.0"

from .api import ANALYSES, Experiment, Results, load  # noqa: E402

__all__ = ["ANALYSES", "Experiment", "Results", "load", "__version__"]
