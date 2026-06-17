"""
MEW-OG: Minimum-Excess-Work Observable Guidance for Diffusion Models.

A minimal implementation of observable-guided generative modeling for the
Prinz potential toy system.
"""

__version__ = "0.1.0"

from mew_og.guidance.mew_og_model import MewOGModel
from mew_og.reweighting.maxent import MaxEntReweightingEstimator
from mew_og.models.score_network import ScoreBasedDDPM
from mew_og.models.oggm_compat import OGGMScoreBasedDDPM, load_oggm_score_model
from mew_og.samplers.vp_sde import VPSDESampler
from mew_og.observables.gmm import GaussianMixtureObservable

__all__ = [
    "MewOGModel",
    "MaxEntReweightingEstimator",
    "ScoreBasedDDPM",
    "OGGMScoreBasedDDPM",
    "load_oggm_score_model",
    "VPSDESampler",
    "GaussianMixtureObservable",
]

