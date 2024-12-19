from webapp_client import AppAccessConfig, AppConfig, ComputeEnvironment

from . import __version__
from .app import FeStokesRePair

_DESCRIPTION = """Evaluator for https://fe-nerd-games.github.io/FEStokesRePair/"""

_DOCKERFILE = """
FROM python:3.12
RUN python3 -m pip install ngsolve==6.2.2406
"""

config = AppConfig(
    name="FE-Stokes RE-Pair",
    version=__version__,
    python_class=FeStokesRePair,
    frontend_pip_dependencies=["netgen", "ngsolve", "scipy"],
    frontend_dependencies=[],
    description=_DESCRIPTION,
    compute_environments=[
        ComputeEnvironment(env_type="docker", dockerfile=_DOCKERFILE)
    ],
    access=AppAccessConfig(),
)
