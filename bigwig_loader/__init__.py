import logging

try:
    from bigwig_loader._version import version as __version__
except ImportError:
    # this protects against an edge case where a user tries to import
    # the module without installing it, by adding it manually to
    # sys.path or trying to run it directly from the git checkout.
    __version__ = "not-installed"

from bigwig_loader.settings import Settings as _Settings

__author__ = ["Joren Retel"]
__email__ = ["joren.retel@pfizer.com"]

logger = logging.getLogger("bigwig_loader")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)


logger.warning(
    "Since bigwig-loader v0.3.0 the order of the output "
    "has changed to (batch x sequence x tracks)"
)

config = _Settings()
