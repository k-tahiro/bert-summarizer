import importlib.util
import sys
from logging import getLogger

logger = getLogger(__name__)

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

_onmt_avaliable = importlib.util.find_spec("onmt") is not None
try:
    _onmt_version = importlib_metadata.version("onmt")
    logger.debug(f"Successfully imported onmt version {_onmt_version}")
except importlib_metadata.PackageNotFoundError:
    _onmt_avaliable = False


def is_onmt_available() -> bool:
    return _onmt_avaliable
