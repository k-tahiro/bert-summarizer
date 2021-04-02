import warnings

from .abs import BertSumAbs
from .decoder import BertSumAbsDecoder

__all__ = [
    "BertSumAbs",
    "BertSumAbsDecoder",
]

try:
    from .onmt_decoder import BertSumAbsOpenNMTDecoder

    __all__.append("BertSumAbsOpenNMTDecoder")
except ModuleNotFoundError:
    warnings.warn("Failed to import BertSumAbsOpenNMTDecoder")
