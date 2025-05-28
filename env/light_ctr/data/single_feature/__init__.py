from .as_type import AsType
from .clip import ClipLayer
from .fillna_string import FillNaString
from .log1p import Log1pTransform
from .segment_transform import SegmentTransform
from .split_embedding import SplitEmbedding
from .split_processor import SplitProcessor
from .split_transform import SplitTransform
from .str_embedding import StrEmbedding
from .sum_pooling import SumPooling
from .to_map_transform import ToMapTransform

SINGLE_PROCESSOR_DICT = {
    "AsType": AsType,
    "ClipLayer": ClipLayer,
    "FillNaString": FillNaString,
    "Log1pTransform": Log1pTransform,
    "SegmentTransform": SegmentTransform,
    "SplitProcessor": SplitProcessor,
    "StrEmbedding": StrEmbedding,
    "ToMapTransform": ToMapTransform,
    "SplitEmbedding": SplitEmbedding,
    "SumPooling": SumPooling,
    "SplitTransform": SplitTransform,
}
