from .as_type import AsType
from .bert_embedding import BertEmbedding
from .clip import ClipLayer
from .fillna_string import FillNaString
from .log1p import Log1pTransform
from .precomputed_embedding import PrecomputedEmbedding
from .segment_transform import SegmentTransform
from .split_embedding import SplitEmbedding
from .split_processor import SplitProcessor
from .split_transform import SplitTransform
from .str_embedding import StrEmbedding
from .string_concat import StringConcat
from .sum_pooling import SumPooling
from .to_map_transform import ToMapTransform

SINGLE_PROCESSOR_DICT = {
    "AsType": AsType,
    "BertEmbedding": BertEmbedding,
    "ClipLayer": ClipLayer,
    "FillNaString": FillNaString,
    "Log1pTransform": Log1pTransform,
    "PrecomputedEmbedding": PrecomputedEmbedding,
    "SegmentTransform": SegmentTransform,
    "SplitProcessor": SplitProcessor,
    "StrEmbedding": StrEmbedding,
    "StringConcat": StringConcat,
    "ToMapTransform": ToMapTransform,
    "SplitEmbedding": SplitEmbedding,
    "SumPooling": SumPooling,
    "SplitTransform": SplitTransform,
}
