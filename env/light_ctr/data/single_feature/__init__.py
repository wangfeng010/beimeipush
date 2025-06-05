from .as_type import AsType
from .bert_embedding import BertEmbedding
from .clip import ClipLayer
from .fillna_string import FillNaString
from .json_array_processor import (
    JsonArrayProcessor, 
    StockCodeProcessor, 
    TagProcessor, 
    ImportanceTagProcessor
)
from .json_object_to_list import JsonObjectToList, Padding
from .log1p import Log1pTransform
from .precomputed_embedding import PrecomputedEmbedding
from .segment_transform import SegmentTransform
from .split_embedding import SplitEmbedding, EntityOnlyEmbedding
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
    "EntityOnlyEmbedding": EntityOnlyEmbedding,
    "FillNaString": FillNaString,
    "ImportanceTagProcessor": ImportanceTagProcessor,
    "JsonArrayProcessor": JsonArrayProcessor,
    "JsonObjectToList": JsonObjectToList,
    "Log1pTransform": Log1pTransform,
    "Padding": Padding,
    "PrecomputedEmbedding": PrecomputedEmbedding,
    "SegmentTransform": SegmentTransform,
    "SplitEmbedding": SplitEmbedding,
    "SplitProcessor": SplitProcessor,
    "SplitTransform": SplitTransform,
    "StockCodeProcessor": StockCodeProcessor,
    "StrEmbedding": StrEmbedding,
    "StringConcat": StringConcat,
    "SumPooling": SumPooling,
    "TagProcessor": TagProcessor,
    "ToMapTransform": ToMapTransform,
}
