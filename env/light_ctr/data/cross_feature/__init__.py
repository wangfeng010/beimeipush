from .intersection import Intersection
from .string_concat import StringConcat

CROSS_PROCESSOR_DICT = {
    "StringConcat": StringConcat,
    "Intersection": Intersection,
}
