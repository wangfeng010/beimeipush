from typing import Any, Dict, Iterable, List, Set, Union

from .constants import MISSING_VALUE


def intersection(
    x: Union[List, Set, Dict],
    y: Union[List, Set, Dict],
    fail_value: Union[str, int],
    exclude: List = MISSING_VALUE,
) -> List:
    """求两个集合的交集

    Parameters
    ----------
    x : Union[List, Set, Dict]
        [description]
    y : Union[List, Set, Dict]
        [description]
    exclude : List, optional
        [description], by default MISSING_VALUE

    Returns
    -------
    Set
        [description]
    """
    if isinstance(x, dict):
        x = set(x.keys())
    elif isinstance(x, list):
        x = set(x)
    if isinstance(y, dict):
        y = set(y.keys())
    elif isinstance(y, list):
        y = set(y)
    res = x.intersection(y) - set(exclude)
    if not res:
        return [fail_value]
    return list(res)


def is_in(a: Any, b: Iterable, exclude: List = MISSING_VALUE) -> int:
    """判断元素a是否在集合B中

    Parameters
    ----------
    a : Any
        [description]
    b : Iterable
        [description]

    Returns
    -------
    int
        [description]
    """
    if a in exclude:
        return 0
    flag = a in b
    return int(flag)


def intersection_num(
    f1: Union[List, Set],
    f2: Union[List, Set],
    fail_value: Union[str, int],
    exclude: List = MISSING_VALUE,
) -> int:
    """求两个集合交集的元素数量

    Parameters
    ----------
    f1 : Union[List, Set]
        [description]
    f2 : Union[List, Set]
        [description]
    exclude : List, optional
        [description], by default MISSING_VALUE

    Returns
    -------
    int
        [description]
    """
    res = intersection(f1, f2, fail_value, exclude)
    if fail_value in res:
        return 0
    return len(res)


def concat_str(a: str, b: str) -> str:
    return str(a) + str(b)


def intersection_list_dict(
    a: List[Any], b: Dict, exclude: List = MISSING_VALUE
) -> List[Any]:
    """求列表a与字典b的keys的交集

    Parameters
    ----------
    a : List[Any]
        [description]
    b : Dict
        [description]

    Returns
    -------
    List[Any]
        [description]
    """
    return list(set(a) & set(b.keys()) - set(exclude))


def ratio(a: float, b: float, a_smoothing: float, b_smoothing: float) -> float:
    """a / b

    Parameters
    ----------
    a : float
        [description]
    b : float
        [description]
    a_smoothing : float
        [description]
    b_smoothing : float
        [description]

    Returns
    -------
    float
        [description]
    """
    return (a + a_smoothing) / (b + b_smoothing)


def union(a: Iterable, b: Iterable) -> List[Any]:
    return list(set(a).union(set(b)))
