import json
import math
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Union
from .constants import MISSING_VALUE


def log(x: float, base: float = math.e) -> float:
    """log_{base}(x)

    Parameters
    ----------
    x : float
        [description]
    base : float, optional
        [description], by default math.e

    Returns
    -------
    float
        [log(x)]
    """
    assert x > 0, "input value must above zero."
    return math.log(x, base)


def astype(x: Union[int, float, str], targe_type: str) -> Union[int, float, str]:
    """[summary]

    Parameters
    ----------
    x : Union[int, float, str]
        [description]
    targe_type : str
        [description]

    Returns
    -------
    Union[int, float, str]
        [description]

    Raises
    ------
    ValueError
        [description]
    """

    if targe_type in ["int", "integer"]:
        return int(x)
    elif targe_type in ["float"]:
        return float(x)
    elif targe_type in ["str", "string"]:
        return str(x)
    else:
        return x


def list_astype(x: List, target_type: str) -> List:
    res = list()
    for item in x:
        tmp = astype(item, target_type)
        res.append(tmp)
    return res


def split(x: str, sep: str) -> List:
    """[summary]

    Parameters
    ----------
    x : str
        [description]
    sep : str
        [description]

    Returns
    -------
    List
        [description]
    """
    return x.split(sep)


def seperation(x: List[str], sep: str) -> List[List[str]]:
    """[summary]

    Parameters
    ----------
    x : List[str]
        [description]
    sep : str
        [description]

    Returns
    -------
    List[List[str]]
        [description]
    """
    assert isinstance(x, list), "input is not a list"
    res = []
    for i in range(len(x)):
        res.append(x[i].split(sep))
    return res


def get_decimal_places(x: float) -> int:
    """求小数点后位数

    Parameters
    ----------
    x : Union[str, float]
        [description]

    Returns
    -------
    int
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    num_str = repr(x)
    if "." in num_str:
        return len(num_str) - num_str.index(".") - 1
    else:
        return 0


def str_len(x: str) -> int:
    """求字符串长度

    Parameters
    ----------
    x : str
        [description]

    Returns
    -------
    int
        [description]
    """
    return len(x)


def list_len(x: List) -> int:
    """求列表长度

    Parameters
    ----------
    x : List
        [description]

    Returns
    -------
    int
        [description]
    """
    return len(x)


def scale(x: float, scale_ratio: float) -> float:
    """x / scale_ratio

    Parameters
    ----------
    x : float
        [description]
    scale_ratio : float
        [description]

    Returns
    -------
    float
        [description]
    """
    assert scale_ratio != 0, "DivisionByZero error"
    return x / scale_ratio


def padding(x: List[Any], pad_value: Union[str, float, int], max_len: int) -> List[Any]:
    """[summary]

    Parameters
    ----------
    x : List[Any]
        [description]
    pad_value : Union[str, float, int]
        [description]
    max_len : int
        [description]

    Returns
    -------
    List[Any]
        [description]

    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    """
    if isinstance(x, list):
        if isinstance(pad_value, (str, int, float)):
            if len(x) > max_len:
                return x[:max_len]
            else:
                return x + [pad_value] * (max_len - len(x))
        else:
            raise ValueError("Unsported pad value type.")
    else:
        raise ValueError("input x in not list.")


def fillna(
    x: Union[float, int, str], na_value: Union[float, int, str]
) -> Union[float, int, str]:
    """[summary]

    Parameters
    ----------
    x : Union[float, int, str]
        [description]
    na_value : Union[float, int, str]
        [description]

    Returns
    -------
    Union[float, int, str]
        [description]
    """
    if x in MISSING_VALUE:
        return na_value
    elif x != x:
        return na_value
    return x


def str_hash(x: str, vocabulary_size: int) -> int:
    """[summary]

    Parameters
    ----------
    x : str
        [description]
    vocabulary_size : int
        [description]

    Returns
    -------
    int
        [description]
    """
    index = int(md5(str(x).encode()).hexdigest(), 16) % vocabulary_size
    return index


def list_hash(x: List[str], vocabulary_size: int) -> List[int]:
    """[summary]

    Parameters
    ----------
    x : List[str]
        [description]
    vocabulary_size : int
        [description]

    Returns
    -------
    List[int]
        [description]
    """
    res = []
    for i in range(len(x)):
        res.append(str_hash(x[i], vocabulary_size))
    return res


def map_to_int(
    x: Union[str, list], map_dict: Dict[str, int], default_code: int = 0
) -> Union[List[int], int]:
    """[summary]

    Parameters
    ----------
    x : str
        [description]
    map_dict : Dict[str, int]
        [description]
    default_code : int, optional
        [description], by default 0

    Returns
    -------
    int
        [description]
    """
    if isinstance(x, list):
        res = []
        for item in x:
            res.append(map_dict.get(item, default_code))
        return res
    elif isinstance(x, str):
        return map_dict.get(x, default_code)
    else:
        raise ValueError


def to_dict(x: List[List[Any]], key_index: int, value_index: int) -> Dict[Any, Any]:
    """[summary]

    Parameters
    ----------
    x : List[List[Any]]
        [description]
    key_index : int
        [description]
    value_index : int
        [description]

    Returns
    -------
    Dict[Any, Any]
        [description]
    """
    out = dict()
    for i, item in enumerate(x):
        try:
            key = item[key_index]
            value = item[value_index]
        except IndexError as ie:
            print(
                f"An error {ie} happened. \
                input key_index {key_index} or key_index {value_index} out of bound at {i}-th item:{item} "
            )
        out[key] = value
    return out


def get_max_key(
    x: Dict[str, Union[float, int]], key_num: int = 1
) -> Union[str, List[str]]:
    """[summary]

    Parameters
    ----------
    x : Dict[str, Union[float, int]]
        [description]
    key_num : int, optional
        [description], by default 1

    Returns
    -------
    Union[str, List[str]]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    if key_num == 1:
        return max(x, key=x.get)  # type: ignore
    elif key_num > 1:
        import heapq

        return heapq.nlargest(key_num, x, key=x.get)  # type: ignore
    else:
        raise NotImplementedError


def list_get(x: List[List[Any]], item_index: int) -> List[Any]:
    """[summary]

    Parameters
    ----------
    x : List[List[Any]]
        [description]
    item_index : int
        [description]

    Returns
    -------
    List[Any]
        [description]
    """
    out = list()
    for i, item in enumerate(x):
        try:
            get_item = item[item_index]
        except IndexError as e:
            print(
                f"An error {e} happened. \
            input item_index {item_index} out of bound at {i}-th item:{item}"
            )
        out.append(get_item)
    return out


def to_bucket(x: float, bin_boundaries: List[float]) -> int:
    """将数值分桶

    Parameters
    ----------
    x : float
        [description]
    bin_boundaries : List[float]
        递增列表. 第一个元素要求大于0

    Returns
    -------
    int
        [description]
    """
    res = len(bin_boundaries) + 1  # 默认值
    for i in range(len(bin_boundaries)):
        if x > bin_boundaries[i]:
            return i + 1
    return res


def to_date(x: str, format: str) -> datetime:
    """将字符串转化为日期对象

    Parameters
    ----------
    x : str
        [date string]
    format : str
        [date format]
        - i.e.  '%Y-%m-%d %H:%M:%S'

    Returns
    -------
    datetime
        [description]
    """
    return datetime.strptime(x, format)


def weekday(x: str, format: str, fail_value: int = 1) -> int:
    """获得星期: 1, 2, 3, 4, 5, 6, 7

    Parameters
    ----------
    x : str
        [date string]
    format : str
        [date format]
        - i.e.  '%Y-%m-%d %H:%M:%S'
    Returns
    -------
    int
        [description]
    """
    try:
        date = datetime.strptime(x, format)
    except Exception as e:
        print(e)
        return fail_value
    return date.weekday()


def get_hour(x: str, format: str, fail_value: int = 1) -> int:
    """获得小时: 0-24

    Parameters
    ----------
    x : str
        [date string]
    format : str
        [date format]
        - i.e.  '%Y-%m-%d %H:%M:%S'
    Returns
    -------
    int
        [description]
    """
    try:
        date = datetime.strptime(x, format)
    except Exception as e:
        print(e)
        return fail_value
    return date.hour


def json_to_list(x: str, fail_value: str) -> List[Any]:
    """将列表形式的json字符串转化为Python列表

    i.e.
    input: '["DTST","NVDA","META","GOOGL","MSFT"]'
    output: ['DTST', 'NVDA', 'META', 'GOOGL', 'MSFT']

    如果转化失败，默认返回[fail_value]

    Parameters
    ----------
    x : str
        [description]
    fail_value : str
        [description]

    Returns
    -------
    List[Any]
        [description]
    """
    try:
        x = json.loads(x)
    except Exception as e:
        print(e)
        return [fail_value]
    return x


def list_get_join(
    x: List[List[Any]], indices: List[int], sep: str, fail_value: str = "null"
):
    """将输入的二级列表转化为一级列表

    例子：
    in: [['300308', '33', 'aStock'], ['000001', '16', 'index']]
    out: ['300308,33', '000001,16']

    Parameters
    ----------
    x : List[List[Any]]
        _description_
    indices : List[int]
        _description_
    sep : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    out = list()
    for _, item in enumerate(x):
        try:
            target_elements = list()
            for idx in indices:
                target_elements.append(item[idx])
            get_item = f"{sep}".join(target_elements)
        except IndexError as e:
            get_item = fail_value
            print(f"An error {e} happened. got item value {item}")
        out.append(get_item)
    return out


def int_max(x: int, max_value: int) -> int:
    return max(x, max_value)


formats = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO8601 带毫秒
    "%Y-%m-%dT%H:%M:%S",  # ISO8601 不带毫秒
    "%Y/%m/%d %H:%M",  # 自定义格式
]


def to_hour(x: str):
    for fmt in formats:
        try:
            dt = datetime.strptime(x, fmt)
            return dt.hour
        except ValueError:
            continue
    else:
        print(f"unable to decoder{x}")
        return 0


def to_weekday(x):
    for fmt in formats:
        try:
            dt = datetime.strptime(x, fmt)
            return dt.weekday()
        except ValueError:
            continue
    else:
        print(f"unable to decoder{x}")
        return 0


def remove_items(x: List[Any], target_values: List[Any]):
    residual = set(x) - set(target_values)
    if not residual:
        return x
    return [i for i in x if i not in target_values]


def json_object_to_list(x: str, key: str, fail_value: str = "null"):
    try:
        x_obj = json.loads(x)
    except Exception as e:
        print(e)
        return [fail_value]
    x_obj = json.loads(x)
    y = [z.get(key, fail_value) for z in x_obj]
    return y


def has_intersection(x: List[Any], y: List[Any], exclude=MISSING_VALUE) -> int:
    a = set(x) - set(exclude)
    b = set(y) - set(exclude)
    return int(len(set(a) & set(b)) > 0)
