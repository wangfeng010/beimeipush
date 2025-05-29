import json
import math
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Union, Optional
from .constants import MISSING_VALUE


def log(x: float, base: float = math.e) -> float:
    """计算对数 log_{base}(x)

    Args:
        x: 输入数值，必须大于0
        base: 对数的底数，默认为自然对数e

    Returns:
        对数值

    Raises:
        AssertionError: 当输入值不大于0时
    """
    assert x > 0, "输入值必须大于0"
    return math.log(x, base)


def astype(x: Union[int, float, str], target_type: str) -> Union[int, float, str]:
    """转换数据类型

    Args:
        x: 需要转换类型的值
        target_type: 目标类型，支持 'int'/'integer', 'float', 'str'/'string'

    Returns:
        转换后的值

    Raises:
        ValueError: 当目标类型不受支持时
    """
    if target_type in ["int", "integer"]:
        return int(x)
    elif target_type in ["float"]:
        return float(x)
    elif target_type in ["str", "string"]:
        return str(x)
    else:
        return x


def list_astype(x: List, target_type: str) -> List:
    """转换列表中所有元素的数据类型

    Args:
        x: 输入列表
        target_type: 目标类型

    Returns:
        转换类型后的列表
    """
    return [astype(item, target_type) for item in x]


def split(x: str, sep: str) -> List:
    """使用分隔符拆分字符串

    Args:
        x: 输入字符串
        sep: 分隔符

    Returns:
        拆分后的字符串列表
    """
    return x.split(sep)


def seperation(x: List[str], sep: str) -> List[List[str]]:
    """对列表中的每个字符串使用分隔符进行拆分

    Args:
        x: 字符串列表
        sep: 分隔符

    Returns:
        二维字符串列表，每个内部列表是对应输入字符串的拆分结果

    Raises:
        AssertionError: 当输入不是列表时
    """
    assert isinstance(x, list), "输入必须是列表类型"
    return [item.split(sep) for item in x]


def get_decimal_places(x: float) -> int:
    """计算小数点后的位数

    Args:
        x: 输入浮点数

    Returns:
        小数点后的位数
    """
    num_str = repr(x)
    if "." in num_str:
        return len(num_str) - num_str.index(".") - 1
    else:
        return 0


def str_len(x: str) -> int:
    """计算字符串长度

    Args:
        x: 输入字符串

    Returns:
        字符串长度
    """
    return len(x)


def list_len(x: List) -> int:
    """计算列表长度

    Args:
        x: 输入列表

    Returns:
        列表长度
    """
    return len(x)


def scale(x: float, scale_ratio: float) -> float:
    """对数值进行缩放 (x / scale_ratio)

    Args:
        x: 输入数值
        scale_ratio: 缩放比例，不能为0

    Returns:
        缩放后的数值

    Raises:
        AssertionError: 当缩放比例为0时
    """
    assert scale_ratio != 0, "缩放比例不能为0，除数不能为0"
    return x / scale_ratio


def padding(x: List[Any], pad_value: Union[str, float, int], max_len: int) -> List[Any]:
    """对列表进行填充或截断，使其长度为max_len

    Args:
        x: 输入列表
        pad_value: 填充值，可以是字符串、整数或浮点数
        max_len: 目标长度

    Returns:
        填充或截断后的列表

    Raises:
        ValueError: 当输入不是列表或填充值类型不受支持时
    """
    if not isinstance(x, list):
        raise ValueError("输入x必须是列表类型")
    
    if not isinstance(pad_value, (str, int, float)):
        raise ValueError("填充值类型不支持，只支持字符串、整数或浮点数")
    
    # 截断列表
    if len(x) > max_len:
        return x[:max_len]
    
    # 填充列表
    return x + [pad_value] * (max_len - len(x))


def fillna(x: Any, na_value: Any) -> Any:
    """用指定值替换缺失值

    Args:
        x: 输入值，可以是任何类型
        na_value: 用于替换缺失值的值

    Returns:
        如果输入是缺失值则返回na_value，否则返回原值
    """
    # 合并判断：检查是否在缺失值列表中或者是NaN
    if x in MISSING_VALUE or x != x:  # x != x 检查是否为NaN
        return na_value
    return x


def str_hash(x: str, vocabulary_size: int) -> int:
    """计算字符串的哈希值，并映射到指定范围内

    Args:
        x: 输入字符串
        vocabulary_size: 词汇表大小，哈希值会被映射到[0, vocabulary_size-1]

    Returns:
        映射后的哈希值
    """
    return int(md5(str(x).encode()).hexdigest(), 16) % vocabulary_size


def list_hash(x: List[str], vocabulary_size: int) -> List[int]:
    """计算字符串列表中每个元素的哈希值

    Args:
        x: 字符串列表
        vocabulary_size: 词汇表大小

    Returns:
        哈希值列表
    """
    return [str_hash(item, vocabulary_size) for item in x]


def map_to_int(
    x: Union[str, List[str]], map_dict: Dict[str, int], default_code: int = 0
) -> Union[int, List[int]]:
    """将字符串或字符串列表映射为整数或整数列表

    Args:
        x: 输入字符串或字符串列表
        map_dict: 映射字典，键为字符串，值为整数
        default_code: 当键不在字典中时的默认值，默认为0

    Returns:
        映射后的整数或整数列表

    Raises:
        ValueError: 当输入类型既不是字符串也不是列表时
    """
    if isinstance(x, list):
        return [map_dict.get(item, default_code) for item in x]
    
    if isinstance(x, str):
        return map_dict.get(x, default_code)
    
    raise ValueError("输入必须是字符串或字符串列表")


def to_dict(x: List[List[Any]], key_index: int, value_index: int) -> Dict[Any, Any]:
    """将二维列表转换为字典，使用指定索引的元素作为键和值

    Args:
        x: 二维列表
        key_index: 用作键的元素索引
        value_index: 用作值的元素索引

    Returns:
        转换后的字典
    """
    out = {}
    for i, item in enumerate(x):
        try:
            key = item[key_index]
            value = item[value_index]
            out[key] = value
        except IndexError as ie:
            print(
                f"错误 {ie}：输入的key_index {key_index}或value_index {value_index} "
                f"超出范围，在第{i}个元素: {item}"
            )
    return out


def get_max_key(
    x: Dict[str, Union[float, int]], key_num: int = 1
) -> Union[str, List[str]]:
    """获取字典中值最大的键

    Args:
        x: 输入字典，键为字符串，值为数值
        key_num: 返回键的数量，默认为1

    Returns:
        值最大的键或键列表

    Raises:
        NotImplementedError: 当key_num小于1时
    """
    if key_num < 1:
        raise NotImplementedError("key_num必须大于或等于1")
    
    if key_num == 1:
        return max(x, key=x.get)  # type: ignore
    
    import heapq
    return heapq.nlargest(key_num, x, key=x.get)  # type: ignore


def list_get(x: List[List[Any]], item_index: int) -> List[Any]:
    """从二维列表中提取指定索引的元素，组成新列表

    Args:
        x: 二维列表
        item_index: 要提取的元素索引

    Returns:
        提取出的元素列表
    """
    result = []
    for i, item in enumerate(x):
        try:
            result.append(item[item_index])
        except IndexError as e:
            print(
                f"错误 {e}：输入的item_index {item_index} "
                f"超出范围，在第{i}个元素: {item}"
            )
            # 在错误情况下，你可能想要添加一个默认值
    return result


def to_bucket(x: float, bin_boundaries: List[float]) -> int:
    """将数值分桶

    Args:
        x: 输入数值
        bin_boundaries: 分桶边界，必须是递增列表

    Returns:
        桶索引，从1开始
    """
    for i, boundary in enumerate(bin_boundaries):
        if x <= boundary:
            return i + 1
    return len(bin_boundaries) + 1


def to_date(x: str, format: str) -> datetime:
    """将字符串转换为日期对象

    Args:
        x: 日期字符串
        format: 日期格式，例如 '%Y-%m-%d %H:%M:%S'

    Returns:
        datetime对象
    """
    return datetime.strptime(x, format)


def weekday(x: str, format: str, fail_value: int = 1) -> int:
    """获取日期字符串对应的星期几（1-7）

    Args:
        x: 日期字符串
        format: 日期格式
        fail_value: 转换失败时返回的默认值，默认为1

    Returns:
        星期几，取值为1-7
    """
    try:
        date = datetime.strptime(x, format)
        return date.weekday() + 1  # 将0-6转换为1-7
    except Exception as e:
        print(f"日期转换错误: {e}")
        return fail_value


def get_hour(x: str, format: str, fail_value: int = 0) -> int:
    """从日期字符串中获取小时（0-23）

    Args:
        x: 日期字符串
        format: 日期格式
        fail_value: 转换失败时返回的默认值，默认为0

    Returns:
        小时，取值为0-23
    """
    try:
        date = datetime.strptime(x, format)
        return date.hour
    except Exception as e:
        print(f"日期转换错误: {e}")
        return fail_value


# 预定义的日期格式列表
formats = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO8601 带毫秒
    "%Y-%m-%dT%H:%M:%S",      # ISO8601 不带毫秒
    "%Y/%m/%d %H:%M",         # 自定义格式
]


def to_hour(x: str) -> int:
    """尝试使用多种格式解析日期字符串并返回小时

    Args:
        x: 日期字符串

    Returns:
        小时(0-23)，解析失败时返回0
    """
    for fmt in formats:
        try:
            dt = datetime.strptime(x, fmt)
            return dt.hour
        except ValueError:
            continue
    
    print(f"无法解析日期: {x}")
    return 0


def to_weekday(x: str) -> int:
    """尝试使用多种格式解析日期字符串并返回星期几

    Args:
        x: 日期字符串

    Returns:
        星期几(0-6，0表示周一)，解析失败时返回0
    """
    for fmt in formats:
        try:
            dt = datetime.strptime(x, fmt)
            return dt.weekday()
        except ValueError:
            continue
    
    print(f"无法解析日期: {x}")
    return 0


def json_to_list(x: str, fail_value: str) -> List[Any]:
    """将JSON格式的字符串转换为Python列表

    例如：
    输入: '["DTST","NVDA","META","GOOGL","MSFT"]'
    输出: ['DTST', 'NVDA', 'META', 'GOOGL', 'MSFT']

    Args:
        x: JSON格式的字符串
        fail_value: 转换失败时返回的默认值

    Returns:
        转换后的列表，失败时返回包含fail_value的单元素列表
    """
    try:
        return json.loads(x)
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return [fail_value]


def remove_items(x: List[Any], target_values: List[Any]) -> List[Any]:
    """从列表中移除指定的值

    Args:
        x: 输入列表
        target_values: 要移除的值列表

    Returns:
        移除指定值后的列表，如果移除后为空则返回原列表
    """
    residual = set(x) - set(target_values)
    if not residual:
        return x
    return [i for i in x if i not in target_values]


def json_object_to_list(x: str, key: str, fail_value: str = "null") -> List[Any]:
    """从JSON对象字符串中提取指定键的值列表

    Args:
        x: JSON对象字符串
        key: 要提取的键
        fail_value: 转换失败时返回的默认值

    Returns:
        提取的值列表
    """
    try:
        x_obj = json.loads(x)
        return [z.get(key, fail_value) for z in x_obj]
    except Exception as e:
        print(f"JSON解析错误: {e}")
        return [fail_value]


def list_get_join(
    x: List[List[Any]], indices: List[int], sep: str, fail_value: str = "null"
) -> List[str]:
    """从二维列表中提取指定索引的元素并用分隔符连接

    例如：
    输入: [['300308', '33', 'aStock'], ['000001', '16', 'index']]
    indices: [0, 1]
    sep: ","
    输出: ['300308,33', '000001,16']

    Args:
        x: 二维列表
        indices: 要提取的元素索引列表
        sep: 连接分隔符
        fail_value: 提取失败时返回的默认值

    Returns:
        连接后的字符串列表
    """
    result = []
    for item in x:
        try:
            target_elements = [item[idx] for idx in indices]
            result.append(f"{sep}".join(target_elements))
        except IndexError as e:
            print(f"索引错误 {e}，元素值: {item}")
            result.append(fail_value)
    return result


def int_max(x: int, max_value: int) -> int:
    """返回输入值和最大值中的较大者

    Args:
        x: 输入整数
        max_value: 最大值限制

    Returns:
        x和max_value中的较大值
    """
    return max(x, max_value)


def has_intersection(x: List[Any], y: List[Any], exclude=MISSING_VALUE) -> int:
    """检查两个列表在排除特定值后是否有交集

    Args:
        x: 第一个列表
        y: 第二个列表
        exclude: 要排除的值列表，默认为MISSING_VALUE

    Returns:
        如果有交集返回1，否则返回0
    """
    a = set(x) - set(exclude)
    b = set(y) - set(exclude)
    return int(len(set(a) & set(b)) > 0)
