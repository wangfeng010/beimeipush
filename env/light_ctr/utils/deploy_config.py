from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ColumnQualifier:
    """
    表示列限定符的类。

    属性:
    - qualifier: str，限定符字符串。
    - field: str，字段名称字符串。
    """

    qualifier: str
    field: str


@dataclass
class OfflineFeature:
    """
    表示离线特征的数据类。

    属性:
    - source: 特征的来源，可选，默认为None。
    - column_family: 列族名称，可选，默认为None。
    - column_qualifiers: 列限定符列表，可选，默认为一个空列表。
    """

    source: Optional[str] = None
    column_family: Optional[str] = None
    column_qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class Sourceside:
    """
    定义了一个数据类，表示数据源及其离线特征。

    属性:
    - sourceside (str): 数据源的名称或标识。
    - offlineFeature (OfflineFeature): 离线特征对象，包含与数据源相关的离线特征信息。
    """

    sourceside: str
    offlineFeature: OfflineFeature


@dataclass
class RequestContext:
    """
    请求上下文类，用于存储列限定符。

    属性:
    - qualifiers: 列限定符列表，默认为空列表。
      该列表用于存储与请求相关的列限定符信息。
    """

    qualifiers: List[ColumnQualifier] = field(default_factory=list)


# 特殊的item_context，表示当前数据的具体类型：发现场景。
@dataclass
class ItemStreamingPage:
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


# 特殊的item_context，表示当前数据的具体类型：feed场景。
@dataclass
class ItemStreamingFeed:
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


# 特殊的item_context，表示当前数据的具体类型：论股堂场景。
@dataclass
class ItemStreamingLungutang:
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class ItemContext:
    item_streaming_page: Optional[ItemStreamingPage] = None
    item_streaming_feed: Optional[ItemStreamingFeed] = None
    item_streaming_lungutang: Optional[ItemStreamingLungutang] = None


@dataclass
class Context:
    """
    上下文类，包含请求上下文和项目上下文。

    属性:
    - request_context (List[ColumnQualifier]): 请求上下文列表，默认为空列表。
    - item_context (ItemContext): 项目上下文，默认为一个新的ItemContext实例。
    """

    request_context: List[ColumnQualifier] = field(default_factory=list)
    item_context: ItemContext = field(default_factory=ItemContext)


@dataclass
class RawData:
    """
    原始数据类，包含数据源和上下文信息。

    属性:
    - sourcesides (List[Sourceside]): 数据源列表，默认为空列表。
    - context (Context): 上下文信息，默认为一个新的Context实例。
    """

    sourcesides: List[Sourceside] = field(default_factory=list)
    context: Context = field(default_factory=Context)


def parse_yaml_to_dataclass(yaml_file: str) -> RawData:
    """
    解析YAML文件并将其内容转换为RawData数据类实例。

    :param yaml_file: 要解析的YAML文件的路径。
    :return: 包含YAML文件内容的RawData实例。
    """
    with open(yaml_file, "r") as file:
        # 使用PyYAML库安全地加载YAML文件内容。
        data = yaml.safe_load(file)

    raw_data = RawData(
        sourcesides=[
            Sourceside(
                sourceside=s["sourceside"],
                offlineFeature=OfflineFeature(
                    source=s.get("offlineFeature", {}).get("source"),
                    column_family=s.get("offlineFeature", {}).get("column_family"),
                    # 解析并生成列限定符列表，用于离线特征。
                    column_qualifiers=[
                        ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
                        for q in s.get("offlineFeature", {}).get(
                            "column_qualifiers", []
                        )
                    ],
                ),
            )
            for s in data.get("raw_data", {}).get("sourcesides", [])
        ],
        context=Context(
            # 解析并生成请求上下文中的列限定符列表。
            request_context=[
                ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
                for q in data.get("raw_data", {})
                .get("context", {})
                .get("request_context", [])
            ],
            item_context=ItemContext(
                item_streaming_page=ItemStreamingPage(
                    # 解析并生成item_streaming_page中的列限定符列表。
                    qualifiers=[
                        ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
                        for q in data.get("raw_data", {})
                        .get("context", {})
                        .get("item_context", {})
                        .get("item_streaming_page", [])
                    ]
                ),
                item_streaming_feed=ItemStreamingFeed(
                    # 解析并生成item_streaming_feed中的列限定符列表。
                    qualifiers=[
                        ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
                        for q in data.get("raw_data", {})
                        .get("context", {})
                        .get("item_context", {})
                        .get("item_streaming_feed", [])
                    ]
                ),
                item_streaming_lungutang=ItemStreamingLungutang(
                    # 解析并生成item_streaming_lungutang中的列限定符列表。
                    qualifiers=[
                        ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
                        for q in data.get("raw_data", {})
                        .get("context", {})
                        .get("item_context", {})
                        .get("item_streaming_lungutang", [])
                    ]
                ),
            ),
        ),
    )
    return raw_data


if __name__ == "__main__":
    # 使用示例
    from pprint import pprint

    yaml_file = "/home/jovyan/lightctr/config/deploy.yml"
    raw_data = parse_yaml_to_dataclass(yaml_file)
    pprint(raw_data)
