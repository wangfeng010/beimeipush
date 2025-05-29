from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import yaml


@dataclass
class ColumnQualifier:
    """
    表示列限定符的类。

    Attributes:
        qualifier: 限定符字符串
        field: 字段名称字符串
    """
    qualifier: str
    field: str


@dataclass
class OfflineFeature:
    """
    表示离线特征的数据类。

    Attributes:
        source: 特征数据来源
        column_family: HBase列族名称
        column_qualifiers: 列限定符列表
    """
    source: Optional[str] = None
    column_family: Optional[str] = None
    column_qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class DataSource:
    """
    定义数据源及其离线特征。

    Attributes:
        name: 数据源的名称或标识
        offline_feature: 离线特征对象，包含与数据源相关的离线特征信息
    """
    name: str  # 原来是sourceside，避免与类名重复
    offline_feature: OfflineFeature  # 改为下划线命名风格


@dataclass
class ItemStreamingPage:
    """特殊的item_context，表示当前数据类型：发现场景。"""
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class ItemStreamingFeed:
    """特殊的item_context，表示当前数据类型：feed场景。"""
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class ItemStreamingLungutang:
    """特殊的item_context，表示当前数据类型：论股堂场景。"""
    qualifiers: List[ColumnQualifier] = field(default_factory=list)


@dataclass
class ItemContext:
    """
    项目上下文类，包含不同场景的上下文信息。
    
    Attributes:
        item_streaming_page: 发现页面流数据上下文
        item_streaming_feed: Feed流数据上下文
        item_streaming_lungutang: 论股堂流数据上下文
    """
    item_streaming_page: Optional[ItemStreamingPage] = None
    item_streaming_feed: Optional[ItemStreamingFeed] = None
    item_streaming_lungutang: Optional[ItemStreamingLungutang] = None


@dataclass
class Context:
    """
    上下文类，包含请求上下文和项目上下文。

    Attributes:
        request_context: 请求上下文中的列限定符列表
        item_context: 项目上下文对象
    """
    request_context: List[ColumnQualifier] = field(default_factory=list)
    item_context: ItemContext = field(default_factory=ItemContext)


@dataclass
class RawData:
    """
    原始数据类，包含数据源和上下文信息。

    Attributes:
        data_sources: 数据源列表
        context: 上下文信息
    """
    data_sources: List[DataSource] = field(default_factory=list)  # 原来是sourcesides
    context: Context = field(default_factory=Context)


def _parse_column_qualifiers(qualifiers_data: List[Dict[str, str]]) -> List[ColumnQualifier]:
    """
    解析列限定符数据并转换为ColumnQualifier对象列表。
    
    Args:
        qualifiers_data: 包含qualifier和field键值对的字典列表
        
    Returns:
        ColumnQualifier对象列表
    """
    return [
        ColumnQualifier(qualifier=q["qualifier"], field=q["field"])
        for q in qualifiers_data or []
    ]


def _parse_item_context(context_data: Dict[str, Any]) -> ItemContext:
    """
    解析项目上下文数据并转换为ItemContext对象。
    
    Args:
        context_data: 包含项目上下文数据的字典
        
    Returns:
        ItemContext对象
    """
    item_context_data = context_data.get("item_context", {})
    
    return ItemContext(
        item_streaming_page=ItemStreamingPage(
            qualifiers=_parse_column_qualifiers(
                item_context_data.get("item_streaming_page", [])
            )
        ),
        item_streaming_feed=ItemStreamingFeed(
            qualifiers=_parse_column_qualifiers(
                item_context_data.get("item_streaming_feed", [])
            )
        ),
        item_streaming_lungutang=ItemStreamingLungutang(
            qualifiers=_parse_column_qualifiers(
                item_context_data.get("item_streaming_lungutang", [])
            )
        ),
    )


def _parse_data_sources(sources_data: List[Dict[str, Any]]) -> List[DataSource]:
    """
    解析数据源数据并转换为DataSource对象列表。
    
    Args:
        sources_data: 包含数据源信息的字典列表
        
    Returns:
        DataSource对象列表
    """
    data_sources = []
    
    for source in sources_data or []:
        offline_feature_data = source.get("offlineFeature", {})
        
        data_sources.append(
            DataSource(
                name=source["sourceside"],  # 这里保持原字段名，以保证YAML解析兼容
                offline_feature=OfflineFeature(
                    source=offline_feature_data.get("source"),
                    column_family=offline_feature_data.get("column_family"),
                    column_qualifiers=_parse_column_qualifiers(
                        offline_feature_data.get("column_qualifiers", [])
                    ),
                ),
            )
        )
    
    return data_sources


def parse_yaml_to_dataclass(yaml_file: str) -> RawData:
    """
    解析YAML文件并将其内容转换为RawData数据类实例。

    Args:
        yaml_file: 要解析的YAML文件的路径
        
    Returns:
        包含YAML文件内容的RawData实例
    """
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    raw_data_dict = data.get("raw_data", {})
    
    # 创建RawData实例，使用辅助函数处理嵌套结构
    raw_data = RawData(
        data_sources=_parse_data_sources(raw_data_dict.get("sourcesides", [])),
        context=Context(
            request_context=_parse_column_qualifiers(
                raw_data_dict.get("context", {}).get("request_context", [])
            ),
            item_context=_parse_item_context(raw_data_dict.get("context", {})),
            ),
    )
    
    return raw_data


if __name__ == "__main__":
    # 使用示例
    import os
    from pprint import pprint

    # 使用相对路径，更加灵活
    yaml_file_path = os.environ.get(
        "DEPLOY_CONFIG_PATH", 
        "/home/jovyan/lightctr/config/deploy.yml"
    )
    
    try:
        raw_data = parse_yaml_to_dataclass(yaml_file_path)
        print(f"成功加载配置文件: {yaml_file_path}")
    pprint(raw_data)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
