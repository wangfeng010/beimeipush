# 特征配置使用指南

## 🎯 简单使用方法

通过修改 `config/feat.yml` 文件来控制特征排除，然后正常运行训练脚本。

## 📁 配置文件位置

`config/feat.yml` 文件中的 `exclude_features` 段：

```yaml
exclude_features:
  current: default  # 当前使用的配置
  
  # 可选配置
  default: []  # 不排除任何特征
  exclude_user_propernoun: [user_propernoun]  # 排除用户实体词特征
  exclude_user_info: [user_id, user_propernoun]  # 排除用户信息特征
  exclude_user_behavior: [user_watch_stk_code, prefer_bid_code, hold_bid_code, user_propernoun]  # 排除用户行为特征
```

## 🚀 使用步骤

### 1. 修改配置文件

编辑 `config/feat.yml`，修改 `current` 字段：

```yaml
exclude_features:
  current: exclude_user_propernoun  # 改为想要的配置
```

### 2. 运行训练脚本

```bash
# 树模型训练
python src/train.py

# MLP模型训练  
python src/train_MLP.py
```

## 📊 预定义配置

| 配置名称 | 排除特征 | 用途 |
|----------|----------|------|
| `default` | 无 | 基准实验，包含所有特征 |
| `exclude_user_propernoun` | `user_propernoun` | 测试用户实体词特征影响 |
| `exclude_user_info` | `user_id`, `user_propernoun` | 隐私保护场景 |
| `exclude_user_behavior` | 所有用户行为特征 | 测试纯内容特征效果 |

## 🔧 添加自定义配置

在 `config/feat.yml` 中添加新的配置：

```yaml
exclude_features:
  current: my_experiment
  
  # 原有配置...
  
  # 新增自定义配置
  my_experiment: [feature1, feature2, feature3]
```

## ✅ 验证配置生效

运行训练脚本时，控制台会输出：

```
使用当前配置 [exclude_user_propernoun]: ['user_propernoun']
特征过滤结果: 总共12个管道，排除了1个管道，保留11个管道
```

确认配置已正确加载和应用。 