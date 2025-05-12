基于树模型的Push排序原版

项目本身的Readme
基于树模型的Push排序
一、服务流程
1. 在算力中心离线训练模型；
2. 同顺云运行排序服务。
二、部署流程
2.1 离线训练
1. 同顺云打包当前镜像；
2. 镜像同步至北美；
3. 算力中心的可视化建模中建立任务；
4. 启动训练任务，启动脚本为：
set -eu

cd /root/port_service

cat ./hosts >> /etc/hosts

export PYTHONPATH=`pwd`
/opt/conda/bin/python3.8 utils/download.py
ls -lh data/train
/opt/conda/bin/python3.8 train.py 
2.2 运行服务
在同顺云走正式发布流程。
三、开发
3.1 开发环境
Python >= 3.8
pip install -r requirements.txt
3.2 数据准备
下载训练数据，放在训练数据的目录data/train下。
下载用户数据，放在用户数据目录下data/user。
3.3 本地模型训练
python train.py
3.4 本地启动服务
uvicorn infer:app --reload --port=12333
3.5 本地测试服务
curl -X POST \
http://localhost:12333/predict/ \
-H 'cache-control: no-cache' \
-H 'content-type: application/json' \
-H 'postman-token: d9a5645d-6868-a5a4-5891-31f6ca4cfb23' \
-d '{
    "items": [
        {
            "item_id": "0710e8b65d68863f",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Vietnamese Coast Guard Delegation Visits Philippines for Joint Training and Maritime Cooperation",
            "item_code": "[{\"market\":\"169\",\"score\":1,\"code\":\"WAT\",\"name\":\"Waters\",\"type\":0,\"parentId\":\"0710e8b65d68863f\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "0710e8b65d68863f",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Vietnamese Coast Guard Delegation Visits Philippines for Joint Training and Maritime Cooperation",
            "item_code": "[{\"market\":\"169\",\"score\":1,\"code\":\"WAT\",\"name\":\"Waters\",\"type\":0,\"parentId\":\"0710e8b65d68863f\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "0710e8b65d68863f",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Vietnamese Coast Guard Delegation Visits Philippines for Joint Training and Maritime Cooperation",
            "item_code": "[{\"market\":\"169\",\"score\":1,\"code\":\"WAT\",\"name\":\"Waters\",\"type\":0,\"parentId\":\"0710e8b65d68863f\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "0710e8b65d68863f",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Vietnamese Coast Guard Delegation Visits Philippines for Joint Training and Maritime Cooperation",
            "item_code": "[{\"market\":\"169\",\"score\":1,\"code\":\"WAT\",\"name\":\"Waters\",\"type\":0,\"parentId\":\"0710e8b65d68863f\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07194a9106c5708a",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Germany Proposes Reforms for Efficient and Climate-Neutral Electricity System",
            "item_code": "[{\"market\":\"185\",\"score\":1,\"code\":\"FOSL\",\"name\":\"Fossil Group\",\"type\":0,\"parentId\":\"07194a9106c5708a\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07194a9106c5708a",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Germany Proposes Reforms for Efficient and Climate-Neutral Electricity System",
            "item_code": "[{\"market\":\"185\",\"score\":1,\"code\":\"FOSL\",\"name\":\"Fossil Group\",\"type\":0,\"parentId\":\"07194a9106c5708a\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07194a9106c5708a",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Germany Proposes Reforms for Efficient and Climate-Neutral Electricity System",
            "item_code": "[{\"market\":\"185\",\"score\":1,\"code\":\"FOSL\",\"name\":\"Fossil Group\",\"type\":0,\"parentId\":\"07194a9106c5708a\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07194a9106c5708a",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Germany Proposes Reforms for Efficient and Climate-Neutral Electricity System",
            "item_code": "[{\"market\":\"185\",\"score\":1,\"code\":\"FOSL\",\"name\":\"Fossil Group\",\"type\":0,\"parentId\":\"07194a9106c5708a\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "072a65ac75e751b4",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "Antelope Enterprise Hits Triple Whammy: MACD, KDJ Death Crosses, Bearish Marubozu",
            "item_code": "[{\"market\":\"186\",\"score\":1,\"code\":\"AEHL\",\"name\":\"Antelope Enterprise\",\"type\":0,\"parentId\":\"072a65ac75e751b4\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "072a65ac75e751b4",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "Antelope Enterprise Hits Triple Whammy: MACD, KDJ Death Crosses, Bearish Marubozu",
            "item_code": "[{\"market\":\"186\",\"score\":1,\"code\":\"AEHL\",\"name\":\"Antelope Enterprise\",\"type\":0,\"parentId\":\"072a65ac75e751b4\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "072a65ac75e751b4",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "Antelope Enterprise Hits Triple Whammy: MACD, KDJ Death Crosses, Bearish Marubozu",
            "item_code": "[{\"market\":\"186\",\"score\":1,\"code\":\"AEHL\",\"name\":\"Antelope Enterprise\",\"type\":0,\"parentId\":\"072a65ac75e751b4\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "072a65ac75e751b4",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "Antelope Enterprise Hits Triple Whammy: MACD, KDJ Death Crosses, Bearish Marubozu",
            "item_code": "[{\"market\":\"186\",\"score\":1,\"code\":\"AEHL\",\"name\":\"Antelope Enterprise\",\"type\":0,\"parentId\":\"072a65ac75e751b4\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "075e2bfd83972a4b",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "ATN International 15-Min Chart Shows Bullish KDJ Cross, Marubozu Signal",
            "item_code": "[{\"code\":\"ATNI\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"185\",\"score\":1,\"name\":\"ATN International\"},{\"code\":\"VZ\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"169\",\"score\":1,\"name\":\"Verizon\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "075e2bfd83972a4b",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "ATN International 15-Min Chart Shows Bullish KDJ Cross, Marubozu Signal",
            "item_code": "[{\"code\":\"ATNI\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"185\",\"score\":1,\"name\":\"ATN International\"},{\"code\":\"VZ\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"169\",\"score\":1,\"name\":\"Verizon\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "075e2bfd83972a4b",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "ATN International 15-Min Chart Shows Bullish KDJ Cross, Marubozu Signal",
            "item_code": "[{\"code\":\"ATNI\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"185\",\"score\":1,\"name\":\"ATN International\"},{\"code\":\"VZ\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"169\",\"score\":1,\"name\":\"Verizon\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "075e2bfd83972a4b",
            "come_from": "999680",
            "push_title": "Breaking News",
            "push_content": "ATN International 15-Min Chart Shows Bullish KDJ Cross, Marubozu Signal",
            "item_code": "[{\"code\":\"ATNI\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"185\",\"score\":1,\"name\":\"ATN International\"},{\"code\":\"VZ\",\"type\":0,\"parentId\":\"075e2bfd83972a4b\",\"market\":\"169\",\"score\":1,\"name\":\"Verizon\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07679edf9b0eac5e",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Why Boston Scientific Corporation (BSX) Is Attracting Investment According to Baron Funds",
            "item_code": "[{\"code\":\"BSX\",\"type\":0,\"parentId\":\"07679edf9b0eac5e\",\"market\":\"169\",\"score\":1,\"name\":\"Boston Scientific\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07679edf9b0eac5e",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Why Boston Scientific Corporation (BSX) Is Attracting Investment According to Baron Funds",
            "item_code": "[{\"code\":\"BSX\",\"type\":0,\"parentId\":\"07679edf9b0eac5e\",\"market\":\"169\",\"score\":1,\"name\":\"Boston Scientific\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07679edf9b0eac5e",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Why Boston Scientific Corporation (BSX) Is Attracting Investment According to Baron Funds",
            "item_code": "[{\"code\":\"BSX\",\"type\":0,\"parentId\":\"07679edf9b0eac5e\",\"market\":\"169\",\"score\":1,\"name\":\"Boston Scientific\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        },
        {
            "item_id": "07679edf9b0eac5e",
            "come_from": "999679",
            "push_title": "Breaking News",
            "push_content": "Why Boston Scientific Corporation (BSX) Is Attracting Investment According to Baron Funds",
            "item_code": "[{\"code\":\"BSX\",\"type\":0,\"parentId\":\"07679edf9b0eac5e\",\"market\":\"169\",\"score\":1,\"name\":\"Boston Scientific\"}]",
            "submit_type": "autoFlash",
            "create_time": "2021-08-05 09:05:03"
        }
    ]
}'
这是一个讲解很好的项目概览（核心组件 + 主要功能流程 + 技术栈 是一个很好的梳理框架）
02ainvest-push-recall-group-master (推送系统排序模型)
这是一个基于树模型的推送排序系统，用于对金融市场的推送消息进行排序和个性化推送。该项目使用LightGBM构建了一个二分类模型，用于预测用户对不同金融消息的点击可能性。
核心组件
1. 模型架构：
  ○ 使用LightGBM作为基础模型，配置为二分类任务(objective: binary)
  ○ 评估指标为AUC，使用GBDT作为提升方法
  ○ 具有早停机制，最大迭代次数为300
2. 数据处理与特征工程：
  ○ 处理用户特征：观察列表(watchlists)、持仓(holdings)、国家(country)、偏好出价(prefer_bid)
  ○ 处理推送内容特征：推送标题(push_title)、推送内容(push_content)、项目代码(item_code)、标签(item_tags)
  ○ 时间特征：小时(hour)、星期(weekday)
  ○ 使用哈希编码方法处理高基数分类特征
  ○ 处理变长度稀疏特征(varlen_sparse_feat)，如用户观察的股票代码列表
3. 服务架构：
  ○ 模型训练在算力中心离线进行
  ○ 推理服务使用FastAPI实现
  ○ 定时任务：每日更新用户数据，每小时更新新用户数据
  ○ 支持批量预测和异步预测
4. 部署流程：
  ○ 训练后模型存储在HDFS中
  ○ 在同顺云上运行排序服务
  ○ 支持模型动态更新
主要功能流程
1. 训练流程：
  ○ 从Hive下载训练数据
  ○ 数据预处理，特征转换和工程
  ○ 使用LightGBM训练模型
  ○ 将模型上传到HDFS存储
2. 推理流程：
  ○ 加载用户特征数据和模型
  ○ 接收推送项目列表(items)
  ○ 为每个用户-项目对生成特征
  ○ 预测点击概率
  ○ 根据预测分数排序推送项目
  ○ 支持同步和异步预测接口
3. 定时任务：
  ○ 每日任务：更新用户基础数据
  ○ 每小时任务：更新新增用户数据
  ○ 定期检查和更新模型
技术栈
● 编程语言：Python 3.8+
● 机器学习框架：LightGBM
● Web服务：FastAPI
● 调度系统：APScheduler
● 数据处理：Pandas
● 配置系统：YAML
● 部署：Docker
这个项目是一个金融推送消息的个性化排序系统，结合了用户行为数据和金融消息内容，通过机器学习模型预测用户对不同消息的兴趣程度，以优化推送效果和用户体验。系统采用了模块化设计，支持离线训练和在线推理，并实现了定期更新用户数据和模型的机制。
你这个框架梳理的非常好 你可以按照这个框架的逻辑 和项目的具体代码做一个对照再梳理一下嘛 你不用给出代码细节 告诉我是哪个代码文件就可以了 或者代码文件哪个类或函数就可以
我将按照前面的框架逻辑，结合项目的具体代码文件、类和函数进行更详细的对照梳理：
核心组件对照
1. 模型架构
● 配置文件: config/model.yml - 定义LightGBM模型参数（boosting_type, num_leaves, max_depth等）
● 模型类: model.py 中的 PushClassifier 类
● 训练函数: PushClassifier.train() - 在 model.py 中实现模型训练
● 超参数设置: 二分类任务、AUC评估、GBDT提升、早停等都在 config/model.yml 中定义
2. 数据处理与特征工程
● 数据预处理: utils/preprocess.py 中的 data_preprocess() 和 run_one_op_pd() 函数
● 特征定义: config/config.yml 中的 process.pipelines 部分定义各种特征处理流程
● 自定义操作: utils/preprocess.py 中的 SELF_OP_HUB 字典定义了特殊操作函数
● 变长特征处理: model.py 中的 _feat_selection() 和 utils/preprocess.py 中的 data_preprocess() 函数末尾部分
3. 服务架构
● Web服务: infer.py 中使用FastAPI实现的 API 端点
● 定时任务: 
  ○ 每日任务: infer.py 中的 daily_task() 函数
  ○ 每小时任务: infer.py 中的 hour_task() 函数
● 应用生命周期: infer.py 中的 app_lifespan() 函数管理服务启动和关闭时的资源加载
● 配置管理: utils/dtypes.py 中的 InferConfig 类定义推理服务的配置
4. 部署流程
● 模型加载: infer.py 中的 load_model() 函数
● HDFS交互: utils/hdfs.py 中的 download_hdfs() 和 upload_hdfs() 函数
● 容器化: 根目录的 Dockerfile 文件定义容器构建方式
● 训练脚本: train.sh 包含训练环境的初始化和训练执行命令
主要功能流程对照
1. 训练流程
● 数据下载: train.py 中的 data_download() 函数和 utils/download.py 中的 get_multi_data()
● 模型训练入口: train.py 中的 main() 函数
● 数据准备: model.py 中的 _prepare_input() 函数
● 模型训练实现: model.py 中的 train() 函数
● 模型上传: train.py 中使用 utils/hdfs.py 的 upload_hdfs() 函数
2. 推理流程
● API 端点: 
  ○ infer.py 中的 /predict/、/predict_v2/、/predict_async/、/predict_async_v2/ 多个接口
● 预测核心: 
  ○ infer.py 中的 ml_predict() 函数整合预测流程
  ○ predict_items() 函数实现具体预测逻辑
● 特征生成: 
  ○ infer.py 中的 data_to_feat() 函数
● 异步处理: 
  ○ predict_async() 和 predict_async_v2() 函数结合 BackgroundTasks 实现异步操作
● 推送发送:
  ○ predict_send() 和 predict_send_v2() 函数实现预测结果推送
3. 定时任务
● 任务调度: infer.py 中 app_lifespan() 函数中的 scheduler 设置
● 日任务实现: infer.py 中的 daily_task() 函数
● 小时任务实现: infer.py 中的 hour_task() 函数
● 用户数据下载: utils/download.py 中的 download_user_data() 函数
● 模型更新: infer.py 中的 /update_model 接口
辅助工具和组件
● 数据类型定义: utils/dtypes.py 中的 PushItems、InferConfig 等类
● 向量数据库交互: utils/vector_db.py 提供向量检索功能
● SQL查询: utils/download.py 中的 execute_sql_args() 函数
● 健康检查: infer.py 中的 /readiness 接口
这个系统的代码组织遵循了职责分离原则，模型定义、训练、推理、数据处理和API服务都有清晰的模块划分。配置也被合理地分离到不同的YAML文件中，方便在不修改代码的情况下调整系统行为。
你这个框架梳理的非常好 你可以按照这个框架的逻辑 和项目的具体代码做一个对照再梳理一下嘛 你不用给出代码细节 告诉我是哪个代码文件就可以了 或者代码文件哪个类或函数就可以
推荐代码阅读顺序
1. 项目整体结构和配置
● README.md - 了解项目功能、使用方法和部署流程
● config/config.yml - 全局配置，特别关注数据集定义和特征处理流程
● config/model.yml - 模型超参数配置
● config/infer.yml - 推理服务配置
2. 基础数据结构和类型
● utils/dtypes.py - 了解 InferConfig、PushItems 等核心数据类型
● requirements.txt - 查看项目依赖，了解技术栈
3. 训练流程
● train.py - 训练入口，了解整体训练流程和数据下载逻辑
● model.py - 核心模型类 PushClassifier，重点关注：
  ○ _data_preprocess() - 数据预处理
  ○ _feat_selection() - 特征选择和变长特征处理
  ○ train() - 训练实现
4. 特征工程和预处理
● utils/preprocess.py - 关注以下部分：
  ○ SELF_OP_HUB - 自定义操作函数
  ○ run_one_op_pd() - 单个操作执行逻辑
  ○ data_preprocess() - 完整预处理流程
5. 数据获取流程
● utils/download.py - 了解数据下载机制，特别是：
  ○ download_user_data() - 用户数据下载
  ○ get_multi_data() - 训练数据获取
6. 推理服务架构
● infer.py - 推理服务主文件，按以下顺序关注：
  ○ app_lifespan() - 服务生命周期管理
  ○ load_model() - 模型加载
  ○ daily_task() 和 hour_task() - 定时任务实现
7. 推理核心逻辑
● infer.py 中的核心函数：
  ○ data_to_feat() - 特征生成
  ○ predict_items() - 预测逻辑实现
  ○ ml_predict() - 整合预测流程
8. API端点和服务接口
● infer.py 中的API端点：
  ○ /predict/ 和 /predict_v2/ - 同步预测接口
  ○ /predict_async/ 和 /predict_async_v2/ - 异步预测接口
  ○ /update_model - 模型更新接口
  ○ /readiness - 健康检查接口
9. 辅助工具
● utils/hdfs.py - HDFS交互，模型存储和加载
● utils/vector_db.py - 向量数据库交互（如有需要）
10. 部署配置
● Dockerfile - 了解容器化配置
● train.sh - 训练脚本配置
● hosts - 网络配置
阅读技巧
1. 先概览，后深入：每个文件先快速浏览，了解整体结构和主要组件，再深入细节
2. 跟随数据流：按照数据在系统中的流动顺序（获取→预处理→训练/推理→输出）理解代码
3. 配置与实现对照：阅读代码实现时，同时参考配置文件，理解配置如何影响实现
4. 关注关键类和函数：优先关注核心类（如PushClassifier）和关键函数（如ml_predict）
5. 理解特征工程：特别关注config.yml中的特征定义和preprocess.py中的特征处理逻辑
这个阅读顺序从整体概念到具体实现，遵循了系统的数据流和逻辑流，能帮助你有条理地理解这个基于树模型的推送排序系统的设计思路和实现细节。
config/config.yml —— sql下来的数据集预处理（训练集基本信息 + 特征处理pipelines + 构建交叉特征处理pipelines + label处理pipelines）这个讲的挺清楚的
config.yml 是整个项目的核心配置文件，决定了模型训练和推理的大部分行为。我将从几个主要部分来解析这个配置文件：
1. 数据集配置 (datasets)
文件开头定义了训练数据集的基本信息：
● data_dir: 训练数据存放位置 (data/train)
● sep: 分隔符使用制表符 (\t)
● header: 无表头 (null)
● file_num: 使用7个文件进行训练
● label_columns: 标签列为 "log_type"（用户行为类型）
● raw_columns: 原始数据列定义，包括：
  ○ 用户信息: user_id, country, watchlists(关注列表), holdings(持仓), prefer_bid(偏好)
  ○ 推送内容: push_title, push_content, item_code, item_tags
  ○ 行为数据: log_type, create_time, submit_type
2. 特征处理流程 (process)
这部分定义了详细的特征工程流程，每个特征都有独立的处理管道(pipeline)：
时间特征
● hour: 从create_time提取小时信息，作为24类别的稀疏特征
● weekday: 从create_time提取星期信息，作为7类别的稀疏特征
用户特征
● user_watch_stk_code_hash: 用户关注的股票代码，处理流程包括：
  ○ 填充缺失值
  ○ 按 "&" 分割多个股票
  ○ 按 "_" 分割代码和市场
  ○ 提取股票代码
  ○ 移除特定股票('AAPL', 'AMZN', 'GOOGL', 'TSLA')
  ○ 填充至固定长度(5)
  ○ 哈希编码至10000维空间
● country_hash: 用户国家，哈希编码至200维空间
● prefer_bid_code_hash: 用户偏好股票，处理流程类似关注列表
● hold_bid_code_hash: 用户持仓股票，处理流程类似关注列表
内容特征
● push_title_hash: 推送标题，哈希编码至8维空间
● title_len: 推送内容长度，按空格分割后计算词数，限制最大值为31
● item_code_hash: 推送相关股票代码，从JSON中提取，哈希编码
● submit_type_hash: 提交类型，哈希编码至10维空间
● tag_id_hash: 推送标签ID，从JSON中提取，哈希编码至100维空间
3. 特征交叉 (interactions)
这部分创建了用户特征与推送内容的交叉特征，核心是检测交集：
● preder_bid_cross: 推送内容的股票与用户偏好股票的交集
● watch_bid_cross: 推送内容的股票与用户关注股票的交集
● hold_bid_cross: 推送内容的股票与用户持仓股票的交集
这些交叉特征本质上是二元特征(0/1)，表示推送内容是否与用户的股票相关。
4. 标签处理 (label_process)
定义了标签(log_type)的处理方式：
● 填充缺失值为"PR"
● 将类别映射为整数：PR->0(未点击), PC->1(点击)
配置文件的核心意义
这个配置文件体现了项目的核心思想：
1. 个性化匹配: 通过用户的关注列表、持仓和偏好，与推送内容的相关股票进行匹配
2. 特征编码标准化: 统一使用哈希编码处理高基数分类特征
3. 变长特征处理: 对变长列表特征(如用户关注多只股票)进行统一处理
4. 交叉特征: 明确定义了用户兴趣与推送内容的交互方式
整个配置表明这是一个面向金融市场的个性化推送系统，核心预测目标是用户是否会点击(PC)特定的市场消息，基于用户的投资兴趣和消息的相关性。
config/model.yml —— 就是LightGBM模型参数的设置（算法，目标，评价指标，树结构，训练参数，正则与随机性）
model.yml 是 LightGBM 模型的具体配置文件，相比较 config.yml 更简洁，但包含了决定模型性能的关键参数。我来详细解析这个文件：
LightGBM 模型参数配置
1. 基础算法选择
  ○ boosting_type: "gbdt" - 选择梯度提升决策树(GBDT)作为集成学习方法。这是LightGBM的默认且最常用的提升方式，通过顺序构建决策树来拟合残差。
2. 目标和评估指标
  ○ objective: "binary" - 设置为二分类任务，对应预测用户是否会点击推送消息(PC vs PR)
  ○ metric: "auc" - 使用AUC(曲线下面积)作为评估指标，适合不平衡的二分类问题
3. 树结构参数
  ○ num_leaves: 31 - 单棵树的最大叶子节点数，较大的值增加模型复杂度
  ○ max_depth: 5 - 树的最大深度限制，防止过拟合
4. 训练过程参数
  ○ learning_rate: 0.01 - 较小的学习率(0.01)，使模型训练更稳定，适合迭代次数较多的场景
  ○ num_iterations: 300 - 构建300棵树，即模型迭代300轮
  ○ early_stopping_rounds: 50 - 如果验证集性能在50轮内没有提升，则提前终止训练，防止过拟合
5. 正则化和随机性参数
  ○ feature_fraction: 0.9 - 每次迭代随机选择90%的特征来构建树，增加模型随机性，预防过拟合
  ○ subsample: 0.8 - 使用80%的数据样本来构建树，也是一种正则化技术
  ○ colsample_bytree: 0.8 - 构建每棵树时使用80%的特征，进一步增加模型多样性
模型配置的策略解读
这个配置体现了以下模型训练策略：
1. 平衡拟合与泛化
  ○ 通过限制树的复杂度(max_depth, num_leaves)防止过拟合
  ○ 同时使用足够的树数量(num_iterations)确保模型有足够的学习能力
2. 增加模型鲁棒性
  ○ 使用特征和样本的随机抽样(feature_fraction, subsample)
  ○ 这些随机化策略类似于随机森林的思想，但在梯度提升框架下
3. 优化训练效率
  ○ 采用较小的学习率(0.01)和早停机制，寻找最优训练轮次
  ○ 早停机制(early_stopping_rounds)避免不必要的计算，同时选择最佳模型
4. 聚焦排序质量
  ○ 选择AUC作为评估指标，更关注样本的排序正确性而非精确的概率值
  ○ 这与推送排序的业务需求高度一致，因为最终目标是将用户最可能点击的内容排在前面
这个配置文件虽然简短，但涵盖了LightGBM的核心参数设置，是一个相对保守且平衡的配置。这种配置适合处理用户行为数据这类噪声较大、分布不均的数据集，既不会过度拟合训练数据，又能保持足够的表达能力。
与config.yml结合来看，整个模型是一个典型的"特征工程+树模型"方案，先通过精细的特征工程构建有区分度的特征，再用树模型学习这些特征与点击行为的关系。
config/infer.yml —— 部署：本地模型pth文件存到HDFS里哪个位置；数据获取：sql代码在哪，sql出来的数据存哪，怎么定时运行 day h，一次下多少 文件 user个数，数据留多久，报错尝试几次，下下来的数据结构是什么样子的；服务API端口定义
infer.yml 是推理服务的配置文件，定义了模型部署和在线预测的关键参数。我来详细解析这个配置：
模型和存储路径
● ml_model_dir: "checkpoint/model.pth" - 本地模型文件路径
● ml_model_hdfs_dir: "dongshaojie/ainvest_push/tree_model" - HDFS上的模型存储路径
● user_data_dir: data/user - 用户数据存储目录
● new_user_data_dir: data/new_user - 新用户数据存储目录
定时任务配置
● dowload_start_hour: 3 - 每日下载任务的开始小时（凌晨3点）
● dowload_start_minute: 0 - 开始分钟（0分）
● hour_task_minute: 35 - 每小时的第35分钟执行小时任务
数据管理参数
● user_data_file_num: 5 - 用户数据文件数量
● max_user_num_per_iter: 20000 - 每次迭代处理的最大用户数量，用于分批处理
● user_data_save_days: 5 - 保留最近5天的用户数据文件，超过会被删除
● download_max_try: 3 - 下载数据的最大尝试次数
SQL查询配置
● user_data_sql_dir: data/user.sql - 用户数据SQL查询文件路径
● new_user_data_sql_dir: data/new_user.sql - 新用户数据SQL查询文件路径
数据格式配置
● sep: '\t' - 数据分隔符为制表符
● header: null - 数据文件无表头
● data_index: user_id - 以user_id作为数据索引
● data_columns - 用户数据文件的列定义：
  ○ user_id - 用户标识
  ○ watchlists - 用户关注的股票列表
  ○ holdings - 用户持仓股票
  ○ country - 用户所在国家
  ○ prefer_bid - 用户偏好出价/股票
模型特征配置
● varlen_max_col_num: 5 - 变长特征的最大列数，与config.yml中的填充长度对应
API端点配置
● push_server_url - 推送服务器V2版API地址
● push_server_url_v3 - 推送服务器V3版API地址
配置文件的核心意义
这个配置文件揭示了在线推理服务的几个关键设计点：
1. 数据更新机制
  ○ 采用定时任务架构，每日凌晨3点更新基础用户数据
  ○ 每小时第35分钟更新新用户数据，确保实时性
  ○ 保留5天历史数据，既满足回溯需求，又控制存储空间
2. 数据处理策略
  ○ 分批处理用户数据(最大20000用户/批)，避免内存溢出
  ○ 失败重试机制(最多尝试3次)，提高系统稳定性
  ○ 当前日期数据不可用时，可使用最近可用数据作为回退方案
3. 服务集成
  ○ 与推送服务器集成，支持多版本API(v2和v3)，便于系统升级
  ○ 通过HDFS连接离线训练和在线推理环境，实现模型更新
4. 资源优化
  ○ 通过定期清理旧数据文件，优化存储使用
  ○ 通过配置变长特征的最大列数，控制特征空间大小
与config.yml和model.yml相比，infer.yml更关注系统层面的运行参数，而非模型本身的结构或特征。这体现了推荐系统的完整架构思想：模型结构、特征工程和服务部署三者缺一不可。
特别值得注意的是，配置中体现了一个生产级推荐系统的鲁棒性设计：多次尝试下载、数据回退策略、定期更新机制等，这些都是确保系统在面对数据问题时仍能继续运行的关键机制。
关于三个配置文件的一些问题：可以的 你讲的很清楚 有几个细节问题你可以给我再讲讲嘛：1.API端口在这个推荐系统架构中是怎么被使用的，是我给一个用户id 给一堆文章 我请求这个API服务 他给我返回这个一堆文章中的几篇这个意思嘛2.那这个API服务和最后存到HDFS上的模型是什么关系呢 是这个API其实就是取HDFS上启用模型 拿到一些结果 最后整理用服务的形式返回满足业务需求的结果吗 3.varlen_max_col_num: 5这个变量的意义其实我没有理解 4.embedding_dim: 8 # 可选，以下方数据管道优先 config.yml里面定义的embedding大小是什么意义 我最后不都是输出的一个多少维度的哈希编码吗 5.哈希编码是什么 为什么不生成一个同样维度的一个embedding 6.这个pipeline是一种yml文件阅读速成的可以执行的结构吗
我来回答你提出的这些细节问题：
1. API端口的使用方式
是的，基本符合你的理解。API服务的基本流程是：
● 客户端发送请求，包含用户ID和候选推送项目列表(items)
● 服务根据用户特征和候选项目特征，计算每个项目的得分
● 返回排序后的项目列表，通常是得分最高的几个
具体在infer.py中，有几个关键API端点：
● /predict/: 同步预测接口
● /predict_async/: 异步预测接口
● /predict_v2/ 和 /predict_async_v2/: 新版本接口
这种架构使推荐系统可以作为微服务集成到更大的系统中，负责排序决策但不负责内容获取和最终推送。
2. API服务与HDFS模型的关系
准确的。流程是：
1. 模型在离线环境训练完成后保存到HDFS
2. API服务启动时从HDFS下载最新模型(load_model函数)
3. API服务加载模型到内存，用于实时预测
4. 通过/update_model接口可以触发模型热更新，不停服务的情况下更新模型
这种分离架构实现了训练与推理的解耦，允许模型定期更新而不影响服务可用性。
3. varlen_max_col_num的意义
这个参数定义了变长特征(如用户关注的股票列表)处理后的最大列数。
例如，用户可能关注10只股票，但系统只保留前5只进行处理。具体实现中：
● 变长特征(如watchlists)被拆分成多个列(watchlists_0, watchlists_1, ...)
● 最多保留5个这样的列，超出部分被截断
这样做是为了：
1. 控制特征维度，防止模型输入过于稀疏
2. 统一不同长度输入的处理方式
3. 提高批处理效率
4. embedding_dim的意义
embedding_dim: 8是一个配置参数，但在这个基于树模型的系统中并未实际用于生成嵌入向量。
它更可能是为了：
1. 保持配置的一致性，可能用于其他版本的模型架构
2. 为未来可能的深度学习模型替换做准备
3. 在某些特征处理步骤中可能会用到的维度参数
在当前的树模型实现中，特征最终还是以哈希编码形式输入，而非嵌入向量。
5. 哈希编码与嵌入的区别
哈希编码是一种简单的特征转换方法：
● 将分类变量(如股票代码)映射到固定范围的整数(0~vocabulary_size-1)
● 使用哈希函数确定映射，无需维护映射表
● 可能存在哈希冲突(不同值映射到相同整数)
与嵌入的区别：
● 嵌入会学习每个类别的低维向量表示，需要训练
● 嵌入通常用于深度学习模型，能捕捉类别之间的语义关系
● 哈希编码更简单，无需训练，适合树模型
使用哈希编码而非嵌入的原因：
1. LightGBM等树模型适合处理离散特征，不需要嵌入
2. 计算效率高，无需额外的向量计算
3. 内存效率高，不需要存储大型嵌入矩阵
4. 处理新类别时无需更新模型
6. pipeline的含义
pipeline不是一个可执行的结构，而是一种配置方式，定义了特征处理的流程：
● 每个pipeline描述一个特征的处理流程
● 包含多个operations(操作)，按顺序执行
● 每个operation定义输入列、输出列、操作函数和参数
实际执行依赖Python代码解析这些配置并应用到数据上，主要在：
● utils/preprocess.py中的run_one_op_pd函数
● model.py中的_data_preprocess函数
这种设计将"特征如何处理"从代码中分离到配置文件，使得：
1. 可以轻松添加/修改特征而无需改代码
2. 处理逻辑更清晰，易于维护
3. 多环境部署时保持一致性
总的来说，这些细节反映了一个成熟的推荐系统架构设计，既考虑了算法效果，也兼顾了工程实现和系统可维护性。
utils/dtypes.py —— 定义模型在模型参数配置 推理配置的数据结构；Tag Stock等内容实体数据结构；ItemVector TargetGroup等与外部系统（向量化数据库）交互的数据结构；以及接受API请求的Pushitems item推送item列表 单个item数据结构。来实现API契约定义、Pydantic数据结构验证和初始值设定、以及系统集成接口定义
utils/dtypes.py 是一个定义数据结构的文件，使用了Pydantic库来创建具有类型验证功能的数据模型。这些模型定义了系统中各种数据实体的结构。我来详细解析这个文件：
核心数据模型概览
1. 配置类模型
● ModelArgs: 深度学习模型的超参数配置（注意：在当前基于树的系统中可能未被实际使用）
  ○ emb_dim: 嵌入维度 (8)
  ○ query_dim_hidden, user_dim_hidden: 隐藏层维度
  ○ dim_out: 输出维度
  ○ margin, bias, activation: 模型训练参数
● InferConfig: 推理服务的配置参数，对应infer.yml文件
  ○ 包含模型路径、数据目录、下载参数、服务URL等
  ○ 对应我们之前讨论的infer.yml配置文件各项内容
2. 内容实体模型
● Tag: 标签实体
  ○ score: 分数/权重
  ○ tagId: 标签ID
  ○ name: 标签名称
  ○ type: 标签类型
  ○ parentId: 父项ID（关联到推送项目）
● Stock: 股票实体
  ○ market: 市场代码
  ○ score: 相关性分数
  ○ code: 股票代码
  ○ name: 股票名称
  ○ type: 类型
  ○ parentId: 父项ID（关联到推送项目）
● ProperNoun: 专有名词实体
  ○ 用于处理文本中的特殊实体
  ○ 包含name, id, type等字段
3. 向量检索相关模型
● ItemVector: 项目向量
  ○ code: 状态码
  ○ item_id: 项目ID
  ○ success: 是否成功
  ○ item_vector: 向量表示（浮点数列表）
  ○ usedTime: 处理时间
● User: 用户实体
  ○ docScore: 文档相关性分数
  ○ user_id: 用户ID
  ○ docId: 文档ID（默认为user_id）
  ○ 包含一个验证器确保docId有值
● TargetGroup 和 DBResult: 数据库查询结果
  ○ 包装用户列表和操作结果信息
4. 推送相关模型
● Item: 推送项目
  ○ item_id: 项目唯一标识
  ○ create_time: 创建时间
  ○ push_title: 推送标题
  ○ push_content: 推送内容
  ○ item_code: 相关股票代码（JSON字符串）
  ○ come_from: 来源
  ○ submit_type: 提交类型
  ○ item_tags: 标签信息（JSON字符串）
● PushItems: 推送项目列表
  ○ items: Item对象列表
  ○ 这是API接口接收的主要输入数据结构
文件的关键作用
dtypes.py 在系统中扮演以下角色：
1. API契约定义
  ○ 定义了API接口的请求和响应数据结构
  ○ PushItems是主要的请求数据结构，用于/predict等接口
2. 数据验证
  ○ 使用Pydantic进行自动数据验证
  ○ 定义默认值，确保系统在数据不完整时也能工作
3. 类型安全
  ○ 提供类型提示，使代码更易于理解和维护
  ○ 防止类型错误，提高系统稳定性
4. 系统集成接口
  ○ 定义了与外部系统（如向量数据库）交互的数据结构
  ○ ItemVector, TargetGroup等可能用于与专用存储系统的集成
与其他模块的关系
● 与infer.py的关系: InferConfig和PushItems直接用于API接口
● 与配置文件的关系: InferConfig对应infer.yml的结构
● 与特征处理的关系: Item定义了特征处理的输入字段，与config.yml中的特征定义对应
特别值得注意的设计
1. 可选字段处理
  ○ 大多数字段设为Optional，提高系统容错性
  ○ 为关键字段提供默认值，确保数据一致性
2. 模型验证器
  ○ User类使用model_validator确保docId字段有值
  ○ 这种设计使数据满足业务逻辑要求
3. 多用途设计
  ○ 既支持纯机器学习模型(LightGBM)，也有深度学习相关的数据结构(ModelArgs)
  ○ 表明系统可能支持或计划支持多种模型类型
dtypes.py虽然代码简单，但对整个系统至关重要，它定义了数据在系统各部分之间流动的格式，确保了系统组件之间的有效通信。从这个文件也可以看出系统的模块化设计思想，各个组件之间通过明确定义的数据结构进行交互。
requirements.txt —— 
requirements.txt 文件列出了项目的依赖包及其版本，这对理解项目的技术栈和限制非常重要。我来详细解析这个文件：
核心依赖分类
1. Web服务相关
● fastapi==0.111.0: 现代化、高性能的Python API框架，用于构建API端点
● uvicorn==0.29.0: ASGI服务器，用于运行FastAPI应用
● requests==2.32.2: HTTP客户端库，用于发送HTTP请求
● aiohttp==3.8.3: 异步HTTP客户端/服务器，用于异步API调用
● watchfiles==0.24.0: 文件监控工具，可能用于开发环境中的热重载
2. 数据处理与科学计算
● numpy==1.22.3: 科学计算基础库，提供多维数组支持
● pandas==2.0.0: 数据分析库，用于表格数据处理
● scipy==1.10.0: 科学计算库，提供各种数学算法
● scikit-learn==0.24.2: 机器学习库，提供预处理、评估等功能
● pandarallel==1.6.5: Pandas并行计算扩展，加速数据处理
3. 机器学习模型
● lightgbm==3.3.5: 高效梯度提升框架，项目的核心模型
● rec-preprocess==0.2.5: 推荐系统预处理库，可能是内部开发的包
4. 配置与日志
● pyyaml==6.0.1: YAML解析库，用于配置文件处理
● loguru==0.7.0: 增强的日志库，比标准logging更易用
5. 任务调度
● APScheduler==3.10.4: 高级Python调度库，用于定时任务
6. 其他工具
● networkx==3.0: 复杂网络分析库，可能用于某些特征工程或分析
技术栈解读
通过requirements.txt可以清晰看出这个项目的技术栈特点：
1. 现代Web技术
  ○ 使用FastAPI构建API，这是当前Python生态中最快的Web框架之一
  ○ 支持异步处理(aiohttp)，适合高并发场景
2. 高效数据处理
  ○ 使用pandas、numpy进行数据操作
  ○ 引入pandarallel提高并行处理能力，适合大规模数据
3. 轻量级机器学习
  ○ 使用LightGBM而非更重的深度学习框架
  ○ 这符合我们之前看到的基于树模型的设计
4. 生产级部署考虑
  ○ APScheduler用于定时任务管理
  ○ loguru提供更好的日志支持
5. 版本锁定
  ○ 所有包都指定了确切版本，确保环境一致性
  ○ 注意某些版本不是最新的(如scikit-learn 0.24.2)，表明系统稳定性优先于更新
项目架构洞察
从依赖包可以推断出项目的一些设计决策：
1. 轻量级选择
  ○ 选择LightGBM而非TensorFlow/PyTorch，表明对推理效率的重视
  ○ 整体依赖较少(只有16个直接依赖)，降低了维护复杂度
2. 可扩展性考虑
  ○ FastAPI支持自动文档、异步处理，便于系统扩展
  ○ APScheduler支持复杂的调度策略，满足业务需求变化
3. 内部生态集成
  ○ rec-preprocess==0.2.5可能是内部开发的预处理库
  ○ 版本精确控制表明与特定环境集成
4. 平衡新旧技术
  ○ 使用较新的FastAPI(0.111.0)和较旧的scikit-learn(0.24.2)
  ○ 表明在不同组件上有不同的更新策略
与项目其他部分的关联
● 与model.py的关系: 使用lightgbm实现模型训练
● 与infer.py的关系: 使用fastapi、APScheduler构建服务
● 与配置文件的关系: 使用pyyaml解析配置
● 与工程实践的关系: 精确版本控制表明DevOps规范
requirements.txt虽然简单，但提供了项目技术选择的全景图，反映了开发团队在效率、可维护性和可扩展性之间的平衡。从依赖可以看出，这是一个注重实用性和生产稳定性的系统，而非研究导向的实验项目。
train.py —— 模型训练入口：数据下载utils/download给他定义好sql文件路径就可以去取数了，定义HDFS路径和当前目录路径；train 创建model实例（model.py），配置模型（config.yml,model.yml)，之后train这个实例；模型部署：model上传HDFS（utils/hdfs.py),并尝试下载刚才上传的model来check是否上传成功
train.py 是模型训练的主入口文件，虽然代码简洁，但包含了完整的训练流程。我来详细解析这个文件：
文件结构概览
train.py 包含两个主要函数：
1. data_download: 下载训练数据
2. main: 训练流程的主函数
详细解析
导入部分
from pathlib import Path
from loguru import logger
from model import PushClassifier
● 使用Path进行跨平台路径处理
● 使用loguru进行日志记录
● 从model.py导入核心模型类PushClassifier
数据下载函数
def data_download(cur_path: Path, data_num: int):
    logger.info("downloading data from hive.")
    from utils.download import get_multi_data
    
    sql_path = cur_path / "data" / "train.sql"
    get_multi_data(sql_path, past_day=data_num)
● 这个函数负责从Hive数据仓库下载训练数据
● cur_path: 当前工作目录
● data_num: 需要下载的数据天数
● 实际下载逻辑在utils/download.py的get_multi_data中实现
● 使用SQL查询(从data/train.sql文件)获取数据
主函数
def main():
    HDFS_DIR = "dongshaojie/ainvest_push/tree_model"
    # 1.下载数据
    cur_path = Path.cwd()
    # 2. 模型训练
    model = PushClassifier(
        config_dir=cur_path / "config" / "config.yml",
        model_config_dir=cur_path / "config" / "model.yml",
        mode="trian",
        varlen_max=5,
    )
    model.train()
    # 3. 上传模型
    logger.info("start uploading.")
    from utils.hdfs import upload_hdfs
    
    upload_hdfs(
        HDFS_DIR,
        "model.pth",
    )
    
    model_path = "tmp.pth"
    from utils.hdfs import download_hdfs
    
    download_hdfs(HDFS_DIR, model_path)
    if Path(model_path).exists():
        logger.info("upload success.")
    else:
        logger.error("download fail.")
主函数分三个明确的步骤：
1. 准备阶段
  ○ 定义HDFS目录路径
  ○ 获取当前工作目录
  ○ 注意这里并没有实际调用data_download函数，可能在实际使用中需要手动调用或通过其他方式触发
2. 模型训练
  ○ 创建PushClassifier实例
  ○ 传入配置文件路径: config.yml和model.yml
  ○ 设置变长特征最大列数为5
  ○ 调用train()方法执行训练
3. 模型部署
  ○ 将训练好的模型上传到HDFS指定目录
  ○ 尝试下载刚上传的模型到本地临时文件
  ○ 验证上传是否成功
  ○ 这里的验证很实用：通过尝试下载来确认上传成功
执行入口
if __name__ == "__main__":
    main()
● 标准Python脚本入口，确保文件被直接执行时才会运行main()
关键设计特点
1. 职责分离
  ○ 训练逻辑封装在PushClassifier类中
  ○ 主文件只负责流程编排
  ○ 配置与代码分离，通过文件路径传入
2. 端到端流程
  ○ 从数据获取到模型上传的完整流程
  ○ 体现了MLOps的思想，自动化模型部署
3. 错误处理与验证
  ○ 通过下载验证上传结果
  ○ 使用logging记录关键步骤
4. 配置一致性
  ○ varlen_max=5与infer.yml中的varlen_max_col_num: 5保持一致
  ○ 确保训练和推理使用相同的特征处理逻辑
与项目其他部分的关联
● 与model.py的关系: 调用PushClassifier类进行实际训练
● 与配置文件的关系: 使用config.yml和model.yml配置模型
● 与HDFS的关系: 通过utils/hdfs.py处理模型存储
● 与数据获取的关系: 准备调用data_download但未执行，可能在外部脚本中
潜在的改进空间
1. 数据下载集成
  ○ data_download函数已定义但未在main中调用
  ○ 可能需要添加参数控制是否执行数据下载
2. 错误处理增强
  ○ 缺少对训练过程可能出现的异常处理
  ○ 可以添加更完善的错误处理和重试机制
train.py虽然简短，但展示了一个完整的模型训练和部署流程，体现了工业级机器学习系统的基本架构：数据获取、模型训练和模型部署。虽然实际的训练逻辑主要在PushClassifier类中，但这个文件提供了整个训练流水线的入口和编排。
model.py —— 核心就是两个方法 ☝️初始化 传入config.yml和model.yml来生成model实例 ☝️train() 分成4步 1. 生成X Y 特征和标签（读取yml文件，处理成cfg对象，让他处理各个pipline具体的pipline里的每一个op怎么操作untils/preprocess.py里定）读csv为df来进行特征处理和特征选择 生成XY 2.sklearn包来分训练集 验证集 3.用lightGBM包来定义Dataset和train 4.模型保存为model.pth
model.py 是项目的核心文件，实现了模型训练的主要逻辑。这个文件定义了 PushClassifier 类，负责从数据准备到模型训练的整个流程。我来详细解析这个文件：
导入部分
import lightgbm as lgb
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from uniprocess.config import Config

from utils.preprocess import run_one_op_pd
● 核心模型: 使用lightgbm作为模型框架
● 数据处理: 使用pandas进行数据操作
● 配置管理: 使用yaml解析配置，使用自定义的Config类管理结构化配置
● 工具函数: 从utils.preprocess导入run_one_op_pd用于数据预处理
● 训练工具: 使用sklearn的train_test_split划分训练集和验证集
PushClassifier类
初始化方法
def __init__(
    self,
    config_dir: str,
    model_config_dir: str,
    mode: str = "train",
    varlen_max: int = 5,
) -> None:
    self.cfg = self._get_config(config_dir)
    self.varlen_max = varlen_max

    with open(model_config_dir, encoding="utf-8", mode="r") as f:
        raw_config = yaml.safe_load(f)
    self.model_config = raw_config
● 接收两个配置文件路径：config_dir(一般特征配置)和model_config_dir(模型参数)
● mode参数可能用于区分训练和推理模式，但在代码中未见使用
● varlen_max设置变长特征的最大列数，默认为5
● 加载两种配置：一般配置(self.cfg)和模型参数(self.model_config)
配置加载方法
def _get_config(self, config_dir: str) -> Config:
    with open(config_dir, encoding="utf-8", mode="r") as f:
        raw_config = yaml.safe_load(f)
    cfg = Config(**raw_config)
    logger.info(f"base config: \n{cfg}")
    return cfg
● 从文件加载YAML配置
● 使用Config类将原始字典转换为结构化对象
● 记录配置信息到日志
数据预处理方法
def _data_preprocess(self, x: pd.DataFrame) -> pd.DataFrame:
    # 特征
    pipelines = self.cfg.process.pipelines + self.cfg.interactions.pipelines
    for pipe in pipelines:
        for op in pipe.operations:
            x = run_one_op_pd(x, op)
    # label
    for pipe in self.cfg.label_process.pipelines:
        for op in pipe.operations:
            x = run_one_op_pd(x, op)
    out_columns = self.cfg.feat_names + self.cfg.datasets.trainset.label_columns
    return x[out_columns]
● 这是实现配置驱动的特征工程的关键方法
● 处理分两步：
  a. 对特征执行处理管道(pipelines)
  b. 对标签执行处理管道
● 每个管道包含多个操作，每个操作都通过run_one_op_pd函数执行
● 最后只保留配置中指定的特征列和标签列
特征选择方法
def _feat_selection(self, x: pd.DataFrame, max_col_num: int) -> pd.DataFrame:
    names_set = set(self.cfg.feat_names)

    for feat_name in self.cfg.model.varlen_sparse_feat_names:
        x_explode = x[feat_name].apply(pd.Series)
        out_names = [feat_name + f"_{i}" for i in range(x_explode.columns.stop)][:max_col_num]
        in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
        x[out_names] = pd.DataFrame(x_explode[in_columns], index=x.index)
        names_set.remove(feat_name)
        names_set = names_set.union(set(out_names))
    return x[list(names_set)]
● 这个方法处理变长特征(如用户关注的多只股票)
● 对每个变长特征(如watchlists)：
  a. 使用apply(pd.Series)将列表展开为多列
  b. 生成新的列名(如watchlists_0, watchlists_1...)
  c. 限制最大列数为max_col_num(varlen_max参数)
  d. 将展开的列添加到DataFrame中
  e. 更新特征名称集合
● 返回只包含所需特征的DataFrame
输入准备方法
def _prepare_input(self, trainset_cfg):
    xs, ys = [], []
    logger.debug(f"trainset_cfg = {trainset_cfg}")
    for data_path in trainset_cfg.data_path:
        logger.info(f"loading data from {data_path}.")
        df = pd.read_csv(
            data_path,
            sep=trainset_cfg.sep,
            chunksize=trainset_cfg.chunksize,
            header=trainset_cfg.header,
            names=trainset_cfg.raw_columns,
        )
        logger.debug(f"train data: {df.head(10)}")
        x = df
        x = self._data_preprocess(x)
        feat = self._feat_selection(x, self.varlen_max)
        label = x[self.cfg.datasets.trainset.label_columns]
        xs.append(feat)
        ys.append(label)
    X = pd.concat(xs, axis=0)
    Y = pd.concat(ys, axis=0)
    logger.info(f"X.shape={X.shape}.Y.shape={Y.shape}.")
    return X, Y
● 此方法准备模型训练所需的特征(X)和标签(Y)
● 支持处理多个数据文件(trainset_cfg.data_path)
● 对每个数据文件：
  a. 使用pandas读取CSV文件
  b. 调用_data_preprocess进行预处理
  c. 调用_feat_selection进行特征选择
  d. 提取标签列
● 最后合并所有数据并返回X和Y
训练方法
def train(self):
    X, Y = self._prepare_input(self.cfg.datasets.trainset)
    X_train, X_val, y_train, y_val = train_test_split(
        X.values, Y.values[:, 0], test_size=0.2, random_state=42
    )
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        self.model_config,
        train_data,
        valid_sets=[val_data],
        feature_name=list(X.columns),
    )
    model.save_model("model.pth")
● 这是模型训练的主要方法
● 流程如下：
  a. 调用_prepare_input获取特征和标签
  b. 使用train_test_split将数据分为训练集(80%)和验证集(20%)
  c. 创建LightGBM数据集格式
  d. 使用lgb.train训练模型，传入之前加载的模型配置
  e. 在训练过程中使用验证集评估
  f. 将训练好的模型保存为"model.pth"
主执行块
if __name__ == "__main__":
    push_classifier = PushClassifier("config/config.yml", "config/model.yml")
    push_classifier.train()
● 允许直接执行此文件进行模型训练
● 使用默认配置路径
核心设计特点
1. 配置驱动
  ○ 整个训练流程由配置文件控制
  ○ 特征处理流程、模型参数都在外部配置
2. 模块化处理
  ○ 数据加载、预处理、特征选择、训练各自独立
  ○ 每个方法职责明确
3. 变长特征处理
  ○ 专门处理列表型特征的机制
  ○ 将变长特征展开为固定列数格式
4. 多文件支持
  ○ 支持从多个数据文件训练
  ○ 方便增量训练或处理大数据集
与项目其他部分的关联
● 与config.yml的关系: 特征工程流程完全遵循配置文件定义
● 与model.yml的关系: 模型参数直接从配置加载
● 与utils/preprocess.py的关系: 使用run_one_op_pd执行具体操作
● 与train.py的关系: 被train.py调用进行实际训练
model.py是整个项目的核心，它实现了从原始数据到训练模型的完整流程。它的设计体现了配置与代码分离、模块化设计等优秀的工程实践，使得系统能够灵活适应不同的特征处理需求，同时保持代码的简洁性和可维护性。
utils/preprocess.py —— 三个核心方法：1.SELF_OP_HUB：自定义操作函数 2.run_one_op_pd：train的时候我把config里的piplines展开成pipline展开到op，对每个op进行处理的函数 3.infei的时候一样对来的输出处理的函数（相当于model里数据预处理和特征选择方法的集合 也就相当于执行很多步run_one_op_pd实现infer特征处理的函数；其他的就是一些数据处理的基本操作
utils/preprocess.py 是项目中负责数据预处理的核心工具文件，实现了配置驱动的特征工程流程。我来详细解析这个文件：
导入部分
import json
from functools import partial
from typing import Any, List

import pandas as pd
from loguru import logger
from uniprocess.config import Config
from uniprocess.operations import OP_HUB

from utils.dtypes import InferConfig
● 标准库: 使用json处理JSON字符串，partial支持函数部分应用
● 数据处理: 使用pandas进行数据操作
● 配置类型: 导入自定义的Config和InferConfig类型
● 操作注册表: 导入OP_HUB字典，这是预定义操作的注册表
常量和工具函数
MISSING_VALUE = (None, "N/A", "-", "", "NULL", "Null", "null")
● 定义了各种缺失值的表示形式，用于数据清洗
工具函数
def int_max(x: int, max_value: int) -> int:
    return max(x, max_value)
● 获取一个值和指定最大值之间的较大者，用于限制特征上限
def remove_items(x: List[Any], target_values: List[Any]):
    residual = set(x) - set(target_values)
    if not residual:
        return x
    return [i for i in x if i not in target_values]
● 从列表中删除指定元素，若删除后为空则保留原列表
● 用于过滤不需要的值，如高频股票代码
def json_object_to_list(x: str, key: str, fail_value: str = "null"):
    try:
        x_obj = json.loads(x)
    except Exception as e:
        logger.error(e)
        logger.debug(f"json parse error.got input {x}.")
        return [fail_value]
    x_obj = json.loads(x)
    y = [z.get(key, fail_value) for z in x_obj]
    return y
● 从JSON字符串中提取指定键的值列表
● 包含错误处理，确保即使JSON格式错误也能返回默认值
● 用于处理诸如item_code和item_tags这样的JSON格式特征
def has_intersection(x: List[Any], y: List[Any], exclude=MISSING_VALUE) -> int:
    a = set(x) - set(exclude)
    b = set(y) - set(exclude)
    return int(len(set(a) & set(b)) > 0)
● 判断两个列表在排除缺失值后是否有交集
● 返回0或1，适合作为二元特征
● 用于创建如preder_bid_cross、watch_bid_cross等交叉特征
操作注册
SELF_OP_HUB = {
    "int_max": int_max,
    "json_object_to_list": json_object_to_list,
    "has_intersection": has_intersection,
    "remove_items": remove_items,
}
OP_HUB.update(SELF_OP_HUB)
● 将自定义操作函数注册到操作中心OP_HUB
● 这使得配置文件中可以通过函数名引用这些操作
核心处理函数
单操作执行函数
def run_one_op_pd(x, op):
    # 获取输入列、输出列、函数名和函数参数
    col_in = op.col_in
    col_out = op.col_out
    func_name = op.func_name
    parameters = op.func_parameters if op.func_parameters else dict()

    # 使用partial函数创建一个部分应用的函数，固定函数参数
    partial_func = partial(OP_HUB[func_name], **parameters)

    # 如果输入列是一个列表，对每一行应用函数，否则对单列应用函数
    if isinstance(col_in, list):
        x[col_out] = x[col_in].apply(lambda row: partial_func(*row), axis=1)
    else:
        x[col_out] = x[col_in].apply(partial_func)

    return x
● 这是配置驱动特征工程的核心函数
● 流程如下：
  a. 从操作配置中提取需要的信息（输入列、输出列、函数名、参数）
  b. 使用partial创建一个预设参数的函数
  c. 根据输入列类型选择不同的应用方式:
    ■ 若输入是多列(列表)，则按行应用并传递多个参数
    ■ 若输入是单列，直接应用函数
  d. 将结果存储到新列并返回修改后的DataFrame
● 这个设计使得可以在配置中以声明式方式定义特征转换管道
完整预处理函数
def data_preprocess(df: pd.DataFrame, config: Config, infer_config: InferConfig):
    # 遍历配置中的每个处理管道
    for pipe in config.process.pipelines:
        # 遍历管道中的每个操作
        for op in pipe.operations:
            # 如果操作的输入列存在于DataFrame中，则执行该操作
            if op.col_in in df.columns:
                df = run_one_op_pd(df, op)
        # 记录当前处理管道的特征名称
        logger.info(f"processing {pipe.feat_name}")

    # 获取变长特征的最大列数
    max_col_num = infer_config.varlen_max_col_num
    # 获取配置中的特征名称与DataFrame列名的交集
    names_set = set(config.feat_names).intersection(df.columns)
    # 获取模型配置中的变长稀疏特征名称
    varlen_sparse_feat_names = config.model.varlen_sparse_feat_names
    logger.info("start process valen sparse feat")

    # 获取DataFrame列名与变长稀疏特征名称的交集
    feat_names = set(df.columns).intersection(varlen_sparse_feat_names)
    # 遍历每个变长稀疏特征名称
    for feat_name in feat_names:
        # 将特征列展开为多列
        x_explode = df[feat_name].apply(pd.Series)
        # 生成输出列名
        out_names = [feat_name + f"_{i}" for i in range(x_explode.columns.stop)][:max_col_num]
        logger.info(f"processing {feat_name}")
        # 生成输入列索引
        in_columns = [i for i in range(x_explode.columns.stop)][:max_col_num]
        # 将展开后的列添加到DataFrame中
        df[out_names] = pd.DataFrame(x_explode[in_columns], index=df.index)
        # 从names_set中移除原始特征名称，并添加新的输出列名
        names_set.remove(feat_name)
        names_set = names_set.union(set(out_names))

    return df
● 这个函数是推理阶段的预处理入口，与model.py中的_data_preprocess和_feat_selection功能类似
● 主要分两部分：
  a. 执行配置中的处理管道:
    ■ 遍历config.process.pipelines中的每个管道
    ■ 对每个管道中的操作，如果输入列存在，则执行之
    ■ 使用run_one_op_pd应用具体操作
  b. 处理变长特征:
    ■ 获取变长特征的最大列数max_col_num
    ■ 找出需要处理的变长特征
    ■ 对每个变长特征：
      ● 将列表展开为多列
      ● 生成新列名(如feature_0, feature_1...)
      ● 限制列数到max_col_num
      ● 更新特征名集合
    ■ 返回处理后的DataFrame
设计特点和技术亮点
1. 配置驱动架构
  ○ 处理逻辑由配置决定，代码只负责执行框架
  ○ 这种设计使得添加新特征只需修改配置，无需改代码
2. 函数注册表模式
  ○ 使用OP_HUB作为函数注册表
  ○ 可以轻松扩展新操作，只需添加函数并注册到SELF_OP_HUB
3. 适应性处理
  ○ 自动跳过缺失的输入列
  ○ 自动处理变长特征和定长特征
4. 错误处理
  ○ 包含异常捕获和日志记录
  ○ 保证即使某些数据异常，处理流程也能继续
5. 函数式编程思想
  ○ 使用partial进行函数部分应用
  ○ 每个操作都是纯函数，输入DataFrame，输出新DataFrame
与项目其他部分的关联
● 与config.yml的关系: 实现了配置文件中定义的所有特征处理管道
● 与model.py的关系: 提供run_one_op_pd函数，是_data_preprocess的核心
● 与infer.py的关系: 提供推理时使用的data_preprocess函数
● 与dtypes.py的关系: 使用InferConfig类型获取配置参数
utils/preprocess.py是项目中特征工程的核心实现，它创建了一个灵活且可扩展的框架，使得复杂的特征处理逻辑可以通过配置文件清晰地表达。这种配置驱动的设计大大提高了系统的可维护性和可扩展性，是整个项目架构中的亮点之一。
补充：函数注册表模式是什么
函数注册表模式是一种设计模式，我来详细解释它在这个项目中的工作原理：
函数注册表模式详解
函数注册表（Function Registry）是一种设计模式，它本质上是一个字典（或映射），将函数名（字符串）映射到实际的函数对象。在这个项目中，OP_HUB就是这样一个函数注册表。
工作原理
1. 核心概念
  ○ 注册表：一个字典，键是函数名，值是函数对象
  ○ 注册：将函数添加到注册表中
  ○ 查找：通过名称获取函数
2. 在项目中的实现
# 从外部库导入的注册表
from uniprocess.operations import OP_HUB

# 自定义函数
def int_max(x: int, max_value: int) -> int:
    return max(x, max_value)

# 创建自己的注册表并添加函数
SELF_OP_HUB = {
    "int_max": int_max,
    "json_object_to_list": json_object_to_list,
    "has_intersection": has_intersection,
    "remove_items": remove_items,
}

# 将自定义注册表合并到主注册表
OP_HUB.update(SELF_OP_HUB)
3. 使用方式
# 从注册表中通过名称获取函数
func_name = op.func_name  # 例如 "int_max"
parameters = op.func_parameters  # 例如 {"max_value": 31}

# 使用partial创建新函数，预设参数
partial_func = partial(OP_HUB[func_name], **parameters)

# 执行函数
result = partial_func(some_value)
为什么使用函数注册表
1. 与配置解耦
  ○ 配置文件中只需写函数名字符串（如"int_max"）
  ○ 不需要在配置中直接引用Python函数
2. 动态函数调用
  ○ 允许根据配置动态选择要执行的函数
  ○ 配置可以在运行时更改，而不需要修改代码
3. 扩展性
  ○ 可以轻松添加新函数而无需修改现有代码
  ○ 只需定义函数并注册到注册表即可
4. 可配置性
  ○ 完全通过配置文件驱动特征工程流程
  ○ 相同的代码可以执行不同的特征处理流程
举例说明
假设你想添加一个新的特征处理函数，比如计算两个数字的加权平均：
1. 定义函数
def weighted_average(value, weight, default=0):
    try:
        return float(value) * float(weight)
    except (ValueError, TypeError):
        return default
2. 注册函数
SELF_OP_HUB["weighted_average"] = weighted_average
# 或者直接在SELF_OP_HUB定义时添加
3. 在配置中使用
operations:
  - col_in: value_column
    col_out: weighted_value
    func_name: weighted_average
    func_parameters:
      weight: 0.75
      default: 0
4. 无需修改核心代码
  ○ run_one_op_pd函数不需要任何改变
  ○ 它会从OP_HUB获取名为"weighted_average"的函数
  ○ 使用配置的参数创建partial函数
  ○ 将函数应用到数据上
源代码中的实际流程
1. 配置文件(config.yml)定义了操作:
operations:
  - col_in: title_len
    col_out: title_len
    func_name: int_max
    func_parameters:
      max_value: 31
2. 代码读取配置并执行:
# 获取函数名和参数
func_name = "int_max"
parameters = {"max_value": 31}

# 从注册表获取函数
func = OP_HUB["int_max"]  # 得到int_max函数对象

# 创建预设参数的函数
partial_func = partial(func, max_value=31)

# 应用到数据
df["title_len"] = df["title_len"].apply(partial_func)
OP_HUB的完整内容
这个项目中，OP_HUB包含两类函数:
1. 从uniprocess.operations导入的预定义函数
  ○ 可能包含通用操作如fillna, split, list_hash等
2. 在SELF_OP_HUB中定义的特定于项目的函数
  ○ int_max: 限制整数最大值
  ○ remove_items: 从列表中移除指定项
  ○ json_object_to_list: 从JSON提取值列表
  ○ has_intersection: 检查两个列表的交集
函数注册表模式使这个项目能够实现高度可配置的特征工程流程，是实现"配置驱动"架构的关键技术。这种设计让系统可以灵活应对新的特征需求，而无需修改核心处理逻辑。
untils/download.py —— 
utils/download.py 是项目中负责数据获取的关键组件，实现了从Hive数据仓库下载训练和用户数据的功能。我来详细解析这个文件：
导入部分
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict

from hxhive.cli import hexecute
from loguru import logger
● 标准库: 使用datetime处理日期，os和shutil处理文件操作
● 路径处理: 使用Path进行跨平台路径操作
● 数据查询: 导入hexecute，这是一个内部工具，用于执行Hive查询
● 日志记录: 使用loguru的logger进行日志记录
文件管理函数
def remove_dir(path: Path, max_num: int):
    assert max_num > 0, "max_num must be greater than 0"
    # 当目录下文件数量超过max_num保留最新的max_num个文件
    file_num = len(list(path.iterdir()))
    logger.info(f"current path contains {file_num} files; max num: {max_num}")
    if file_num > max_num:
        for f in sorted(path.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[
            max_num:  # 删除最旧的文件
        ]:
            if f.is_file():
                os.remove(f)
            else:
                shutil.rmtree(f)
            logger.warning(f"remove file: {f}")
● 这个函数用于管理数据目录，确保文件数量不超过指定的最大值
● 工作流程:
  a. 检查目录中的文件数量
  b. 如果超过max_num，则按修改时间排序
  c. 保留最新的max_num个文件，删除其他文件
● 这种机制确保了:
  ○ 存储空间控制，防止数据过多占用磁盘
  ○ 保留最新数据，确保模型使用最近的数据训练
SQL处理函数
def parse_sql(path: Path, now: str, last_1_day: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content_formatted = content.format(now=now, last_1_day=last_1_day)
    return content_formatted
● 读取SQL模板文件并替换日期参数
● 使用Python的格式化字符串功能，将{now}和{last_1_day}替换为实际日期
● 这使得SQL查询可以动态应用于不同日期的数据
训练数据获取函数
def get_multi_data(
    sql_path: Path,
    past_day: int,
    shift_day: int = 2,
    download: bool = True,
    max_file_num: int = 10,
    min_file_size: int = 1024,
) -> Dict[str, str]:
    """
    获取多天的训练数据
    
    Parameters
    ----------
    sql_path : Path
        sql文本路径
    past_day : int
        取前N天的数据
    shift_day : int, optional
        时间向前偏移, by default 2
    download : bool, optional
        是否从hive下载数据, by default True
    max_file_num : int, optional
        训练文件的最大数量, by default 10
    
    Returns
    -------
    Dict[str, str]
        日期和对应SQL的字典
    """
● 这个函数是训练数据下载的核心，负责获取多天的历史数据
● 主要参数:
  ○ sql_path: SQL模板文件路径
  ○ past_day: 需要获取的历史天数
  ○ shift_day: 时间偏移，避免获取当天不完整的数据
  ○ max_file_num: 保留的最大文件数
  ○ min_file_size: 有效文件的最小大小(KB)
● 工作流程:
  a. 计算基准日期(data_date) = 当前日期 - shift_day
  b. 对于需要的每一天:
    ■ 计算日期和前一天的日期
    ■ 解析SQL模板，替换日期参数
    ■ 如果download=True，执行SQL并下载数据
    ■ 检查文件大小，太小则删除(避免无效数据)
  c. 调用remove_dir清理过多文件
  d. 返回日期-SQL映射字典
● 这个函数确保:
  ○ 获取足够多天的历史数据进行训练
  ○ 避免获取当天可能不完整的数据(通过shift_day)
  ○ 控制存储空间使用(通过max_file_num)
  ○ 过滤无效数据(通过min_file_size)
用户数据下载函数
def download_user_data(data_dir: Path, sql_path: Path, sql_params: dict) -> bool:
    """
    下载用户数据并保存到指定目录。
    
    该函数首先读取SQL文件，将其内容格式化为特定的日期，然后执行SQL查询并将结果保存到指定目录。
    如果下载的数据文件大小大于4096字节，则认为下载成功；否则，删除文件并记录警告信息。
    如果目录不存在，记录错误信息。
    
    :param data_dir: 保存下载数据的目录路径。
    :param sql_path: 包含SQL查询的文件路径。
    :param sql_params: 用于格式化SQL查询的日期字符串。
    :return: 如果下载成功返回True，否则返回False。
    """
● 这个函数用于推理服务中的用户数据更新
● 与get_multi_data不同，它一次只处理一个日期的数据
● 工作流程:
  a. 读取SQL模板并用提供的参数格式化
  b. 使用hexecute执行SQL查询并下载结果
  c. 验证下载结果:
    ■ 如果文件存在且大于4KB，返回成功
    ■ 如果文件过小，删除文件并返回失败
    ■ 如果发生异常，记录错误并返回失败
● 安全特性:
  ○ 异常捕获和处理
  ○ 文件有效性检查
  ○ 详细的日志记录
入口点
if __name__ == "__main__":
    current_path = Path.cwd()
    train_path = current_path / "data" / "train.sql"
    content_formatted = get_multi_data(train_path, 5, max_file_num=5)
● 如果直接运行此文件，将下载最近5天的训练数据
● 保留最大5个数据文件
设计亮点
1. 灵活的时间管理
  ○ 支持多天历史数据获取
  ○ 动态时间偏移，避免不完整数据
  ○ 日期参数化SQL查询
2. 存储空间优化
  ○ 自动清理过期数据
  ○ 过滤无效(过小)数据文件
  ○ 限制最大文件数量
3. 错误处理和恢复
  ○ 异常捕获和详细日志
  ○ 失败情况的优雅处理
  ○ 返回清晰的成功/失败状态
4. 模块化设计
  ○ 训练数据和用户数据获取分离
  ○ SQL模板与执行逻辑分离
  ○ 文件管理作为独立功能
与项目其他部分的关联
● 与train.py的关系: 通过get_multi_data提供训练数据
● 与infer.py的关系: 通过download_user_data提供用户特征数据
● 与定时任务的关系: 被daily_task和hour_task调用进行数据更新
● 与数据源的关系: 通过hexecute连接Hive获取数据
utils/download.py是项目中数据流的起点，负责从数据仓库获取训练数据和用户数据。它的设计确保了数据的时效性、完整性和存储效率，为模型训练和在线推理提供了可靠的数据基础。这种数据获取机制支持了系统的定期更新能力，是实现模型持续迭代的关键组件。
infer.py
infer.py 是整个项目的核心服务文件，负责模型推理、API服务和数据定时更新。这个文件实现了完整的推荐系统服务端逻辑。我来详细解析这个文件：
导入部分
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import lightgbm
import pandas as pd
import yaml
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI
from loguru import logger
from uniprocess.config import Config

from model import run_one_op_pd
from utils.download import download_user_data
from utils.dtypes import InferConfig, PushItems
from utils.hdfs import download_hdfs
from utils.preprocess import data_preprocess
● 标准库: 日期处理、路径操作、异步上下文管理
● 数据科学: pandas数据处理、LightGBM模型
● API框架: FastAPI构建Web服务
● 定时任务: APScheduler实现定时数据更新
● 配置管理: YAML解析配置
● 项目模块: 导入各个工具函数和数据类型
全局资源
resource = {}
● 使用字典作为全局资源缓存，存储配置、模型和用户数据
定时任务
每日任务
def daily_task():
    """每日任务：更新用户数据。"""
● 主要功能：更新所有用户的基础数据
● 工作流程：
  a. 获取配置与当前日期
  b. 尝试下载当天用户数据，最多尝试download_max_try次
  c. 清理旧数据文件，只保留最近user_data_save_days天的数据
  d. 如果当天下载失败但有历史数据，使用最新的历史数据
  e. 读取数据并预处理
  f. 更新全局资源中的用户数据
● 容错设计：
  ○ 多次尝试下载
  ○ 历史数据作为回退方案
  ○ 详细日志记录
每小时任务
def hour_task():
    """每小时任务：下载新用户数据并进行预处理。"""
● 主要功能：增量更新新用户数据
● 工作流程：
  a. 下载新用户数据
  b. 预处理新用户数据
  c. 与现有用户数据合并
  d. 处理可能的重复数据(使用groupby().agg(max))
  e. 更新全局资源
● 设计意图：
  ○ 确保新用户能快速被纳入推荐系统
  ○ 保持用户数据的实时性，不必等到每日更新
模型管理
def load_model(infer_config: InferConfig) -> bool:
    """加载模型。"""
● 从HDFS下载模型到本地
● 使用LightGBM加载模型文件
● 将模型和特征名称存入全局资源
● 返回加载是否成功
应用生命周期管理
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # 初始化阶段
    # ...
    yield
    # 清理阶段
    # ...
● 使用FastAPI的生命周期管理
● 初始化阶段：
  a. 配置日志系统
  b. 加载配置文件（infer.yml和config.yml）
  c. 设置定时任务（每日和每小时）
  d. 立即执行一次任务，确保数据可用
  e. 加载模型
● 清理阶段：
  a. 清空资源缓存
  b. 关闭调度器
主应用实例
app = FastAPI(lifespan=app_lifespan)
● 创建FastAPI应用实例
● 配置生命周期管理器
核心预测逻辑
特征转换
def data_to_feat(df_join: pd.DataFrame, base_config: Config, infer_config: InferConfig) -> pd.DataFrame:
    """将数据转换为特征 DataFrame。"""
● 将合并的用户-项目数据转换为模型需要的特征
● 应用配置中定义的特征工程管道
● 处理变长特征，展开为固定列数
● 返回符合模型输入要求的特征DataFrame
预测执行
def predict_and_append(df, model, df_join, results):
    """预测并添加结果到结果列表中。"""
● 使用模型预测给定特征
● 将预测结果与用户ID和项目ID关联
● 添加到结果列表
项目预测
def predict_items(item_df: pd.DataFrame, filter: bool = False) -> pd.DataFrame:
    """预测项目推荐结果。"""
● 预测核心函数，处理给定项目的推荐
● 工作流程：
  a. 预处理项目数据
  b. 将用户数据分批处理（控制内存使用）
  c. 对每批用户：
    ■ 与项目数据做笛卡尔积连接
    ■ 转换为特征
    ■ 使用模型预测得分
  d. 根据filter参数决定是否使用交叉特征过滤：
    ■ 如果为True，优先推荐有交集（用户关注/持有/偏好的股票与项目相关股票有交集）的项目
    ■ 如果为False，纯粹基于得分推荐
  e. 合并所有批次结果并返回
● 批处理设计：
  ○ 避免一次处理过多用户导致内存溢出
  ○ 每批用户数量由max_user_num_per_iter控制
主预测函数
def ml_predict(items: PushItems, filter: bool = False):
    """预测给定项目的推荐结果。"""
● API端点的核心实现
● 将输入的PushItems转换为DataFrame
● 调用predict_items获取每个用户的最佳项目
● 将结果重组为"项目->[用户列表]"的字典形式
● 记录每个项目的推荐用户数量
API端点
同步预测接口
@app.post("/predict/")
async def predict(items: PushItems):
    """项目预测接口"""
    return ml_predict(items)

@app.post("/predict_v2/")
async def predict_v2(items: PushItems):
    """启用交叉过滤的项目预测接口"""
    return ml_predict(items, filter=True)
● 两个同步预测接口：
  ○ /predict/：基础预测接口
  ○ /predict_v2/：启用交叉特征过滤的预测接口
异步预测处理
async def predict_send(items: PushItems):
    """异步发送预测结果到指定的回调URL。"""
    
async def predict_send_v2(items: PushItems):
    """异步发送预测结果到指定的回调URL（启用交叉过滤）。"""

@app.post("/predict_async/")
async def predict_async(items: PushItems, background_tasks: BackgroundTasks):
    """异步处理项目预测请求。"""

@app.post("/predict_async_v2/")
async def predict_async_v2(items: PushItems, background_tasks: BackgroundTasks):
    """异步处理项目预测请求（启用交叉过滤）。"""
● 异步处理预测请求并将结果发送到配置的回调URL
● 两组异步接口：
  ○ 基础版（/predict_async/）
  ○ 启用交叉过滤版（/predict_async_v2/）
● 使用FastAPI的BackgroundTasks在后台执行预测和推送
● 立即返回接收确认，不阻塞客户端
管理接口
@app.post("/update_model")
async def update_model():
    """更新模型。"""

@app.get("/readiness")
async def readiness():
    """检查服务的就绪状态。"""
● /update_model：触发模型热更新，不停机更新模型
● /readiness：健康检查接口，用于监控系统可用性
设计亮点
1. 资源共享与缓存
  ○ 使用全局resource字典缓存模型和用户数据
  ○ 避免重复加载和内存浪费
2. 定时任务机制
  ○ 每日全量更新 + 每小时增量更新
  ○ 确保数据新鲜度的同时控制计算开销
3. 批处理设计
  ○ 用户数据分批处理，控制内存使用
  ○ 适合大规模用户群体
4. 多层容错
  ○ 数据下载多次尝试
  ○ 历史数据作为回退方案
  ○ 异常捕获和日志记录
5. 异步处理流程
  ○ 同步/异步接口并存
  ○ 使用异步HTTP客户端处理回调
  ○ 使用背景任务避免阻塞
6. 模型热更新
  ○ 专用接口触发模型更新
  ○ 不停机更新，保持服务可用性
7. 灵活的过滤机制
  ○ 提供基础推荐和交叉过滤推荐
  ○ 允许业务方根据需求选择不同接口
与项目其他部分的关联
● 与model.py的关系: 使用相同的特征处理逻辑，确保训练和推理一致
● 与utils/download.py的关系: 调用数据下载函数获取用户数据
● 与utils/preprocess.py的关系: 使用相同的预处理函数处理特征
● 与配置文件的关系: 完全依赖配置驱动行为，不需修改代码
infer.py是整个推荐系统的服务层核心，它不仅实现了模型推理，还包含了完整的服务生命周期管理、数据更新机制和API接口。这个文件体现了一个成熟推荐系统的工程实践：数据定期更新、批量处理、异步通信、容错机制和灵活配置。
用户特征是kyc 也就是推荐中台user_tag维护好的 所以我们在infer的时候 其实就是定时把kyc里的user_tag爬到本地data/下面就可以了 这就实现了user特征的更新 —— 所以核心就是调用utils/download里定义的download_user_data函数hexecute（sql，x）来sql从kyc平台里取特征—— 这就是前200行代码 hour_task和daily_task的部分；定时执行这部分逻辑是在@asynccontextmanager async def app_lifespan(app: FastAPI):生命周期管理这个函数里实现的 生命周期管理包含三部分 1.模型加载 2.每日用户kyc更新 3.每小时用户kyc更新 模型有了 user特征有了 给我item数据 我就可以给你你想要的item：users的结果
是的，你的理解非常准确。根据这个系统的设计，用户特征是在外部（数据仓库）维护的，推荐系统只需定时获取这些数据，无需关心特征是如何生成和更新的。我来详细解释这个机制：
用户特征维护与获取流程
1. 特征生成与存储（外部系统）
  ○ 用户特征存储在Hive数据仓库的表中（如db_dws.dws_crd_lb_v_dd）
  ○ 这些特征（watchlists、holdings、country、prefer_bid等）由其他业务系统维护和更新
  ○ 数据表有分区（p_date字段），保存了不同日期的特征快照
2. 特征获取（推荐系统）
  ○ 推荐系统只负责通过SQL查询获取最新特征
  ○ daily_task和hour_task实质上是定时执行的特征提取任务
  ○ 系统不关心特征如何生成，只关心如何获取和使用
关键代码证据
从SQL查询可以看出，特征是从已有的数据表中查询的：
-- 从这个表中获取用户特征
from
    db_dws.dws_crd_lb_v_dd
where
    p_date = '{now}'
    and l_code in ('BM79', 'BM176', 'BM210', 'BM55', 'BM70')
    and person_id > 0
这里的特征是通过代码标识（'BM176'、'BM210'等）从特征表中获取的，表明这些特征是预先计算好并存储在数据仓库中的。
这种设计的优势
1. 职责分离
  ○ 特征工程与推荐算法解耦
  ○ 各系统只负责自己的核心功能，降低复杂度
2. 数据一致性
  ○ 多个系统可以使用相同的特征源
  ○ 避免重复计算和数据不一致
3. 计算效率
  ○ 特征计算可能很复杂，集中在数据仓库处理更高效
  ○ 推荐服务只需查询结果，减轻服务器负担
4. 便于维护
  ○ 特征定义变更只需修改数据仓库处理流程
  ○ 推荐系统代码无需更改，减少维护工作
实际应用中的工作方式
在这个系统的实际运行中：
1. 数据工程团队负责维护用户行为数据收集和特征计算流程
2. 定期（可能是每天）更新数据仓库中的特征表
3. 推荐系统通过定时任务（daily_task和hour_task）从数据仓库拉取最新特征
4. 推荐系统使用这些特征和预训练模型进行推荐
你的总结非常准确："只需要定时去取新的数据，就可以相当于更新了最新的用户特征"。这正是该系统采用的设计模式，它将特征工程与推荐模型服务分离，使每个部分都能专注于自己的职责，同时通过数据仓库作为中介确保数据一致性。
中间几个函数就是数据到特征 特征到结果 最后整合返回成 每个item给一堆user的结构返回 这个就是服务给一个items列表 我return的结果 —— def ml_predict（items：PushItems，filter：bool=False）：

怎么返回结果（原理）代码上就是一行 @app.post("/predict/") async def predict(items: PushItems):return ml_predict(items) 这里展现的是工作原理
你提出了一个很好的问题，让我来澄清同步和异步接口的区别及其实现方式：
同步返回实现 (/predict/ 和 /predict_v2/)
同步接口是通过直接返回函数结果实现的：
@app.post("/predict/")
async def predict(items: PushItems):
    return ml_predict(items)  # 直接返回ml_predict的结果
在这个实现中：
1. 客户端发送请求到/predict/
2. 服务器执行ml_predict(items)，这可能需要几秒到几十秒
3. 整个过程中，HTTP连接保持打开
4. 计算完成后，结果直接作为HTTP响应返回给客户端
5. 客户端必须等待整个计算过程完成才能获得响应
即使函数标记为async，在没有await关键字的情况下，ml_predict依然是同步执行的，会阻塞整个请求处理流程直到计算完成。
异步处理实现 (/predict_async/ 和 /predict_async_v2/)
异步接口使用FastAPI的BackgroundTasks机制实现：
@app.post("/predict_async_v2/")
async def predict_async_v2(items: PushItems, background_tasks: BackgroundTasks):
    background_tasks.add_task(predict_send_v2, items)  # 添加到后台任务
    return {"message": f"{len(items.items)} received.", "state": "200"}  # 立即返回
在这个实现中：
1. 客户端发送请求到/predict_async_v2/
2. 服务器将predict_send_v2(items)添加到后台任务队列
3. 立即返回确认消息，不等待计算完成
4. HTTP连接关闭，客户端收到确认响应
5. 服务器在后台继续执行计算任务
6. 计算完成后，结果通过回调URL推送到指定服务
background_tasks.add_task()是关键 - 它让任务在HTTP响应发送后继续执行，不会阻塞响应的返回。
两种方式的区别
以请求/响应时序图说明：
同步方式 (/predict/, /predict_v2/):
客户端                    服务器
   |                        |
   |--- HTTP请求 ---------> |
   |                        | --- 执行ml_predict (可能需要几秒到几十秒)
   |       (等待中...)       |
   |                        |
   |<-- HTTP响应(预测结果) -- |
   |                        |
异步方式 (/predict_async/, /predict_async_v2/):
客户端                    服务器                    回调服务器
   |                        |                         |
   |--- HTTP请求 ---------> |                         |
   |<-- HTTP响应(确认收到) -- |                         |
   |                        | --- 执行ml_predict       |
   |                        | --- 计算完成             |
   |                        | --- 发送结果 ----------> |
   |                        |                         |
总结
● 同步接口：直接返回计算结果，客户端必须等待整个计算过程完成
● 异步接口：立即返回确认，将任务添加到后台，计算完成后通过另一个HTTP请求发送结果
/predict_async_v2/是异步接口，不是同步返回实现，它的特点正是不需要客户端等待计算完成。这种方式特别适合处理耗时较长的计算任务，避免了客户端超时问题，同时也使服务器能够更有效地管理计算资源。
关于hosts Dockerfile 和 train.sh
我来详细解析这三个文件，它们共同构成了项目的部署和训练环境配置：
1. hosts 文件
10.240.132.150 bmpri-cbas-hadoop-150
10.240.132.151 bmpri-cbas-hadoop-151
10.240.132.152 bmpri-cbas-hadoop-152
10.104.13.18 ainvest-recsys-resource-pool-server-14660
hosts 文件定义了主机名到IP地址的映射：
● Hadoop集群：前三行定义了Hadoop集群节点的IP地址和主机名映射（bmpri-cbas-hadoop-150/151/152）
● 资源池服务器：最后一行定义了推荐系统资源池服务器的映射
用途：
● 允许系统通过主机名而非IP地址访问这些服务器
● 确保在容器环境中能正确解析这些内部服务器名称
● 支持数据下载（从Hadoop）和模型训练/部署过程
2. Dockerfile 文件
FROM hub-dev.hexin.cn/jupyterhub/script_crontab:beimeipri_py38
USER root
COPY ./ /root/port_service
WORKDIR /root/port_service
RUN pip install -r requirements.txt && cat ./hosts >> /etc/hosts
ENTRYPOINT [ "uvicorn", "infer:app", "--reload", "--port=12333", "--host=0.0.0.0"]
这个Dockerfile定义了项目的容器化部署环境：
● 基础镜像：使用内部Python 3.8基础镜像，预装了crontab和JupyterHub
● 用户设置：使用root用户运行（在容器内是常见做法）
● 代码复制：将当前目录下所有文件复制到容器的/root/port_service目录
● 工作目录：设置工作目录为/root/port_service
● 依赖安装：
  ○ 安装requirements.txt中定义的Python依赖
  ○ 将主机映射添加到容器的/etc/hosts文件
● 启动命令：
  ○ 使用uvicorn启动FastAPI应用（infer:app）
  ○ 启用--reload选项（检测到代码变更时自动重启，适合开发环境）
  ○ 在端口12333上提供服务
  ○ 监听所有网络接口（--host=0.0.0.0）
用途：
● 定义推理服务的运行环境和启动方式
● 确保所有依赖包和主机映射都正确配置
● 提供一个可重复部署的容器化环境
3. train.sh 脚本
set -eu

cd /root/port_service

cat ./hosts >> /etc/hosts

export PYTHONPATH=`pwd`
/opt/conda/bin/python3.8 utils/download.py
ls -lh data/train
/opt/conda/bin/python3.8 train.py 
这是一个训练脚本，用于在算力中心执行模型训练：
● 错误处理：set -eu 确保脚本在遇到错误或未定义变量时立即退出
  ○ e表示遇到命令返回非零状态时立即退出
  ○ u表示遇到未定义变量时立即退出
● 工作目录：切换到/root/port_service
● 主机映射：将hosts文件内容追加到系统的/etc/hosts
● 环境设置：将当前目录添加到PYTHONPATH环境变量
● 数据下载：执行utils/download.py脚本下载训练数据
● 数据检查：列出data/train目录内容，验证数据下载
● 模型训练：执行train.py脚本进行模型训练
用途：
● 提供一个标准化的训练流程
● 确保训练环境正确配置（主机映射、Python路径）
● 自动执行数据下载和模型训练步骤
● 适合在算力中心或CI/CD流程中调用
三个文件的协同工作方式
这三个文件共同构成了一个完整的训练和部署流程：
1. 环境配置：
  ○ hosts文件提供网络访问配置
  ○ 在容器环境和训练环境中都需应用这些配置
2. 训练流程：
  ○ train.sh在算力中心执行，完成数据下载和模型训练
  ○ 训练结果（模型文件）存储到HDFS，便于后续部署
3. 服务部署：
  ○ Dockerfile定义推理服务的容器环境
  ○ 容器启动时会从HDFS加载训练好的模型
  ○ 服务以API形式提供推荐功能
这种设计实现了训练和服务的分离，符合MLOps的最佳实践：
● 在高性能算力中心进行资源密集型训练
● 使用轻量级容器部署推理服务
● 通过文件共享（HDFS）实现模型在环境间的传递
● 保持配置一致性（如主机映射）以确保各环境中的网络访问
这种架构使得模型可以定期训练更新，而不影响线上服务的可用性。
