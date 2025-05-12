# 基于树模型的Push排序

## 一、服务流程

1. 在算力中心离线训练模型；
2. 同顺云运行排序服务。

## 二、部署流程

### 2.1 离线训练

1. 同顺云打包当前镜像；
2. 镜像同步至北美；
3. 算力中心的可视化建模中建立任务；
4. 启动训练任务，启动脚本为：

```sh
set -eu

cd /root/port_service

cat ./hosts >> /etc/hosts

export PYTHONPATH=`pwd`
/opt/conda/bin/python3.8 utils/download.py
ls -lh data/train
/opt/conda/bin/python3.8 train.py 
```

### 2.2 运行服务

在同顺云走正式发布流程。

## 三、开发

### 3.1 开发环境

Python >= 3.8

```sh
pip install -r requirements.txt
```

### 3.2 数据准备

下载训练数据，放在训练数据的目录`data/train`下。

下载用户数据，放在用户数据目录下`data/user`。

### 3.3 本地模型训练

```sh
python train.py
```

### 3.4 本地启动服务

```sh
uvicorn infer:app --reload --port=12333
```

### 3.5 本地测试服务

```sh
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
```

# beimeipush
本项目是基于LightGBM的美股推送消息排序系统
