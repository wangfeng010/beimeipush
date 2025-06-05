# 算子介绍

算子根据输入变量的数量分为常规算子与交叉算子。

- 常规算子接收1个输入，返回一个输出。

- 交叉算子介绍2个及以上输入，返回一个输出。

## 常规算子

| 名称                                           | 函数名             | 参数                       |
| ---------------------------------------------- | ------------------ | -------------------------- |
| 求对数                                         | log                | base                       |
| 类型转换                                       | astype             | target_type                |
| 字符串拆分                                     | split              | sep                        |
| 列表内字符串拆分                               | seperation         | sep                        |
| 除以一个数                                     | scale              | *scale_ratio*              |
| 列表填充至指定长度                             | padding            | *pad_value*,*max_len*      |
| 填充缺失值                                     | fillna             | *na_value*                 |
| 字符串哈希                                     | str_hash           | *vocabulary_size*          |
| 列表内字符串哈希                               | list_hash          | *vocabulary_size*          |
| 根据字典进行值映射                             | map_to_int         | *map_dict*，*default_code* |
| 将双重列表转化为字典                           | to_dict            | *key_index*,*value_index*  |
| 求字典中Top的key                               | get_max_key        | *key_num*                  |
| 根据下标取出双重列表中的指定下标元素，组成列表 | list_get           | *item_index*               |
| 对列表元素进行类型转换                         | list_astype        | target_type                |
| 数值分桶                                       | to_bucket          | *bin_boundaries*           |
| 求列表长度                                     | list_len           | /                          |
| 求字符串长度                                   | str_len            | /                          |
| 求小数点后位数                                 | get_decimal_places | /                          |
| 取小时时刻                                     | get_hour           | *format*                   |
| 字符串转日期对象                               | to_date            | *format*                   |
| 获取当前星期                                   | weekday            | *format*                   |
| 将列表形式的json字符串转化为Python列表         | json_to_list       | *fail_value*               |
| 取二重列表的部分元素，拼接为字符串列表         | list_get_join      | *indices,sep,fail_value*   |

## 交叉算子

| 名称                       | 函数名                 | 参数      |
| -------------------------- | ---------------------- | --------- |
| 交集                       | intersection           | *exclude* |
| 交集的元素数量             | intersection_num       | *exclude* |
| 判断B的元素内是否有a       | is_in                  | *exclude* |
| 拼接两个字符串             | concat_str             |           |
| 求列表a与字典b的keys的交集 | intersection_list_dict | *exclude* |
| 求列表a与列表b并集         | union                  | /         |
