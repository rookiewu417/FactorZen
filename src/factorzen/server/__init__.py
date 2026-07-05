"""只读服务层。

在既有 workspace 产物之上提供 REST API 与 Web Dashboard。纯读层:只扫描
manifest/指标文件建索引,不 import 研究模块、不触发任何计算——服务崩不了研究,
研究改不崩服务。
"""
