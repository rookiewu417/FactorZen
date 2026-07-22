"""LLM 基础设施（挖掘 / agents 共用）。

因子报告 LLM 解读（service/schema/cache/snapshot/prompt）已移除。
消费方请直接从子模块导入，例如::

    from factorzen.llm.client import request_chat, LLMClientError
    from factorzen.llm.config import load_llm_config
    from factorzen.llm.generation import LLMFn, extract_json_items
"""
