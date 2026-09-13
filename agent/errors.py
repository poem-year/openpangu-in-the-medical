"""智能体异常定义（对应《智能体接口规范.md》§1.5）。"""


class AgentError(Exception):
    """智能体模块异常基类。"""


class AgentUnavailableError(AgentError):
    """模型服务不可用或请求超时（重试后仍失败）。由调用方决定如何提示用户。"""
