def escape_milvus_string(value: str) -> str:
    """转义字符串以安全嵌入 Milvus filter 表达式。

    反斜杠和双引号按 Milvus 规则转义，换行符替换为空格以保证表达式单行有效。
    """
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
