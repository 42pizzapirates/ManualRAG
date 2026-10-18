from modelscope.hub.snapshot_download import snapshot_download

# 下载到固定的本地缓存目录，路径需与 BGE_M3_PATH 保持一致
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='D:/ai_models/modelscope_cache/models')
print(f"模型已下载到: {model_dir}")
