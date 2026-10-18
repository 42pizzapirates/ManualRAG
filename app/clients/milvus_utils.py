import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

_milvus_client = None


def get_milvus_client():
    """返回共享的 Milvus 客户端，连接失败时返回 None。"""
    try:
        global _milvus_client
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 服务端未开启认证时，空凭证即可连接
            _milvus_client = MilvusClient(
                uri=milvus_uri,
                user=milvus_config.milvus_user or "",
                password=milvus_config.milvus_password or ""
            )
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """把 chunk_id 转为 INT64，并分离无法转换的值。"""
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """按主键批量查询切片；get 失败时回退到 query 表达式过滤。"""

    if client is None:
        return []
    if not collection_name:
        return []
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")

    if not ok_ids:
        return []

    results = []
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    return results


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """构造稠密与稀疏两路 ANN 检索请求，默认 limit=5。"""

    if dense_params is None:
        dense_params = {"metric_type": "IP"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """执行稠密 + 稀疏加权混合检索，失败时返回 None。"""
    try:
        # norm_score 会先把两路分数归一化到 0-1，避免量级差异让权重失效
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        if output_fields is None:
            output_fields = ["item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None
