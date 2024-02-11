# first line: 38
@CacheMemory.cache
def colbertv2_get_request_v2(url: str, query: str, k: int):
    assert (
        k <= 100
    ), "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."

    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)

    topk = res.json()["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]
