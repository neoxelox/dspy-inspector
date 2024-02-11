# first line: 250
@functools.lru_cache(maxsize=None if cache_turn_on else 0)
@NotebookCacheMemory.cache
def v1_cached_gpt3_request_v2_wrapped(**kwargs):
    return v1_cached_gpt3_request_v2(**kwargs)
