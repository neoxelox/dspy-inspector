# first line: 246
@CacheMemory.cache
def v1_cached_gpt3_request_v2(**kwargs):
    return openai.completions.create(**kwargs)
