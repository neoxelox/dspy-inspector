import superinvoke

from .tags import Tags


class Tools(superinvoke.Tools):
    Git = superinvoke.Tool(
        name="git",
        version=">=2.34.1",
        tags=[*Tags.As("*")],
        path="git",
    )

    Python = superinvoke.Tool(
        name="python",
        version=">=3.11",
        tags=[*Tags.As("*")],
        path="python",
    )

    Poetry = superinvoke.Tool(
        name="poetry",
        version=">=1.7.1",
        tags=[*Tags.As("*")],
        path="poetry",
    )
