import superinvoke

from .tags import Tags


class Envs(superinvoke.Envs):
    Default = lambda cls: cls.Dev

    Dev = superinvoke.Env(
        name="dev",
        tags=[*Tags.As("dev*")],
    )

    Ci = superinvoke.Env(
        name="ci",
        tags=[*Tags.As("ci*")],
    )

    Prod = superinvoke.Env(
        name="prod",
        tags=[*Tags.As("prod*")],
    )
