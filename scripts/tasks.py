import os

from superinvoke import task

from .envs import Envs
from .tools import Tools


@task(
    help={
        "test": "[<FILE_PATH>]::[<TEST_NAME>]. If empty, it will run all tests.",
    },
)
def test(context, test="./dspy_inspector/"):
    """Run tests."""

    context.run(f"{Tools.Poetry} run pytest -p no:warnings -n auto {test}")


@task(
    help={
        "file": "[<FILE_PATH>]. If empty, it will lint all files.",
    },
)
def lint(context, file="./dspy_inspector/"):
    """Run linter."""

    context.run(f"{Tools.Poetry} run flakehell lint {file}")
    context.run(f"{Tools.Poetry} run mypy {file}")


@task(
    help={
        "file": "[<FILE_PATH>]. If empty, it will format all files.",
    },
)
def format(context, file="./dspy_inspector/"):
    """Run formatter."""

    context.run(f"{Tools.Poetry} run autoflake {file}")
    context.run(f"{Tools.Poetry} run isort {file}")
    context.run(f"{Tools.Poetry} run black {file}")


@task()
def build(context):
    """Build package."""

    if Envs.Current != Envs.Ci:
        context.fail(f"build command only available in {Envs.Ci} environment!")

    context.run(f"{Tools.Poetry} build")


@task()
def publish(context):
    """Publish package."""

    if Envs.Current != Envs.Ci:
        context.fail(f"publish command only available in {Envs.Ci} environment!")

    version = context.tag()
    if not version:
        latest_version = context.tag(current=False) or "v0.0.0"
        major, minor, patch = tuple(map(str, (latest_version.split("."))))
        version = f"{major}.{str(int(minor) + 1)}.{0}"
        context.info(f"Version tag not set, generating one from {latest_version}: {version}")
        context.run(f"{Tools.Git} tag {version}")
        context.run(f"{Tools.Git} push origin {version}")
    else:
        context.info(f"Version tag already set: {version}")

    context.run(f"{Tools.Poetry} config pypi-token.pypi {os.environ['PYPI_TOKEN']}")
    context.run(f"{Tools.Poetry} publish")
