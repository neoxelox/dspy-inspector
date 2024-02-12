# dspy-inspector 🔍 ![BETA](https://img.shields.io/badge/BETA-8b5cf6)

**`DSPy program/pipeline inspector widget for Jupyter/VSCode Notebooks.`** 🔭

![Demo](demo.gif)

> Dark theme is coming soon™.

## Install

👉 `pip install dspy-inspector`

## Usage

```python
from dspy_inspector import inspect

class MultiHopRAG(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        ...

    def forward(self, question):
        ...

inspect(MultiHopRAG)

# Or 👇

program = MultiHopRAG()
inspect(program) # Automatically updates
program("How many storeys are in the castle that David Gregory inherited?")
```

Check an example in the [`multi_hop_rag.ipynb`](examples/multi_hop_rag.ipynb) notebook.

> All example notebooks are cached, no calls to LLMs are needed to run them.

## Roadmap

This is a personal project I made to better understand what were my DSPy programs actually doing underneath. This extension is currently in **`BETA`**, so, expect bugs. For a roadmap on what things must be fixed or done check all the `TODOs` inside the source code.

## Developing

To setup the virtual developing environment, follow these steps:

1. Install dependencies: `poetry install`
2. Install tools: `poetry run inv tool.install --include "dev*"`
3. Create kernel: `poetry run python -m ipykernel install --user --name dspy-inspector --display-name "DSPy Inspector"`
4. Select `dspy-inspector` kernel in all notebooks

Run `poetry run inv help` for further commands and `poetry run inv <command> --help` for their usage.

## Contribute

Feel free to contribute to this project : ) .

## License

This project is licensed under the [GPL-3.0 License](https://opensource.org/licenses/GPL-3) - read the [LICENSE](LICENSE) file for details.
