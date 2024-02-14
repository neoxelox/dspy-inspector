import html
import inspect
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pprint import pprint
from threading import Timer
from typing import Callable, List, Optional, Tuple

import dsp
import dspy
import ipycytoscape as cytoscape
import ipywidgets as widgets
import orjson
import tiktoken

with open(os.path.join(os.path.dirname(__file__), "styles.css"), "r") as file:
    _styles_css = file.read()


def _run_html_script(script: str) -> str:
    return f"""<style onload="(function(){{{script}}})();"></style>"""


def _escape_html_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br/>")


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


class Color(_StrEnum):
    emerald_content = "#10b981"
    emerald_background = "#d1fae5"
    amber_content = "#eab308"
    amber_background = "#fef9c3"
    rose_content = "#f43f5e"
    rose_background = "#ffe4e6"
    violet_content = "#8b5cf6"
    violet_background = "#a78bfa"
    stone_content = "#a8a29e"
    stone_background = "#f5f5f4"
    edge = "#1E73E9"


@dataclass
class Node:
    class Type(_StrEnum):
        PARAMETER = "parameter"
        PREDICTOR = "predictor"
        RETRIEVER = "retriever"
        PROGRAM = "program"

    id: str
    parent: Optional[str]
    variable: str
    type: Type
    metadata: dict
    dspy_instance: object
    cyto_instance: cytoscape.Node


@dataclass
class ParameterNode(Node):
    class Direction(_StrEnum):
        INPUT = "input"
        OUTPUT = "output"

    @dataclass
    class Value:
        text: str
        tokens: int

    parameter: str
    direction: Direction
    description: str
    prefix: str
    format: Optional[Callable]
    value: Optional[Value]

    @property
    def label(self) -> str:
        return self.parameter

    @property
    def title(self) -> str:
        return self.parameter

    @property
    def subtitle(self) -> Optional[str]:
        return None


@dataclass
class PredictorNode(Node):
    @dataclass
    class Signature:
        name: str
        syntax: str
        instructions: str

    @dataclass
    class Model:
        name: str
        settings: dict
        instance: object

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int

    module: str
    signature: Signature
    model: Model
    usage: Usage
    prompt: Optional[str]
    completion: Optional[str]
    demos: Optional[List[str]]

    @property
    def label(self) -> str:
        list_marker = self.variable.find("[")
        if list_marker > 0:
            return f"{self.signature.name}\n{self.module}{self.variable[list_marker:]}"
        return f"{self.signature.name}\n{self.module}"

    @property
    def title(self) -> str:
        return self.signature.name

    @property
    def subtitle(self) -> Optional[str]:
        return self.module


@dataclass
class RetrieverNode(Node):
    @dataclass
    class Model:
        name: str
        settings: dict
        instance: object

    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int

    module: str
    model: Model
    usage: Usage

    @property
    def label(self) -> str:
        list_marker = self.variable.find("[")
        if list_marker > 0:
            return f"{self.module}{self.variable[list_marker:]}"
        return self.module

    @property
    def title(self) -> str:
        return self.module

    @property
    def subtitle(self) -> Optional[str]:
        return None


@dataclass
class ProgramNode(Node):
    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int

    program: str
    compiled: bool
    usage: Usage

    @property
    def label(self) -> str:
        return self.program

    @property
    def title(self) -> str:
        return self.program

    @property
    def subtitle(self) -> Optional[str]:
        return None


@dataclass
class Edge:
    class Direction(_StrEnum):
        INPUT = "input"
        OUTPUT = "output"

    class Endpoint(_StrEnum):
        TOP = "top"
        MID = "mid"
        BOT = "bot"

    source: str
    target: str
    direction: Direction
    endpoint: Endpoint


class Inspector:
    def __init__(self, debug: bool, tokenizer_of_model: str) -> None:
        self.debug = debug
        encoder = tiktoken.encoding_for_model(tokenizer_of_model)
        self.tokenizer = partial(encoder.encode, allowed_special="all")

    _graph_widget_style = [
        {
            "selector": "core",
            "style": {
                "active-bg-opacity": 0,
            },
        },
        {
            "selector": "node",
            "style": {
                "width": "1em",
                "height": "1em",
                "shape": "round-rectangle",
                "background-color": Color.rose_background,
                "border-width": "0em",
                "border-color": Color.rose_content,
                "font-family": 'BlinkMacSystemFont, -apple-system, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", "Helvetica", "Arial", sans-serif',  # noqa: E501
                "font-size": "0.75em",
                "font-weight": "600",
                "label": "data(label)",
                "color": Color.rose_content,
                "text-valign": "center",
                "text-halign": "center",
                "text-justification": "center",
                "text-max-width": "99em",
                "text-wrap": "wrap",
                "compound-sizing-wrt-labels": "include",
                "padding": "10%",
            },
        },
        {
            "selector": "node[type='parameter']",
            "style": {
                "color": Color.stone_content,
                "background-color": Color.stone_background,
                "border-color": Color.stone_content,
            },
        },
        {
            "selector": "node[type='retriever']",
            "style": {
                "color": Color.amber_content,
                "background-color": Color.amber_background,
                "border-color": Color.amber_content,
            },
        },
        {
            "selector": "node[type='predictor']",
            "style": {
                "color": Color.emerald_content,
                "background-color": Color.emerald_background,
                "border-color": Color.emerald_content,
            },
        },
        {
            "selector": "node[type='program']",
            "style": {
                "background-opacity": 0,
                "color": Color.violet_content,
                "border-style": "dashed",
                "border-color": Color.violet_background,
                "border-width": "0.05em",
                "font-size": "1.25em",
                "text-valign": "top",
                "text-halign": "center",
                "text-margin-y": "-1em",
                "padding": "2.5%",
            },
        },
        {
            "selector": "node:active",
            "style": {
                "overlay-opacity": 0,
                "border-width": "0em",
            },
        },
        {
            "selector": "node:active[type='program']",
            "style": {
                "overlay-opacity": 0,
                "border-width": "0.15em",
            },
        },
        {
            "selector": "node.selected",
            "style": {
                "overlay-opacity": 0,
                "border-width": "0.15em",
            },
        },
        {
            "selector": "node.selected[inner]",
            "style": {
                "overlay-opacity": 0,
                "border-width": "0em",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "unbundled-bezier",
                "width": "0.075em",
                "line-color": Color.edge,
                "arrow-scale": "0.75",
                "target-arrow-shape": "circle",
                "target-arrow-color": Color.edge,
                "target-arrow-width": "1em",
                "source-endpoint": "90deg",
                "target-endpoint": "270deg",
                "edge-distances": "endpoints",
                "control-point-distances": "-10 10",
                "control-point-weights": "0.25 0.75",
            },
        },
        {
            "selector": "edge[hidden]",
            "style": {
                "visibility": "hidden",
            },
        },
        {
            "selector": "edge:active",
            "style": {
                "overlay-opacity": 0,
            },
        },
        {
            "selector": "edge[direction='input'][endpoint='top']",
            "style": {
                "source-endpoint": "90deg",
                "target-endpoint": "315deg",
                "control-point-distances": "-30",
                "control-point-weights": "0.75",
            },
        },
        {
            "selector": "edge[direction='input'][endpoint='mid']",
            "style": {
                "source-endpoint": "90deg",
                "target-endpoint": "270deg",
                "control-point-distances": "-10 10",
                "control-point-weights": "0.25 0.75",
            },
        },
        {
            "selector": "edge[direction='input'][endpoint='bot']",
            "style": {
                "source-endpoint": "90deg",
                "target-endpoint": "225deg",
                "control-point-distances": "30",
                "control-point-weights": "0.75",
            },
        },
        {
            "selector": "edge[direction='output'][endpoint='top']",
            "style": {
                "source-endpoint": "45deg",
                "target-endpoint": "270deg",
                "control-point-distances": "-30",
                "control-point-weights": "0.25",
            },
        },
        {
            "selector": "edge[direction='output'][endpoint='mid']",
            "style": {
                "source-endpoint": "90deg",
                "target-endpoint": "270deg",
                "control-point-distances": "-10 10",
                "control-point-weights": "0.25 0.75",
            },
        },
        {
            "selector": "edge[direction='output'][endpoint='bot']",
            "style": {
                "source-endpoint": "135deg",
                "target-endpoint": "270deg",
                "control-point-distances": "30",
                "control-point-weights": "0.25",
            },
        },
    ]
    _panel_widget_style = f"<style>{_styles_css}</style>"

    def inspect(self, program: dspy.Program) -> widgets.Widget:
        if inspect.isclass(program) and issubclass(program, dspy.Program):
            program = program()

        if not isinstance(program, dspy.Program):
            raise TypeError(f"provided program of type {type(program)} is not a DSPy program")

        # TODO: Maybe change nodes to a dict by id
        graph = {"nodes": [], "edges": []}
        selected_node = None

        graph_widget = cytoscape.CytoscapeWidget()
        graph_widget.set_style(self._graph_widget_style)
        graph_widget.set_layout(name="dagre", directed=True, animate=True, rankDir="LR", align=None)

        style_widget = widgets.HTML(self._panel_widget_style)

        export_button_widget = widgets.Button(
            description="",
            icon="picture-o",
            tooltip="Export",
            button_style="",
            layout=widgets.Layout(width="auto", height="auto"),
        )
        save_button_widget = widgets.Button(
            description="",
            icon="floppy-o",
            tooltip="Save",
            button_style="",
            layout=widgets.Layout(width="auto", height="auto"),
        )
        reset_button_widget = widgets.Button(
            description="",
            icon="refresh",
            tooltip="Reset View",
            button_style="danger",
            layout=widgets.Layout(width="auto", height="auto"),
        )

        panel_info_widget = widgets.HTML()
        panel_action_executor_widget = widgets.HTML(
            layout=widgets.Layout(display="none"),
        )
        panel_actions_widget = widgets.HBox(
            [panel_action_executor_widget, export_button_widget, save_button_widget, reset_button_widget],
        )
        panel_widget = widgets.VBox(
            [panel_info_widget, panel_actions_widget],
        )

        inspector_widget = widgets.HBox(
            [
                graph_widget,
                panel_widget,
                style_widget,
            ],
            layout=widgets.Layout(width="100%", height="400px", overflow="hidden"),
        )

        def _draw_graph_widget() -> None:
            cytoscape_graph = {"nodes": [], "edges": []}

            for node in graph["nodes"]:
                if node.type == Node.Type.PROGRAM:
                    cytoscape_graph["nodes"].append(
                        {
                            "id": node.id,
                            "type": node.type,
                            "label": node.label,
                            **({"parent": node.parent} if node.parent else {}),
                        },
                    )
                else:  # A hack to make nodes as large as their labels
                    cytoscape_graph["nodes"].extend(
                        [
                            {
                                "id": f"{node.id}-outer",
                                "type": node.type,
                                **({"parent": node.parent} if node.parent else {}),
                            },
                            {
                                "id": node.id,
                                "type": node.type,
                                "label": node.label,
                                "parent": f"{node.id}-outer",
                                "inner": True,
                            },
                        ]
                    )

            for edge in graph["edges"]:
                cytoscape_graph["edges"].extend(
                    [
                        {  # A hack to make edges endpoints be outside of compound nodes without breaking the layout
                            "source": edge.source,
                            "target": edge.target,
                            "direction": edge.direction,
                            "endpoint": edge.endpoint,
                            "hidden": True,
                        },
                        {
                            "source": f"{edge.source}-outer",
                            "target": f"{edge.target}-outer",
                            "direction": edge.direction,
                            "endpoint": edge.endpoint,
                        },
                    ]
                )

            graph_widget.graph.clear()
            graph_widget.graph.add_graph_from_json(cytoscape_graph, directed=True)

            # TODO: Replace this quadratic for...
            for cyto_node in graph_widget.graph.nodes:
                for node in graph["nodes"]:
                    if cyto_node.data["id"] == f"{node.id}-outer" or (
                        node.type == Node.Type.PROGRAM and cyto_node.data["id"] == node.id
                    ):
                        node.cyto_instance = cyto_node

        def _draw_node_panel_info_widget(node: Node) -> str:
            return f"""
<div class="dspy-inspector-panel-info-node-title"><h1>{node.title}</h1></div>
{f'<div class="dspy-inspector-panel-info-node-subtitle"><h2>{node.subtitle}</h2></div>' if node.subtitle else ''}
<div class="dspy-inspector-panel-info-node-tags">
    <div><span>type</span><span data-type-{node.type}>{node.type}</span></div>
    <div><span>variable</span><span data-variable>{node.variable}</span></div>
</div>
"""  # noqa: E501

        def _draw_parameter_panel_info_widget(parameter: ParameterNode) -> str:
            return f"""
<div><dt>Direction</dt><dd>{parameter.direction}</dd></div>
<div><dt>Prefix</dt><dd>{_escape_html_text(parameter.prefix)}</dd></div>
<div><dt>Description</dt><dd>{_escape_html_text(parameter.description)}</dd></div>
<div><dt>Format</dt><dd>{parameter.format.__name__ if parameter.format else 'None'}</dd></div>
<div><dt>Value</dt><dd>{_escape_html_text(parameter.value.text) if parameter.value else 'None'}</dd></div>
<div><dt>Tokens</dt><dd>{parameter.value.tokens if parameter.value else '0'}</dd></div>
"""  # noqa: E501

        def _draw_predictor_panel_info_widget(predictor: PredictorNode) -> str:
            # TODO: Put demos inside the Result field and add a button to show/hide them
            return f"""
<div><dt>Signature</dt><dd>{predictor.signature.syntax}<br/><i>(From: {predictor.signature.name})</i></dd></div>
<div><dt>Instructions</dt><dd>{_escape_html_text(predictor.signature.instructions)}</dd></div>
<div><dt>Module</dt><dd>{predictor.module}</dd></div>
<div><dt>Result</dt><dd>{_escape_html_text(predictor.prompt or 'None')}<b style="color: {Color.emerald_content};">{_escape_html_text(predictor.completion or '')}</b></dd></div>
<div><dt>Tokens</dt><dd>Input: {predictor.usage.input_tokens} | Output: {predictor.usage.output_tokens}</dd></div>
<div><dt>Demos</dt><dd>{'<br/><br/>'.join([_escape_html_text(demo.toDict().__str__()) for demo in predictor.demos]) if predictor.demos else 'None'}</dd></div>
<div><dt>Model</dt><dd>{predictor.model.name}</dd></div>
<div><dt>Settings</dt><dd>{predictor.model.settings}</dd></div>
"""  # noqa: E501

        def _draw_retriever_panel_info_widget(retriever: RetrieverNode) -> str:
            return f"""
<div><dt>Module</dt><dd>{retriever.module}</dd></div>
<div><dt>Tokens</dt><dd>Input: {retriever.usage.input_tokens} | Output: {retriever.usage.output_tokens}</dd></div>
<div><dt>Model</dt><dd>{retriever.model.name}</dd></div>
<div><dt>Settings</dt><dd>{retriever.model.settings}</dd></div>
"""  # noqa: E501

        def _draw_program_panel_info_widget(program: ProgramNode) -> str:
            return f"""
<div><dt>Compiled</dt><dd>{program.compiled}</dd></div>
<div><dt>Subprogram</dt><dd>{program.parent is not None}</dd></div>
<div><dt>Tokens</dt><dd>Input: {program.usage.input_tokens} | Output: {program.usage.output_tokens}</dd></div>
"""  # noqa: E501

        def _update_graph() -> None:
            # Find the best endpoint for each edge
            for edge in graph["edges"]:
                siblings = list(
                    filter(
                        lambda other: (
                            other.target == edge.target
                            if edge.direction == Edge.Direction.INPUT
                            else other.source == edge.source
                        ),
                        graph["edges"],
                    )
                )

                if len(siblings) > 1:
                    for count, sibling in enumerate(siblings, start=1):
                        if count == 1:
                            sibling.endpoint = Edge.Endpoint.TOP
                        elif count == len(siblings):
                            sibling.endpoint = Edge.Endpoint.BOT
                        else:
                            sibling.endpoint = Edge.Endpoint.MID
                else:
                    edge.endpoint = Edge.Endpoint.MID

            if self.debug:
                pprint(graph)

            _draw_graph_widget()

            # Force a graph relayout after a new draw, but let some time for the ui to settle
            for n in range(3):
                Timer(1 + n, lambda: graph_widget.relayout()).start()

        def _select_node(node: Node, force: bool = False) -> None:
            nonlocal selected_node

            if self.debug:
                pprint(node)

            if selected_node:
                selected_node.cyto_instance.classes = ""

                if selected_node.id == node.id and not force:
                    selected_node = None
                    return

            selected_node = node

            # TODO: Center graph view on selected cytoscape node
            selected_node.cyto_instance.classes = "selected"

            info_node_html = _draw_node_panel_info_widget(node)

            if isinstance(node, ParameterNode):
                info_table_html = _draw_parameter_panel_info_widget(node)
            elif isinstance(node, PredictorNode):
                info_table_html = _draw_predictor_panel_info_widget(node)
            elif isinstance(node, RetrieverNode):
                info_table_html = _draw_retriever_panel_info_widget(node)
            elif isinstance(node, ProgramNode):
                info_table_html = _draw_program_panel_info_widget(node)

            panel_info_widget.value = f"""
<div class="dspy-inspector-panel-info-node">
    {info_node_html}
</div>
<div class="dspy-inspector-panel-info-table">
    <dl>
        {info_table_html}
    </dl>
</div>
"""

        def _graph_widget_on_node_click(event: dict) -> None:
            node_id = event["data"]["id"].split("-outer")[0]
            node = next(filter(lambda node: node.id == node_id, graph["nodes"]), None)
            if node:
                _select_node(node)

        def _export_button_widget_on_click(*args, **kwargs) -> None:
            panel_action_executor_widget.value = ""

            # TODO: Export graph in PNG
            panel_action_executor_widget.value = _run_html_script("""alert('Export not implemented');""")

        def _save_button_widget_on_click(*args, **kwargs) -> None:
            panel_action_executor_widget.value = ""

            filepath = os.path.abspath(f"./{graph['nodes'][0].program}.json")
            with open(filepath, "w") as file:
                file.write(
                    orjson.dumps(
                        graph, default=lambda o: None, option=orjson.OPT_SERIALIZE_DATACLASS | orjson.OPT_INDENT_2
                    ).decode()
                )

            panel_action_executor_widget.value = _run_html_script(f"""alert('Saved graph to {filepath}');""")

        def _reset_button_widget_on_click(*args, **kwargs) -> None:
            graph_widget.relayout()

        def _update_parameters(this: Node, parameters: list, nodes: List[Node], edges: List[Edge]) -> None:
            for parameter, parameter_value in parameters:
                parameter_node = next(filter(lambda node: node.id == f"{this.id}.{parameter}", nodes), None)
                if not parameter_node:
                    continue

                if parameter_node.format:
                    parameter_value = parameter_node.format(parameter_value)

                parameter_node.value = ParameterNode.Value(
                    text=parameter_value, tokens=len(self.tokenizer(parameter_value))
                )

                # TODO: Create edges between parameter's traversed nodes

        def _update_predictor(predictor: PredictorNode) -> None:
            # TODO: This probably won't work if model calls are done in parallel
            call = predictor.model.instance.history[-1]

            # TODO: Fix, this is model-dependant

            predictor.usage.input_tokens = call["response"]["usage"]["prompt_tokens"]
            predictor.usage.output_tokens = call["response"]["usage"]["completion_tokens"]

            predictor.prompt = call["prompt"]
            predictor.completion = call["response"]["choices"][0]["text"]

            predictor.demos = predictor.dspy_instance.demos

        def _update_retriever(retriever: RetrieverNode) -> None:
            # TODO: Can't update usage because dspy does not support it yet

            pass

        def _get_total_tokens_for_program(program: ProgramNode) -> Tuple[int, int]:
            total_input_tokens = 0
            total_output_tokens = 0

            for node in filter(lambda node: node.parent == program.id, graph["nodes"]):
                if isinstance(node, PredictorNode):
                    total_input_tokens += node.usage.input_tokens
                    total_output_tokens += node.usage.output_tokens
                elif isinstance(node, RetrieverNode):
                    total_input_tokens += node.usage.input_tokens
                    total_output_tokens += node.usage.output_tokens
                elif isinstance(node, ProgramNode):
                    input_tokens, output_tokens = _get_total_tokens_for_program(node)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens

            return total_input_tokens, total_output_tokens

        def _update_program(program: ProgramNode) -> None:
            program.compiled = program.dspy_instance._compiled
            program.usage.input_tokens, program.usage.output_tokens = _get_total_tokens_for_program(program)

        def _parse_parameter(attribute: str, parameter: object, parent: Node) -> Node:
            parameter_name = attribute
            if isinstance(parameter, dspy.InputField):
                parameter_direction = ParameterNode.Direction.INPUT
                # dada
            elif isinstance(parameter, dspy.OutputField):
                parameter_direction = ParameterNode.Direction.OUTPUT
            elif isinstance(parameter, dsp.Type):
                # TODO: How to get direction?
                parameter_direction = ParameterNode.Direction.OUTPUT

            this = ParameterNode(
                id=f"{parent.id}.{attribute}",
                parent=parent.parent,  # Put parameter in the same level as parent
                variable=attribute,
                type=Node.Type.PARAMETER,
                parameter=parameter_name,
                direction=parameter_direction,
                description=parameter.desc,
                prefix=parameter.prefix,
                format=parameter.format
                or dsp.Template("").format_handlers.get(
                    parameter_name, None
                ),  # Some injected parameters have default formatting
                value=None,
                metadata={},
                dspy_instance=parameter,
                cyto_instance=None,
            )

            return this

        def _build_syntax_from_template(template: dsp.Template) -> str:
            inputs = []
            outputs = []

            for name, field in template.kwargs.items():
                if isinstance(field, dspy.InputField):
                    inputs.append(name)
                elif isinstance(field, dspy.OutputField):
                    outputs.append(name)
                elif isinstance(field, dsp.Type):
                    # TODO: How to get direction?
                    outputs.append(name)

            return f"{', '.join(inputs)} -> {', '.join(outputs)}"

        def _parse_predictor(attribute: str, module: object, parent: Node) -> Tuple[List[Node], List[Edge]]:
            nodes = []
            edges = []

            module_name = module.__class__.__name__
            module_forward = getattr(module, "forward")
            module_model = module.lm or dsp.settings.lm

            if inspect.isclass(module.signature) and issubclass(module.signature, dspy.Signature):
                module_signature_syntax = _build_syntax_from_template(module.signature._template)
                module_signature_name = module.signature.__name__
            elif isinstance(module.signature, dsp.Template):
                module_signature_syntax = _build_syntax_from_template(module.signature)
                module_signature_name = module_signature_syntax
            module_signature_instructions = module.signature.instructions
            module_signature_parameters = module.signature.kwargs
            if hasattr(module, "extended_signature"):
                module_signature_syntax = _build_syntax_from_template(module.extended_signature)
                module_signature_parameters = module.extended_signature.kwargs

            this = PredictorNode(
                id=f"{parent.id}.{attribute}",
                parent=parent.id,
                variable=attribute,
                type=Node.Type.PREDICTOR,
                module=module_name,
                signature=PredictorNode.Signature(
                    name=module_signature_name,
                    syntax=module_signature_syntax,
                    instructions=module_signature_instructions,
                ),
                model=PredictorNode.Model(
                    name=module_model.kwargs["model"],
                    settings=module_model.copy().kwargs,
                    instance=module_model,
                ),
                usage=PredictorNode.Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
                prompt=None,
                completion=None,
                demos=module.demos,
                metadata={},
                dspy_instance=module,
                cyto_instance=None,
            )
            nodes.append(this)

            for attribute, parameter in module_signature_parameters.items():
                parameter_node = _parse_parameter(attribute, parameter, this)
                nodes.append(parameter_node)
                if parameter_node.direction == ParameterNode.Direction.INPUT:
                    edges.append(
                        Edge(
                            source=parameter_node.id,
                            target=this.id,
                            direction=Edge.Direction.INPUT,
                            endpoint=Edge.Endpoint.MID,
                        )
                    )
                elif parameter_node.direction == ParameterNode.Direction.OUTPUT:
                    edges.append(
                        Edge(
                            source=this.id,
                            target=parameter_node.id,
                            direction=Edge.Direction.OUTPUT,
                            endpoint=Edge.Endpoint.MID,
                        )
                    )

            def wrap_forward(*args, **kwargs):
                result = module_forward(*args, **kwargs)
                parameters = list(kwargs.items()) + result.items()
                _update_parameters(this, parameters, nodes, edges)
                _update_predictor(this)
                return result

            setattr(module, "forward", wrap_forward)

            return nodes, edges

        def _parse_retriever(attribute: str, module: object, parent: Node) -> Tuple[List[Node], List[Edge]]:
            nodes = []
            edges = []

            module_name = module.__class__.__name__
            module_forward = getattr(module, "forward")
            module_model = dsp.settings.rm  # TODO: Cannot get module.rm because dspy does not support it yet

            this = RetrieverNode(
                id=f"{parent.id}.{attribute}",
                parent=parent.id,
                variable=attribute,
                type=Node.Type.RETRIEVER,
                module=module_name,
                model=RetrieverNode.Model(
                    name=module_model.__class__.__name__,  # TODO: Cannot get module_model.kwargs["model"] because dspy does not support it yet # noqa: E501
                    settings={},  # TODO: Cannot get module_model.copy().kwargs because dspy does not support it yet
                    instance=module_model,
                ),
                usage=RetrieverNode.Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
                metadata={},
                dspy_instance=module,
                cyto_instance=None,
            )
            nodes.append(this)

            # Create signature manually as retrievers don't have it
            class RetrieverSignature(dspy.Signature):
                """Take the given search query and return one or more potentially relevant passages from a corpus."""

                query = dspy.InputField()
                passages = dspy.OutputField()

            for attribute, parameter in RetrieverSignature.kwargs.items():
                parameter_node = _parse_parameter(attribute, parameter, this)
                nodes.append(parameter_node)
                if parameter_node.direction == ParameterNode.Direction.INPUT:
                    edges.append(
                        Edge(
                            source=parameter_node.id,
                            target=this.id,
                            direction=Edge.Direction.INPUT,
                            endpoint=Edge.Endpoint.MID,
                        )
                    )
                elif parameter_node.direction == ParameterNode.Direction.OUTPUT:
                    edges.append(
                        Edge(
                            source=this.id,
                            target=parameter_node.id,
                            direction=Edge.Direction.OUTPUT,
                            endpoint=Edge.Endpoint.MID,
                        )
                    )

            def wrap_forward(*args, **kwargs):
                result = module_forward(*args, **kwargs)
                # Get query parameter manually as it is not a kwarg
                parameters = [("query", args[0])] + list(kwargs.items()) + result.items()
                _update_parameters(this, parameters, nodes, edges)
                _update_retriever(this)
                return result

            setattr(module, "forward", wrap_forward)

            return nodes, edges

        def _get_program_modules(program: object) -> List[Tuple[str, object]]:
            modules = []

            # Following almost the same logic as in dspy.BaseModule.named_parameters

            def append_module(attr: str, obj: object) -> None:
                if isinstance(obj, dspy.Retrieve) or isinstance(obj, dspy.Predict) or isinstance(obj, dspy.Program):
                    modules.append((attr, obj))

            visited = set()
            for attr, obj in program.__dict__.items():
                if id(obj) in visited:
                    continue
                visited.add(id(obj))

                if isinstance(obj, (list, tuple)):
                    for idx, item in enumerate(obj):
                        append_module(f"{attr}[{idx}]", item)
                elif isinstance(obj, dict):
                    for key, item in obj.items():
                        append_module(f"{attr}['{key}']", item)
                else:
                    append_module(attr, obj)

            return modules

        def _parse_program(
            attribute: str, program: object, parent: Optional[Node] = None
        ) -> Tuple[List[Node], List[Edge]]:
            nodes = []
            edges = []

            program_name = program.__class__.__name__
            program_forward = getattr(program, "forward")

            this = ProgramNode(
                id=f"{parent.id}.{attribute}" if parent else attribute,
                parent=parent.id if parent else None,
                variable=attribute,
                type=Node.Type.PROGRAM,
                program=program_name,
                compiled=program._compiled,
                usage=ProgramNode.Usage(input_tokens=0, output_tokens=0),
                metadata={},
                dspy_instance=program,
                cyto_instance=None,
            )
            nodes.append(this)

            # TODO: Get (forward) parameters

            modules = _get_program_modules(program)
            for attribute, module in modules:
                if isinstance(module, dspy.Retrieve):
                    retriever_nodes, retriever_edges = _parse_retriever(attribute, module, this)
                    nodes.extend(retriever_nodes)
                    edges.extend(retriever_edges)
                elif isinstance(module, dspy.Predict):
                    predictor_nodes, predictor_edges = _parse_predictor(attribute, module, this)
                    nodes.extend(predictor_nodes)
                    edges.extend(predictor_edges)
                elif isinstance(module, dspy.Program):
                    program_nodes, program_edges = _parse_program(attribute, module, this)
                    nodes.extend(program_nodes)
                    edges.extend(program_edges)

            def wrap_forward(*args, **kwargs):
                result = program_forward(*args, **kwargs)
                parameters = list(kwargs.items()) + result.items()
                _update_parameters(this, parameters, nodes, edges)
                _update_program(this)
                if not parent:  # Only update graph once after root program finishes
                    _update_graph()
                    if selected_node:
                        _select_node(selected_node, force=True)
                return result

            setattr(program, "forward", wrap_forward)

            return nodes, edges

        graph_widget.on("node", "click", _graph_widget_on_node_click)
        export_button_widget.on_click(_export_button_widget_on_click)
        save_button_widget.on_click(_save_button_widget_on_click)
        reset_button_widget.on_click(_reset_button_widget_on_click)
        panel_info_widget.add_class("dspy-inspector-panel-info")
        panel_actions_widget.add_class("dspy-inspector-panel-actions")
        panel_widget.add_class("dspy-inspector-panel")
        inspector_widget.add_class("dspy-inspector")

        # Initial graph update
        graph["nodes"], graph["edges"] = _parse_program(program.__class__.__name__.lower(), program)
        _update_graph()
        _select_node(graph["nodes"][0], force=True)

        return inspector_widget
