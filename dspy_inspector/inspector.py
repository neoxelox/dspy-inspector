import html
import inspect
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pprint import pprint
from typing import Callable, List, Optional, Tuple

import dsp
import dspy
import ipycytoscape as cytoscape
import ipywidgets as widgets
import tiktoken

with open(os.path.join(os.path.dirname(__file__), "styles.css"), "r") as file:
    _styles_css = file.read()


def _RunHTMLScript(script: str) -> str:
    return f"""<style onload="(function(){{{script}}})();"></style>"""


class _StrEnum(str, Enum):
    def __str__(self) -> str:
        return str(self.value)


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
    cyto_instance: object


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
    completions: Optional[List[str]]
    demos: Optional[List[str]]

    @property
    def label(self) -> str:
        list_marker = self.variable.find("[")
        if list_marker > 0:
            return f"{self.signature.name}\n{self.module}\n{self.variable[list_marker:]}"
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
    passages: Optional[List[str]]

    @property
    def label(self) -> str:
        list_marker = self.variable.find("[")
        if list_marker > 0:
            return f"{self.module}\n{self.variable[list_marker:]}"
        return self.module

    @property
    def title(self) -> str:
        return self.module

    @property
    def subtitle(self) -> Optional[str]:
        return None


@dataclass
class ProgramNode(Node):
    program: str
    compiled: bool

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
    source: str
    target: str


class Inspector:
    def __init__(self, debug: bool, tokenizer_of_model: str) -> None:
        self.debug = debug
        encoder = tiktoken.encoding_for_model(tokenizer_of_model)
        self.tokenizer = partial(encoder.encode, allowed_special="all")

    _graph_widget_style = [
        {
            "selector": "core",
            "style": {
                "active-bg-color": "yellow",
                "active-bg-opacity": 1,
                "active-bg-size": "2em",
            },
        },
        {
            "selector": "node",
            "style": {
                "width": "2em",
                "height": "2em",
                "shape": "round-rectangle",
                "background-color": "blue",
                "font-family": 'BlinkMacSystemFont, -apple-system, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", "Helvetica", "Arial", sans-serif',  # noqa: E501
                "font-size": "1em",
                "label": "data(label)",
                "color": "purple",
                "text-valign": "top",
                "text-halign": "right",
                "text-max-width": "30em",
                "text-wrap": "wrap",
            },
        },
        {
            "selector": "node[type='parameter']",
            "style": {
                "background-color": "grey",
            },
        },
        {
            "selector": "node[type='retriever']",
            "style": {
                "background-color": "yellow",
            },
        },
        {
            "selector": "node[type='predictor']",
            "style": {
                "background-color": "green",
            },
        },
        {
            "selector": "node[type='program']",
            "style": {
                "background-color": "#93c5fd",
            },
        },
        {
            "selector": "node:active",
            "style": {
                "overlay-color": "blue",
                "overlay-opacity": 0.1,
                "overlay-shape": "rectangle",
            },
        },
        {
            "selector": "node.selected",
            "style": {
                "overlay-color": "blue",
                "overlay-opacity": 0.1,
                "overlay-shape": "rectangle",
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "bezier",
                "width": "0.20em",
                "line-color": "pink",
                #  'arrow-scale': '0.75',
                #  'source-arrow-shape': 'circle',
                #  'source-arrow-color': 'red',
                #  'source-endpoint': 'inside-to-node',
                #  'source-distance-from-node': "0.80em",
                #  'target-arrow-shape': 'circle',
                #  'target-arrow-color': 'red',
                #  'target-endpoint': 'inside-to-node',
                #  'target-distance-from-node': "0.80em",
                "arrow-scale": "1",
                "target-arrow-shape": "chevron",
                "target-arrow-color": "red",
                "target-endpoint": "outside-to-node",
                "target-distance-from-node": "0",
            },
        },
        {
            "selector": "edge:active",
            "style": {
                "overlay-opacity": 0,
            },
        },
    ]
    _panel_widget_style = f"<style>{_styles_css}</style>"

    def inspect(self, program: dspy.Program) -> widgets.Widget:
        # TODO: The first load is very slow

        if inspect.isclass(program) and issubclass(program, dspy.Program):
            program = program()

        if not isinstance(program, dspy.Program):
            raise TypeError(f"provided program of type {type(program)} is not a DSPy program")

        graph = {"nodes": [], "edges": []}
        selected_node = None

        graph_widget = cytoscape.CytoscapeWidget()
        graph_widget.set_style(self._graph_widget_style)
        graph_widget.set_layout(name="dagre", nodeSpacing=10, edgeLengthVal=10)

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
                cytoscape_graph["nodes"].append(
                    {
                        "id": node.id,
                        "type": node.type,
                        "label": node.label,
                        **({"parent": node.parent} if node.parent else {}),
                    }
                )

            for edge in graph["edges"]:
                cytoscape_graph["edges"].append(
                    {
                        "source": edge.source,
                        "target": edge.target,
                    }
                )

            graph_widget.graph.clear()
            graph_widget.graph.add_graph_from_json(cytoscape_graph, directed=True)

            # TODO: Replace this quadratic for...
            for cyto_node in graph_widget.graph.nodes:
                for node in graph["nodes"]:
                    if cyto_node.data["id"] == node.id:
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
<div><dt>Prefix</dt><dd>{html.escape(parameter.prefix)}</dd></div>
<div><dt>Description</dt><dd>{html.escape(parameter.description)}</dd></div>
<div><dt>Format</dt><dd>{parameter.format.__name__ if parameter.format else 'None'}</dd></div>
<div><dt>Value</dt><dd>{html.escape(parameter.value.text) if parameter.value else 'None'}</dd></div>
<div><dt>Tokens</dt><dd>{parameter.value.tokens if parameter.value else '0'}</dd></div>
"""  # noqa: E501

        def _draw_predictor_panel_info_widget(predictor: PredictorNode) -> str:
            return f"""
<div><dt>Signature</dt><dd>{predictor.signature.syntax}<br/><i>(From: {predictor.signature.name})</i></dd></div>
<div><dt>Instructions</dt><dd>{html.escape(predictor.signature.instructions)}</dd></div>
<div><dt>Module</dt><dd>{predictor.module}</dd></div>
<div><dt>Prompt</dt><dd>{html.escape(predictor.prompt or 'None')}<b>{html.escape(predictor.completions[0]) if predictor.completions and len(predictor.completions) > 0 else ''}</b></dd></div>
<div><dt>Tokens</dt><dd>Input: {predictor.usage.input_tokens} | Output: {predictor.usage.output_tokens}</dd></div>
<div><dt>Demos</dt><dd>{[f'{html.escape(demo)}<br/><br/>' for demo in predictor.demos] if predictor.demos else 'None'}</dd></div>
<div><dt>Model</dt><dd>{predictor.model.name}</dd></div>
<div><dt>Settings</dt><dd>{predictor.model.settings}</dd></div>
"""  # noqa: E501

        def _draw_retriever_panel_info_widget(retriever: RetrieverNode) -> str:
            return f"""
<div><dt>Module</dt><dd>{retriever.module}</dd></div>
<div><dt>Passages</dt><dd>{[f'{html.escape(passage)}<br/><br/>' for passage in retriever.passages] if retriever.passages else 'None'}</dd></div>
<div><dt>Tokens</dt><dd>Input: {retriever.usage.input_tokens} | Output: {retriever.usage.output_tokens}</dd></div>
<div><dt>Model</dt><dd>{retriever.model.name}</dd></div>
<div><dt>Settings</dt><dd>{retriever.model.settings}</dd></div>
"""  # noqa: E501

        def _draw_program_panel_info_widget(program: ProgramNode) -> str:
            return f"""
<div><dt>Compiled</dt><dd>{program.compiled}</dd></div>
<div><dt>Subprogram</dt><dd>{program.parent is not None}</dd></div>
"""  # noqa: E501

        def _update_graph() -> None:
            if self.debug:
                pprint(graph)

            _draw_graph_widget()

        def _select_node(node: Node) -> None:
            nonlocal selected_node

            if self.debug:
                pprint(node)

            if selected_node:
                selected_node.cyto_instance.classes = ""
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
            node = next(filter(lambda node: node.id == event["data"]["id"], graph["nodes"]), None)
            if node:
                _select_node(node)

        def _export_button_widget_on_click(*args, **kwargs) -> None:
            panel_action_executor_widget.value = ""

            # TODO: Export graph in PNG
            panel_action_executor_widget.value = _RunHTMLScript("""console.log('export not implemented');""")

        def _save_button_widget_on_click(*args, **kwargs) -> None:
            panel_action_executor_widget.value = ""

            # TODO: Save graph in json
            panel_action_executor_widget.value = _RunHTMLScript("""console.log('save not implemented');""")

        def _reset_button_widget_on_click(*args, **kwargs) -> None:
            graph_widget.relayout()

        def _update_parameters(this: Node, parameters: list, nodes: List[Node], edges: List[Edge]) -> None:
            for parameter, parameter_value in parameters:
                parameter_node = next(filter(lambda node: node.id == f"{this.id}.{parameter}", nodes), None)
                if parameter_node:
                    if parameter_node.format:
                        parameter_value = parameter_node.format(parameter_value)

                    parameter_node.value = ParameterNode.Value(
                        text=parameter_value, tokens=len(self.tokenizer(parameter_value))
                    )

                    # TODO: Create edges between parameter's traversed nodes

        def _update_predictor(predictor: PredictorNode) -> None:
            # TODO: This probably won't work if model calls are done in parallel
            call = predictor.model.instance.history[-1]

            predictor.usage.input_tokens = call["response"]["usage"]["prompt_tokens"]
            predictor.usage.output_tokens = call["response"]["usage"]["completion_tokens"]

            predictor.prompt = call["prompt"]
            predictor.completions = [choice["text"] for choice in call["response"]["choices"]]

            predictor.demos = predictor.dspy_instance.demos

        def _update_retriever(retriever: RetrieverNode) -> None:
            # TODO: Can't update usage because dspy does not support it yet
            # TODO: Can't update passages because dspy does not support it yet

            pass

        def _update_program(program: ProgramNode) -> None:
            pass

        def _parse_parameter(attribute: str, parameter: object, parent: Node) -> Node:
            parameter_name = attribute
            if isinstance(parameter, dspy.InputField):
                parameter_direction = ParameterNode.Direction.INPUT
                # dada
            elif isinstance(parameter, dspy.OutputField):
                parameter_direction = ParameterNode.Direction.OUTPUT
            elif isinstance(parameter, dsp.Type):
                # TODO: how to get direction??
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
                    # TODO: how to get direction??
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
                completions=None,
                demos=None,
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
                        )
                    )
                elif parameter_node.direction == ParameterNode.Direction.OUTPUT:
                    edges.append(
                        Edge(
                            source=this.id,
                            target=parameter_node.id,
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
                    name="unknown",  # TODO: Cannot get module.kwargs["model"] because dspy does not support it yet
                    settings={},  # TODO: Cannot get module.copy().kwargs because dspy does not support it yet
                    instance=module_model,
                ),
                usage=RetrieverNode.Usage(
                    input_tokens=0,
                    output_tokens=0,
                ),
                passages=None,
                metadata={},
                dspy_instance=module,
                cyto_instance=None,
            )
            nodes.append(this)

            # TODO: Get (forward) parameters

            def wrap_forward(*args, **kwargs):
                result = module_forward(*args, **kwargs)
                parameters = list(kwargs.items()) + result.items()
                _update_parameters(this, parameters, nodes, edges)
                _update_retriever(this)
                return result

            setattr(module, "forward", wrap_forward)

            return nodes, edges

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
                metadata={},
                dspy_instance=program,
                cyto_instance=None,
            )
            nodes.append(this)

            # TODO: Get (forward) parameters

            for attribute, module in program.named_parameters():
                if isinstance(module, dspy.Retrieve):
                    retriever_nodes, retriever_edges = _parse_retriever(attribute, module, this)
                    nodes.extend(retriever_nodes)
                    edges.extend(retriever_edges)
                elif isinstance(module, dspy.Predict):
                    predictor_nodes, predictor_edges = _parse_predictor(attribute, module, this)
                    nodes.extend(predictor_nodes)
                    edges.extend(predictor_edges)
                elif isinstance(module, dspy.Program):
                    # TODO: how to get sub-programs?? Check: dspy.BaseModule.named_parameters func
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
                    _select_node(selected_node)
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
        _select_node(graph["nodes"][0])

        return inspector_widget
