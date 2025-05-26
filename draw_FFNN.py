import dash
import dash_cytoscape as cyto
from dash import html, Input, Output, State
from FFNN import FFNN
import numpy as np

def visualize_FFNN(ffnn: FFNN):
    nodes, edges = [], []

    n_inputs = ffnn.layers[0].W.data.shape[1]
    input_nodes = [f"x{i+1}" for i in range(n_inputs)]

    y_offset = 0
    max_nodes_per_layer = max([len(layer.W.data) for layer in ffnn.layers] + [n_inputs])
    layer_height = max_nodes_per_layer * 150

    layer_start_y = y_offset + (layer_height - n_inputs * 150) / 2
    prev_layer_nodes = input_nodes
    nodes.extend([{"data": {"id": node, "label": node}, "classes": "input", "position": {"x": 0, "y": i * 150 + layer_start_y}} for i, node in enumerate(input_nodes)])

    layer_distance = 100 * max_nodes_per_layer
    for layer_idx, layer in enumerate(ffnn.layers):
        n_neurons = layer.W.data.shape[0]
        sum_nodes = [f"S{layer_idx + 1}_N{n_idx + 1}" for n_idx in range(n_neurons)]
        act_nodes = [f"A{layer_idx + 1}_N{n_idx + 1}" for n_idx in range(n_neurons)]

        layer_start_y = y_offset + (layer_height - n_neurons * 150) / 2

        nodes.extend([{"data": {"id": node, "label": node}, "classes": "sum", "position": {"x": (layer_idx + 1) * layer_distance, "y": i * 150 + layer_start_y}} for i, node in enumerate(sum_nodes)])
        nodes.extend([{"data": {"id": node, "label": node}, "classes": "activation", "position": {"x": (layer_idx + 1) * layer_distance + 150, "y": i * 150 + layer_start_y}} for i, node in enumerate(act_nodes)])

        for neuron_idx in range(n_neurons):
            for prev_idx, prev_node in enumerate(prev_layer_nodes):
                weight_value = layer.W.data[neuron_idx, prev_idx]
                weight_grad = layer.W.grad
                label_extension = f"\ndw={weight_grad[neuron_idx, prev_idx]}" if weight_grad is not None else ""
                edges.append({
                    "data": {"id": f"{prev_node}-{sum_nodes[neuron_idx]}", "source": prev_node, "target": sum_nodes[neuron_idx], "weight": f"w={weight_value:.2f}{label_extension}"},
                    "classes": "hidden-label"
                })

            bias_value = layer.b.data[neuron_idx]
            bias_grad = layer.b.grad
            label_extension = f"\ndb={np.sum(bias_grad, axis=0)[neuron_idx]}" if bias_grad is not None else ""
            bias_node = f"b{layer_idx}_N{neuron_idx}"
            nodes.append({"data": {"id": bias_node, "label": bias_node}, "classes": "bias", "position": {"x": (layer_idx + 1) * layer_distance - (layer_distance) // 2, "y": neuron_idx * 150 + layer_start_y}})
            edges.append({
                "data": {"id": f"{bias_node}-{sum_nodes[neuron_idx]}", "source": bias_node, "target": sum_nodes[neuron_idx], "weight": f"b={bias_value:.2f}{label_extension}"},
                "classes": "hidden-label"
            })

            edges.append({
                "data": {"id": f"{sum_nodes[neuron_idx]}-{act_nodes[neuron_idx]}", "source": sum_nodes[neuron_idx], "target": act_nodes[neuron_idx], "weight": layer.activation.__name__},
                "classes": "hidden-label"
            })

        prev_layer_nodes = act_nodes

    output_nodes = [f"y{i+1}" for i in range(len(prev_layer_nodes))]
    layer_start_y = y_offset + (layer_height - len(prev_layer_nodes) * 150) / 2
    nodes.extend([{"data": {"id": node, "label": node}, "classes": "output", "position": {"x": (len(ffnn.layers) + 1) * layer_distance, "y": i * 150 + layer_start_y}} for i, node in enumerate(output_nodes)])

    for prev_node, out_node in zip(prev_layer_nodes, output_nodes):
        edges.append({
            "data": {"id": f"{prev_node}-{out_node}", "source": prev_node, "target": out_node, "weight": ""},
            "classes": "hidden-label"
        })

    app = dash.Dash(__name__)

    app.layout = html.Div([
        cyto.Cytoscape(
            id='ffnn-graph',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '800px'},
            elements=nodes + edges,
            stylesheet=[
                {"selector": "node", "style": {
                    "content": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "color": "white",
                    "width": "60px",
                    "height": "60px",
                    "font-size": "18px",
                    "border-width": "2px",
                    "border-color": "black"
                }},
                {"selector": ".input", "style": {"background-color": "#FFA500"}},
                {"selector": ".sum", "style": {"background-color": "#2ca02c"}},
                {"selector": ".activation", "style": {"background-color": "#1f77b4"}},
                {"selector": ".output", "style": {"background-color": "#FF4500"}},
                {"selector": ".bias", "style": {"background-color": "#d62728"}},
                {"selector": "edge", "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "label": "",
                    "font-size": "0px"
                }},
                {"selector": ".show-label", "style": {
                    "label": "data(weight)",
                    "font-size": "14px",
                    "color": "black",
                    "text-background-opacity": 1,
                    "text-background-color": "white",
                    "text-border-opacity": 1,
                    "text-border-width": 1,
                    "text-border-color": "black"
                }}
            ]
        ),
        html.Div(id='edge-data')
    ])

    @app.callback(
        Output('ffnn-graph', 'stylesheet'),
        Input('ffnn-graph', 'tapEdgeData'),
        State('ffnn-graph', 'stylesheet')
    )
    def display_edge_data(edge_data, stylesheet):
        if edge_data:
            edge_id = edge_data['id']
            found = False
            for style in stylesheet:
                if style['selector'] == f'edge[id = "{edge_id}"]':
                    if 'label' in style['style'] and style['style']['label'] == 'data(weight)':
                        style['style']['label'] = ''
                        style['style']['font-size'] = '0px'
                    else:
                        style['style'] = {
                            "label": "data(weight)",
                            "font-size": "14px",
                            "color": "black",
                            "text-background-opacity": 1,
                            "text-background-color": "white",
                            "text-border-opacity": 1,
                            "text-border-width": 1,
                            "text-border-color": "black"
                        }
                    found = True
                    break
            if not found:
                stylesheet.append({
                    "selector": f'edge[id = "{edge_id}"]',
                    "style": {
                        "label": "data(weight)",
                        "font-size": "14px",
                        "color": "black",
                        "text-background-opacity": 1,
                        "text-background-color": "white",
                        "text-border-opacity": 1,
                        "text-border-width": 1,
                        "text-border-color": "black"
                    }
                })
            return stylesheet
        else:
            return stylesheet

    return app

# Example usage in Jupyter Notebook:
def test_main() :
    from activation import sigmoid, tanh
    from init import he_init
    from Layer import Layer
    layers = [Layer(3, 4, sigmoid, he_init), Layer(4, 3, sigmoid, he_init), Layer(3, 4, tanh, he_init)]
    ffnn = FFNN(layers=layers, layer_sizes=[3,4,3,4])
    app = visualize_FFNN(ffnn)
    # app.run(mode='inline')
