import copy
import warnings

import graphviz


def draw_net(genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for ng_id, ng in genome.node_genes.items():
        if ng.type == 'INPUT':
            inputs.add(ng_id)
            name = node_names.get(ng_id, str(ng_id))
            input_attrs = {'style': 'filled',
                           'shape': 'box'}
            input_attrs['fillcolor'] = node_colors.get(ng_id, 'lightgray')
            dot.node(name, _attributes=input_attrs)

    outputs = set()
    for ng_id, ng in genome.node_genes.items():
        if ng.type == 'OUTPUT':
            outputs.add(ng_id)
            name = node_names.get(ng_id, str(ng_id))
            node_attrs = {'style': 'filled'}
            node_attrs['fillcolor'] = node_colors.get(ng_id, 'lightblue')

            dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.conn_genes.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            # print(pending, used_nodes)
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.node_genes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled'}
        attrs['fillcolor'] = node_colors.get(n, 'white')
        dot.node(str(n), _attributes=attrs)

    for cg in genome.conn_genes.values():
        if cg.enabled or show_disabled:
            if cg.in_node_id not in used_nodes or cg.out_node_id not in used_nodes:
                continue

            a = node_names.get(cg.in_node_id, str(cg.in_node_id))
            b = node_names.get(cg.out_node_id, str(cg.out_node_id))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
