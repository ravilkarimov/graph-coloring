#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import typing

from copy import deepcopy
from collections import namedtuple


Node = namedtuple("Node",["id","color","edges","unpossible_colors"])
Solver_Node = namedtuple("Solver_Node",["id","parent_id","nodes_remain","solution"])


# Dummy solution
def baseline(node_count: int) -> typing.List[int]:
    return list(range(node_count))

def parse_graph(lines: typing.List[str]) -> typing.Dict[int, Node]:
    graph = {}
    for line in lines:
        nodes = list(map(int, line.split()))
        for node in nodes:
            edges = {i for i in nodes if i != node}
            if node not in graph:
                graph[node] = Node(
                    id=node,
                    color=None,
                    edges=edges,
                    unpossible_colors=set())
            else:
                graph[node].edges.update(edges)
    return graph

def get_node_copy_with_color(node: Node, color: int) -> Node:
    return Node(id=node.id,
                color=color,
                edges=node.edges,
                unpossible_colors=node.unpossible_colors)

def get_color(
    last_color: int,
    used_colors: typing.Set[int],
    unpossible_colors: typing.Set[int]
) -> typing.Tuple[int, int]:
    if not used_colors:
        color = last_color
    else:
        # choose any color from used_colors - unpossible
        choices = used_colors - unpossible_colors
        if len(choices) > 0:
            color = min(choices)
        else:
            last_color += 1
            color = last_color
    return last_color, color

def greedy(graph: typing.Dict[int, Node]) -> typing.Tuple[int, typing.List[int]]:
    # init variables
    last_color = 0
    used_colors: typing.Set[int] = set()
    graph_c = deepcopy(graph)

    # take any node
    sorted_idx = sorted(
        graph_c,
        key=lambda x: len(graph_c[x].edges),
        reverse=True)

    for idx in sorted_idx:
        node = graph_c[idx]
        last_color, color = get_color(last_color, used_colors, node.unpossible_colors)
        node = get_node_copy_with_color(node, color)

        # node.color = color
        used_colors.add(color)
        # remove colors from nodes on edges
        for edge in node.edges:
            graph_c[edge].unpossible_colors.add(color)
        graph_c[idx] = node

    # preapare result
    result = [0]*len(graph_c)
    for i, _ in enumerate(result):
        result[i] = graph_c[i].color
    return len(used_colors), result


def initialize_root(graph: typing.Dict[int, Node]) -> Solver_Node:
    # Break symmetry, force first color to be zero
    color = 0
    # Sort node by number of neighbours
    nodes_remain = [graph[x] for x in sorted(graph, key=lambda x: len(graph[x].edges))]
    # Take the first one
    first_node = nodes_remain.pop()
    first_node = Node(
        id=first_node.id,
        color=color,
        edges=first_node.edges,
        unpossible_colors=set())
    # Set constraints to neighbours
    for nbh_idx in first_node.edges:
       neighbour = list(filter(lambda x: x.id == nbh_idx, nodes_remain))
       neighbour[0].unpossible_colors.add(color)
    return Solver_Node(
        id=0,
        parent_id=None,
        nodes_remain=nodes_remain,
        solution={first_node.id: first_node})

def get_solution(
    graph_node: Node,
    solver_node: Solver_Node,
    choices: typing.Set[int]
) -> typing.Tuple[int, typing.List[int]]:

    color = min(choices)
    graph_node = get_node_copy_with_color(graph_node, color)
    solution = solver_node.solution
    solution[graph_node.id] = graph_node
    solution = [solution[x] for x in sorted(solution, key=lambda x: solution[x].id)]
    solution_colors = [x.color for x in solution]
    len_colors = len(set(solution_colors))
    return len_colors, solution_colors

def get_nodes_remain(
    color: int,
    graph_node: Node,
    solver_node: Solver_Node
) -> typing.List[Node]:

    nodes_remain = []
    for node in solver_node.nodes_remain:
        if node.id in graph_node.edges:
            unpossible_colors = node.unpossible_colors.copy()
            node = Node(node.id, node.color, node.edges, unpossible_colors)
            node.unpossible_colors.add(color)
        nodes_remain.append(node)
    # DSATUR logic
    nodes_remain = sorted(nodes_remain, key=lambda x: len(x.unpossible_colors))
    return nodes_remain

def update_queue(
    queue: typing.List[Solver_Node],
    graph_node: Node,
    solver_node: Solver_Node,
    choices: typing.Set[int],
    min_len_colors: int
) -> None:

    max_color = max(x.color for _, x in solver_node.solution.items())
    for color in choices:
        # Break symmetry
        # A new color J is only allowed to appear after colors 0..J-1 have been seen before (in any order)
        # if color - sum(x in range(len(used_colors)) for x in used_colors) > 0:
        if color - max_color > 1:
            continue
        graph_node_copy = get_node_copy_with_color(graph_node, color)

        # adding to all neighbours constraint with cur color
        nodes_remain_c = get_nodes_remain(color, graph_node_copy, solver_node)

        solver_node_sol_copy = solver_node.solution.copy()
        solver_node_sol_copy[graph_node_copy.id] = graph_node_copy
        len_colors = len({solver_node_sol_copy[x].color for x in solver_node_sol_copy})

        if len_colors < min_len_colors:
            queue.append(Solver_Node(
                id = solver_node.id + 1,
                parent_id = solver_node.id,
                nodes_remain = nodes_remain_c,
                solution = solver_node_sol_copy))
        # queue.append(S_Node(id = idx, parent_id = s_node.id, nodes_remain=p_nodes_remain_c, solution=s_node_solution_c))

def opt(graph: typing.Dict[int, Node]) -> typing.Tuple[int, typing.List[int]]:
    start_time = time.time()

    # init
    min_len_colors, best_solution = greedy(graph)
    colors = set(range(min_len_colors))

    queue = [initialize_root(graph)]

    while queue and time.time() - start_time < 60*5:
        solver_node = queue.pop()

        nodes_remain = solver_node.nodes_remain
        graph_node = nodes_remain.pop()

        choices = colors - graph_node.unpossible_colors
        # if we do not have choices we will get more colors than in greedy algo so branch is bad
        if not choices:
            continue

        if not nodes_remain:
            len_colors, solution_colors = get_solution(graph_node, solver_node, choices)
            if min_len_colors > len_colors:
                min_len_colors = len_colors
                best_solution = solution_colors
                # print("New minimum:", min_len_colors)
            continue

        # create node for every choice
        update_queue(queue, graph_node, solver_node, choices, min_len_colors)

    # print("Nodes processed: ",nodes_processed_n)
    result = best_solution
    return min_len_colors, result


def solve_it(input_data, method="opt"):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    graph = parse_graph(lines[1:])

    # build a trivial solution
    # every node has its own color
    # solution = baseline(node_count)
    if method == "greedy":
        len_colors, solution = greedy(graph)
    elif method == "opt":
        len_colors, solution = opt(graph)

    # prepare the solution in the specified output format
    output_data = str(len_colors) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')