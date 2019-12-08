import os
import argparse
import numpy as np
from collections import Counter, defaultdict


def leaf_node_proportion(parent):
    n = len(parent)
    # Start with all nodes being leafs.
    is_leaf = [1 for i in range(n)]
    for i in range(1, n): # starts at -1 for the first artificial token.
        # Set the array to 0 for nodes that are someone's head.
        is_leaf[parent[i]] = 0
    return sum(is_leaf)/(len(is_leaf) - 1) # -1 for the first artificial node

# This functio fills depth of i'th element in parent[]
# The depth is filled in depth[i]
def fill_depth(parent, i , depth):

    # If depth[i] is already filled
    if depth[i] != 0:
        return

    # If node at index i is root
    if parent[i] == -1:
        depth[i] = 0
        return

    # If depth of parent is not evaluated before,
    # then evaluate depth of parent first
    if depth[parent[i]] == 0:
        fill_depth(parent, parent[i] , depth)

    # Depth of this node is depth of parent plus 1
    depth[i] = depth[parent[i]] + 1

# This function reutns height of binary tree represented
# by parent array
def find_height(parent):
    n = len(parent)
    # Create an array to store depth of all nodes and
    # initialize depth of every node as 0
    # Depth of root is 1
    depth = [0 for i in range(n)]

    # fill depth of all nodes
    for i in range(n):
        fill_depth(parent, i, depth)

    # The height of binary tree is maximum of all
    # depths. Find the maximum in depth[] and assign
    # it to ht
    ht = depth[0]
    for i in range(1,n):
        ht = max(ht, depth[i])

    return ht

def trees_from_structure_folder(folder_path):
    tree_structures = []
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        tree = read_tree_from_file(os.path.join(folder_path, file_name))
        tree_structures.append(tree)
    return tree_structures

def read_tree_from_file(file_path):
    tree_list = None
    with open(file_path) as struct_file:
        lines = struct_file.readlines()
        for line in lines:
            if not line.startswith('['):
                continue
            else:
                # Line looks like:
                # [-1, 0, 1, 1] 2.0449346899986267
                tree_string = line[1:line.find(']')]
                tree_list = [int(val) for val in tree_string.split(', ')]
                break
        assert tree_list != None, "Problem with the tree structure extracted."
        return tree_list

def tree_distance(latent_tree, explicit_tree):
    """
    Computes precision and recall of shared links between latent tree and explicit structure.
    latent_tree should be represent with a parent array of size n + 1
    explicit_tree should be a numpy matrix of size n x n (undirected).

    The weights don't matter, we consider any non zero entry to correspond to a link.
    """
    assert np.sum(explicit_tree.diagonal()) == 0, "Non zero diagonal elements."

    undirected = explicit_tree + explicit_tree.T
    # undirected = explicit_tree
    n = len(latent_tree) - 1
    shared_links = [0 for _ in range(n)]
    for i in range(1, n):
        if undirected[i][latent_tree[i+1]-1] != 0 or undirected[latent_tree[i+1]-1][i] != 0:
            shared_links[i] = 1

    print(shared_links)
    # Precision over the edges in the latent tree.
    precision = sum(shared_links) / (n - 1)
    # Recall over the edges in the explicit structure.
    recall = sum(shared_links) / (np.sum(undirected != 0)  / 2)

    return precision, recall

if __name__ == '__main__':
    """
    Provide the path to the structures folder.
    Extract metrics about the structure folder.

    Depth
    Proportion of leaf nodes
    TODO: Normalized arc length?

    """
    parser = argparse.ArgumentParser(description='anlyze tree structures.')
    parser.add_argument('--structures_path', type=str, default=None, help='location of the structures folder')
    parser.add_argument('--result_file_path', type=str, default='stats_tree.txt', help='file path of results stats.')
    args = parser.parse_args()

    assert os.path.isdir(args.structures_path), 'Provided structures_path argument is not a directory:\n {}'.format(args.structures_path)

    # 1. Read trees.
    trees = trees_from_structure_folder(args.structures_path)

    # 2. Calculate metrics.
    heights = []
    proportion_leafs = []
    for tree in trees:
        heights.append(find_height(tree))
        proportion_leafs.append(leaf_node_proportion(tree))

    avg_height = sum(heights) / len(heights)
    avg_proportion_leafs = sum(proportion_leafs) / len(proportion_leafs)

    # Write metrics to file.
    with open(args.result_file_path, 'w') as stats_file:
        stats_file.write('Average depth of tree: {}\n'.format(avg_height))
        stats_file.write('Average proportion of leaf nodes in tree: {}\n'.format(avg_proportion_leafs))

    # Dump file to stout.
    with open(args.result_file_path, 'r') as stats_file:
        print(stats_file.read())


# adj_matrix = np.array([
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [1, 0, 0]])
    # adj_matrix_2 = np.array([
    #     [0, 0, 0],
    #     [1, 0, 1],
    #     [1, 0, 0]])
    # adj_matrix_3 = np.array([
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [1, 0, 0]])

    # heads = [-1, 0, 1, 1]
    # heads_2 = [-1, 0, 0, 2]
    # heads_3 = [-1, 0, 0, 1]

    # print(tree_distance(heads, adj_matrix))
    # print(tree_distance(heads, adj_matrix_2))
    # print(tree_distance(heads, adj_matrix_3))

    # print()
    # print(tree_distance(heads_2, adj_matrix))
    # print(tree_distance(heads_3, adj_matrix))