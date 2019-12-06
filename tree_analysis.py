import os
import argparse
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
