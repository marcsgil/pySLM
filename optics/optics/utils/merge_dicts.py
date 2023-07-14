import copy


def merge_dicts(dictionary_tree: dict, defaults: dict) -> dict:
    """
    Recursively sets the values in dictionary according to the specified defaults.

    :param defaults: The input tree of dictionaries
    :param dictionary_tree:  The dictionary with the default values
    :return: A new dictionary tree with all the values of the input dictionary_tree and default values from
    the defaults dictionary tree.
    """
    merged = dictionary_tree.copy()
    for k, v in defaults.items():
        # If this branch already exists, check that it is complete
        if k in dictionary_tree:
            # If both are branches, continue merging recursively
            if isinstance(v, dict) and isinstance(dictionary_tree[k], dict):
                merged[k] = merge_dicts(dictionary_tree[k], v)
            # If either is a leaf, just stick with that
        else:  # No branch found for this key, copy the default branch
            merged[k] = copy.deepcopy(v)

    return merged
