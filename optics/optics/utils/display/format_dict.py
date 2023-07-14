def format_dict(tree: dict, depth: int = 0) -> str:
    """Formats a dictionary with indentation."""
    result = ''
    for key in tree:
        if isinstance(key, type):
            result += '\t' * depth + key.__name__
            child = tree[key]
            if isinstance(child, dict):
                if len(child) > 0:
                    # recurse
                    result += ':\n' + format_dict(child, depth + 1)
                else:
                    result += '()\n'
            else:
                result += '=' + str(child) + ', '
        else:
            result += '\t' * depth + "'" + str(key) + "': "
            child = tree[key]
            if isinstance(child, dict):
                arguments = [str(child_key) + '=' + str(child[child_key]) for child_key in child]
                result += '(' + ', '.join(arguments) + ')\n'
            else:
                result += ': ' + str(child) + '\n'
    return result
