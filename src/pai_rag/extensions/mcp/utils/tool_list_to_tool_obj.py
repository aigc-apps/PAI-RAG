from collections import defaultdict


def tool_list_to_tool_obj(tools):
    # Initialize a dictionary with default values
    tool_calls_dict = defaultdict(
        lambda: {
            "id": "",
            "function": {"arguments": "", "name": ""},
            "type": "",
            "index": 0,
        }
    )

    # Iterate over the tool calls
    for tool_call in tools:
        # If the id is not None, set it
        if tool_call.id is not None:
            tool_calls_dict[tool_call.index]["id"] += tool_call.id
            tool_calls_dict[tool_call.index]["index"] = tool_call.index

        # If the function name is not None, set it
        if tool_call.function.name is not None:
            tool_calls_dict[tool_call.index]["function"][
                "name"
            ] += tool_call.function.name

        # Append the arguments
        if tool_call.function.arguments is not None:
            tool_calls_dict[tool_call.index]["function"][
                "arguments"
            ] += tool_call.function.arguments

        # If the type is not None, set it
        if tool_call.type is not None:
            tool_calls_dict[tool_call.index]["type"] = tool_call.type

    # Convert the dictionary to a list
    tool_calls_list = list(tool_calls_dict.values())

    # Return the result
    return tool_calls_list
