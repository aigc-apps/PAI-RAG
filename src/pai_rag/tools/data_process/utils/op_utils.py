from pai_rag.tools.data_process.ops.base_op import OPERATORS


def load_ops(process_list):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param op_fusion: whether to fuse ops that share the same intermediate
        variables.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []
    op_names = []
    for process in process_list:
        op_name, args = list(process.items())[0]
        if op_name == "pai_rag_parser":
            ops.append(OPERATORS.modules[op_name].remote(**args))
        else:
            ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)
        op_names.append(op_name)

    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    return ops, op_names
