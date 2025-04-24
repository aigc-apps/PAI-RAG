from loguru import logger

RAG_PARSER_CPU_REQUIRED = 4
RAG_PARSER_MEM_REQUIRED = 8  # GB

RAG_SPLITTER_CPU_REQUIRED = 2
RAG_SPLITTER_MEM_REQUIRED = 2  # GB

RAG_EMBEDDER_CPU_REQUIRED = 8
RAG_EMBEDDER_MEM_REQUIRED = 10  # GB

min_resource_requirements = {
    "rag_parser": {
        "cpu_required": RAG_PARSER_CPU_REQUIRED,
        "mem_required": RAG_PARSER_MEM_REQUIRED,
    },
    "rag_splitter": {
        "cpu_required": RAG_SPLITTER_CPU_REQUIRED,
        "mem_required": RAG_SPLITTER_MEM_REQUIRED,
    },
    "rag_embedder": {
        "cpu_required": RAG_EMBEDDER_CPU_REQUIRED,
        "mem_required": RAG_EMBEDDER_MEM_REQUIRED,
    },
}


def check_and_set_min_cpu(original_cpu, required_cpu, op_name):
    if original_cpu < required_cpu:
        logger.info(
            f"{op_name}: cpu_required {original_cpu} is less than {required_cpu}，"
            f"set to {required_cpu} as minimal requirements."
        )
        return required_cpu
    else:
        logger.info(
            f"{op_name}: cpu_required {original_cpu} meets the minimal requirements {required_cpu}."
        )
        return original_cpu


def check_and_set_min_mem(original_mem, required_mem, op_name):
    if original_mem < required_mem:
        logger.info(
            f"{op_name}: mem_required {original_mem}GB is less than {required_mem}GB，"
            f"set to {required_mem}GB as minimal requirements."
        )
        return f"{required_mem}GB"
    else:
        logger.info(
            f"{op_name}: mem_required {original_mem}GB meets the minimal requirements {required_mem}GB."
        )
        return f"{original_mem}GB"


def enforce_min_requirements(op_name, args):
    if op_name not in min_resource_requirements:
        logger.error(f"Invalid operation name: {op_name}")
        return

    requirements = min_resource_requirements[op_name]

    # 类型转换和错误处理
    try:
        cpu_key = "cpu_required"
        original_cpu = int(args.get(cpu_key, 0))
        mem_key = "mem_required"
        original_mem = int(args.get(mem_key, 0))
    except (ValueError, TypeError) as e:
        logger.error(f"{op_name}: Invalid parameters - {e}")
        return

    # 检查并设置 CPU 和内存
    args[cpu_key] = check_and_set_min_cpu(original_cpu, requirements[cpu_key], op_name)
    args[mem_key] = check_and_set_min_mem(original_mem, requirements[mem_key], op_name)
    logger.info(
        f"{op_name}: Final Resources - cpu_required: {args[cpu_key]}, mem_required: {args[mem_key]}."
    )
    return args


# 示例使用
if __name__ == "__main__":
    # 假设这些是用户提供的资源需求
    parser_args = {"cpu_required": 4, "mem_required": 6}  # CPU 和内存均低于要求
    splitter_args = {"cpu_required": 3, "mem_required": 2}  # CPU 高于要求，内存满足
    embedder_args = {"cpu_required": 8, "mem_required": 12}  # CPU 和内存均满足要求

    # 应用资源限制
    parser_args = enforce_min_requirements("rag_parser", parser_args)
    splitter_args = enforce_min_requirements("rag_splitter", splitter_args)
    embedder_args = enforce_min_requirements("rag_embedder", embedder_args)

    # 输出结果
    logger.info(f"Parser Args: {parser_args}")
    logger.info(f"Splitter Args: {splitter_args}")
    logger.info(f"Embedder Args: {embedder_args}")
