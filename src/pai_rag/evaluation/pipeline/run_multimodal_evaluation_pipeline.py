import asyncio
from pai_rag.evaluation.utils.create_components import (
    get_rag_components,
    get_rag_config_and_mode,
    get_multimodal_eval_components,
)


def run_multimodal_evaluation_pipeline(
    config_file=None,
    oss_path=None,
    qca_dataset_path=None,
    data_path=None,
    pattern=None,
    exp_name="default",
    eval_model_source=None,
    eval_model_name=None,
    tested_multimodal_llm_config=None,
):
    assert (oss_path is not None) or (
        data_path is not None
    ), "Must provide either local path or oss path."
    assert (oss_path is None) or (
        data_path is None
    ), f"Can not provide both local path '{data_path}' and oss path '{oss_path}'."

    config, mode, exist_flag = get_rag_config_and_mode(config_file, exp_name)
    data_loader, vector_index, query_engine = get_rag_components(config)
    multimodal_qca_generator, evaluator = get_multimodal_eval_components(
        config,
        vector_index,
        query_engine,
        mode,
        eval_model_source,
        eval_model_name,
        exp_name,
        tested_multimodal_llm_config,
        qca_dataset_path,
    )
    if qca_dataset_path:
        _ = asyncio.run(
            multimodal_qca_generator.agenerate_qca_dataset(stage="predicted")
        )
        response_result = asyncio.run(evaluator.aevaluation(stage="response"))
        return {"response": response_result}

    if not exist_flag:
        data_loader.load_data(
            file_path_or_directory=data_path,
            filter_pattern=pattern,
            oss_path=oss_path,
            from_oss=oss_path is not None,
            enable_raptor=False,
        )

    _ = asyncio.run(multimodal_qca_generator.agenerate_qca_dataset(stage="labelled"))
    _ = asyncio.run(multimodal_qca_generator.agenerate_qca_dataset(stage="predicted"))
    response_result = asyncio.run(evaluator.aevaluation(stage="response"))
    return {"response": response_result}
