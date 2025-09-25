from loguru import logger
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.evaluation.dataset import DatasetEntity
from db.models.evaluation.dataset import DatasetSampleEntity
from db.models.evaluation.experiment import ExperimentSampleEntity, ExperimentEntity
from db.models.evaluation.run_config import RunConfigEntity
from db.models.evaluation.evaluator_config import EvaluatorConfigEntity
from db.models.llm import LlmModelEntity
from typing import List
from datetime import datetime, timezone
from common.chat.models import ChatAgentRequest
from evaluation.run import run_agent, run_evaluator
from chat.openai.openai_like import OpenAILike
from sqlmodel import select
from common.encrypt_utils import decrypt_key
from fastapi import UploadFile
import json
@with_async_db_session
async def get_exp_run_entity(
    session: AsyncSession,
    exp_run_id: str,
) -> ExperimentSampleEntity:
    exp_run_entity = await session.get(ExperimentSampleEntity, exp_run_id)
    assert exp_run_entity is not None, f"Evaluation experiment run entity {exp_run_id} not found."
    return exp_run_entity

@with_async_db_session
async def get_dataset_entity(
    session: AsyncSession,
    dataset_id: str,
) -> DatasetEntity:
    dataset_entity = await session.get(DatasetEntity, dataset_id)
    assert dataset_entity is not None, f"Evaluation {dataset_id} not found."
    return dataset_entity

@with_async_db_session
async def get_dataset_sample_entity(
    session: AsyncSession,
    sample_id: str,
) -> DatasetSampleEntity:
    dataset_sample_entity = await session.get(DatasetSampleEntity, sample_id)
    assert dataset_sample_entity is not None, f"Dataset sample {sample_id} not found."
    return dataset_sample_entity

@with_async_db_session
async def get_experiment_entity(
    session: AsyncSession,
    experiment_id: str,
) -> ExperimentEntity:
    exp_entity = await session.get(ExperimentEntity, experiment_id)
    assert exp_entity is not None, f"Experiment {experiment_id} not found."
    return exp_entity

@with_async_db_session
async def get_run_config_entity(
    session: AsyncSession,
    run_config_id: str,
) -> RunConfigEntity:
    run_config_entity = await session.get(RunConfigEntity, run_config_id)
    assert run_config_entity is not None, f"RunConfig {run_config_id} not found."
    return run_config_entity

@with_async_db_session
async def get_evaluator_config_entity(
    session: AsyncSession,
    evaluator_config_id: str,
) -> EvaluatorConfigEntity:
    evaluator_config_entity = await session.get(EvaluatorConfigEntity, evaluator_config_id)
    assert evaluator_config_entity is not None, f"EvaluatorConfig {evaluator_config_id} not found."
    return evaluator_config_entity

@with_async_db_session
async def get_llm_model(
    session: AsyncSession,
    model_id: str,
) -> OpenAILike:
    model_entity = (await session.exec(
        select(LlmModelEntity).where(LlmModelEntity.model_id == model_id)
    )).first()

    assert model_entity is not None, f"Model ID {model_id} not found."
    return OpenAILike(
        model=model_entity.model,
        api_base=model_entity.base_url,
        api_key=decrypt_key(model_entity.encrypted_api_key),
        temperature=model_entity.temperature,
        context_window=model_entity.context_window,
        max_tokens=4000,
        is_chat_model=True,
        is_function_calling_model=True,
        additional_kwargs={"extra_body":{"chat_template_kwargs":{"enable_thinking": model_entity.enable_thinking}}},
    )

@with_async_db_session
async def update_experiment_run_result(
    session: AsyncSession,
    exp_run_id: str,
    actual_output: str,
    status: str,
    score: float = 0.0,
    reason: str = "",
    execution_metadata: List[dict] = []
):
    exp_run_entity = await session.get(ExperimentSampleEntity, exp_run_id)
    if exp_run_entity.status == "pending" and status == "running":
        exp_run_entity.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    exp_run_entity.actual_output = actual_output
    exp_run_entity.status = status
    exp_run_entity.score = score
    exp_run_entity.reason = reason
    if execution_metadata:
        exp_run_entity.execution_metadata = execution_metadata
    exp_run_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    session.add(exp_run_entity)
    await session.commit()
    await session.refresh(exp_run_entity)

@with_async_db_session
async def update_experiment_status(
    session: AsyncSession,
    experiment_id: str,
    status: str,
    avg_score: float = 0.0
):
    exp_entity = await session.get(ExperimentEntity, experiment_id)
    exp_entity.status = status
    exp_entity.avg_score = avg_score
    exp_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add(exp_entity)
    await session.commit()
    await session.refresh(exp_entity)

class PaiEvaluationClient:
    def __init__(self):
        pass

    def load_dataset_from_local_path(self, file_path: str):
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        entry_data = json.loads(line)
                        if "input" in entry_data:  # 只有包含 "input" 的才保留
                            results.append(entry_data)
                        else:
                            print(f"Warning: Line {line_num} missing 'input' field, skipped.")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            raise
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            raise

        return results

    async def load_dataset_from_upload_file(self, file: UploadFile):
        results = []
        try:
            # 异步读取整个文件内容并按行分割（适用于中小文件）
            content = await file.read()
            lines = content.decode('utf-8').splitlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    entry_data = json.loads(line)
                    if "input" in entry_data:  # 只保留包含 "input" 的条目
                        results.append(entry_data)
                    else:
                        print(f"Warning: Line {line_num} missing 'input' field, skipped.")
                except json.JSONDecodeError as e:
                    print(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")
        except Exception as e:
            print(f"Error reading uploaded file: {e}")
            raise

        return results

    async def create_evaluation_task(self, dataset_id: str, experiment_id: str, exp_run_ids: List[str]):
        logger.info(f"[WORKER] creating evaluation dataset for dataset_id {dataset_id} in background.")
        await update_experiment_status(
            experiment_id=experiment_id,
            status="running"
        )
        run_scores = []
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id)
        run_config_entity: RunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id)
        evaluator_config: EvaluatorConfigEntity = await get_evaluator_config_entity(evaluator_config_id=experiment_entity.evaluator_config_id)
        logger.info(f"[WORKER]run_config_entity: {run_config_entity} \n evaluator_config: {evaluator_config}")
        eval_llm = None
        if evaluator_config.type == "LLMJudge":
            eval_llm = await get_llm_model(model_id=evaluator_config.model_id)
        for exp_run_id in exp_run_ids:
            exp_run_entity: ExperimentSampleEntity = await get_exp_run_entity(exp_run_id=exp_run_id)
            dataset_sample_entity: DatasetSampleEntity = await get_dataset_sample_entity(sample_id=exp_run_entity.sample_id)

            logger.info(f"[WORKER] get exp_run_entity {exp_run_entity} and dataset_sample_entity {dataset_sample_entity}.")
            logger.info("[WORKER] processing evaluation task...")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output="",
                status="running",
                score=0.0
            )
            input_messages = [
                {"role": "user", "content": dataset_sample_entity.input}
            ]
            chat_request = ChatAgentRequest(
                model=run_config_entity.model_id,
                messages=input_messages,
                stream=True,
                mcp_ids=run_config_entity.mcp_ids,
                enable_search=run_config_entity.enable_search,
                enable_agent=run_config_entity.enable_agent,
                kb_ids=run_config_entity.kb_ids,
                enable_input_guardrail=run_config_entity.enable_input_guardrail,
                enable_output_guardrail=run_config_entity.enable_output_guardrail,
                guardrail_hint=run_config_entity.guardrail_hint,
            )
            try:
                output, execution_metadata, status = await run_agent(chat_request)
                logger.info(f"=== Agent output: {output} ===")
                eval_res = await run_evaluator(dataset_sample_entity.input, output, dataset_sample_entity.expected_output, evaluator_config.model_dump(), eval_llm)
                logger.info(f"=== Evaluation output: {eval_res} === evaluator_config: {evaluator_config}")
                if status and eval_res:
                    logger.info(f"[WORKER] completed evaluation task for exp_run_id {exp_run_id} in background.")
                    score = eval_res.get("score", 0.0)
                    reason = eval_res.get("reason", "Empty Reason")
                    await update_experiment_run_result(
                        exp_run_id=exp_run_id,
                        actual_output=output,
                        status="success",
                        score=score,
                        reason=reason,
                        execution_metadata=execution_metadata
                    )
                    run_scores.append(score)
                else:
                    logger.error("GAIA agent failed to get valid response.")
                    await update_experiment_run_result(
                        exp_run_id=exp_run_id,
                        actual_output=output,
                        status="failed",
                        score=0.0
                    )
                    run_scores.append(0.0)
            except Exception as e:
                output = f"Error: {e}"
                logger.error(f"[WORKER] evaluation task for exp_run_id {exp_run_id} failed with error: {e}")
                await update_experiment_run_result(
                    exp_run_id=exp_run_id,
                    actual_output=output,
                    status="failed",
                    score=0.0
                )
                run_scores.append(0.0)
                continue

        logger.info(f"[WORKER] completed all evaluation task for experiment_id {experiment_id} in background.")
        await update_experiment_status(
            experiment_id=experiment_id,
            status="success",
            avg_score=sum(run_scores) / len(run_scores) if run_scores else 0
        )

eval_client = PaiEvaluationClient()
