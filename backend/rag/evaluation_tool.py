import asyncio
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
from sqlmodel import select, update, func, case
from common.encrypt_utils import decrypt_key
from fastapi import UploadFile
import json
from utils.attachment_utils import AttachmentFile, upload_gaia_attachment_file


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
    trace_id: str = "",
    reason: str = "",
    entity_status: str = "",
    execution_metadata: List[dict] = []
):
    exp_run_entity = await session.get(ExperimentSampleEntity, exp_run_id)
    if (exp_run_entity.status == "pending" or entity_status == "running") and status == "running":
        exp_run_entity.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    exp_run_entity.actual_output = actual_output
    exp_run_entity.status = status
    exp_run_entity.score = score
    exp_run_entity.reason = reason
    exp_run_entity.trace_id = trace_id
    if execution_metadata:
        exp_run_entity.execution_metadata = execution_metadata
    exp_run_entity.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    session.add(exp_run_entity)
    await session.commit()
    await session.refresh(exp_run_entity)


@with_async_db_session
async def is_evaluation_completed(
    session: AsyncSession,
    experiment_id: str,
) -> bool:
    """
    检查 experiment_id 对应的所有样本是否都已完成
    """
    experiment = await session.get(ExperimentEntity, experiment_id)
    if not experiment:
        # experiment not exist, both true/false sounds correct.
        # However, to prevent the loop from hanging, return True if the experiment not exist.
        return True

    statement = (
        select(func.count(ExperimentSampleEntity.id))
        .where(
            ExperimentSampleEntity.experiment_id == experiment_id,
            ExperimentSampleEntity.status == "running"
        )
    )
    result = await session.execute(statement)
    running_count = result.scalar()
    return running_count == 0

@with_async_db_session
async def update_experiment(
    session: AsyncSession,
    experiment_id: str,
    status: str,
):
    """ Update average score & status"""
    statement = (
        select(
            func.count(ExperimentSampleEntity.id).label("total_count"),
            func.sum(func.coalesce(ExperimentSampleEntity.score, 0.0)).label("total_score")
        )
        .where(ExperimentSampleEntity.experiment_id == experiment_id)
    )
    try:
        result = await session.execute(statement)
        row = result.one()
        if row.total_count > 0:
            avg_score = row.total_score / row.total_count
        else:
            avg_score = 0.0

        # if experiment already finished, don't update updated_at&status
        update_statement = (
            update(ExperimentEntity)
            .where(ExperimentEntity.id == experiment_id)
            .values(avg_score=avg_score,
                    updated_at=case(
                        (ExperimentEntity.status.not_in(["success", "failed"]), datetime.now(timezone.utc).replace(tzinfo=None)),
                        else_=ExperimentEntity.updated_at,
                        ),
                    status=case(
                        (ExperimentEntity.status.not_in(["success", "failed"]), status),
                        else_=ExperimentEntity.status
                        )
                    )
        )
        await session.execute(update_statement)
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Error: update_evaluation_summary exception: {e}")


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
                            logger.warning(f"Warning: Line {line_num} missing 'input' field, skipped.")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")
        except FileNotFoundError:
            logger.error(f"Error: File '{file_path}' not found.")
            raise
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            raise

        return results

    async def load_dataset_from_upload_file(self, file: UploadFile):
        results = []
        # 异步读取整个文件内容并按行分割（适用于中小文件）
        content = await file.read()
        lines = content.decode('utf-8').splitlines()

        if not lines:
            raise ValueError("上传文件为空，请重新上传。")

        valid_lines = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                entry_data = json.loads(line)
                if "input" in entry_data:  # 只保留包含 "input" 的条目
                    results.append(entry_data)
                    valid_lines += 1
                else:
                    logger.warning(f"Warning: Line {line_num} missing 'input' field, skipped.")
            except json.JSONDecodeError as e:
                logger.warning(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")

        if valid_lines == 0:
            raise ValueError("文件解析失败，请检查schema。")
        return results

    async def evaluate_one_sample(self, experiment_id: str, exp_run_id: str, trace_id: str = ""):
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id)
        run_config_entity: RunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id)
        exp_run_entity: ExperimentSampleEntity = await get_exp_run_entity(exp_run_id=exp_run_id)
        dataset_sample_entity: DatasetSampleEntity = await get_dataset_sample_entity(sample_id=exp_run_entity.sample_id)

        logger.info(f"[WORKER] get exp_run_entity {exp_run_entity} and dataset_sample_entity {dataset_sample_entity}.")
        logger.info("[WORKER] processing evaluation task...")
        await update_experiment_run_result(
            exp_run_id=exp_run_id,
            actual_output="",
            trace_id=trace_id,
            status="running",
            entity_status="running",
            score=0.0
        )

        input_messages = [
            {"role": "user", "content": dataset_sample_entity.input}
        ]
        if dataset_sample_entity.eval_metadata is not None and dataset_sample_entity.eval_metadata.get("file_name"):
            try:
                file_entity: AttachmentFile = await upload_gaia_attachment_file(file_name=dataset_sample_entity.eval_metadata.get("file_name"))
                logger.info("[WORKER] get file_entity", file_entity)
                input_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": dataset_sample_entity.input}
                        ],
                        "attachments": [
                            {
                                "id": file_entity.id,
                                "name": file_entity.name,
                                "contentType": file_entity.contentType,
                            }
                        ],
                    }
                ]
            except Exception as ex:
                logger.error(f"Get gaia attachment file failed: {ex}")
                await update_experiment_run_result(
                    exp_run_id=exp_run_id,
                    actual_output="",
                    trace_id=trace_id,
                    status="failed",
                    score=0.0
                )

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
            prompts=run_config_entity.prompts,
        )
        try:
            import app.worker as background_worker
            logger.info(f"=== Agent Run Input {chat_request} ===")
            output, execution_metadata, trace_id, status = await run_agent(chat_request)
            logger.info(f"=== Agent output: {output} ===")
            background_worker.evaluate_sample_result.delay(experiment_id=experiment_id,
                                                           exp_run_id=exp_run_id,
                                                           sample_id=exp_run_entity.sample_id,
                                                           trace_id=trace_id,
                                                           evaluator_config_id=experiment_entity.evaluator_config_id,
                                                           execution_metadata=execution_metadata,
                                                           output=output)
        except Exception as e:
            output = f"Error: {e}"
            logger.error(f"[WORKER] evaluation task for exp_run_id {exp_run_id} failed with error: {e}")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output=output,
                trace_id=trace_id,
                status="failed",
                score=0.0
            )


    async def create_evaluation_task(self, dataset_id: str, experiment_id: str, exp_run_ids: List[str], is_evaluate_single_sample:bool=False):
        # is_evaluate_single_sample=False, create a brand new evaluation task
        # is_evaluate_single_sample=True, meaning evaluate one-single sample of given experiment
        #   e.g., when experiment_id already finished, however some cases failed due to exception, we need re-run the single case.

        logger.info(f"[WORKER] creating evaluation dataset for dataset_id {dataset_id}, experiment_id {experiment_id} in background.")
        if not is_evaluate_single_sample:
            await update_experiment(
                experiment_id=experiment_id,
                status="running"
            )
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id)
        run_config_entity: RunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id)
        evaluator_config: EvaluatorConfigEntity = await get_evaluator_config_entity(evaluator_config_id=experiment_entity.evaluator_config_id)
        logger.info(f"[WORKER]run_config_entity: {run_config_entity} \n evaluator_config: {evaluator_config}")
        for exp_run_id in exp_run_ids:
            await self.evaluate_one_sample(experiment_id=experiment_id, exp_run_id=exp_run_id)


        if not is_evaluate_single_sample:
            while not await is_evaluation_completed(experiment_id=experiment_id):
                await asyncio.sleep(10.0)
            await update_experiment(experiment_id=experiment_id,
                                    status="success")

    async def evaluate_sample_result(self, experiment_id: str, exp_run_id:str, sample_id:str, trace_id:str, evaluator_config_id:str, execution_metadata:str, output:str):
        dataset_sample_entity: DatasetSampleEntity = await get_dataset_sample_entity(sample_id=sample_id)
        evaluator_config: EvaluatorConfigEntity = await get_evaluator_config_entity(evaluator_config_id=evaluator_config_id)
        eval_llm = None
        if evaluator_config.type == "LLMJudge":
            eval_llm = await get_llm_model(model_id=evaluator_config.model_id)

        eval_res = await run_evaluator(dataset_sample_entity.input, output, dataset_sample_entity.expected_output, evaluator_config.model_dump(), eval_llm)
        logger.info(f"=== Evaluation output: {eval_res} === evaluator_config: {evaluator_config}")
        if eval_res:
            logger.info(f"[WORKER] completed evaluation task for exp_run_id {exp_run_id} in background.")
            score = eval_res.get("score", 0.0)
            reason = eval_res.get("reason", "Empty Reason")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output=output,
                status="success",
                score=score,
                trace_id=trace_id,
                reason=reason,
                execution_metadata=execution_metadata
            )
            await update_experiment(
                experiment_id=experiment_id,
                status="running",
            )
        else:
            logger.error("GAIA agent failed to get valid response.")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output=output,
                trace_id=trace_id,
                status="failed",
                score=0.0
            )

eval_client = PaiEvaluationClient()
