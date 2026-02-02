import asyncio
import traceback
from loguru import logger
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.evaluation.dataset import DatasetEntity
from db.models.evaluation.dataset import DatasetSampleEntity
from db.models.evaluation.experiment import ExperimentSampleEntity, ExperimentEntity
from db.models.evaluation.run_config import RunConfigEntity
from db.models.evaluation.evaluator_config import EvaluatorConfigEntity
from typing import List
from datetime import datetime, timezone
from common.chat.models import ChatAgentRequest
from evaluation.run import run_agent, run_evaluator
from sqlmodel import select, update, func, case
from fastapi import UploadFile
from rag.offline_db_helper import get_openailike_llm_from_db
import json
from utils.attachment_utils import AttachmentFile, upload_gaia_attachment_file


@with_async_db_session
async def get_exp_run_entity(
    session: AsyncSession,
    exp_run_id: str,
    tenant_id: str = None,
) -> ExperimentSampleEntity:
    query = select(ExperimentSampleEntity).where(ExperimentSampleEntity.id == exp_run_id, ExperimentSampleEntity.tenant_id == tenant_id)
    exp_run_entity_execution = await session.exec(query)
    exp_run_entity = exp_run_entity_execution.first()
    assert exp_run_entity is not None, f"Evaluation experiment run entity {exp_run_id} not found."
    return exp_run_entity

@with_async_db_session
async def get_dataset_entity(
    session: AsyncSession,
    dataset_id: str,
    tenant_id: str = None,
) -> DatasetEntity:
    query = select(DatasetEntity).where(DatasetEntity.id == dataset_id, DatasetEntity.tenant_id == tenant_id)
    dataset_entity_execution = await session.exec(query)
    dataset_entity = dataset_entity_execution.first()
    assert dataset_entity is not None, f"Evaluation dataset {dataset_id} not found."
    return dataset_entity

@with_async_db_session
async def get_dataset_sample_entity(
    session: AsyncSession,
    sample_id: str,
    tenant_id: str = None,
) -> DatasetSampleEntity:
    query = select(DatasetSampleEntity).where(DatasetSampleEntity.id == sample_id, DatasetSampleEntity.tenant_id == tenant_id)
    dataset_sample_entity_execution = await session.exec(query)
    dataset_sample_entity = dataset_sample_entity_execution.first()
    assert dataset_sample_entity is not None, f"Evaluation dataset sample {sample_id} not found."
    return dataset_sample_entity

@with_async_db_session
async def get_experiment_entity(
    session: AsyncSession,
    experiment_id: str,
    tenant_id: str = None,
) -> ExperimentEntity:
    query = select(ExperimentEntity).where(ExperimentEntity.id == experiment_id, ExperimentEntity.tenant_id == tenant_id)
    exp_entity_execution = await session.exec(query)
    exp_entity = exp_entity_execution.first()
    assert exp_entity is not None, f"Experiment {experiment_id} not found."
    return exp_entity

@with_async_db_session
async def get_run_config_entity(
    session: AsyncSession,
    run_config_id: str,
    tenant_id: str = None,
) -> RunConfigEntity:
    query = select(RunConfigEntity).where(RunConfigEntity.id == run_config_id, RunConfigEntity.tenant_id == tenant_id)
    run_config_entity_execution = await session.exec(query)
    run_config_entity = run_config_entity_execution.first()
    assert run_config_entity is not None, f"RunConfig {run_config_id} not found."
    return run_config_entity

@with_async_db_session
async def get_evaluator_config_entity(
    session: AsyncSession,
    evaluator_config_id: str,
    tenant_id: str = None,
) -> EvaluatorConfigEntity:
    query = select(EvaluatorConfigEntity).where(EvaluatorConfigEntity.id == evaluator_config_id, EvaluatorConfigEntity.tenant_id == tenant_id)
    evaluator_config_entity_execution = await session.exec(query)
    evaluator_config_entity = evaluator_config_entity_execution.first()
    assert evaluator_config_entity is not None, f"EvaluatorConfig {evaluator_config_id} not found."
    return evaluator_config_entity


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
    execution_metadata: List[dict] = [],
    tenant_id: str = None,
):
    query = select(ExperimentSampleEntity).where(ExperimentSampleEntity.id == exp_run_id, ExperimentSampleEntity.tenant_id == tenant_id)
    exp_run_entity_execution = await session.exec(query)
    exp_run_entity = exp_run_entity_execution.first()
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


@with_async_db_session
async def is_evaluation_completed(
    session: AsyncSession,
    experiment_id: str,
    tenant_id: str = None,
) -> bool:
    """
    检查 experiment_id 对应的所有样本是否都已完成
    """
    query = select(ExperimentEntity).where(ExperimentEntity.id == experiment_id, ExperimentEntity.tenant_id == tenant_id)
    experiment_execution = await session.exec(query)
    experiment = experiment_execution.first()
    if not experiment:
        # experiment not exist, both true/false sounds correct.
        # However, to prevent the loop from hanging, return True if the experiment not exist.
        return True

    statement = (
        select(func.count(ExperimentSampleEntity.id))
        .where(
            ExperimentSampleEntity.experiment_id == experiment_id,
            ExperimentSampleEntity.status == "running",
            ExperimentSampleEntity.tenant_id == tenant_id,
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
    tenant_id: str = None,
):
    """ Update average score & status"""
    statement = (
        select(
            func.count(ExperimentSampleEntity.id).label("total_count"),
            func.sum(func.coalesce(ExperimentSampleEntity.score, 0.0)).label("total_score")
        )
        .where(ExperimentSampleEntity.experiment_id == experiment_id, ExperimentSampleEntity.tenant_id == tenant_id)
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
            .where(ExperimentEntity.id == experiment_id, ExperimentEntity.tenant_id == tenant_id)
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
    except Exception as e:
        logger.error(f"Update_evaluation_summary exception: {e}")
        raise


class PaiEvaluationClient:
    def __init__(self):
        pass


    async def load_dataset_from_upload_file(self, file: UploadFile):
        results = []
        # 异步读取整个文件内容并按行分割（适用于中小文件）
        content = await file.read()
        lines = content.decode('utf-8').splitlines()

        if not lines:
            raise ValueError("Uploaded file is empty, please upload again.")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                entry_data = json.loads(line)
                if "input" in entry_data:  # 只保留包含 "input" 的条目
                    results.append(entry_data)
                else:
                    logger.warning(f"Warning: Line {line_num} missing 'input' field, skipped.")
            except json.JSONDecodeError as e:
                logger.warning(f"Warning: Line {line_num} is not valid JSON, skipped. Error: {e}")

        if not results:
            raise ValueError("File parsing failed, please check schema.")
        return results

    async def evaluate_one_sample(self, experiment_id: str, exp_run_id: str, trace_id: str = "", tenant_id: str = None):
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id, tenant_id=tenant_id)
        run_config_entity: RunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id, tenant_id=tenant_id)
        exp_run_entity: ExperimentSampleEntity = await get_exp_run_entity(exp_run_id=exp_run_id, tenant_id=tenant_id)
        dataset_sample_entity: DatasetSampleEntity = await get_dataset_sample_entity(sample_id=exp_run_entity.sample_id, tenant_id=tenant_id)

        logger.info(f"[WORKER] get exp_run_entity {exp_run_entity} and dataset_sample_entity {dataset_sample_entity}.")
        logger.info("[WORKER] processing evaluation task...")
        await update_experiment_run_result(
            exp_run_id=exp_run_id,
            tenant_id=tenant_id,
            actual_output="",
            trace_id=trace_id,
            status="running",
            entity_status="running",
            score=0.0,
        )

        input_messages = [
            {"role": "user", "content": dataset_sample_entity.input}
        ]
        if dataset_sample_entity.eval_metadata and dataset_sample_entity.eval_metadata.get("file_name"):
            try:
                file_entity: AttachmentFile = await upload_gaia_attachment_file(file_name=dataset_sample_entity.eval_metadata.get("file_name"), tenant_id=tenant_id)
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
                error_traceback = traceback.format_exc()
                logger.error(f"Get gaia attachment file failed: {ex}")
                logger.error(f"[WORKER] error traceback:\n{error_traceback}")
                await update_experiment_run_result(
                    exp_run_id=exp_run_id,
                    actual_output=f"Error: Failed to upload attachment file: {ex}",
                    trace_id=trace_id,
                    status="failed",
                    score=0.0,
                    tenant_id=tenant_id
                )
                return  # Exit early if attachment upload fails

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
            logger.info(f"=== Agent Run Input {chat_request} ===")
            output, execution_metadata, trace_id, status = await run_agent(chat_request)
            logger.info(f"=== Agent output: {output} ===")
            await self.evaluate_sample_result(experiment_id=experiment_id,
                                            exp_run_id=exp_run_id,
                                            sample_id=exp_run_entity.sample_id,
                                            trace_id=trace_id,
                                            evaluator_config_id=experiment_entity.evaluator_config_id,
                                            execution_metadata=execution_metadata,
                                            output=output,
                                            tenant_id=tenant_id)
        except Exception as e:
            error_msg = str(e) if e else "Unknown error"
            error_traceback = traceback.format_exc()
            output = f"Error: {error_msg}"
            logger.error(f"[WORKER] evaluation task for exp_run_id {exp_run_id} failed with error: {error_msg}")
            logger.error(f"[WORKER] error traceback:\n{error_traceback}")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output=output,
                trace_id=trace_id,
                status="failed",
                score=0.0,
                tenant_id=tenant_id,
            )


    async def create_evaluation_task(self, dataset_id: str, experiment_id: str, exp_run_ids: List[str], is_evaluate_single_sample:bool=False, tenant_id: str = None):
        # is_evaluate_single_sample=False, create a brand new evaluation task
        # is_evaluate_single_sample=True, meaning evaluate one-single sample of given experiment
        #   e.g., when experiment_id already finished, however some cases failed due to exception, we need re-run the single case.

        logger.info(f"[WORKER] creating evaluation dataset for dataset_id {dataset_id}, experiment_id {experiment_id} tenant_id {tenant_id} in background.")
        if not is_evaluate_single_sample:
            await update_experiment(
                experiment_id=experiment_id,
                status="running",
                tenant_id=tenant_id,
            )
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id, tenant_id=tenant_id)
        run_config_entity: RunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id, tenant_id=tenant_id)
        evaluator_config: EvaluatorConfigEntity = await get_evaluator_config_entity(evaluator_config_id=experiment_entity.evaluator_config_id, tenant_id=tenant_id)
        logger.info(f"[WORKER]run_config_entity: {run_config_entity} \n evaluator_config: {evaluator_config}")
        for exp_run_id in exp_run_ids:
            await self.evaluate_one_sample(experiment_id=experiment_id, exp_run_id=exp_run_id, tenant_id=tenant_id)

        if not is_evaluate_single_sample:
            while not await is_evaluation_completed(experiment_id=experiment_id, tenant_id=tenant_id):
                await asyncio.sleep(10.0)
            await update_experiment(
                experiment_id=experiment_id,
                status="success",
                tenant_id=tenant_id,
            )


    async def evaluate_sample_result(self, experiment_id: str, exp_run_id:str, sample_id:str, trace_id:str, evaluator_config_id:str, execution_metadata:str, output:str, tenant_id: str = None):
        dataset_sample_entity: DatasetSampleEntity = await get_dataset_sample_entity(sample_id=sample_id, tenant_id=tenant_id)
        evaluator_config: EvaluatorConfigEntity = await get_evaluator_config_entity(evaluator_config_id=evaluator_config_id, tenant_id=tenant_id)
        eval_llm = None
        if evaluator_config.type == "LLMJudge":
            eval_llm = await get_openailike_llm_from_db(model_id=evaluator_config.model_id, tenant_id=tenant_id, provider_name=evaluator_config.model_provider_name)

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
                execution_metadata=execution_metadata,
                tenant_id=tenant_id
            )
            await update_experiment(
                experiment_id=experiment_id,
                status="running",
                tenant_id=tenant_id
            )
        else:
            logger.error("GAIA agent failed to get valid response.")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output=output,
                trace_id=trace_id,
                status="failed",
                score=0.0,
                tenant_id=tenant_id
            )

eval_client = PaiEvaluationClient()
