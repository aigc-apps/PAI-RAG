from loguru import logger
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.evaluation.evaluation import EvalEntity
from db.models.evaluation.dataset import EvalDatasetEntity
from db.models.evaluation.experiment import ExperimentRunResultEntity, ExperimentEntity
from db.models.evaluation.run_config import EvalRunConfigEntity
from db.models.llm import LlmModelEntity
from typing import List
from datetime import datetime, timezone
from common.chat.models import ChatAgentRequest
from evaluation.run import run_agent, run_evaluator
from chat.openai.openai_like import OpenAILike
from sqlmodel import select
from db.encrypt_utils import decrypt_key

@with_async_db_session
async def get_exp_run_entity(
    session: AsyncSession,
    exp_run_id: str,
) -> ExperimentRunResultEntity:
    exp_run_entity = await session.get(ExperimentRunResultEntity, exp_run_id)
    assert exp_run_entity is not None, f"Evaluation experiment run entity {exp_run_id} not found."
    return exp_run_entity

@with_async_db_session
async def get_evaluation_entity(
    session: AsyncSession,
    eval_id: str,
) -> EvalEntity:
    eval_entity = await session.get(EvalEntity, eval_id)
    assert eval_entity is not None, f"Evaluation {eval_id} not found."
    return eval_entity

@with_async_db_session
async def get_dataset_entity(
    session: AsyncSession,
    dataset_id: str,
) -> EvalDatasetEntity:
    dataset_entity = await session.get(EvalDatasetEntity, dataset_id)
    assert dataset_entity is not None, f"Evaluation dataset {dataset_id} not found."
    return dataset_entity

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
) -> EvalRunConfigEntity:
    run_config_entity = await session.get(EvalRunConfigEntity, run_config_id)
    assert run_config_entity is not None, f"EvalRunConfig {run_config_id} not found."
    return run_config_entity

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
    exp_run_entity = await session.get(ExperimentRunResultEntity, exp_run_id)
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

    async def create_evaluation_task(self, eval_id: str, experiment_id: str, exp_run_ids: List[str]):
        logger.info(f"[WORKER] creating evaluation task for eval_id {eval_id} in background.")
        eval_entity: EvalEntity = await get_evaluation_entity(eval_id=eval_id)
        logger.info(f"[WORKER] get eval_entity {eval_entity}.")
        await update_experiment_status(
            experiment_id=experiment_id,
            status="running"
        )
        run_scores = []
        experiment_entity: ExperimentEntity = await get_experiment_entity(experiment_id=experiment_id)
        run_config_entity: EvalRunConfigEntity = await get_run_config_entity(run_config_id=experiment_entity.run_config_id)
        print("run_config_entity.evaluator_config", run_config_entity.evaluator_config)
        eval_llm = None
        if run_config_entity.evaluator_config.get("name") == "LLMJudge":
            eval_llm = await get_llm_model(model_id=run_config_entity.evaluator_config.get("model_id"))
        for exp_run_id in exp_run_ids:
            exp_run_entity: ExperimentRunResultEntity = await get_exp_run_entity(exp_run_id=exp_run_id)
            dataset_entity: EvalDatasetEntity = await get_dataset_entity(dataset_id=exp_run_entity.dataset_id)

            logger.info(f"[WORKER] get exp_run_entity {exp_run_entity} and dataset_entity {dataset_entity}.")
            logger.info("[WORKER] processing evaluation task...")
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output="",
                status="running",
                score=0.0
            )
            input_messages = [
                {"role": "user", "content": dataset_entity.input}
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
                print(f"=== Agent output: {output} ===")
                eval_res = await run_evaluator(dataset_entity.input, output, dataset_entity.expected_output, run_config_entity.evaluator_config, eval_llm)
                print(f"=== Evaluation output: {eval_res} === evaluator_config: {run_config_entity.evaluator_config}")
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
