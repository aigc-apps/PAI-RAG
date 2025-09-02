from loguru import logger
from db.db_context import with_async_db_session
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.evaluation.evaluation import EvalEntity
from db.models.evaluation.dataset import EvalDatasetEntity
from db.models.evaluation.experiment import ExperimentRunResultEntity, ExperimentEntity
from typing import List
import random


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
async def update_experiment_run_result(
    session: AsyncSession,
    exp_run_id: str,
    actual_output: str,
    status: str,
    score: float = 0.0
):
    exp_run_entity = await session.get(ExperimentRunResultEntity, exp_run_id)
    exp_run_entity.actual_output = actual_output
    exp_run_entity.status = status
    exp_run_entity.score = score
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
        for exp_run_id in exp_run_ids:
            exp_run_entity: ExperimentRunResultEntity = await get_exp_run_entity(exp_run_id=exp_run_id)
            dataset_entity: EvalEntity = await get_dataset_entity(dataset_id=exp_run_entity.dataset_id)

            logger.info(f"[WORKER] get exp_run_entity {exp_run_entity} and dataset_entity {dataset_entity}.")
            logger.info("[WORKER] mock processing evaluation task...")

            import asyncio
            await asyncio.sleep(20)
            logger.info(f"[WORKER] completed evaluation task for exp_run_id {exp_run_id} in background.")
            random_score = random.uniform(0, 1)
            await update_experiment_run_result(
                exp_run_id=exp_run_id,
                actual_output="This is a mocked actual output.",
                status="success",
                score=random_score
            )
            run_scores.append(random_score)
        logger.info(f"[WORKER] completed all evaluation task for experiment_id {experiment_id} in background.")
        await update_experiment_status(
            experiment_id=experiment_id,
            status="success",
            avg_score=sum(run_scores) / len(run_scores) if run_scores else 0
        )

eval_client = PaiEvaluationClient()
