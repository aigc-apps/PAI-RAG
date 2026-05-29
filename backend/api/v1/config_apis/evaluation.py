### Evaluation configuration API ###
from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.evaluation.dataset import DatasetEntity, DatasetCreate, DatasetSampleEntity
from db.models.evaluation.experiment import (
    ExperimentSampleEntity,
    ExperimentCreate,
)
from db.models.evaluation.run_config import RunConfigEntity, RunConfigCreate
from db.db_context import get_db_session
from common.chat.response_model import (
    ResponseModel,
    success_response,
)
from db.models.evaluation.evaluator_config import (
    EvaluatorConfigCreate,
    EvaluatorConfigEntity,
)
from rag.evaluation_tool import eval_client
from service.tool.evaluation_service import EvaluationService
from service.injection import get_evaluation_service, get_tenant_id
from api.api_exception import ApiException, handle_api_exceptions
from loguru import logger


evaluation_router = APIRouter()

@evaluation_router.post("", response_model=ResponseModel[DatasetEntity])
@handle_api_exceptions(action="create dataset")
async def create_dataset(
    dataset_data: DatasetCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    dataset_entity = await evaluation_service.create_dataset(dataset_data=dataset_data, tenant_id=tenant_id)
    await session.refresh(dataset_entity)
    return success_response(data=dataset_entity, message="Created dataset successfully")


@evaluation_router.get("")
@handle_api_exceptions(action="list datasets")
async def list_datasets(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Listing datasets with tenant_id: {tenant_id}, page: {page}, size: {size}.")
    datasets = await evaluation_service.list_datasets(tenant_id=tenant_id, page=page, size=size)
    return success_response(data=datasets, message="Listed datasets successfully")


@evaluation_router.get("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
@handle_api_exceptions(action="get dataset")
async def read_dataset(
    dataset_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Reading dataset: {dataset_id} with tenant_id: {tenant_id}.")
    dataset_entity = await evaluation_service.get_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
    if not dataset_entity:
        raise ApiException(code=404, message=f"Dataset '{dataset_id}' does not exist.")
    return success_response(data=dataset_entity, message="Dataset retrieved successfully")


@evaluation_router.put("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
@handle_api_exceptions(action="update dataset")
async def update_dataset(
    dataset_id: str,
    update_data: DatasetCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Updating dataset: {dataset_id} with tenant_id: {tenant_id}.")
    dataset_entity = await evaluation_service.update_dataset(dataset_id=dataset_id, update_data=update_data, tenant_id=tenant_id)
    await session.refresh(dataset_entity)
    return success_response(data=dataset_entity, message="Dataset updated successfully")


@evaluation_router.delete("/{dataset_id}")
@handle_api_exceptions(action="delete dataset")
async def delete_dataset(
    dataset_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Deleting dataset: {dataset_id} with tenant_id: {tenant_id}.")
    await evaluation_service.delete_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(message="Dataset deleted successfully")


@evaluation_router.post("/{dataset_id}/upload")
@handle_api_exceptions(action="upload dataset samples", default_code=400)
async def upload_dataset_samples(
    dataset_id: str,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Uploading dataset samples to dataset {dataset_id}.")
    if not file:
        raise ApiException(code=400, message="No file uploaded.")

    # Validate dataset exists
    dataset = await evaluation_service.get_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
    if not dataset:
        raise ApiException(code=404, message=f"Dataset '{dataset_id}' does not exist.")

    # Load data from file
    file_results = await eval_client.load_dataset_from_upload_file(file=file)

    # Batch create dataset samples
    dataset_entities = await evaluation_service.batch_create_dataset_samples(
        dataset_id=dataset_id, samples=file_results, tenant_id=tenant_id
    )

    return success_response(data=dataset_entities, message="File uploaded successfully")

@evaluation_router.get("/{dataset_id}/samples")
@handle_api_exceptions(action="list dataset samples")
async def list_dataset_samples(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    samples = await evaluation_service.list_dataset_samples(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
    return success_response(data=samples, message="Listed dataset samples successfully")


@evaluation_router.put(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
@handle_api_exceptions(action="update dataset sample")
async def update_dataset_sample(
    dataset_id: str,
    sample_id: str,
    new_sample: DatasetSampleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Updating dataset sample: {sample_id}.")
    dataset_sample_entity = await evaluation_service.update_dataset_sample(
        sample_id=sample_id,
        tenant_id=tenant_id,
        input=new_sample.input,
        expected_output=new_sample.expected_output,
        eval_metadata=new_sample.eval_metadata,
    )
    await session.commit()
    await session.refresh(dataset_sample_entity)
    return success_response(data=dataset_sample_entity, message="Dataset sample updated successfully")

@evaluation_router.get(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
@handle_api_exceptions(action="get dataset sample")
async def get_dataset_sample(
    dataset_id: str,
    sample_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Getting dataset sample: {sample_id}.")
    dataset_sample_entity = await evaluation_service.get_dataset_sample(sample_id=sample_id, tenant_id=tenant_id)
    if not dataset_sample_entity:
        raise ApiException(code=404, message=f"Dataset sample '{sample_id}' does not exist.")
    return success_response(data=dataset_sample_entity, message="Dataset sample retrieved successfully")

@evaluation_router.delete("/{dataset_id}/samples/{sample_id}")
@handle_api_exceptions(action="delete dataset sample")
async def delete_dataset_sample(
    dataset_id: str,
    sample_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Deleting dataset sample: {sample_id}.")
    await evaluation_service.delete_dataset_sample(sample_id=sample_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(message="Dataset sample deleted successfully")

@evaluation_router.post("/{dataset_id}/experiments")
@handle_api_exceptions(action="create experiment")
async def create_experiment(
    dataset_id: str,
    experiment_create: ExperimentCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Create experiment for dataset {dataset_id}.")
    if not experiment_create.sample_ids or len(experiment_create.sample_ids) == 0:
        raise ApiException(code=400, message="No sample IDs provided.")

    import app.worker as background_worker

    experiment_entity, exp_sample_ids = await evaluation_service.create_experiment(
        dataset_id=dataset_id, experiment_data=experiment_create, tenant_id=tenant_id
    )
    await session.commit()
    await session.refresh(experiment_entity)

    logger.info(f"Experiment {experiment_entity.id} created successfully.")
    background_worker.execute_evaluation_task.delay(
        dataset_id=dataset_id, experiment_id=experiment_entity.id, exp_run_ids=exp_sample_ids, tenant_id=tenant_id
    )

    return success_response(data=experiment_entity, message="Experiment created successfully")


@evaluation_router.get("/{dataset_id}/experiments")
@handle_api_exceptions(action="list experiments")
async def get_experiments(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiments for {dataset_id}.")
    experiments = await evaluation_service.list_experiments(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
    return success_response(data=experiments, message="Experiments listed successfully")


@evaluation_router.get("/{dataset_id}/experiments/{experiment_id}")
@handle_api_exceptions(action="get experiment")
async def get_experiment(
    dataset_id: str,
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiment for dataset_id {dataset_id} and experiment_id {experiment_id}.")
    experiment_entity = await evaluation_service.get_experiment(experiment_id=experiment_id, tenant_id=tenant_id)
    if not experiment_entity:
        raise ApiException(code=404, message=f"Experiment '{experiment_id}' does not exist.")
    return success_response(data=experiment_entity, message="Experiment retrieved successfully")

@evaluation_router.get("/{dataset_id}/experiments/{experiment_id}/samples")
@handle_api_exceptions(action="list experiment samples")
async def get_experiment_samples(
    dataset_id: str,
    experiment_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    status: str = Query(default=None, description="Filter by status: running, success, failed, pending"),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiment details for dataset_id {dataset_id}, experiment_id {experiment_id}, status={status}.")
    experiment_samples = await evaluation_service.get_experiment_samples(experiment_id=experiment_id, tenant_id=tenant_id, page=page, size=size, status=status)
    return success_response(data=experiment_samples, message="Experiment samples listed successfully")

@evaluation_router.put("/{dataset_id}/experiments/{experiment_id}/samples")
@handle_api_exceptions(action="evaluate experiment sample")
async def evaluate_experiment_sample(
    dataset_id: str,
    experiment_id: str,
    experiment_sample_entity: ExperimentSampleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"evaluate_single_sample for dataset_id: {dataset_id}, experiment_id: {experiment_id}, exp_run_id: {experiment_sample_entity.id}")
    import app.worker as background_worker

    logger.info(f"Re-evaluating sample {experiment_sample_entity.id} successful.")
    background_worker.execute_evaluation_task.delay(
        dataset_id=dataset_id, experiment_id=experiment_id, exp_run_ids=[experiment_sample_entity.id], tenant_id=tenant_id
    )
    return success_response(message="Re-evaluation of the sample started successfully.")

@evaluation_router.delete("/{dataset_id}/experiments/{experiment_id}")
@handle_api_exceptions(action="delete experiment")
async def delete_experiment(
    dataset_id: str,
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for dataset_id {dataset_id} and experiment_id {experiment_id}.")
    await evaluation_service.delete_experiment(experiment_id=experiment_id, tenant_id=tenant_id)
    await session.commit()
    return success_response(message="Experiment deleted successfully")


@evaluation_router.post("/{dataset_id}/runconfigs")
@handle_api_exceptions(action="create run config", default_code=400)
async def create_run_config(
    dataset_id: str,
    config_data: RunConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info("Create run_config.")
    run_config_entity = await evaluation_service.create_run_config(dataset_id=dataset_id, config_data=config_data, tenant_id=tenant_id)
    await session.refresh(run_config_entity)

    logger.info(f"Run config {run_config_entity.id} created successfully.")
    return success_response(data=run_config_entity, message="Run config created successfully")


@evaluation_router.put(
    "/{dataset_id}/runconfigs/{config_id}", response_model=ResponseModel[RunConfigEntity]
)
@handle_api_exceptions(action="update run config")
async def update_run_config(
    dataset_id: str,
    config_id: str,
    update_data: RunConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    run_config = await evaluation_service.update_run_config(config_id=config_id, update_data=update_data, tenant_id=tenant_id)
    await session.refresh(run_config)

    return success_response(data=run_config, message="Run config updated successfully")


@evaluation_router.get("/{dataset_id}/runconfigs")
@handle_api_exceptions(action="list run configs")
async def list_run_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get run_configs for {dataset_id}.")
    run_configs = await evaluation_service.list_run_configs(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
    return success_response(data=run_configs, message="Run configs listed successfully")


@evaluation_router.get("/{dataset_id}/runconfigs/{config_id}")
@handle_api_exceptions(action="get run config")
async def get_config_details(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiment for config_id {config_id}.")
    run_config = await evaluation_service.get_run_config(config_id=config_id, tenant_id=tenant_id)
    if not run_config:
        raise ApiException(code=404, message=f"Run config '{config_id}' does not exist.")
    return success_response(data=run_config, message="Run config retrieved successfully")


@evaluation_router.delete("/{dataset_id}/runconfigs/{config_id}")
@handle_api_exceptions(action="delete run config")
async def delete_config(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    await evaluation_service.delete_run_config(config_id=config_id, tenant_id=tenant_id)
    await session.commit()
    logger.info(f"run_config {config_id} has been deleted.")
    return success_response(message=f"Run config '{config_id}' deleted successfully.")



@evaluation_router.post("/{dataset_id}/evalconfigs")
@handle_api_exceptions(action="create eval config", default_code=400)
async def create_evaluator_config(
    dataset_id: str,
    config_data: EvaluatorConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info("Create eval_config_entity.")
    eval_config_entity = await evaluation_service.create_evaluator_config(
        dataset_id=dataset_id, config_data=config_data, tenant_id=tenant_id
    )
    await session.refresh(eval_config_entity)
    logger.info(f"Eval config {eval_config_entity.id} created successfully.")
    return success_response(data=eval_config_entity, message="Eval config created successfully")


@evaluation_router.put(
    "/{dataset_id}/evalconfigs/{config_id}", response_model=ResponseModel[EvaluatorConfigEntity]
)
@handle_api_exceptions(action="update eval config")
async def update_evaluator_config(
    dataset_id: str,
    config_id: str,
    update_data: EvaluatorConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    eval_config = await evaluation_service.update_evaluator_config(
        config_id=config_id, update_data=update_data, tenant_id=tenant_id
    )
    await session.refresh(eval_config)

    return success_response(data=eval_config, message="Eval config updated successfully")


@evaluation_router.get("/{dataset_id}/evalconfigs")
@handle_api_exceptions(action="list eval configs")
async def list_eval_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get eval configs for {dataset_id}.")
    eval_configs = await evaluation_service.list_evaluator_configs(
        dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size
    )
    return success_response(data=eval_configs, message="Eval configs listed successfully")


@evaluation_router.get("/{dataset_id}/evalconfigs/{config_id}")
@handle_api_exceptions(action="get eval config")
async def get_eval_config_details(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get evalconfigs for config_id {config_id}.")
    eval_config = await evaluation_service.get_evaluator_config(config_id=config_id, tenant_id=tenant_id)
    if not eval_config:
        raise ApiException(code=404, message=f"Eval config '{config_id}' does not exist.")
    return success_response(data=eval_config, message="Eval config retrieved successfully")


@evaluation_router.delete("/{dataset_id}/evalconfigs/{config_id}")
@handle_api_exceptions(action="delete eval config")
async def delete_eval_config(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    await evaluation_service.delete_evaluator_config(config_id)
    await session.commit()
    logger.info(f"eval_config {config_id} has been deleted.")
    return success_response(message=f"Eval config '{config_id}' deleted successfully.")
