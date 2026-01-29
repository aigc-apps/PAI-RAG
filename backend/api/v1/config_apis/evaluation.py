### Evaluation configuration API ###
import traceback
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
from api.api_exception import ApiException
from loguru import logger


evaluation_router = APIRouter()

@evaluation_router.post("", response_model=ResponseModel[DatasetEntity])
async def create_dataset(
    dataset_data: DatasetCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    try:
        dataset_entity = await evaluation_service.create_dataset(dataset_data=dataset_data, tenant_id=tenant_id)
        await session.refresh(dataset_entity)
        return success_response(data=dataset_entity, message="数据集创建成功。")
    except ValueError as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to create dataset: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"数据集创建失败: '{e}'.")


@evaluation_router.get("")
async def list_datasets(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Listing datasets with tenant_id: {tenant_id}, page: {page}, size: {size}.")
    try:
        datasets = await evaluation_service.list_datasets(tenant_id=tenant_id, page=page, size=size)
        return success_response(data=datasets, message="获取数据集列表成功。")
    except Exception as e:
        logger.error(f"Failed to list datasets: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取数据集列表失败: '{e}'.")


@evaluation_router.get("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
async def read_dataset(
    dataset_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Reading dataset: {dataset_id} with tenant_id: {tenant_id}.")
    try:
        dataset_entity = await evaluation_service.get_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
        if not dataset_entity:
            raise ApiException(code=404, message=f"数据集 '{dataset_id}' 不存在。")
        return success_response(data=dataset_entity, message="获取数据集成功。")
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to read dataset: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取数据集失败: '{e}'.")


@evaluation_router.put("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
async def update_dataset(
    dataset_id: str,
    update_data: DatasetCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Updating dataset: {dataset_id} with tenant_id: {tenant_id}.")
    try:
        dataset_entity = await evaluation_service.update_dataset(dataset_id=dataset_id, update_data=update_data, tenant_id=tenant_id)
        await session.refresh(dataset_entity)
        return success_response(data=dataset_entity, message="更新数据集成功。")
    except ValueError as e:
        logger.error(f"Failed to update dataset: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update dataset: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"更新数据集失败: '{e}'.")


@evaluation_router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Deleting dataset: {dataset_id} with tenant_id: {tenant_id}.")
    try:
        await evaluation_service.delete_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
        await session.commit()
        return success_response(message="数据集删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete dataset: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"数据集删除失败: '{e}'.")


@evaluation_router.post("/{dataset_id}/upload")
async def upload_dataset_samples(
    dataset_id: str,
    file: UploadFile = File(...),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"正在上传数据样本到数据集 {dataset_id}.")
    if not file:
        raise ApiException(code=400, message="没有上传任何文件。")

    try:
        # Validate dataset exists
        dataset = await evaluation_service.get_dataset(dataset_id=dataset_id, tenant_id=tenant_id)
        if not dataset:
            raise ApiException(code=404, message=f"没有找到数据集 {dataset_id}。")

        # Load data from file
        file_results = await eval_client.load_dataset_from_upload_file(file=file)

        # Batch create dataset samples
        dataset_entities = await evaluation_service.batch_create_dataset_samples(
            dataset_id=dataset_id, samples=file_results, tenant_id=tenant_id
        )

        return success_response(data=dataset_entities, message="文件上传成功")
    except ValueError as e:
        logger.error(f"Failed to upload eval dataset: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to upload eval dataset: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=400, message=f"评估数据集添加失败: {e}")

@evaluation_router.get("/{dataset_id}/samples")
async def list_dataset_samples(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    try:
        samples = await evaluation_service.list_dataset_samples(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
        return success_response(data=samples, message="获取评估数据集列表成功")
    except Exception as e:
        logger.error(f"Failed to list dataset samples: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估数据集列表失败: '{e}'.")


@evaluation_router.put(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
async def update_dataset_sample(
    dataset_id: str,
    sample_id: str,
    new_sample: DatasetSampleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Updating dataset sample: {sample_id}.")
    try:
        dataset_sample_entity = await evaluation_service.update_dataset_sample(
            sample_id=sample_id,
            tenant_id=tenant_id,
            input=new_sample.input,
            expected_output=new_sample.expected_output,
            eval_metadata=new_sample.eval_metadata,
        )
        await session.commit()
        await session.refresh(dataset_sample_entity)
        return success_response(data=dataset_sample_entity, message="更新数据集样本成功。")
    except ValueError as e:
        logger.error(f"Failed to update dataset sample: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to update dataset sample: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"更新数据集样本失败: '{e}'.")

@evaluation_router.get(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
async def get_dataset_sample(
    dataset_id: str,
    sample_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Getting dataset sample: {sample_id}.")
    try:
        dataset_sample_entity = await evaluation_service.get_dataset_sample(sample_id=sample_id, tenant_id=tenant_id)
        if not dataset_sample_entity:
            raise ApiException(code=404, message=f"数据样本 '{sample_id}' 不存在。")
        return success_response(data=dataset_sample_entity, message="获取数据集样本信息成功。")
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset sample: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取数据集样本信息失败: '{e}'.")

@evaluation_router.delete("/{dataset_id}/samples/{sample_id}")
async def delete_dataset_sample(
    dataset_id: str,
    sample_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Deleting dataset sample: {sample_id}.")
    try:
        await evaluation_service.delete_dataset_sample(sample_id=sample_id, tenant_id=tenant_id)
        await session.commit()
        return success_response(message="数据样本删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete dataset sample: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete dataset sample: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"数据样本删除失败: '{e}'.")

@evaluation_router.post("/{dataset_id}/experiments")
async def create_experiment(
    dataset_id: str,
    experiment_create: ExperimentCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Create experiment for dataset {dataset_id}.")
    if not experiment_create.sample_ids or len(experiment_create.sample_ids) == 0:
        raise ApiException(code=400, message="没有选择任何数据集。")

    try:
        import app.worker as background_worker

        experiment_entity, exp_sample_ids = await evaluation_service.create_experiment(
            dataset_id=dataset_id, experiment_data=experiment_create, tenant_id=tenant_id
        )
        await session.commit()
        await session.refresh(experiment_entity)

        logger.info(f"创建实验 {experiment_entity.id} 成功.")
        background_worker.execute_evaluation_task.delay(
            dataset_id=dataset_id, experiment_id=experiment_entity.id, exp_run_ids=exp_sample_ids, tenant_id=tenant_id
        )

        return success_response(data=experiment_entity, message="创建实验成功")
    except ValueError as e:
        logger.error(f"创建实验失败: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"创建实验失败: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"创建实验失败: '{e}'.")


@evaluation_router.get("/{dataset_id}/experiments")
async def get_experiments(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiments for {dataset_id}.")
    try:
        experiments = await evaluation_service.list_experiments(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
        return success_response(data=experiments, message="获取评估实验列表成功。")
    except Exception as e:
        logger.error(f"Failed to get experiments: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估实验列表失败: '{e}'.")


@evaluation_router.get("/{dataset_id}/experiments/{experiment_id}")
async def get_experiment(
    dataset_id: str,
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiment for dataset_id {dataset_id} and experiment_id {experiment_id}.")
    try:
        experiment_entity = await evaluation_service.get_experiment(experiment_id=experiment_id, tenant_id=tenant_id)
        if not experiment_entity:
            raise ApiException(code=404, message=f"实验 '{experiment_id}' 不存在。")
        return success_response(data=experiment_entity, message="获取评估实验详情成功。")
    except ApiException:
        raise
    except Exception as e:
        logger.error(f"Failed to get experiment: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估实验详情失败: '{e}'.")

@evaluation_router.get("/{dataset_id}/experiments/{experiment_id}/samples")
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
    try:
        experiment_samples = await evaluation_service.get_experiment_samples(experiment_id=experiment_id, tenant_id=tenant_id, page=page, size=size, status=status)
        return success_response(data=experiment_samples, message="获取评估实验样本列表成功。")
    except Exception as e:
        logger.error(f"Failed to get experiment samples: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估实验样本列表失败: '{e}'.")

@evaluation_router.put("/{dataset_id}/experiments/{experiment_id}/samples")
async def evaluate_experiment_sample(
    dataset_id: str,
    experiment_id: str,
    experiment_sample_entity: ExperimentSampleEntity,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"evaluate_single_sample for dataset_id: {dataset_id}, experiment_id: {experiment_id}, exp_run_id: {experiment_sample_entity.id}")
    try:
        import app.worker as background_worker

        logger.info(f"重新评估 {experiment_sample_entity.id} 成功.")
        background_worker.execute_evaluation_task.delay(
            dataset_id=dataset_id, experiment_id=experiment_id, exp_run_ids=[experiment_sample_entity.id], tenant_id=tenant_id
        )
        return success_response(message="重新开始评估该样本成功。")
    except ValueError as e:
        logger.error(f"Failed to evaluate experiment sample: {str(e)}")
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to evaluate experiment sample: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"重新开始评估该样本失败: '{e}'.")

@evaluation_router.delete("/{dataset_id}/experiments/{experiment_id}")
async def delete_experiment(
    dataset_id: str,
    experiment_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for dataset_id {dataset_id} and experiment_id {experiment_id}.")
    try:
        await evaluation_service.delete_experiment(experiment_id=experiment_id, tenant_id=tenant_id)
        await session.commit()
        return success_response(message="删除实验成功。")
    except ValueError as e:
        logger.error(f"Failed to delete experiment: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete experiment: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"删除实验失败: '{e}'.")


@evaluation_router.post("/{dataset_id}/runconfigs")
async def create_run_config(
    dataset_id: str,
    config_data: RunConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info("Create run_config.")
    try:
        run_config_entity = await evaluation_service.create_run_config(dataset_id=dataset_id, config_data=config_data, tenant_id=tenant_id)
        await session.refresh(run_config_entity)

        logger.info(f"创建实验设置 {run_config_entity.id} 成功.")
        return success_response(data=run_config_entity, message="创建实验设置成功")
    except ValueError as e:
        logger.error(f"创建实验设置失败: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"创建实验设置失败: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=400, message=f"创建实验设置失败: {e}")


@evaluation_router.put(
    "/{dataset_id}/runconfigs/{config_id}", response_model=ResponseModel[RunConfigEntity]
)
async def update_run_config(
    dataset_id: str,
    config_id: str,
    update_data: RunConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    try:
        run_config = await evaluation_service.update_run_config(config_id=config_id, update_data=update_data, tenant_id=tenant_id)
        await session.refresh(run_config)

        return success_response(data=run_config, message="更新运行配置成功。")
    except ValueError as e:
        logger.error(f"Failed to update run config {config_id}: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(
            f"Failed to update run config {config_id}: {traceback.format_exc()}"
        )
        await session.rollback()
        raise ApiException(code=500, message=f"更新运行配置失败：{str(e)}")


@evaluation_router.get("/{dataset_id}/runconfigs")
async def list_run_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get run_configs for {dataset_id}.")
    try:
        run_configs = await evaluation_service.list_run_configs(dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size)
        return success_response(data=run_configs, message="获取实验设置列表成功")
    except Exception as e:
        logger.error(f"Failed to list run configs: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取实验设置列表失败: '{e}'.")


@evaluation_router.get("/{dataset_id}/runconfigs/{config_id}")
async def get_config_details(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get experiment for config_id {config_id}.")
    try:
        run_config = await evaluation_service.get_run_config(config_id=config_id, tenant_id=tenant_id)
        if not run_config:
            raise ApiException(code=404, message=f"获取运行配置失败: '{config_id}'不存在。")
        return success_response(data=run_config, message="获取实验设置详情成功")
    except Exception as e:
        logger.error(f"Failed to get run config: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取实验设置详情失败: '{e}'.")


@evaluation_router.delete("/{dataset_id}/runconfigs/{config_id}")
async def delete_config(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    try:
        await evaluation_service.delete_run_config(config_id=config_id, tenant_id=tenant_id)
        await session.commit()
        logger.info(f"run_config {config_id} has been deleted.")
        return success_response(message=f"实验设置'{config_id}'删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete run config: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete run config: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"删除实验设置失败: '{e}'.")



@evaluation_router.post("/{dataset_id}/evalconfigs")
async def create_evaluator_config(
    dataset_id: str,
    config_data: EvaluatorConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info("Create eval_config_entity.")
    try:
        eval_config_entity = await evaluation_service.create_evaluator_config(
            dataset_id=dataset_id, config_data=config_data, tenant_id=tenant_id
        )
        await session.refresh(eval_config_entity)
        logger.info(f"创建评估器设置 {eval_config_entity.id} 成功.")
        return success_response(data=eval_config_entity, message="创建评估器设置成功")
    except ValueError as e:
        logger.error(f"创建评估器设置失败: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"创建评估器设置失败: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=400, message=f"创建评估器设置失败: {e}")


@evaluation_router.put(
    "/{dataset_id}/evalconfigs/{config_id}", response_model=ResponseModel[EvaluatorConfigEntity]
)
async def update_evaluator_config(
    dataset_id: str,
    config_id: str,
    update_data: EvaluatorConfigCreate,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    try:
        eval_config = await evaluation_service.update_evaluator_config(
            config_id=config_id, update_data=update_data, tenant_id=tenant_id
        )
        await session.refresh(eval_config)

        return success_response(data=eval_config, message="更新评估器设置成功。")
    except ValueError as e:
        logger.error(f"Failed to update evaluator config {config_id}: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(
            f"Failed to update evaluator config {config_id}: {traceback.format_exc()}"
        )
        await session.rollback()
        raise ApiException(code=500, message=f"更新评估器设置失败：{str(e)}")


@evaluation_router.get("/{dataset_id}/evalconfigs")
async def list_eval_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get eval configs for {dataset_id}.")
    try:
        eval_configs = await evaluation_service.list_evaluator_configs(
            dataset_id=dataset_id, tenant_id=tenant_id, page=page, size=size
        )
        return success_response(data=eval_configs, message="获取评估器设置列表成功")
    except Exception as e:
        logger.error(f"Failed to list evaluator configs: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估器设置列表失败: '{e}'.")


@evaluation_router.get("/{dataset_id}/evalconfigs/{config_id}")
async def get_eval_config_details(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Get evalconfigs for config_id {config_id}.")
    try:
        eval_config = await evaluation_service.get_evaluator_config(config_id=config_id, tenant_id=tenant_id)
        if not eval_config:
            raise ApiException(code=404, message=f"获取评估器设置失败: '{config_id}'不存在。")
        return success_response(data=eval_config, message="获取评估器设置详情成功")
    except Exception as e:
        logger.error(f"Failed to get evaluator config: {traceback.format_exc()}")
        raise ApiException(code=500, message=f"获取评估器设置详情失败: '{e}'.")


@evaluation_router.delete("/{dataset_id}/evalconfigs/{config_id}")
async def delete_eval_config(
    dataset_id: str,
    config_id: str,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_db_session),
    evaluation_service: EvaluationService = Depends(get_evaluation_service),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    try:
        await evaluation_service.delete_evaluator_config(config_id)
        await session.commit()
        logger.info(f"eval_config {config_id} has been deleted.")
        return success_response(message=f"评估器设置'{config_id}'删除成功。")
    except ValueError as e:
        logger.error(f"Failed to delete evaluator config: {str(e)}")
        await session.rollback()
        raise ApiException(code=400, message=str(e))
    except Exception as e:
        logger.error(f"Failed to delete evaluator config: {traceback.format_exc()}")
        await session.rollback()
        raise ApiException(code=500, message=f"删除评估器设置失败: '{e}'.")
