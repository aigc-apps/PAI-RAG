### Evaluation configuration API ###
import traceback
from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.evaluation.dataset import DatasetEntity, DatasetCreate
from db.models.evaluation.dataset import DatasetSampleEntity
from db.models.evaluation.experiment import (
    ExperimentEntity,
    ExperimentSampleEntity,
    ExperimentCreate,
)
from db.models.evaluation.run_config import RunConfigEntity, RunConfigCreate
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from api.response_model import (
    ResponseModel,
    PagedResult,
    success_response,
    error_response,
)
from loguru import logger
from api.v1.utils.paginate import get_pagination_meta
from config.providers.evaluation_provider import evaluation_provider
from db.models.evaluation.evaluator_config import (
    EvaluatorConfigCreate,
    EvaluatorConfigEntity,
)
from rag.evaluation_tool import eval_client


evaluation_router = APIRouter()

@evaluation_router.post("", response_model=ResponseModel[DatasetEntity])
async def create_dataset(
    dataset_create: DatasetCreate, session: AsyncSession = Depends(get_session)
):
    try:
        dataset = DatasetEntity.model_validate(dataset_create)
        evaluation_provider.add(dataset)
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EVALUATION,
            event_type=ChangeEventType.ADD,
            source_id=dataset.id,
        )
        return success_response(data=dataset, message="数据集创建成功。")

    except IntegrityError as e:
        logger.exception(f"创建数据集失败。{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return error_response(
                    code=400, message="创建数据集失败: 数据集名称已存在。"
                )
        else:
            return error_response(code=400, message=f"创建数据集失败: {e}.")
    except Exception:
        logger.exception(f"创建数据集失败。{traceback.format_exc()}")
        await session.rollback()
        return error_response(
                code=400, message=f"创建数据集失败: {traceback.format_exc()}."
            )


@evaluation_router.get("")
async def list_datasets(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    # 子查询 1：统计每个 dataset_id 对应的数据样本数量
    dataset_count_subq = (
        select(
            DatasetSampleEntity.dataset_id,
            func.count(DatasetSampleEntity.id).label("dataset_count"),
        )
        .group_by(DatasetSampleEntity.dataset_id)
        .subquery()
    )

    # 子查询 2：统计每个 dataset_id 对应的实验数量
    experiment_count_subq = (
        select(
            ExperimentEntity.dataset_id,
            func.count(ExperimentEntity.id).label("experiments_count"),
        )
        .group_by(ExperimentEntity.dataset_id)
        .subquery()
    )

    # 主查询：左连接两个子查询
    query = (
        select(
            DatasetEntity,
            func.coalesce(dataset_count_subq.c.dataset_count, 0).label("dataset_count"),
            func.coalesce(experiment_count_subq.c.experiments_count, 0).label(
                "experiments_count"
            ),
        )
        .outerjoin(
            dataset_count_subq, DatasetEntity.id == dataset_count_subq.c.dataset_id
        )
        .outerjoin(
            experiment_count_subq,
            DatasetEntity.id == experiment_count_subq.c.dataset_id,
        )
        .order_by(DatasetEntity.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )

    # 获取总数（不变）
    total_results = await session.exec(
        select(func.count()).select_from(DatasetEntity)
    )
    total_num = total_results.one_or_none()

    # 执行主查询
    results = await session.exec(query)
    eval_entities_with_counts = results.all()

    # 构造返回数据（包含两个统计字段）
    items = []
    for eval_entity, dataset_count, experiments_count in eval_entities_with_counts:
        item = eval_entity.model_dump()
        item["dataset_count"] = dataset_count
        item["experiments_count"] = experiments_count
        items.append(item)

    pagination = get_pagination_meta(page, size, total_num)

    return success_response(
        data=PagedResult(
            items=items,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取评估任务列表成功",
    )


@evaluation_router.get("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
async def read_evaluation(dataset_id: str, session: AsyncSession = Depends(get_session)):
    evaluation = await session.get(DatasetEntity, dataset_id)

    if not evaluation:
        return error_response(
                code=404, message=f"查询数据集失败: 数据集'{dataset_id}'不存在。"
            )

    return success_response(data=evaluation, message="查询数据集成功。")


@evaluation_router.put("/{dataset_id}", response_model=ResponseModel[DatasetEntity])
async def update_dataset(
    dataset_id: str,
    new_dataset: DatasetCreate,
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(DatasetEntity, dataset_id)
    if not dataset:
        return error_response(
                code=404, message=f"更新数据集信息失败: 数据集'{dataset_id}'不存在。"
            )

    try:
        dataset.name = new_dataset.name or dataset.name
        dataset.description = new_dataset.description or dataset.description

        evaluation_provider.update(dataset)
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EVALUATION,
            event_type=ChangeEventType.UPDATE,
            source_id=dataset.id,
        )

        logger.info(f"数据集 {dataset_id} 更新成功 {dataset}.")

        return success_response(data=dataset, message="更新数据集信息成功。")
    except Exception:
        logger.error(f"数据集 {dataset_id} 更新失败: {traceback.format_exc()}")
        return error_response(
                code=404, message=f"更新数据集信息失败：{traceback.format_exc()}"
            )


@evaluation_router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    session: AsyncSession = Depends(get_session),
):
    dataset = await session.get(DatasetEntity, dataset_id)

    if not dataset:
        return error_response(
                code=404, message=f"删除数据集失败: 数据集'{dataset_id}'不存在。"
            )

    evaluation_provider.delete(dataset_id)
    await session.delete(dataset)
    await session.commit()

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EVALUATION,
        event_type=ChangeEventType.DELETE,
        source_id=dataset.id,
    )

    logger.info(f"数据集 {dataset_id} 删除成功.")

    return success_response(message=f"数据集'{dataset_id}'删除成功。")


@evaluation_router.post("/{dataset_id}/upload")
async def upload_dataset_samples(
    dataset_id: str,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"正在上传数据样本到数据集 {dataset_id}.")
    if not file:
        return error_response(code=404, message="没有上传任何文件。")

    try:
        try:
            _ = await evaluation_provider.aget_evaluation(dataset_id)
        except ValueError:
            logger.error(f"没找到数据集{dataset_id}")
            return error_response(
                    code=404, message=f"没有找到数据集 {dataset_id}。"
                )

        file_results = await eval_client.load_dataset_from_upload_file(file=file)
        dataset_entities = []
        for line in file_results:
            dataset_sample_entity = DatasetSampleEntity(
                dataset_id=dataset_id,
                input=line["input"],
                expected_output=line.get("expected_output"),
                eval_metadata=line.get("metadata"),
            )
            session.add(dataset_sample_entity)
            dataset_entities.append(dataset_sample_entity)
            logger.info(f"Saved file {dataset_sample_entity} successfully.")
        await session.commit()
        return success_response(data=dataset_entities, message="文件上传成功")
    except Exception as e:
        logger.error(f"Failed to upload eval dataset: {traceback.format_exc()}")
        await session.rollback()
        return error_response(
                code=404, message=f"Failed to save eval dataset to database: {e}"
            )

@evaluation_router.get("/{dataset_id}/samples")
async def list_dataset_samples(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    total_results = await session.exec(
        select(func.count()).select_from(
            select(DatasetSampleEntity).where(
                DatasetSampleEntity.dataset_id == dataset_id
            )
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    file_results = await session.exec(
        select(DatasetSampleEntity)
        .where(DatasetSampleEntity.dataset_id == dataset_id)
        .order_by(DatasetSampleEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    file_entities = file_results.all()

    return success_response(
        data=PagedResult(
            items=file_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取评估数据集列表成功",
    )


@evaluation_router.put(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
async def update_dataset_sample(
    dataset_id: str,
    sample_id: str,
    new_sample: DatasetSampleEntity,
    session: AsyncSession = Depends(get_session),
):
    dataset_sample_entity = await session.get(DatasetSampleEntity, sample_id)
    if not dataset_sample_entity:
        return error_response(
                code=404, message=f"更新数据集样本失败: 样本'{sample_id}'不存在。"
            )

    try:
        dataset_sample_entity.input = new_sample.input
        dataset_sample_entity.expected_output = new_sample.expected_output
        dataset_sample_entity.eval_metadata = new_sample.eval_metadata

        session.add(dataset_sample_entity)
        await session.commit()
        await session.refresh(dataset_sample_entity)

        logger.info(
            f"Dataset Sample {sample_id} for dataset_id {dataset_id} updated to {dataset_sample_entity}."
        )

        return success_response(data=dataset_sample_entity, message="更新数据集样本成功。")
    except Exception:
        logger.error(
            f"Failed to update Dataset Sample {sample_id} for evaluation {dataset_id}: {traceback.format_exc()}"
        )
        return error_response(
                code=404, message=f"更新数据集样本失败：{traceback.format_exc()}"
            )


@evaluation_router.get(
    "/{dataset_id}/samples/{sample_id}",
    response_model=ResponseModel[DatasetSampleEntity],
)
async def get_dataset_sample(
    dataset_id: str,
    sample_id: str,
    session: AsyncSession = Depends(get_session),
):
    sample_entity = await session.get(DatasetSampleEntity, sample_id)
    if not sample_entity:
        return error_response(
                code=404, message=f"获取数据集 {dataset_id} 样本信息失败: 样本'{sample_id}'不存在。"
            )

    return success_response(data=sample_entity, message="获取数据集样本信息成功。")


@evaluation_router.delete("/{dataset_id}/samples/{sample_id}")
async def delete_dataset_sample(
    dataset_id: str,
    sample_id: str,
    session: AsyncSession = Depends(get_session),
):
    sample_entity = await session.get(DatasetSampleEntity, sample_id)

    if not sample_entity:
        return error_response(
                code=404,
                message=f"删除数据样本失败: 数据集 {dataset_id} 样本 '{sample_id}'不存在。",
            )

    await session.delete(sample_entity)
    await session.commit()

    logger.info(f"Dataset Sample {sample_id} has been deleted.")

    return success_response(message=f"数据样本'{dataset_id}'删除成功。")


@evaluation_router.post("/{dataset_id}/experiments")
async def create_experiment(
    dataset_id: str,
    experiment_create: ExperimentCreate,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Create experiment for dataset {dataset_id}.")
    if len(experiment_create.sample_ids) == 0:
        return error_response(code=400, message="没有选择任何数据集。")

    try:
        import app.worker as background_worker

        dataset_sample_results = await session.exec(
            select(DatasetSampleEntity)
            .where(DatasetSampleEntity.id.in_(experiment_create.sample_ids))
            .where(DatasetSampleEntity.dataset_id == dataset_id)
        )
        dataset_sample_entities = dataset_sample_results.all()
        if len(dataset_sample_entities) == 0:
            return error_response(
                    code=400, message=f"没有找到数据集 {dataset_id} 的数据样本。"
                )
        if len(dataset_sample_entities) != len(experiment_create.sample_ids):
            missing_ids = set(experiment_create.sample_ids) - {
                d.id for d in dataset_sample_entities
            }
            return error_response(
                    code=400, message=f"以下样本ID不存在: {missing_ids}"
                )
        experiment_entity = ExperimentEntity(
            dataset_id=dataset_id,
            name=experiment_create.name,
            samples_count=len(dataset_sample_entities),
            run_config_id=experiment_create.run_config_id,
            evaluator_config_id=experiment_create.evaluator_config_id,
            description=experiment_create.description or "Experiment created via API",
            status="pending",
        )
        session.add(experiment_entity)
        await session.commit()
        logger.info(f"创建实验 {experiment_entity} 成功.")
        exp_sample_ids = []
        for sample_id in experiment_create.sample_ids:
            exp_run_entity = ExperimentSampleEntity(
                experiment_id=experiment_entity.id,
                dataset_id=dataset_id,
                sample_id=sample_id,
                status="pending",
            )
            session.add(exp_run_entity)
            await session.commit()
            exp_sample_ids.append(exp_run_entity.id)
        background_worker.execute_evaluation_task.delay(
            dataset_id, experiment_entity.id, exp_sample_ids
        )

        return success_response(data=experiment_entity, message="创建实验成功")
    except Exception as e:
        logger.error(f"创建实验失败: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"创建实验失败: {e}")


@evaluation_router.get("/{dataset_id}/experiments")
async def get_experiments(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiments for {dataset_id}.")
    total_results = await session.exec(
        select(func.count()).select_from(
            select(ExperimentEntity).where(ExperimentEntity.dataset_id == dataset_id)
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    experiment_results = await session.exec(
        select(ExperimentEntity)
        .where(ExperimentEntity.dataset_id == dataset_id)
        .order_by(ExperimentEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    experiment_entities = experiment_results.all()

    return success_response(
        data=PagedResult(
            items=experiment_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取评估实验列表成功",
    )


@evaluation_router.get("/{dataset_id}/experiments/{exp_id}")
async def get_experiment(
    dataset_id: str,
    exp_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment for dataset_id {dataset_id} and exp_id {exp_id}.")
    experiment_results = await session.exec(
        select(ExperimentEntity)
        .where(ExperimentEntity.dataset_id == dataset_id)
        .where(ExperimentEntity.id == exp_id)
    )
    experiment_entity = experiment_results.all()[0]

    return success_response(data=experiment_entity, message="获取评估实验详情成功")


@evaluation_router.get("/{dataset_id}/experiments/{exp_id}/samples")
async def get_experiment_samples(
    dataset_id: str,
    exp_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment details for dataset_id {dataset_id}, exp_id {exp_id}.")
    total_results = await session.exec(
        select(func.count()).select_from(
            select(ExperimentSampleEntity).where(
                ExperimentSampleEntity.experiment_id == exp_id
            )
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    experiment_results = await session.exec(
        select(
            ExperimentSampleEntity,
            DatasetSampleEntity.input,
            DatasetSampleEntity.expected_output,
            DatasetSampleEntity.eval_metadata.label("dataset_metadata"),
        )
        .join(
            DatasetSampleEntity,
            ExperimentSampleEntity.sample_id == DatasetSampleEntity.id,
        )
        .where(ExperimentSampleEntity.experiment_id == exp_id)
        .order_by(ExperimentSampleEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    transformed_results = []
    for row in experiment_results.all():
        result_entity = row[0]
        result_dict = result_entity.model_dump()
        result_dict["input"] = row[1]
        result_dict["expected_output"] = row[2]
        result_dict["dataset_metadata"] = row[3]
        transformed_results.append(result_dict)

    return success_response(
        data=PagedResult(
            items=transformed_results,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取评估实验执行详情信息成功",
    )


@evaluation_router.delete("/{dataset_id}/experiments/{exp_id}")
async def delete_experiment(
    dataset_id: str,
    exp_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Delete experiment for dataset_id {dataset_id} and exp_id {exp_id}.")
    experiment = await session.get(ExperimentEntity, exp_id)
    if not experiment:
        return error_response(
                code=404, message=f"删除实验失败: 实验'{exp_id}'不存在。"
            )

    await session.delete(experiment)
    await session.commit()

    logger.info(f"Experiment {exp_id} has been deleted.")
    return success_response(message=f"实验'{exp_id}'删除成功。")


@evaluation_router.post("/{dataset_id}/runconfigs")
async def create_run_config(
    dataset_id: str,
    run_config: RunConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    logger.info("Create run_config.")
    try:
        run_config_entity = RunConfigEntity(
            name=run_config.name,
            dataset_id=dataset_id,
            model_id=run_config.model_id,
            mcp_ids=run_config.mcp_ids,
            kb_ids=run_config.kb_ids,
            enable_search=run_config.enable_search,
            enable_vision=run_config.enable_vision,
            enable_agent=run_config.enable_agent,
            enable_input_guardrail=run_config.enable_input_guardrail,
            enable_output_guardrail=run_config.enable_output_guardrail,
            guardrail_hint=run_config.guardrail_hint,
        )

        session.add(run_config_entity)
        await session.commit()
        logger.info(f"创建实验设置 {run_config_entity} 成功.")
        return success_response(data=run_config_entity, message="创建实验设置成功")
    except Exception as e:
        logger.error(f"创建实验设置失败: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"创建实验设置失败: {e}")


@evaluation_router.put(
    "/{dataset_id}/runconfigs/{config_id}", response_model=ResponseModel[RunConfigEntity]
)
async def update_run_config(
    dataset_id: str,
    config_id: str,
    new_run_config: RunConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    run_config = await session.get(RunConfigEntity, config_id)
    if not run_config:
        return error_response(
                code=404, message=f"更新运行配置失败: '{config_id}'不存在。"
            )

    try:
        run_config.name = new_run_config.name
        run_config.model_id = new_run_config.model_id
        run_config.mcp_ids = new_run_config.mcp_ids
        run_config.kb_ids = new_run_config.kb_ids
        run_config.enable_search = new_run_config.enable_search
        run_config.enable_vision = new_run_config.enable_vision
        run_config.enable_agent = new_run_config.enable_agent
        run_config.enable_input_guardrail = new_run_config.enable_input_guardrail
        run_config.enable_output_guardrail = new_run_config.enable_output_guardrail
        run_config.guardrail_hint = new_run_config.guardrail_hint

        evaluation_provider.update(run_config)
        session.add(run_config)
        await session.commit()
        await session.refresh(run_config)

        return success_response(data=run_config, message="更新运行配置成功。")
    except Exception:
        logger.error(
            f"Failed to update run config {config_id}: {traceback.format_exc()}"
        )
        return error_response(
                code=404, message=f"更新运行配置失败：{traceback.format_exc()}"
            )


@evaluation_router.get("/{dataset_id}/runconfigs")
async def list_run_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get run_configs for {dataset_id}.")
    total_results = await session.exec(
        select(func.count()).select_from(
            select(RunConfigEntity).where(RunConfigEntity.dataset_id == dataset_id)
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    run_config_results = await session.exec(
        select(RunConfigEntity)
        .where(RunConfigEntity.dataset_id == dataset_id)
        .order_by(RunConfigEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    run_config_entities = run_config_results.all()

    return success_response(
        data=PagedResult(
            items=run_config_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取实验设置列表成功",
    )


@evaluation_router.get("/{dataset_id}/runconfigs/{config_id}")
async def get_config_details(
    dataset_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment for config_id {config_id}.")
    run_config = await session.get(RunConfigEntity, config_id)
    if not run_config:
        return error_response(
                code=404, message=f"获取运行配置失败: '{config_id}'不存在。"
            )

    return success_response(
            data=run_config, message="获取实验设置详情成功"
        )


@evaluation_router.delete("/{dataset_id}/runconfigs/{config_id}")
async def delete_config(
    dataset_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    run_config = await session.get(RunConfigEntity, config_id)
    if not run_config:
        return error_response(
                code=404, message=f"删除实验设置失败: '{config_id}'不存在。"
            )

    await session.delete(run_config)
    await session.commit()

    logger.info(f"run_config {config_id} has been deleted.")
    return success_response(message=f"实验设置'{config_id}'删除成功。")



@evaluation_router.post("/{dataset_id}/evalconfigs")
async def create_evaluator_config(
    dataset_id: str,
    eval_config: EvaluatorConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    logger.info("Create eval_config_entity.")
    try:
        eval_config_entity = EvaluatorConfigEntity(
            name=eval_config.name,
            type=eval_config.type,
            dataset_id=dataset_id,
            model_id=eval_config.model_id,
            case_sensitive=eval_config.case_sensitive,
            ignore_punctuation=eval_config.ignore_punctuation,
        )

        session.add(eval_config_entity)
        await session.commit()
        logger.info(f"创建评估器设置 {eval_config_entity} 成功.")
        return success_response(data=eval_config_entity, message="创建评估器设置成功")
    except Exception as e:
        logger.error(f"创建评估器设置失败: {traceback.format_exc()}")
        await session.rollback()
        return error_response(code=400, message=f"创建评估器设置失败: {e}")


@evaluation_router.put(
    "/{dataset_id}/evalconfigs/{config_id}", response_model=ResponseModel[EvaluatorConfigEntity]
)
async def update_evaluator_config(
    dataset_id: str,
    config_id: str,
    new_eval_config: EvaluatorConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    eval_config = await session.get(EvaluatorConfigEntity, config_id)
    if not eval_config:
        return error_response(
                code=404, message=f"更新评估器设置失败: '{config_id}'不存在。"
            )

    try:
        eval_config.name = new_eval_config.name
        eval_config.type = new_eval_config.type
        eval_config.model_id = new_eval_config.model_id
        eval_config.case_sensitive = new_eval_config.case_sensitive
        eval_config.ignore_punctuation = new_eval_config.ignore_punctuation


        evaluation_provider.update(eval_config)
        session.add(eval_config)
        await session.commit()
        await session.refresh(eval_config)

        return success_response(data=eval_config, message="更新评估器设置成功。")
    except Exception:
        logger.error(
            f"Failed to update evaluator config {config_id}: {traceback.format_exc()}"
        )
        return error_response(
                code=404, message=f"更新评估器设置失败：{traceback.format_exc()}"
            )


@evaluation_router.get("/{dataset_id}/evalconfigs")
async def list_eval_configs(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get eval configs for {dataset_id}.")
    total_results = await session.exec(
        select(func.count()).select_from(
            select(EvaluatorConfigEntity).where(EvaluatorConfigEntity.dataset_id == dataset_id)
        )
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    eval_config_results = await session.exec(
        select(EvaluatorConfigEntity)
        .where(EvaluatorConfigEntity.dataset_id == dataset_id)
        .order_by(EvaluatorConfigEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    eval_config_entities = eval_config_results.all()

    return success_response(
        data=PagedResult(
            items=eval_config_entities,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        ),
        message="获取评估器设置列表成功",
    )


@evaluation_router.get("/{dataset_id}/evalconfigs/{config_id}")
async def get_eval_config_details(
    dataset_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get evalconfigs for config_id {config_id}.")
    eval_config = await session.get(EvaluatorConfigEntity, config_id)
    if not eval_config:
        return error_response(
                code=404, message=f"获取评估器设置失败: '{config_id}'不存在。"
            )

    return success_response(
            data=eval_config, message="获取评估器设置详情成功"
        )


@evaluation_router.delete("/{dataset_id}/evalconfigs/{config_id}")
async def delete_eval_config(
    dataset_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Delete experiment for config_id {config_id}.")
    eval_config = await session.get(EvaluatorConfigEntity, config_id)
    if not eval_config:
        return error_response(
                code=404, message=f"删除评估器设置失败: '{config_id}'不存在。"
            )

    await session.delete(eval_config)
    await session.commit()

    logger.info(f"eval_config {config_id} has been deleted.")
    return success_response(message=f"评估器设置'{config_id}'删除成功。")
