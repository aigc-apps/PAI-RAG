### Evaluation configuration API ###
import traceback
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.evaluation.evaluation import EvalEntity, EvalCreate
from db.models.evaluation.dataset import EvalDatasetEntity
from db.models.evaluation.experiment import ExperimentEntity, ExperimentRunResultEntity, ExperimentCreate
from db.models.evaluation.run_config import EvalRunConfigEntity, EvalRunConfigCreate
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from rag.file.store.file_store_helper import file_store
from api.response_model import ResponseModel, PagedResult, success_response, error_response
from loguru import logger
from rag.file.models.file_item import FileItem
from api.v1.utils.paginate import get_pagination_meta
from config.providers.evaluation_provider import evaluation_provider


evaluation_router = APIRouter()

@evaluation_router.post("", response_model=ResponseModel[EvalEntity])
async def create_evaluation(
    eval_create: EvalCreate, session: AsyncSession = Depends(get_session)
):
    try:
        evaluation = EvalEntity.model_validate(eval_create)
        evaluation_provider.add(evaluation)
        session.add(evaluation)
        await session.commit()
        await session.refresh(evaluation)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EVALUATION,
            event_type=ChangeEventType.ADD,
            source_id=evaluation.id,
        )
        return success_response(data=evaluation, message="评估任务创建成功。")

    except IntegrityError as e:
        logger.exception(f"创建评估任务失败。\nIntegrityError:{e}")
        await session.rollback()

        if "UniqueViolationError" in str(e.orig):
            return JSONResponse(
                content=error_response(code=400, message="创建评估任务失败: 评估任务名称已存在。"),
                status_code=400,
            )
        else:
            return JSONResponse(
                content=error_response(code=400, message=f"创建评估任务失败: {e}."),
                status_code=400,
            )
    except Exception:
        logger.exception(f"创建评估任务失败。\nException:{traceback.format_exc()}")
        await session.rollback()
        return JSONResponse(
            content=error_response(code=400, message=f"创建评估任务失败: {traceback.format_exc()}."),
            status_code=400,
        )

@evaluation_router.get("")
async def list_evaluations(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    # 子查询 1：统计每个 eval_id 对应的数据集数量
    dataset_count_subq = (
        select(
            EvalDatasetEntity.eval_id,
            func.count(EvalDatasetEntity.id).label("dataset_count")
        )
        .group_by(EvalDatasetEntity.eval_id)
        .subquery()
    )

    # 子查询 2：统计每个 eval_id 对应的实验数量
    experiment_count_subq = (
        select(
            ExperimentEntity.eval_id,
            func.count(ExperimentEntity.id).label("experiments_count")
        )
        .group_by(ExperimentEntity.eval_id)
        .subquery()
    )

    # 主查询：左连接两个子查询
    query = (
        select(
            EvalEntity,
            func.coalesce(dataset_count_subq.c.dataset_count, 0).label("dataset_count"),
            func.coalesce(experiment_count_subq.c.experiments_count, 0).label("experiments_count"),
        )
        .outerjoin(
            dataset_count_subq,
            EvalEntity.id == dataset_count_subq.c.eval_id
        )
        .outerjoin(
            experiment_count_subq,
            EvalEntity.id == experiment_count_subq.c.eval_id
        )
        .order_by(EvalEntity.created_at.desc())
        .offset((page - 1) * size)
        .limit(size)
    )

    # 获取总数（不变）
    total_results = await session.exec(
        select(func.count()).select_from(EvalEntity)
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
        item["experiments_count"] = experiments_count  # 👈 新增
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

@evaluation_router.get("/{eval_id}", response_model=ResponseModel[EvalEntity])
async def read_evaluation(eval_id: str, session: AsyncSession = Depends(get_session)):
    evaluation = await session.get(EvalEntity, eval_id)

    if not evaluation:
        return JSONResponse(
            content=error_response(code=404, message=f"查询评估任务失败: 评估任务'{eval_id}'不存在。"),
            status_code=404,
        )

    return success_response(data=evaluation, message="查询评估任务成功。")


@evaluation_router.put("/{eval_id}", response_model=ResponseModel[EvalEntity])
async def update_evaluation(
    eval_id: str,
    new_eval: EvalCreate,
    session: AsyncSession = Depends(get_session),
):
    evaluation = await session.get(EvalEntity, eval_id)
    if not evaluation:
        return JSONResponse(
            content=error_response(code=404, message=f"更新评估任务失败: 评估任务'{eval_id}'不存在。"),
            status_code=404,
        )

    try:
        evaluation.name = new_eval.name or evaluation.name
        evaluation.description = new_eval.description or evaluation.description

        evaluation_provider.update(evaluation)
        session.add(evaluation)
        await session.commit()
        await session.refresh(evaluation)

        await config_change_manager.notify_change_async(
            event_source=ChangeEventSource.EVALUATION,
            event_type=ChangeEventType.UPDATE,
            source_id=evaluation.id,
        )

        logger.info(f"Evaluation {eval_id} updated to {evaluation}.")

        return success_response(data=evaluation, message="更新评估任务成功。")
    except Exception:
        logger.error(f"Failed to update evaluation {eval_id}: {traceback.format_exc()}")
        return error_response(message=f"更新评估任务失败：{traceback.format_exc()}")


@evaluation_router.delete("/{eval_id}")
async def delete_evaluation(
    eval_id: str,
    session: AsyncSession = Depends(get_session),
):
    evaluation = await session.get(EvalEntity, eval_id)

    if not evaluation:
        return JSONResponse(
            content=error_response(code=404, message=f"删除评估任务失败: 知识库'{eval_id}'不存在。"),
            status_code=404,
        )

    evaluation_provider.delete(eval_id)
    await session.delete(evaluation)
    await session.commit()

    await config_change_manager.notify_change_async(
        event_source=ChangeEventSource.EVALUATION,
        event_type=ChangeEventType.DELETE,
        source_id=evaluation.id,
    )

    logger.info(f"Evaluation {eval_id} has been deleted.")

    return success_response(message=f"评估任务'{eval_id}'删除成功。")

@evaluation_router.post("/{eval_id}/dataset")
async def upload_dataset(
    eval_id: str,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Uploading dataset to {eval_id}.")
    if not file:
        return error_response(code=400, message="没有上传任何文件。")

    try:
        try:
            _ = await evaluation_provider.aget_evaluation(eval_id)
        except ValueError:
            logger.error(f"没找到评估任务{eval_id}")
            return error_response(code=400, message=f"没有找到评估任务 {eval_id}。")


        file_name = file.filename
        destination_file_path = f"{eval_id}/datasets/{file_name}"
        file_store.save(
            file=file.file,
            file_path=destination_file_path,
        )
        file_item = FileItem.from_file(
            file=file.file,
            file_path=destination_file_path,
            kb_id=eval_id,
            file_name=file.filename,
        )
        file_results = file_item.get_eval_dataset_from_jsonl_file()
        dataset_entities = []
        for line in file_results:
            dataset_entity = EvalDatasetEntity(
                eval_id=eval_id,
                input=line["input"],
                expected_output=line.get("expected_output"),
                eval_metadata=line.get("metadata")
            )
            session.add(dataset_entity)
            await session.commit()
            logger.info(f"Saved file {dataset_entity} successfully.")
            dataset_entities.append(dataset_entity)

        return success_response(data=dataset_entities, message="文件上传成功")
    except Exception as e:
        logger.error(f"Failed to upload eval dataset: {traceback.format_exc()}")
        await session.rollback()
        return error_response(message=f"Failed to save eval dataset to database: {e}")


@evaluation_router.get("/{eval_id}/dataset")
async def list_dataset(
    eval_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
) :
    total_results = await session.exec(
        select(func.count())
        .select_from(select(EvalDatasetEntity).where(EvalDatasetEntity.eval_id == eval_id))
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    file_results = await session.exec(
        select(EvalDatasetEntity)
        .where(EvalDatasetEntity.eval_id == eval_id)
        .order_by(EvalDatasetEntity.created_at.desc())
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
        message="获取评估数据集列表成功")


@evaluation_router.put("/{eval_id}/dataset/{sample_id}", response_model=ResponseModel[EvalDatasetEntity])
async def update_dataset_sample(
    eval_id: str,
    sample_id: str,
    new_sample: EvalDatasetEntity,
    session: AsyncSession = Depends(get_session),
):
    sample_entity = await session.get(EvalDatasetEntity, sample_id)
    if not sample_entity:
        return JSONResponse(
            content=error_response(code=404, message=f"更新数据集样本失败: 样本'{sample_id}'不存在。"),
            status_code=404,
        )

    try:
        sample_entity.input = new_sample.input
        sample_entity.expected_output = new_sample.expected_output
        sample_entity.eval_metadata = new_sample.eval_metadata

        session.add(sample_entity)
        await session.commit()
        await session.refresh(sample_entity)

        logger.info(f"Dataset Sample {sample_id} for eval_id {eval_id} updated to {sample_entity}.")

        return success_response(data=sample_entity, message="更新数据集样本成功。")
    except Exception:
        logger.error(f"Failed to update Dataset Sample {sample_id} for evaluation {eval_id}: {traceback.format_exc()}")
        return error_response(message=f"更新数据集样本失败：{traceback.format_exc()}")


@evaluation_router.delete("/{eval_id}/dataset/{sample_id}")
async def delete_dataset_sample(
    eval_id: str,
    sample_id: str,
    session: AsyncSession = Depends(get_session),
):
    sample_entity = await session.get(EvalDatasetEntity, sample_id)

    if not sample_entity:
        return JSONResponse(
            content=error_response(code=404, message=f"删除数据样本失败: 数据集 {eval_id} 样本 '{sample_id}'不存在。"),
            status_code=404,
        )

    await session.delete(sample_entity)
    await session.commit()

    logger.info(f"Dataset Sample {sample_id} has been deleted.")

    return success_response(message=f"数据样本'{eval_id}'删除成功。")
@evaluation_router.post("/{eval_id}/experiments")
async def create_experiment(
    eval_id: str,
    experiment_create: ExperimentCreate,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Create experiment for {eval_id}.")
    if len(experiment_create.dataset_ids) == 0:
        return error_response(code=400, message="没有选择任何数据集。")

    try:
        import app.worker as background_worker
        dataset_results = await session.exec(
            select(EvalDatasetEntity)
            .where(EvalDatasetEntity.id.in_(experiment_create.dataset_ids))
            .where(EvalDatasetEntity.eval_id == eval_id)
        )
        dataset_entities = dataset_results.all()
        if len(dataset_entities) == 0:
            return error_response(code=400, message=f"没有找到评估任务 {eval_id} 的数据集。")
        assert len(dataset_entities) == len(experiment_create.dataset_ids), "Some dataset IDs not found in the evaluation."
        experiment_entity = ExperimentEntity(
            eval_id=eval_id,
            name=experiment_create.name,
            samples_count=len(dataset_entities),
            run_config_id=experiment_create.run_config_id,
            description=experiment_create.description or "Experiment created via API",
            status="pending"
        )
        session.add(experiment_entity)
        await session.commit()
        logger.info(f"创建实验 {experiment_entity} 成功.")
        exp_run_ids = []
        for dataset_id in experiment_create.dataset_ids:
            exp_run_entity = ExperimentRunResultEntity(
                experiment_id=experiment_entity.id,
                dataset_id=dataset_id,
                status="pending"
            )
            session.add(exp_run_entity)
            await session.commit()
            exp_run_ids.append(exp_run_entity.id)
        background_worker.execute_evaluation_task.delay(eval_id, experiment_entity.id, exp_run_ids)

        return success_response(data=experiment_entity, message="创建实验成功")
    except Exception as e:
        logger.error(f"Failed to create experiment: {traceback.format_exc()}")
        await session.rollback()
        return error_response(message=f"Failed to create experiment: {e}")


@evaluation_router.get("/{eval_id}/experiments")
async def get_experiments(
    eval_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiments for {eval_id}.")
    total_results = await session.exec(
        select(func.count())
        .select_from(select(ExperimentEntity).where(ExperimentEntity.eval_id == eval_id))
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    experiment_results = await session.exec(
        select(ExperimentEntity)
        .where(ExperimentEntity.eval_id == eval_id)
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
        message="获取评估实验列表成功")

@evaluation_router.get("/{eval_id}/experiments/{exp_id}")
async def get_experiment(
    eval_id: str,
    exp_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment for eval_id {eval_id} and exp_id {exp_id}.")
    experiment_results = await session.exec(
        select(ExperimentEntity)
        .where(ExperimentEntity.eval_id == eval_id)
        .where(ExperimentEntity.id == exp_id)
    )
    experiment_entity = experiment_results.all()[0]

    return success_response(
        data=experiment_entity,
        message="获取评估实验详情成功")

@evaluation_router.get("/{eval_id}/experiments/{exp_id}/details")
async def get_experiment_details(
    eval_id: str,
    exp_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment details for eval_id {eval_id}, exp_id {exp_id}.")
    total_results = await session.exec(
        select(func.count())
        .select_from(select(ExperimentRunResultEntity).where(ExperimentRunResultEntity.experiment_id == exp_id))
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    experiment_results = await session.exec(
        select(
            ExperimentRunResultEntity,
            EvalDatasetEntity.input,
            EvalDatasetEntity.expected_output,
            EvalDatasetEntity.eval_metadata.label("dataset_metadata"),
        )
        .join(EvalDatasetEntity, ExperimentRunResultEntity.dataset_id == EvalDatasetEntity.id)
        .where(ExperimentRunResultEntity.experiment_id == exp_id)
        .order_by(ExperimentRunResultEntity.created_at.desc())
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
        message="获取评估实验执行详情信息成功")

@evaluation_router.delete("/{eval_id}/experiments/{exp_id}")
async def delete_experiment(
    eval_id: str,
    exp_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Delete experiment for eval_id {eval_id} and exp_id {exp_id}.")
    experiment = await session.get(ExperimentEntity, exp_id)
    if not experiment:
        return JSONResponse(
            content=error_response(code=404, message=f"删除实验失败: 实验'{exp_id}'不存在。"),
            status_code=404,
        )

    await session.delete(experiment)
    await session.commit()

    logger.info(f"Experiment {exp_id} has been deleted.")
    return success_response(message=f"实验'{exp_id}'删除成功。")

# evaluation run config
@evaluation_router.post("/{eval_id}/configs")
async def create_run_config(
    eval_id: str,
    run_config: EvalRunConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Create run_config for {eval_id}.")
    try:
        run_config_entity = EvalRunConfigEntity(
            eval_id=eval_id,
            name=run_config.name,
            model_id=run_config.model_id,
            mcp_ids=run_config.mcp_ids,
            kb_ids=run_config.kb_ids,
            enable_search=run_config.enable_search,
            enable_vision=run_config.enable_vision,
            enable_agent=run_config.enable_agent,
            enable_input_guardrail=run_config.enable_input_guardrail,
            enable_output_guardrail=run_config.enable_output_guardrail,
            guardrail_hint=run_config.guardrail_hint,
            evaluator_config=run_config.evaluator_config.model_dump()
        )

        session.add(run_config_entity)
        await session.commit()
        logger.info(f"创建实验设置 {run_config_entity} 成功.")
        return success_response(data=run_config_entity, message="创建实验设置成功")
    except Exception as e:
        logger.error(f"Failed to create run_config: {traceback.format_exc()}")
        await session.rollback()
        return error_response(message=f"Failed to create run_config: {e}")


@evaluation_router.put("/{eval_id}/configs/{config_id}", response_model=ResponseModel[EvalRunConfigEntity])
async def update_run_config(
    eval_id: str,
    config_id: str,
    new_run_config: EvalRunConfigCreate,
    session: AsyncSession = Depends(get_session),
):
    run_config = await session.get(EvalRunConfigEntity, config_id)
    if not run_config:
        return JSONResponse(
            content=error_response(code=404, message=f"更新实验设置失败: '{config_id}'不存在。"),
            status_code=404,
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
        run_config.evaluator_config = new_run_config.evaluator_config.model_dump()


        evaluation_provider.update(run_config)
        session.add(run_config)
        await session.commit()
        await session.refresh(run_config)

        logger.info(f"Evaluation {eval_id} run config  {config_id} updated to {run_config}.")

        return success_response(data=run_config, message="更新实验设置成功。")
    except Exception:
        logger.error(f"Failed to update run config {config_id}: {traceback.format_exc()}")
        return error_response(message=f"更新实验设置失败：{traceback.format_exc()}")

@evaluation_router.get("/{eval_id}/configs")
async def list_run_configs(
    eval_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, le=1000),
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get run_configs for {eval_id}.")
    total_results = await session.exec(
        select(func.count())
        .select_from(select(EvalRunConfigEntity).where(EvalRunConfigEntity.eval_id == eval_id))
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    run_config_results = await session.exec(
        select(EvalRunConfigEntity)
        .where(EvalRunConfigEntity.eval_id == eval_id)
        .order_by(EvalRunConfigEntity.created_at.desc())
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
        message="获取实验设置列表成功")


@evaluation_router.get("/{eval_id}/configs/{config_id}")
async def get_configs(
    eval_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Get experiment for eval_id {eval_id} and config_id {config_id}.")
    run_config_results = await session.exec(
        select(EvalRunConfigEntity)
        .where(EvalRunConfigEntity.eval_id == eval_id)
        .where(EvalRunConfigEntity.id == config_id)
    )
    run_config_entities = run_config_results.all()
    if len(run_config_entities) > 0:
        return success_response(
            data=run_config_entities[0],
            message="获取实验设置详情成功")
    else:
        return success_response(
            data=[],
            message="获取实验设置详情成功")

@evaluation_router.delete("/{eval_id}/configs/{config_id}")
async def delete_config(
    eval_id: str,
    config_id: str,
    session: AsyncSession = Depends(get_session),
):
    logger.info(f"Delete experiment for eval_id {eval_id} and config_id {config_id}.")
    Run_config = await session.get(EvalRunConfigEntity, config_id)
    if not Run_config:
        return JSONResponse(
            content=error_response(code=404, message=f"删除实验设置失败: '{config_id}'不存在。"),
            status_code=404,
        )

    await session.delete(Run_config)
    await session.commit()

    logger.info(f"Run_config {config_id} has been deleted.")
    return success_response(message=f"实验设置'{config_id}'删除成功。")
