### Evaluation configuration API ###
import traceback
from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import JSONResponse
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from db.models.change_event import ChangeEventSource, ChangeEventType
from db.models.evaluation.evaluation import EvalEntity, EvalCreate, EvalRunConfig
from db.models.evaluation.dataset import EvalDatasetEntity
from db.models.evaluation.experiment import ExperimentEntity, ExperimentRunResultEntity, ExperimentCreate
from db.db_context import get_session
from sqlalchemy.exc import IntegrityError
from config.providers.config_change_manager import config_change_manager
from rag.file.store.file_store_helper import file_store
from api.response_model import ResponseModel, PagedResult, success_response, error_response
from loguru import logger
from rag.file.models.file_item import FileItem
from api.v1.utils.paginate import get_pagination_meta
from config.providers.evaluation_provider import evaluation_provider
from db.models.chatbot import ChatBotEntity


evaluation_router = APIRouter()

@evaluation_router.post("", response_model=ResponseModel[EvalEntity])
async def create_evaluation(
    eval_create: EvalCreate, session: AsyncSession = Depends(get_session)
):
    try:
        if eval_create.run_type and eval_create.run_type == "chatbot":
            statement = select(ChatBotEntity).where(
                ChatBotEntity.app_id == eval_create.chatbot_id
            )
            chatbot = (await session.exec(statement)).first()
            eval_create.default_run_config = {
                "model_id": chatbot.model_id,
                "mcp_ids": chatbot.mcp_ids,
                "kb_ids": chatbot.kb_ids,
                "enable_search": chatbot.enable_search,
                "enable_vision": chatbot.enable_vision,
                "enable_agent": chatbot.enable_agent,
                "enable_input_guardrail": chatbot.enable_input_guardrail,
                "enable_output_guardrail": chatbot.enable_output_guardrail,
                "guardrail_hint": chatbot.guardrail_hint,
            }
        else:
            eval_create.default_run_config = eval_create.default_run_config or EvalRunConfig().model_dump()
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
    total_results = await session.exec(
        select(func.count()).select_from(EvalEntity)
    )
    total_num = total_results.one_or_none()
    pagination = get_pagination_meta(page, size, total_num)
    eval_results = await session.exec(
        select(EvalEntity)
        .order_by(EvalEntity.created_at.desc())
        .offset(pagination.offset)
        .limit(size)
    )
    eval_entities = eval_results.all()

    return success_response(
        data=PagedResult(
            items=eval_entities,
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
        evaluation.run_type = new_eval.run_type or evaluation.run_type

        if evaluation.run_type == "chatbot":
            evaluation.chatbot_id = new_eval.chatbot_id
            statement = select(ChatBotEntity).where(
                ChatBotEntity.app_id == evaluation.chatbot_id
            )
            chatbot = (await session.exec(statement)).first()
            evaluation.default_run_config = {
                "model_id": chatbot.model_id,
                "mcp_ids": chatbot.mcp_ids,
                "kb_ids": chatbot.kb_ids,
                "enable_search": chatbot.enable_search,
                "enable_vision": chatbot.enable_vision,
                "enable_agent": chatbot.enable_agent,
                "enable_input_guardrail": chatbot.enable_input_guardrail,
                "enable_output_guardrail": chatbot.enable_output_guardrail,
                "guardrail_hint": chatbot.guardrail_hint,
            }
        else:
            evaluation.chatbot_id = ""
            evaluation.default_run_config = new_eval.default_run_config

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
            run_config=experiment_create.run_config or EvalRunConfig().model_dump(),
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
        .order_by(ExperimentEntity.created_at.asc())
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
