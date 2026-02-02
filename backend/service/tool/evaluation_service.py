"""Evaluation Service layer for database operations."""

from typing import Optional, List, Dict
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger
from api.v1.utils.paginate import get_pagination_meta

from db.models.evaluation.dataset import DatasetEntity, DatasetCreate, DatasetSampleEntity
from db.models.evaluation.experiment import (
    ExperimentEntity,
    ExperimentSampleEntity,
    ExperimentCreate,
)
from db.models.evaluation.run_config import RunConfigEntity, RunConfigCreate
from db.models.evaluation.evaluator_config import (
    EvaluatorConfigEntity,
    EvaluatorConfigCreate,
)
from common.chat.response_model import PagedResult


class EvaluationService:
    """Service layer for Evaluation entity CRUD operations using dependency injection."""

    def __init__(self, session: AsyncSession):
        """
        Initialize EvaluationService with a database session.

        Args:
            session: Database session (injected dependency)
        """
        self.session = session

    # ========== Dataset Operations ==========

    async def get_dataset(self, dataset_id: str, tenant_id: str) -> Optional[DatasetEntity]:
        """
        Get a single Dataset entity by ID.

        Args:
            dataset_id: Dataset entity ID

        Returns:
            DatasetEntity if found, None otherwise
        """
        datasets = await self.session.exec(select(DatasetEntity).where(DatasetEntity.id == dataset_id, DatasetEntity.tenant_id == tenant_id))
        return datasets.first()

    async def get_default_eval_dataset(self, tenant_id: str) -> Optional[DatasetEntity]:
        """
        Get the default GAIA evaluation dataset.
        If it doesn't exist, create it and load samples from the GAIA dataset file.

        Args:
            tenant_id: Tenant ID

        Returns:
            DatasetEntity if found or created, None otherwise
        """
        statement = select(DatasetEntity).where(
            DatasetEntity.name == "GAIA",
            DatasetEntity.tenant_id == tenant_id
        )
        result = await self.session.exec(statement)
        default_dataset = result.first()

        if not default_dataset:
            logger.info(f"No default GAIA dataset was found for tenant {tenant_id}, creating it.")

            # Create GAIA dataset
            gaia_dataset_data = DatasetCreate(
                name="GAIA",
                description="GAIA Evaluation",
                type="built-in"
            )

            try:
                default_dataset = await self.create_dataset(
                    dataset_data=gaia_dataset_data,
                    tenant_id=tenant_id
                )
                await self.session.commit()
                await self.session.refresh(default_dataset)

                logger.info(f"Created default GAIA dataset: {default_dataset.id}")

                GAIA_DATASET_PATH = "./resources/dataset/gaia/gaia_level_1_27.jsonl"
                try:
                    from utils.upload_file_utils import load_eval_dataset_from_local_path
                    file_results = load_eval_dataset_from_local_path(file_path=GAIA_DATASET_PATH)

                    # Prepare samples for batch creation
                    samples = []
                    for line in file_results:
                        samples.append({
                            "input": line["input"],
                            "expected_output": line.get("expected_output"),
                            "metadata": line.get("metadata") or {}
                        })

                    # Batch create dataset samples
                    if samples:
                        await self.batch_create_dataset_samples(
                            dataset_id=default_dataset.id,
                            samples=samples,
                            tenant_id=tenant_id
                        )
                        await self.session.commit()
                        logger.info(f"Loaded {len(samples)} samples into GAIA dataset.")

                except FileNotFoundError:
                    logger.warning(f"GAIA dataset file not found at {GAIA_DATASET_PATH}, dataset created without samples.")
                except Exception as e:
                    logger.error(f"Failed to load GAIA dataset samples: {e}")

            except IntegrityError as e:
                logger.error(f"IntegrityError when creating default GAIA dataset: {e.orig}")
                await self.session.rollback()
                result = await self.session.exec(statement)
                default_dataset = result.first()
                if not default_dataset:
                    raise ValueError(f"Default evaluation dataset creation failed: {e}") from e
            except Exception as e:
                logger.error(f"Error creating default GAIA dataset: {e}")
                await self.session.rollback()
                raise

        return default_dataset

    async def list_datasets(
        self,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[DatasetEntity]]:
        """
        List Dataset entities with pagination and statistics.

        Args:
            page: Page number (1-indexed)
            size: Page size
            tenant_id: Tenant ID

        Returns:
            PagedResult containing list of DatasetEntity
        """
        # Subquery 1: count samples per dataset_id
        dataset_count_subq = (
            select(
                DatasetSampleEntity.dataset_id,
                func.count(DatasetSampleEntity.id).label("dataset_count"),
            )
            .where(DatasetSampleEntity.tenant_id == tenant_id)
            .group_by(DatasetSampleEntity.dataset_id)
            .subquery()
        )

        # Subquery 2: count experiments per dataset_id
        experiment_count_subq = (
            select(
                ExperimentEntity.dataset_id,
                func.count(ExperimentEntity.id).label("experiments_count"),
            )
            .where(ExperimentEntity.tenant_id == tenant_id)
            .group_by(ExperimentEntity.dataset_id)
            .subquery()
        )

        # Main query: LEFT JOIN to get counts
        query = (
            select(
                DatasetEntity,
                func.coalesce(dataset_count_subq.c.dataset_count, 0).label(
                    "dataset_count"
                ),
                func.coalesce(experiment_count_subq.c.experiments_count, 0).label(
                    "experiments_count"
                ),
            )
            .where(DatasetEntity.tenant_id == tenant_id)
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

        # Get total count
        total_results = await self.session.exec(
            select(func.count()).select_from(DatasetEntity).where(DatasetEntity.tenant_id == tenant_id)
        )
        total = total_results.one_or_none() or 0

        # Execute main query
        results = await self.session.exec(query)
        eval_entities_with_counts = results.all()

        # Build result list with counts
        items = []
        for eval_entity, dataset_count, experiments_count in eval_entities_with_counts:
            item = eval_entity.model_dump()
            item["dataset_count"] = dataset_count
            item["experiments_count"] = experiments_count
            items.append(item)

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=items,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_dataset(self, dataset_data: DatasetCreate, tenant_id: str) -> DatasetEntity:
        """
        Create a new Dataset entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_data: Dataset creation data

        Returns:
            Created DatasetEntity (not yet committed)

        Raises:
            ValueError: If name already exists (IntegrityError converted)
        """
        dataset = DatasetEntity.model_validate(dataset_data, update={"tenant_id": tenant_id})
        self.session.add(dataset)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(dataset)

            logger.info(f"Created Dataset entity: {dataset.id} (name: {dataset.name})")
            return dataset

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Dataset: {e.orig}")

            if "UniqueViolationError" in str(e.orig):
                raise ValueError(
                    f"Dataset name '{dataset_data.name}' already exists."
                ) from e
            else:
                raise ValueError(f"Dataset creation failed: {e}") from e

    async def update_dataset(
        self, dataset_id: str, update_data: DatasetCreate, tenant_id: str
    ) -> DatasetEntity:
        """
        Update an existing Dataset entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset entity ID
            update_data: Updated Dataset data

        Returns:
            Updated DatasetEntity (not yet committed)

        Raises:
            ValueError: If Dataset entity not found
        """
        result = await self.session.exec(select(DatasetEntity).where(DatasetEntity.id == dataset_id, DatasetEntity.tenant_id == tenant_id))
        dataset = result.first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        logger.info(f"Updating Dataset {dataset_id} with data: {update_data}")

        # Update fields
        if update_data.name is not None:
            dataset.name = update_data.name
        if update_data.description is not None:
            dataset.description = update_data.description
        if update_data.type is not None:
            dataset.type = update_data.type

        self.session.add(dataset)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(dataset)

        logger.info(f"Updated Dataset entity: {dataset.id} (name: {dataset.name})")
        return dataset

    async def delete_dataset(self, dataset_id: str, tenant_id: str) -> None:
        """
        Delete a Dataset entity.
        Note: This will cascade delete related samples, experiments, etc.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset entity ID

        Raises:
            ValueError: If Dataset entity not found
        """
        result = await self.session.exec(select(DatasetEntity).where(DatasetEntity.id == dataset_id, DatasetEntity.tenant_id == tenant_id))
        dataset = result.first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' does not exist.")

        # Delete from database (staged, not committed)
        # CASCADE will handle related entities
        await self.session.delete(dataset)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Dataset entity: {dataset_id} (name: {dataset.name})")

    # ========== DatasetSample Operations ==========

    async def get_dataset_sample(
        self, sample_id: str, tenant_id: str
    ) -> Optional[DatasetSampleEntity]:
        """
        Get a single DatasetSample entity by ID.

        Args:
            sample_id: DatasetSample entity ID

        Returns:
            DatasetSampleEntity if found, None otherwise
        """
        result = await self.session.exec(select(DatasetSampleEntity).where(DatasetSampleEntity.id == sample_id, DatasetSampleEntity.tenant_id == tenant_id))
        return result.first()

    async def list_dataset_samples(
        self,
        dataset_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[DatasetSampleEntity]]:
        """
        List DatasetSample entities with pagination.

        Args:
            dataset_id: Dataset ID
            tenant_id: Tenant ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of DatasetSampleEntity and pagination metadata
        """
        # Build base query
        base_query = select(DatasetSampleEntity).where(
            DatasetSampleEntity.dataset_id == dataset_id,
            DatasetSampleEntity.tenant_id == tenant_id
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(DatasetSampleEntity.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        samples = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=samples,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_dataset_sample(
        self,
        dataset_id: str,
        input: str,
        tenant_id: str,
        expected_output: Optional[str] = None,
        eval_metadata: Optional[dict] = None,
    ) -> DatasetSampleEntity:
        """
        Create a new DatasetSample entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset ID
            input: Sample input
            expected_output: Optional expected output
            eval_metadata: Optional evaluation metadata

        Returns:
            Created DatasetSampleEntity (not yet committed)
        """
        sample = DatasetSampleEntity(
            dataset_id=dataset_id,
            input=input,
            expected_output=expected_output,
            eval_metadata=eval_metadata or {},
        )

        self.session.add(sample)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(sample)

            logger.info(f"Created DatasetSample entity: {sample.id}")
            return sample

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating DatasetSample: {e.orig}")
            raise ValueError(f"Dataset sample creation failed: {e}") from e

    async def update_dataset_sample(
        self,
        sample_id: str,
        tenant_id: str,
        input: Optional[str] = None,
        expected_output: Optional[str] = None,
        eval_metadata: Optional[dict] = None,
    ) -> DatasetSampleEntity:
        """
        Update an existing DatasetSample entity.
        Note: Caller is responsible for committing the session.

        Args:
            sample_id: DatasetSample entity ID
            input: Updated input
            expected_output: Updated expected output
            eval_metadata: Updated evaluation metadata

        Returns:
            Updated DatasetSampleEntity (not yet committed)

        Raises:
            ValueError: If DatasetSample entity not found
        """
        result = await self.session.exec(select(DatasetSampleEntity).where(DatasetSampleEntity.id == sample_id, DatasetSampleEntity.tenant_id == tenant_id))
        sample = result.first()
        if not sample:
            raise ValueError(f"Dataset sample '{sample_id}' does not exist.")

        logger.info(f"Updating DatasetSample {sample_id}")

        # Update fields
        if input is not None:
            sample.input = input
        if expected_output is not None:
            sample.expected_output = expected_output
        if eval_metadata is not None:
            sample.eval_metadata = eval_metadata

        self.session.add(sample)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(sample)

        logger.info(f"Updated DatasetSample entity: {sample.id}")
        return sample

    async def delete_dataset_sample(self, sample_id: str, tenant_id: str) -> None:
        """
        Delete a DatasetSample entity.
        Note: Caller is responsible for committing the session.

        Args:
            sample_id: DatasetSample entity ID

        Raises:
            ValueError: If DatasetSample entity not found
        """
        result = await self.session.exec(select(DatasetSampleEntity).where(DatasetSampleEntity.id == sample_id, DatasetSampleEntity.tenant_id == tenant_id))
        sample = result.first()
        if not sample:
            raise ValueError(f"Dataset sample '{sample_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(sample)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted DatasetSample entity: {sample_id}")

    async def get_dataset_samples(
        self, dataset_id: str, sample_ids: List[str], tenant_id: str
    ) -> List[DatasetSampleEntity]:
        """
        Get multiple DatasetSample entities by IDs.

        Args:
            dataset_id: Dataset ID
            sample_ids: List of sample IDs

        Returns:
            List of DatasetSampleEntity
        """
        if not sample_ids:
            return []

        statement = select(DatasetSampleEntity).where(
            DatasetSampleEntity.id.in_(sample_ids),
            DatasetSampleEntity.dataset_id == dataset_id,
            DatasetSampleEntity.tenant_id == tenant_id
        )
        results = await self.session.exec(statement)
        return list(results.all())

    async def batch_create_dataset_samples(
        self,
        dataset_id: str,
        samples: List[Dict],
        tenant_id: str,
    ) -> List[DatasetSampleEntity]:
        """
        Batch create DatasetSample entities.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset ID
            samples: List of sample dicts with keys: input, expected_output, eval_metadata

        Returns:
            List of created DatasetSampleEntity (not yet committed)
        """
        dataset_samples = []
        for sample_data in samples:
            sample = DatasetSampleEntity(
                dataset_id=dataset_id,
                input=sample_data["input"],
                expected_output=sample_data.get("expected_output"),
                eval_metadata=sample_data.get("metadata") or {},
                tenant_id=tenant_id,
            )
            self.session.add(sample)
            dataset_samples.append(sample)

        try:
            # Flush to get IDs, but don't commit
            await self.session.flush()
            for sample in dataset_samples:
                await self.session.refresh(sample)

            logger.info(
                f"Created {len(dataset_samples)} DatasetSample entities for dataset {dataset_id}"
            )
            return dataset_samples

        except IntegrityError as e:
            logger.error(f"IntegrityError when batch creating DatasetSamples: {e.orig}")
            raise ValueError(f"Batch creation of dataset samples failed: {e}") from e

    # ========== Experiment Operations ==========

    async def get_experiment(
        self, experiment_id: str, tenant_id: str
    ) -> Optional[ExperimentEntity]:
        """
        Get a single Experiment entity by ID.

        Args:
            experiment_id: Experiment entity ID

        Returns:
            ExperimentEntity if found, None otherwise
        """
        result = await self.session.exec(select(ExperimentEntity).where(ExperimentEntity.id == experiment_id, ExperimentEntity.tenant_id == tenant_id))
        return result.first()

    async def list_experiments(
        self,
        dataset_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[ExperimentEntity]]:
        """
        List Experiment entities with pagination.

        Args:
            dataset_id: Dataset ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of ExperimentEntity and pagination metadata
        """
        # Build base query
        base_query = select(ExperimentEntity).where(
            ExperimentEntity.dataset_id == dataset_id,
            ExperimentEntity.tenant_id == tenant_id
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(ExperimentEntity.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        experiments = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=experiments,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_experiment(
        self, dataset_id: str, experiment_data: ExperimentCreate, tenant_id: str
    ) -> tuple[ExperimentEntity, List[str]]:
        """
        Create a new Experiment entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset ID
            experiment_data: Experiment creation data

        Returns:
            Tuple of (Created ExperimentEntity, List of created ExperimentSampleEntity IDs) (not yet committed)

        Raises:
            ValueError: If sample_ids are invalid
        """
        if not experiment_data.sample_ids or len(experiment_data.sample_ids) == 0:
            raise ValueError("No dataset samples selected.")

        # Validate sample_ids exist and belong to dataset_id
        dataset_sample_results = await self.session.exec(
            select(DatasetSampleEntity)
            .where(DatasetSampleEntity.id.in_(experiment_data.sample_ids))
            .where(DatasetSampleEntity.dataset_id == dataset_id)
            .where(DatasetSampleEntity.tenant_id == tenant_id)
        )
        dataset_sample_entities = list(dataset_sample_results.all())

        if len(dataset_sample_entities) == 0:
            raise ValueError(f"No dataset samples found for dataset {dataset_id}.")

        if len(dataset_sample_entities) != len(experiment_data.sample_ids):
            missing_ids = set(experiment_data.sample_ids) - {
                d.id for d in dataset_sample_entities
            }
            raise ValueError(f"The following sample IDs do not exist: {missing_ids}")

        # Create experiment entity
        experiment_entity = ExperimentEntity(
            dataset_id=dataset_id,
            name=experiment_data.name,
            samples_count=len(dataset_sample_entities),
            run_config_id=experiment_data.run_config_id,
            evaluator_config_id=experiment_data.evaluator_config_id,
            description=experiment_data.description or "Experiment created via API",
            status="pending",
            tenant_id=tenant_id,
        )

        self.session.add(experiment_entity)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(experiment_entity)

            # Create experiment sample entities
            exp_sample_ids = []
            for sample_id in experiment_data.sample_ids:
                exp_run_entity = ExperimentSampleEntity(
                    experiment_id=experiment_entity.id,
                    dataset_id=dataset_id,
                    sample_id=sample_id,
                    status="pending",
                )
                self.session.add(exp_run_entity)
                exp_sample_ids.append(exp_run_entity.id)

            # Flush again to get all IDs
            await self.session.flush()

            logger.info(
                f"Created Experiment entity: {experiment_entity.id} (name: {experiment_entity.name})"
            )
            return experiment_entity, exp_sample_ids

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating Experiment: {e.orig}")
            raise ValueError(f"Experiment creation failed: {e}") from e

    async def delete_experiment(self, experiment_id: str, tenant_id: str) -> None:
        """
        Delete an Experiment entity.
        Note: This will cascade delete related experiment samples.
        Note: Caller is responsible for committing the session.

        Args:
            experiment_id: Experiment entity ID

        Raises:
            ValueError: If Experiment entity not found
        """
        result = await self.session.exec(select(ExperimentEntity).where(ExperimentEntity.id == experiment_id, ExperimentEntity.tenant_id == tenant_id))
        experiment = result.first()
        if not experiment:
            raise ValueError(f"Experiment '{experiment_id}' does not exist.")

        # Delete from database (staged, not committed)
        # CASCADE will handle related experiment samples
        await self.session.delete(experiment)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted Experiment entity: {experiment_id} (name: {experiment.name})")

    async def get_experiment_samples(
        self,
        experiment_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
        status: Optional[str] = None,
    ) -> PagedResult[List[ExperimentSampleEntity]]:
        """
        List ExperimentSample entities with pagination.

        Args:
            experiment_id: Experiment ID
            page: Page number (1-indexed)
            size: Page size
            status: Optional status filter (running, success, failed, pending)

        Returns:
            PagedResult containing list of ExperimentSampleEntity and pagination metadata
        """
        logger.info(f"Get experiment samples for experiment_id {experiment_id}, tenant_id {tenant_id}, status={status}.")

        # Build base conditions
        conditions = [
            ExperimentSampleEntity.experiment_id == experiment_id,
            ExperimentSampleEntity.tenant_id == tenant_id,
        ]

        # Apply status filter if provided
        if status:
            conditions.append(ExperimentSampleEntity.status == status)

        # Get total count
        count_query = select(func.count()).select_from(
            select(ExperimentSampleEntity).where(*conditions).subquery()
        )
        total_result = await self.session.exec(count_query)
        total_num = total_result.one_or_none() or 0

        # Calculate pagination
        pagination = get_pagination_meta(page, size, total_num)

        # Build main query with JOIN
        main_query = (
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
            .where(*conditions)
            .order_by(ExperimentSampleEntity.created_at.desc())
            .offset(pagination.offset)
            .limit(size)
        )

        results = await self.session.exec(main_query)

        # Transform results
        transformed_results = [
            {
                **row[0].model_dump(),
                "input": row[1],
                "expected_output": row[2],
                "dataset_metadata": row[3],
            }
            for row in results.all()
        ]

        return PagedResult(
            items=transformed_results,
            total=pagination.total,
            pages=pagination.pages,
            page=pagination.page,
            size=pagination.size,
        )

    # ========== RunConfig Operations ==========

    async def get_run_config(self, config_id: str, tenant_id: str) -> Optional[RunConfigEntity]:
        """
        Get a single RunConfig entity by ID.

        Args:
            config_id: RunConfig entity ID

        Returns:
            RunConfigEntity if found, None otherwise
        """
        result = await self.session.exec(select(RunConfigEntity).where(RunConfigEntity.id == config_id, RunConfigEntity.tenant_id == tenant_id))
        return result.first()

    async def list_run_configs(
        self,
        dataset_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[RunConfigEntity]]:
        """
        List RunConfig entities with pagination.

        Args:
            dataset_id: Dataset ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of RunConfigEntity and pagination metadata
        """
        # Build base query
        base_query = select(RunConfigEntity).where(
            RunConfigEntity.dataset_id == dataset_id,
            RunConfigEntity.tenant_id == tenant_id
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(RunConfigEntity.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        configs = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=configs,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_run_config(
        self, dataset_id: str, config_data: RunConfigCreate, tenant_id: str
    ) -> RunConfigEntity:
        """
        Create a new RunConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset ID
            config_data: RunConfig creation data

        Returns:
            Created RunConfigEntity (not yet committed)
        """
        run_config_entity = RunConfigEntity(
            name=config_data.name,
            dataset_id=dataset_id,
            model_id=config_data.model_id,
            mcp_ids=config_data.mcp_ids,
            kb_ids=config_data.kb_ids,
            enable_search=config_data.enable_search,
            enable_vision=config_data.enable_vision,
            enable_agent=config_data.enable_agent,
            enable_input_guardrail=config_data.enable_input_guardrail,
            enable_output_guardrail=config_data.enable_output_guardrail,
            guardrail_hint=config_data.guardrail_hint,
            prompts=config_data.prompts,
            tenant_id=tenant_id,
        )

        self.session.add(run_config_entity)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(run_config_entity)

            logger.info(
                f"Created RunConfig entity: {run_config_entity.id} (name: {run_config_entity.name})"
            )
            return run_config_entity

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating RunConfig: {e.orig}")
            raise ValueError(f"Run configuration creation failed: {e}") from e

    async def update_run_config(
        self, config_id: str, update_data: RunConfigCreate, tenant_id: str
    ) -> RunConfigEntity:
        """
        Update an existing RunConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_id: RunConfig entity ID
            update_data: Updated RunConfig data

        Returns:
            Updated RunConfigEntity (not yet committed)

        Raises:
            ValueError: If RunConfig entity not found
        """
        result = await self.session.exec(select(RunConfigEntity).where(RunConfigEntity.id == config_id, RunConfigEntity.tenant_id == tenant_id))
        run_config = result.first()
        if not run_config:
            raise ValueError(f"Run configuration '{config_id}' does not exist.")

        logger.info(f"Updating RunConfig {config_id} with data: {update_data}")

        # Update fields
        if update_data.name is not None:
            run_config.name = update_data.name
        if update_data.model_id is not None:
            run_config.model_id = update_data.model_id
        if update_data.mcp_ids is not None:
            run_config.mcp_ids = update_data.mcp_ids
        if update_data.kb_ids is not None:
            run_config.kb_ids = update_data.kb_ids
        if update_data.enable_search is not None:
            run_config.enable_search = update_data.enable_search
        if update_data.enable_vision is not None:
            run_config.enable_vision = update_data.enable_vision
        if update_data.enable_agent is not None:
            run_config.enable_agent = update_data.enable_agent
        if update_data.enable_input_guardrail is not None:
            run_config.enable_input_guardrail = update_data.enable_input_guardrail
        if update_data.enable_output_guardrail is not None:
            run_config.enable_output_guardrail = update_data.enable_output_guardrail
        if update_data.guardrail_hint is not None:
            run_config.guardrail_hint = update_data.guardrail_hint
        if update_data.prompts is not None:
            run_config.prompts = update_data.prompts

        self.session.add(run_config)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(run_config)

        logger.info(f"Updated RunConfig entity: {run_config.id} (name: {run_config.name})")
        return run_config

    async def delete_run_config(self, config_id: str, tenant_id: str) -> None:
        """
        Delete a RunConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_id: RunConfig entity ID

        Raises:
            ValueError: If RunConfig entity not found
        """
        result = await self.session.exec(select(RunConfigEntity).where(RunConfigEntity.id == config_id, RunConfigEntity.tenant_id == tenant_id))
        run_config = result.first()
        if not run_config:
            raise ValueError(f"Run configuration '{config_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(run_config)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(f"Deleted RunConfig entity: {config_id} (name: {run_config.name})")

    # ========== EvaluatorConfig Operations ==========

    async def get_evaluator_config(
        self, config_id: str, tenant_id: str
    ) -> Optional[EvaluatorConfigEntity]:
        """
        Get a single EvaluatorConfig entity by ID.

        Args:
            config_id: EvaluatorConfig entity ID

        Returns:
            EvaluatorConfigEntity if found, None otherwise
        """
        result = await self.session.exec(select(EvaluatorConfigEntity).where(EvaluatorConfigEntity.id == config_id, EvaluatorConfigEntity.tenant_id == tenant_id))
        return result.first()

    async def list_evaluator_configs(
        self,
        dataset_id: str,
        tenant_id: str,
        page: int = 1,
        size: int = 10,
    ) -> PagedResult[List[EvaluatorConfigEntity]]:
        """
        List EvaluatorConfig entities with pagination.

        Args:
            dataset_id: Dataset ID
            page: Page number (1-indexed)
            size: Page size

        Returns:
            PagedResult containing list of EvaluatorConfigEntity and pagination metadata
        """
        # Build base query
        base_query = select(EvaluatorConfigEntity).where(
            EvaluatorConfigEntity.dataset_id == dataset_id,
            EvaluatorConfigEntity.tenant_id == tenant_id
        )

        # Get total count
        count_query = select(func.count()).select_from(base_query)
        total_result = await self.session.exec(count_query)
        total = total_result.one_or_none() or 0

        # Get paginated results
        offset = (page - 1) * size
        paginated_query = (
            base_query.order_by(EvaluatorConfigEntity.created_at.desc())
            .offset(offset)
            .limit(size)
        )
        results = await self.session.exec(paginated_query)
        configs = list(results.all())

        # Calculate pages
        pages = (total + size - 1) // size if total > 0 else 0

        return PagedResult(
            items=configs,
            total=total,
            pages=pages,
            page=page,
            size=size,
        )

    async def create_evaluator_config(
        self, dataset_id: str, config_data: EvaluatorConfigCreate, tenant_id: str
    ) -> EvaluatorConfigEntity:
        """
        Create a new EvaluatorConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            dataset_id: Dataset ID
            config_data: EvaluatorConfig creation data

        Returns:
            Created EvaluatorConfigEntity (not yet committed)
        """
        eval_config_entity = EvaluatorConfigEntity(
            name=config_data.name,
            type=config_data.type,
            dataset_id=dataset_id,
            model_id=config_data.model_id,
            case_sensitive=config_data.case_sensitive,
            ignore_punctuation=config_data.ignore_punctuation,
            tenant_id=tenant_id,
        )

        self.session.add(eval_config_entity)

        try:
            # Flush to get the ID, but don't commit
            await self.session.flush()
            await self.session.refresh(eval_config_entity)

            logger.info(
                f"Created EvaluatorConfig entity: {eval_config_entity.id} (name: {eval_config_entity.name})"
            )
            return eval_config_entity

        except IntegrityError as e:
            logger.error(f"IntegrityError when creating EvaluatorConfig: {e.orig}")
            raise ValueError(f"Evaluator configuration creation failed: {e}") from e

    async def update_evaluator_config(
        self, config_id: str, update_data: EvaluatorConfigCreate, tenant_id: str
    ) -> EvaluatorConfigEntity:
        """
        Update an existing EvaluatorConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_id: EvaluatorConfig entity ID
            update_data: Updated EvaluatorConfig data

        Returns:
            Updated EvaluatorConfigEntity (not yet committed)

        Raises:
            ValueError: If EvaluatorConfig entity not found
        """
        result = await self.session.exec(select(EvaluatorConfigEntity).where(EvaluatorConfigEntity.id == config_id, EvaluatorConfigEntity.tenant_id == tenant_id))
        eval_config = result.first()
        if not eval_config:
            raise ValueError(f"Evaluator configuration '{config_id}' does not exist.")

        logger.info(f"Updating EvaluatorConfig {config_id} with data: {update_data}")

        # Update fields
        if update_data.name is not None:
            eval_config.name = update_data.name
        if update_data.type is not None:
            eval_config.type = update_data.type
        if update_data.model_id is not None:
            eval_config.model_id = update_data.model_id
        if update_data.case_sensitive is not None:
            eval_config.case_sensitive = update_data.case_sensitive
        if update_data.ignore_punctuation is not None:
            eval_config.ignore_punctuation = update_data.ignore_punctuation

        self.session.add(eval_config)

        # Flush to ensure changes are staged
        await self.session.flush()
        await self.session.refresh(eval_config)

        logger.info(
            f"Updated EvaluatorConfig entity: {eval_config.id} (name: {eval_config.name})"
        )
        return eval_config

    async def delete_evaluator_config(self, config_id: str, tenant_id: str) -> None:
        """
        Delete an EvaluatorConfig entity.
        Note: Caller is responsible for committing the session.

        Args:
            config_id: EvaluatorConfig entity ID

        Raises:
            ValueError: If EvaluatorConfig entity not found
        """
        result = await self.session.exec(select(EvaluatorConfigEntity).where(EvaluatorConfigEntity.id == config_id, EvaluatorConfigEntity.tenant_id == tenant_id))
        eval_config = result.first()
        if not eval_config:
            raise ValueError(f"Evaluator configuration '{config_id}' does not exist.")

        # Delete from database (staged, not committed)
        await self.session.delete(eval_config)

        # Flush to ensure deletion is staged
        await self.session.flush()

        logger.info(
            f"Deleted EvaluatorConfig entity: {config_id} (name: {eval_config.name})"
        )
