import asyncio
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Type,
    Union,
    Coroutine,
    Tuple,
    cast,
    get_origin,
    get_args,
    List,
)
import typing
import concurrent.futures
from inspect import signature


from pydantic import BaseModel, Field, create_model
from .types import ToolMetadata, ToolOutput

AsyncCallable = Callable[..., Awaitable[Any]]


def asyncio_run(coro: Coroutine) -> Any:
    """
    Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    If an event loop is already running, uses threading to run in a separate thread.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # Check if the loop is already running
        if loop.is_running():
            # If loop is already running, run in a separate thread

            def run_coro_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_coro_in_thread)
                return future.result()
        else:
            # If we're here, there's an existing loop but it's not running
            return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )


def sync_to_async(fn: Callable[..., Any]) -> AsyncCallable:
    """Sync to async."""

    async def _async_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    return _async_wrapped_fn


def async_to_sync(func_async: AsyncCallable) -> Callable:
    """Async to sync."""

    def _sync_wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        return asyncio_run(func_async(*args, **kwargs))  # type: ignore[arg-type]

    return _sync_wrapped_fn


# The type that the callback can return: either a ToolOutput instance or a string to override the content.
CallbackReturn = Optional[Union[ToolOutput, str]]


def create_schema_from_function(
    name: str,
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
    ignore_fields: Optional[List[str]] = None,
) -> Type[BaseModel]:
    """Create schema from function."""
    fields = {}
    ignore_fields = ignore_fields or []
    params = signature(func).parameters
    for param_name in params:
        if param_name in ignore_fields:
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default
        description = None

        if get_origin(param_type) is typing.Annotated:
            args = get_args(param_type)
            param_type = args[0]
            if isinstance(args[1], str):
                description = args[1]
            elif isinstance(args[1], Field):
                description = args[1].description

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (param_type, Field(description=description))
        elif isinstance(param_default, Field):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (
                param_type,
                Field(default=param_default, description=description),
            )

    additional_fields = additional_fields or []
    for field_info in additional_fields:
        if len(field_info) == 3:
            field_info = cast(Tuple[str, Type, Any], field_info)
            field_name, field_type, field_default = field_info
            fields[field_name] = (field_type, Field(default=field_default))
        elif len(field_info) == 2:
            # Required field has no default value
            field_info = cast(Tuple[str, Type], field_info)
            field_name, field_type = field_info
            fields[field_name] = (field_type, Field())
        else:
            raise ValueError(
                f"Invalid additional field info: {field_info}. "
                "Must be a tuple of length 2 or 3."
            )

    return create_model(name, **fields)  # type: ignore


class FunctionTool:
    """
    Function Tool.

    A tool that takes in a function, optionally handles workflow context,
    and allows the use of callbacks. The callback can return a new ToolOutput
    to override the default one or a string that will be used as the final content.
    """

    def __init__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[ToolMetadata] = None,
        async_fn: Optional[AsyncCallable] = None,
        callback: Optional[Callable[..., Any]] = None,
        async_callback: Optional[Callable[..., Any]] = None,
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if fn is None and async_fn is None:
            raise ValueError("fn or async_fn must be provided.")

        # Handle function (sync and async)
        self._real_fn = fn or async_fn
        if async_fn is not None:
            self._async_fn = async_fn
            self._fn = fn or async_to_sync(async_fn)
        else:
            assert fn is not None
            if inspect.iscoroutinefunction(fn):
                self._async_fn = fn
                self._fn = async_to_sync(fn)
            else:
                self._fn = fn
                self._async_fn = sync_to_async(fn)

        # Determine if the function requires context by inspecting its signature
        fn_to_inspect = fn or async_fn
        assert fn_to_inspect is not None
        if metadata is None:
            raise ValueError("metadata must be provided")
        self._metadata = metadata

        # Handle callback (sync and async)
        self._callback = None
        if callback is not None:
            self._callback = callback
        elif async_callback is not None:
            self._callback = async_to_sync(async_callback)

        self._async_callback = None
        if async_callback is not None:
            self._async_callback = async_callback
        elif self._callback is not None:
            self._async_callback = sync_to_async(self._callback)

        self.partial_params = partial_params or {}

    def _run_sync_callback(self, result: Any) -> CallbackReturn:
        """
        Runs the sync callback, if provided, and returns either a ToolOutput
        to override the default output or a string to override the content.
        """
        if self._callback:
            ret: CallbackReturn = self._callback(result)
            return ret
        return None

    async def _run_async_callback(self, result: Any) -> CallbackReturn:
        """
        Runs the async callback, if provided, and returns either a ToolOutput
        to override the default output or a string to override the content.
        """
        if self._async_callback:
            ret: CallbackReturn = await self._async_callback(result)
            return ret
        return None

    @classmethod
    def from_defaults(
        cls,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        fn_schema: Optional[Type[BaseModel]] = None,
        async_fn: Optional[AsyncCallable] = None,
        tool_metadata: Optional[ToolMetadata] = None,
        callback: Optional[Callable[[Any], Any]] = None,
        async_callback: Optional[AsyncCallable] = None,
        partial_params: Optional[Dict[str, Any]] = None,
    ) -> "FunctionTool":
        partial_params = partial_params or {}

        if tool_metadata is None:
            fn_to_parse = fn or async_fn
            assert fn_to_parse is not None, "fn must be provided"
            name = name or fn_to_parse.__name__
            docstring = fn_to_parse.__doc__

            # Get function signature
            fn_sig = inspect.signature(fn_to_parse)

            # Handle Field defaults
            fn_sig = fn_sig.replace(
                parameters=[
                    (
                        param.replace(default=inspect.Parameter.empty)
                        if isinstance(param.default, Field)
                        else param
                    )
                    for param in fn_sig.parameters.values()
                    if param.name not in partial_params
                ]
            )

            description = description or f"{name}{fn_sig}\n{docstring}"
            if fn_schema is None:
                ignore_fields = partial_params.keys()

                fn_schema = create_schema_from_function(
                    f"{name}",
                    fn_to_parse,
                    additional_fields=None,
                    ignore_fields=ignore_fields,
                )
            tool_metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=fn_schema,
                return_direct=return_direct,
            )
        return cls(
            fn=fn,
            metadata=tool_metadata,
            async_fn=async_fn,
            callback=callback,
            async_callback=async_callback,
            partial_params=partial_params,
        )

    @property
    def metadata(self) -> ToolMetadata:
        """Metadata."""
        return self._metadata

    @property
    def fn(self) -> Callable[..., Any]:
        """Function."""
        return self._fn

    @property
    def async_fn(self) -> AsyncCallable:
        """Async function."""
        return self._async_fn

    @property
    def real_fn(self) -> Union[Callable[..., Any], AsyncCallable]:
        """Real function."""
        if self._real_fn is None:
            raise ValueError("Real function is not set!")

        return self._real_fn

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        all_kwargs = {**self.partial_params, **kwargs}
        return self.call(*args, **all_kwargs)

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Sync Call."""
        all_kwargs = {**self.partial_params, **kwargs}

        raw_output = self._fn(*args, **all_kwargs)
        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            content=str(raw_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": all_kwargs},
            raw_output=raw_output,
        )
        # Check for a sync callback override
        callback_result = self._run_sync_callback(raw_output)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                return callback_result
            else:
                # Assume callback_result is a string to override the content.
                return ToolOutput(
                    content=str(callback_result),
                    tool_name=self.metadata.name,
                    raw_input={"args": args, "kwargs": all_kwargs},
                    raw_output=raw_output,
                )
        return default_output

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Async Call."""
        all_kwargs = {**self.partial_params, **kwargs}
        raw_output = await self._async_fn(*args, **all_kwargs)
        # Default ToolOutput based on the raw output
        default_output = ToolOutput(
            content=str(raw_output),
            tool_name=self.metadata.name,
            raw_input={"args": args, "kwargs": all_kwargs},
            raw_output=raw_output,
        )
        # Check for an async callback override
        callback_result = await self._run_async_callback(raw_output)
        if callback_result is not None:
            if isinstance(callback_result, ToolOutput):
                return callback_result
            else:
                # Assume callback_result is a string to override the content.
                return ToolOutput(
                    content=str(callback_result),
                    tool_name=self.metadata.name,
                    raw_input={"args": args, "kwargs": all_kwargs},
                    raw_output=raw_output,
                )
        return default_output
