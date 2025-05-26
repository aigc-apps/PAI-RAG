from types import LambdaType
from typing import Any, Dict, Optional, Type, Union, Callable, List, overload, TypeVar


_MISSING_SERVICE = object()


T = TypeVar("T")


def get_origin_type(type_name: Type) -> Optional[Type]:
    return getattr(type_name, "__origin__", None)


def get_type_args(type_name: Type) -> List[Type]:
    return getattr(type_name, "__args__", [])


def is_optional(type_name: Type) -> bool:
    return (
        get_origin_type(type_name) is Union
        and bool(get_type_args(type_name))
        and get_type_args(type_name)[-1] is type(None)  # noqa
    )


def unpack_optional(type_name: Type) -> Type:
    return get_type_args(type_name)[0]


class Container:
    def __init__(self):
        self._memoized_services: Dict[Union[str, Type], Any] = {}
        self._services: Dict[Union[str, Type], Any] = {}
        self._factories: Dict[Union[str, Type], Callable[[Container], Any]] = {}
        self._aliases: Dict[Union[str, Type], List[Union[str, Type]]] = {}

    def __setitem__(self, key: Union[str, Type], value: Any) -> None:
        self._services[key] = value

        if key in self._memoized_services:
            del self._memoized_services[key]

    def add_alias(self, name: Union[str, Type], target: Union[str, Type]):
        if List[target] in self._memoized_services:  # type: ignore
            del self._memoized_services[List[target]]  # type: ignore

        if name not in self._aliases:
            self._aliases[name] = []
        self._aliases[name].append(target)

    @overload
    def __getitem__(self, key: str) -> Any:
        ...

    @overload
    def __getitem__(self, key: Type[T]) -> T:
        ...

    def __getitem__(self, key):
        if key in self._factories:
            return self._factories[key](self)

        if is_optional(key):
            return self[unpack_optional(key)]

        service = self._get(key)

        if service is not _MISSING_SERVICE:
            return service

        if key in self._aliases:
            unaliased_key = self._aliases[key][
                0
            ]  # By default return first aliased service
            if unaliased_key in self._factories:
                return self._factories[unaliased_key](self)
            service = self._get(unaliased_key)

        if service is not _MISSING_SERVICE:
            return service

        # Support aliasing
        if self._has_alias_list_for(key):
            result = [self._get(alias) for alias in self._aliases[key.__args__[0]]]  # type: ignore
            self._memoized_services[key] = result
            return result

        raise ValueError(f"Service '{key}' is not registered.")

    def _get(self, key: Union[str, Type]) -> Any:
        if key in self._memoized_services:
            return self._memoized_services[key]

        if key not in self._services:
            return _MISSING_SERVICE

        value = self._services[key]

        if isinstance(value, LambdaType) and value.__name__ == "<lambda>":
            self._memoized_services[key] = value(self)
            return self._memoized_services[key]

        return value

    def __contains__(self, key) -> bool:
        contains = (
            key in self._services or key in self._factories or key in self._aliases
        )

        if contains:
            return contains

        if self._has_alias_list_for(key):
            return True

        if is_optional(key):
            return unpack_optional(key) in self

        return False

    def _has_alias_list_for(self, key: Union[str, Type]) -> bool:
        return hasattr(key, "__origin__") and hasattr(key, "__args__") and key.__origin__ == list and key.__args__[0] in self._aliases  # type: ignore

    @property
    def factories(self) -> Dict[Union[str, Type], Callable[["Container"], Any]]:
        return self._factories

    def clear_cache(self) -> None:
        self._memoized_services = {}


di: Container = Container()
