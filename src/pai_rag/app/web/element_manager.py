from typing import TYPE_CHECKING, Dict, Generator, List, Tuple

if TYPE_CHECKING:
    from gradio.components import Component


class ElementManager:
    def __init__(self) -> None:
        self._id_to_elem: Dict[str, "Component"] = {}
        self._elem_to_id: Dict["Component", str] = {}

    def add_elems(self, elem_dict: Dict[str, "Component"]) -> None:
        r"""
        Adds elements to manager.
        """
        for elem_id, elem in elem_dict.items():
            self._id_to_elem[elem_id] = elem
            self._elem_to_id[elem] = elem_id

    def get_elem_list(self) -> List["Component"]:
        r"""
        Returns the list of all elements.
        """
        return list(self._id_to_elem.values())

    def get_elem_iter(self) -> Generator[Tuple[str, "Component"], None, None]:
        r"""
        Returns an iterator over all elements with their names.
        """
        for elem_id, elem in self._id_to_elem.items():
            yield elem_id.split(".")[-1], elem

    def get_elem_by_id(self, elem_id: str) -> "Component":
        r"""
        Gets element by id.

        Example: top.lang, train.dataset
        """
        return self._id_to_elem[elem_id]

    def get_id_by_elem(self, elem: "Component") -> str:
        r"""
        Gets id by element.
        """
        return self._elem_to_id[elem]


elem_manager = ElementManager()
