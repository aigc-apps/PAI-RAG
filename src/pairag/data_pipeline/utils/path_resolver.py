from abc import abstractmethod


class MountPathResolver:
    @abstractmethod
    def resolve_destination_path(self, uri: str) -> str:
        """
        Resolve the path to the local file system.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_source_url(self, path: str) -> str:
        """
        Resolve the path to the local file system.
        """
        raise NotImplementedError


# 本地运行，文件路径即为uri
class LocalPathResolver(MountPathResolver):
    def resolve_destination_path(self, uri: str) -> str:
        """
        Resolve the path to the local file system.
        """
        return uri

    def resolve_source_url(self, path: str) -> str:
        """
        Resolve the path to the local file system.
        """
        return path
