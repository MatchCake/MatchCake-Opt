import glob
import os
from pathlib import Path
from typing import List, Optional

from .metadata_file import MetadataFile


class OutputFolder:
    def __init__(self, path: str, metadata_file: Optional[MetadataFile] = None):
        self._path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if metadata_file is None:
            metadata_file = MetadataFile(self.path)
        self._metadata_file = metadata_file

    def update_metadata(self, other: Optional[dict] = None, **kwargs):
        """
        Updates the metadata of the instance with information from the provided dictionary
        and any additional keyword arguments. If a dictionary is not provided, an empty
        dictionary is used. The method updates the instance's metadata with both the
        values from the given dictionary and the keyword arguments.

        :param other: Optional initial dictionary of metadata to update. Defaults to None.
        :type other: Optional[dict]
        :param kwargs: Additional metadata key-value pairs to update.
        :return: None
        """
        other = other or {}
        other.update(kwargs)
        self.metadata_file.update(other)

    def gather_files(
        self,
        pattern: str = "*",
    ) -> List[Path]:
        """
        Gather files in the output folder matching the given pattern.

        :param pattern: The glob pattern to match files.
        :return: A list of file paths matching the pattern.
        """
        return list(self.path.glob(pattern))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata_file(self) -> MetadataFile:
        return self._metadata_file
