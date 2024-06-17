import importlib
import logging
import os
import shutil

from enum import Enum
from pathlib import Path
from typing import NewType, Optional, TypeAlias, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
debug = lambda *args: logger.debug(" ".join([str(arg) for arg in args]))

EXAMPLE_INFO = """Route: army-demo
Recorded Date (MM/DD/YYYY): 10/5/2023
Uploader: Admin
Weather Condition: Sunny, Green
Description: Recorded by Mihir and Thom"""

META_SUFFIX = ".meta.txt"
INFO_SUFFIX = ".info.txt"
ADMIN_ID = "admin"
UserId = NewType("UserId", str)

class User:
    """Represents a user, identified by a unique identifier."""
    GuestId = None

    def __init__(self, id: Union[UserId, "User"] = None):
        """Initializes a User instance.
        
        Args:
            id (Optional[UserId]): The unique identifier for the user. Defaults to None, representing a guest.
        """
        match id:
            case User() as user:
                self._id = user._id
            case _:
                self._id = id if id else User.GuestId

    @property
    def id(self) -> UserId:
        """Gets the user's unique identifier."""
        return self._id

    @id.setter
    def id(self, value):
        """Sets the user's unique identifier."""
        self._id = value

    def __eq__(self, other):
        """Checks equality with another User instance or a string id."""
        match other:
            case User():
                return self._id == other._id
            case str():
                return self._id == other
            case _:
                return NotImplemented

    def __repr__(self):
        """Represents the User instance as a string."""
        match self._id:
            case User.GuestId:
                return "Guest"
            case _:
                return f"<User: {str(self._id)}>"

class Permission(Enum):
    """Defines access permissions for files."""
    READ_ONLY = "read_only"
    POST_ONLY = "post_only"
    FULL_WRITE = "full_write"


class Visibility(Enum):
    """Defines visibility settings for files."""
    SHARED = "shared"
    PRIVATE = "private"


class FileType(Enum):
    """Defines the type of a file system object."""
    FILE = "file"
    DIRECTORY = "directory"


class AnalysisProgress(Enum):
    """Represents the progress of analysis on a file."""
    FULL = "full"
    HALF = "half"
    NONE = "none"

    def __eq__(self, other):
        if isinstance(other, AnalysisProgress):
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, AnalysisProgress):
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, AnalysisProgress):
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, AnalysisProgress):
            return self.value < other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, AnalysisProgress):
            return self.value <= other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)


CreatedBy = NewType("CreatedBy", User)
SpecificType = NewType("SpecificType", str)
MetaData: TypeAlias = Union[
    Permission, Visibility, FileType, AnalysisProgress, CreatedBy, SpecificType
]

class Action(Enum):
    """Defines possible actions that can be performed on files."""
    NEW_DIR = "new_dir"
    UPLOAD_FILE = "upload_file"
    RENAME = "rename"
    SHARE = "share"
    DELETE = "delete"
    DOWNLOAD = "download"
    ANALYZE = "analyze"
    EDIT = "edit"
    COPY = "copy"
    MOVE = "move"
    # WAIT = "wait"


class AccessMode(Enum):
    """Defines access modes based on user relation to the file."""
    OTHERS = "others"
    OWNER = "owner"


def appendText(path: Path, text: str) -> Path:
    """Appends text to the file name, before the file extension.

    Args:
        path (Path): The original file path.
        text (str): The text to append.

    Returns:
        Path: The new file path with the text appended.
    """
    return path.with_name(path.stem + text + path.suffix)


def getCopyName(path: Path) -> Path:
    """The function iteratively appends '_copy' to the base name, before the file extension, of the provided path
    until a non-existing path is generated.

    Args:
        path: A Path object representing the original file or directory.

    Returns:
        A Path object representing the new, unique path with '_copy' appended.
    """
    while path.exists():
        path = appendText(path, "_copy")
    return path


DATABASE_ROOT = "database"
QUERY_ROOT = "query"

reserved_names = [DATABASE_ROOT, QUERY_ROOT]


class QuetzalFile:
    """
    Represents a file or directory within the Quetzal file system, facilitating operations such as creation,
    renaming, sharing, and metadata management. This class serves as a base for specialized file types within Quetzal,
    providing a structured approach to handle common file-related tasks.

    Subclasses of `QuetzalFile` may implement custom behaviors for specific operations (e.g., copying, analyzing,
    sharing) by overriding the internal methods designed for these purposes. For example, see the `Video` class
    for implementations that tailor these operations to the needs of video file management within Quetzal.
    """
    
    FILE_DEFAULT_DESCRIPTION = "Uploader::= default\nRecorded Date (MM/DD/YYYY)::= default\nTime-of-day::= default\nWeather Condition::= default\nDescription::= default"
    PROJECT_DEFAULT_DESCRIPTION = "Route Location::= default\nLast Update (MM/DD/YYYY)::= default\nLast Edit by::= default\nDescription::= default"
    FILE_DEFAULT_META = "FileType::= file\nVisibility::= private\nPermission::= full_write\nAnalysisProgress::= none\nSpecificType::= Video\n"
    PROJECT_DEFAULT_META = (
        "FileType::= directory\nVisibility::= private\nPermission::= full_write\n"
    )
    USER_ROOT_META = (
        "FileType::= directory\nVisibility::= shared\nPermission::= read_only\n"
    )
    USER_ROOT_DESCRIPTION = "Description::=root directory for "
    ROOT_META = "FileType::= directory\nVisibility::= shared\nPermission::= read_only\nCreatedBy::= admin\nOwner::= admin"
    ROOT_DESCRIPTOIN = "Description::=database root"
    

    def __init__(
        self,
        path: Union[str, Path],  # path to the dir/file from root_dir
        root_dir: Union[str, Path],
        metadata_dir: Union[str, Path],
        user: Union[str, User] = User.GuestId,
        home: Union[str, Path] = "./",
        metadata: Optional[dict[str, MetaData]] = None,
        parent: 'QuetzalFile' = None,
    ):
        """        
        Args:
            path (Path): The path of the file or directory, relative to the root directory.
            root_dir (Path): The root directory path of the Quetzal file system.
            metadata_dir (Path): The directory path where metadata files are stored.
            user (User): The user who is interacting with the file.
            home (Path): The base home directory path (relative to rood directory). Defaults to "./".
            metadata (Optional[Dict[str, MetaData]]): The metadata associated with the file.
            parent (Optional[QuetzalFile]): The parent directory of the current file.
        """
        self._path = Path(path)
        self._root_dir = Path(root_dir)
        self._metadata_dir = Path(metadata_dir)
        self._user = User(user)
        self._home = Path(home)
        self._parent = parent

        # assert self._user != "example"

        if not (self.full_path).exists():
            if self._path == Path(self._user.id):
                os.makedirs(self.full_path, exist_ok=True)
                info_path = self.getDescriptionPath(self._path)
                info_path.write_text(self.USER_ROOT_DESCRIPTION)
                mata_path = self.getMetaDataPath(self._path)
                mata_path.write_text(self.USER_ROOT_META + "CreatedBy::= admin\n" )
            else:    
                raise FileNotFoundError(
                    f'File "{self._root_dir / self._path}" Do not Exist'
                )

        if not self.getMetaDataPath(self._path).exists():
            raise FileNotFoundError(
                f'Meta File "{self.getMetaDataPath(self._path)}" Do not Exist'
            )

        if not self._path.is_relative_to(self._home):
            raise ValueError(f'"{path}" is not relative to "{home}"')

        ## load_metadata
        self._parseMetadata(metadata)

        match self._path.parts:
            case [user_id, *_]:
                self._owner = User(user_id)
            case _:
                self._owner = User(ADMIN_ID)

        self._mode = (
            AccessMode.OWNER if self._owner == self._user else AccessMode.OTHERS
        )
        
        self._iterdir = None

    def __hash__(self):
        return hash(str(self.path))

    def __eq__(self, other):
        match other:
            case QuetzalFile():
                return self._path == other._path
            case str():
                return self._path == other
            case _:
                return NotImplemented

    @property
    def _abs_path(self) -> Path:
        """Full absolute path of the file"""
        return (self._root_dir / self._path).absolute()

    @property
    def path(self) -> Path:
        """Relative path to the file or directory from the "home" directory."""
        return self._path.relative_to(self._home)

    @property
    def name(self) -> str:
        """The name of the file or directory."""
        return self._path.name

    @property
    def type(self) -> FileType:
        """The type of the file, either FILE or DIRECTORY."""
        return self._type

    @property
    def full_path(self) -> Path:
        """The full filesystem path to the file or directory."""
        return self._root_dir / self._path

    @property
    def analysis_progress(self) -> AnalysisProgress:
        """The current progress of any ongoing analysis."""
        return self._analysis_progress

    @property
    def visibility(self) -> Visibility:
        """The visibility setting of the file, either SHARED or PRIVATE."""
        return self._visibility

    @property
    def permission(self) -> Permission:
        """The access permissions of the file."""
        return self._permission

    @property
    def createdBy(self) -> User:
        """The user who created the file."""
        return self._created_by.id

    @property
    def user(self) -> User:
        """The current user interacting with the file."""
        return self._user.id

    @property
    def home(self) -> Path:
        """The base home directory path."""
        return self._home

    @property
    def mode(self) -> AccessMode:
        """The access mode, indicating if the current user is the OWNER or OTHERS."""
        return self._mode

    @staticmethod
    def fromFile(file: "QuetzalFile", path: Union[str, Path], home: Optional[Union[str, Path]]=None) -> "QuetzalFile":
        """
        Creates a new instance of QuetzalFile based on an existing instance.

        Args:
            file: The QuetzalFile instance to base the new instance on.
            path: The path for the new QuetzalFile instance. Actual path relative to the root_dir will be "./home/path".
            home: The home directory for the new instance. If None, the home of the 'file' is used.

        Returns:
            A new QuetzalFile instance based on the provided file and path.
        """
        
        if home is None:    
            home = file._home
        else:
            home = Path(home)
        return QuetzalFile(
            path=home / path,
            root_dir=file._root_dir,
            metadata_dir=file._metadata_dir,
            user=file._user,
            home=home
        )

    def loadMetaData(self, file: Union[str, Path]) -> dict[str, MetaData]:
        """
        Loads and returns the metadata for the specified file.
        """
        metadata_path = self.getMetaDataPath(file)
        return self._loadMetaData(metadata_path)
    
    def iterdir(
        self,
        sharedOnly=False,
        directoryOnly=False,
    ) -> list["QuetzalFile"]:
        """
        Lists the contents of the directory represented by this QuetzalFile.

        Args:
            sharedOnly: If True, only shared files/directories are listed.
            directoryOnly: If True, only directories are listed.

        Returns:
            A list of QuetzalFile objects representing the contents of the directory.
        """
        
        assert self._type == FileType.DIRECTORY
        
        if self._iterdir:
            return self._iterdir

        directories = []
        files = []
        directory_path = self._root_dir / self._path

        sorted_items = sorted(directory_path.iterdir(), key=lambda x: x.stem.lower())
        for item in sorted_items:
            # Load MetaData of the file
            try:
                metadata = self.loadMetaData(item.relative_to(self._root_dir))
            except:
                continue

            # Filter Only Directory
            isdir = metadata["FileType"] == FileType.DIRECTORY
            if directoryOnly and not isdir:
                continue

            # Filter only shared
            if sharedOnly and metadata["Visibility"] != Visibility.SHARED:
                continue
            
            file = QuetzalFile._instantiateFile(
                path=item.relative_to(self._root_dir),
                root_dir=self._root_dir,
                metadata_dir=self._metadata_dir,
                user=self._user,
                home=self._home,
                metadata=metadata,
                parent=self,
            )

            # Sort to have directory first
            if isdir:
                directories.append(file)
            else:
                files.append(file)

        self._iterdir = directories + files
        return self._iterdir

    def perform(self, action: Action, input: dict):
        """
        Performs the specified "Action" on this QuetzalFile object.

        Args:
            action: The action to perform, as defined by the Action enum.
            input: A dictionary containing the input parameters required for the action.

        Returns:
            A string message indicating the outcome of the action, or None if no action was performed.
        """
        
        self._iterdir = None
        if self._parent:
            self._parent._iterdir = None
        match action:
            case Action.NEW_DIR:
                return self._newDirectory(**input)
            case Action.UPLOAD_FILE:
                return self._upload(**input)
            case Action.RENAME:
                return self._rename(**input)
            case Action.SHARE:
                return self._share(**input)
            case Action.DELETE:
                return self._delete(**input)
            case Action.ANALYZE:
                return self._analyze(**input)
            case Action.EDIT:
                return self._editDescription(**input)
            case Action.COPY:
                return self._copy(**input)
            case Action.MOVE:
                return self._move(**input)
            case _:
                raise NotImplemented("No Action Implemneted")
    
    @staticmethod
    def _instantiateFile(
        path: Path,
        root_dir: Path,
        metadata_dir: Path,
        user: User,
        home: Path,
        metadata: dict[str, MetaData],
        parent: 'QuetzalFile',
    ) -> "QuetzalFile":
        """
        Instantiate a file into "Specific Type", one of subclass of QuetzalFile.
        
        Args:
            path (Path): The path of the file or directory, relative to the root directory.
            root_dir (Path): The root directory path of the Quetzal file system.
            metadata_dir (Path): The directory path where metadata files are stored.
            user (User): The user who is interacting with the file.
            home (Path): The base home directory path (relative to rood directory). Defaults to "./".
            metadata (Optional[Dict[str, MetaData]]): The metadata associated with the file.
            parent (Optional[QuetzalFile]): The parent directory of the current file.
        
        Returns:
            Instance of "Specific Type" (subclass of QuetzalFile)
        """
        
        specific_file_class = None
        specific_type = metadata.get("SpecificType", None)
        
        if specific_type:
            module = importlib.import_module(f"quetzal.dtos.{specific_type.lower()}")
            specific_file_class = getattr(module, specific_type)

        if specific_file_class and issubclass(specific_file_class, QuetzalFile):
            return specific_file_class(
                path=path,
                root_dir=root_dir,
                metadata_dir=metadata_dir,
                user=user,
                home=home,
                metadata=metadata,
                parent=parent,
            )
        else:
            return QuetzalFile(
                path=path,
                root_dir=root_dir,
                metadata_dir=metadata_dir,
                user=user,
                home=home,
                metadata=metadata,
                parent=parent
            )
            
    @staticmethod
    def _loadMetaData(metadata_path: Path) -> dict[str, MetaData]:
        """
        Load Queztalfile metadata from the given path
        
        Args:
            metadata_dir (Path): The directory path where metadata files are stored.
        
        Returns:
            dict[str, MetaData]
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError

        with open(metadata_path, "r") as file:
            data = file.read().splitlines()
        metadata = {
            line.split("::=")[0].strip(): line.split("::=")[1].strip()
            for line in data
            if line.strip()
        }

        metadata["CreatedBy"] = CreatedBy(User(metadata.get("CreatedBy", None)))
        metadata["FileType"] = FileType(metadata.get("FileType", "file"))
        metadata["Visibility"] = Visibility(metadata.get("Visibility", "private"))
        metadata["AnalysisProgress"] = AnalysisProgress(
            metadata.get("AnalysisProgress", "none")
        )
        metadata["Permission"] = Permission(metadata.get("Permission", "full_write"))
        metadata["SpecificType"] = SpecificType(metadata.get("SpecificType", None))

        return metadata

    def _parseMetadata(self, metadata: Optional[dict[str, MetaData]]):
        """
        Assign Queztalfile metadata into attribute variables
        
        Args:
            metadata (Optional[dict[str, MetaData]]): metadata
        """
        if metadata is None:
            metadata = self.loadMetaData(self._path)

        self._created_by = metadata["CreatedBy"]
        self._type = metadata["FileType"]
        self._visibility = metadata["Visibility"]
        self._analysis_progress = metadata["AnalysisProgress"]
        self._permission = metadata["Permission"]

    
    
    def _updateMetaForRename(self, new_path: Union[str, Path]):
        """
        Updates the metadata and description files' paths for the current QuetzalFile
        instance to reflect a change in the file or directory's name (rename action).

        This method is invoked internally to handle the renaming of metadata and description
        files associated with the QuetzalFile, ensuring consistency within the filesystem.

        Args:
            new_path (Union[str, Path]): The new path (name) for the file or directory, reflecting the rename.
        """
        orig_path_full = self._metadata_dir / self._path
        new_path_full = self._metadata_dir / new_path
        
        ## New Directory if needed
        if self._type == FileType.DIRECTORY:
            if orig_path_full.exists():
                os.rename(orig_path_full, new_path_full)
            
        ## Rename Metadata
        metadata_path = self._getMetaDataPath(orig_path_full)
        new_metadata_path = self._getMetaDataPath(new_path_full)
        if metadata_path.exists():
            os.rename(metadata_path, new_metadata_path)
        else:
            with open(new_metadata_path, "w") as file:
                file.write(
                    self.PROJECT_DEFAULT_META
                    if self._type == FileType.DIRECTORY
                    else self.FILE_DEFAULT_META
                )

        ## Rename Description
        description_path = self._getDescriptionPath(orig_path_full)
        new_description_path = self._getDescriptionPath(new_path_full)
        if description_path.exists():
            os.rename(description_path, new_description_path)
        else:
            with open(new_description_path, "w") as file:
                file.write(
                    self.PROJECT_DEFAULT_DESCRIPTION
                    if self._type == FileType.DIRECTORY
                    else self.FILE_DEFAULT_DESCRIPTION
                )
    
    
    def _rename(self, new_file_name: Union[str, Path]) -> str:
        """
        Renames the current QuetzalFile (file or directory) to the new specified name and updates
        the metadata accordingly.
        
        For given new_file_name, only the name of the file will be considered. Parent_name, and suffix of the name
        will be modified to match the original file.

        Args:
            new_file_name (Union[str, Path]): The new name for the file or directory.

        Returns:
            A confirmation message indicating the successful rename.
        """
        
        new_file_name = Path(new_file_name)
        assert (
            self._mode == AccessMode.OWNER or self._permission == Permission.FULL_WRITE
        )
        assert new_file_name.stem not in reserved_names

        debug(f"\n\n\t{self.name} called on rename {new_file_name}\n")

        # Validate suffix
        if new_file_name.suffix != self._path.suffix:
            new_file_name = Path(new_file_name.name + self._path.suffix)

        # Rename
        new_path = self._path.parent / new_file_name
        new_path_abs = self._root_dir / new_path
        if new_path_abs.exists():
            raise FileExistsError(
                f'File/directory with name "{new_file_name}" already exist at the destination.'
            )
        os.rename(self._abs_path, new_path_abs)
        self._updateMetaForRename(new_path=new_path)

        org_name = self.name
        self._path = new_path
        return f'"{org_name}" renamed to "{self.name}"'


    def _updateMetaForNewFile(
        self, target_path: Path, file_name: Path, meta_data: str, description: str, isDir: bool=False
    ):
        """
        Creates and updates the metadata and description for a new file or directory
        that is being added to the filesystem.

        Args:
            target_path (Path): The target path where the new file or directory is located.
            file_name (Path): The name of the new file or directory.
            meta_data (str): The metadata to be written for the new file or directory.
            description (str): The description to be written for the new file or directory.
            isDir (bool): A flag indicating whether the new entity is a directory. Defaults to False.
        """
        # Crate New Meta Data, Description, and Directory
        new_file_metadata = self._metadata_dir / target_path / file_name
        if isDir:
            os.makedirs(new_file_metadata, exist_ok=True)

        new_file_metadata_path = self._getMetaDataPath(new_file_metadata)
        with open(new_file_metadata_path, "w") as file:
            file.write(meta_data)

        new_file_description_path = self._getDescriptionPath(new_file_metadata)
        with open(new_file_description_path, "w") as file:
            file.write(description)


    def _newDirectory(self, dir_name: Union[str, Path]) -> str:
        """
        Creates a new directory under the current QuetzalFile path and updates the metadata and
        description for it.

        Args:
            dir_name (Union[str, Path]): The name of the new directory to create.

        Returns:
            A confirmation message indicating the successful creation of the directory.
        """
        dir_name = Path(dir_name)
        assert (
            self._mode == AccessMode.OWNER or self._permission != Permission.READ_ONLY
        )
        assert self._type == FileType.DIRECTORY
        assert dir_name.stem not in reserved_names

        debug(f"\n\n\t{self.name} called on New dir {dir_name}\n")
        
        new_dir_path = self._root_dir / self._path / dir_name
        if new_dir_path.exists():
            raise FileExistsError(
                f'Directory with name "{dir_name}" already exist at the destination.'
            )
        os.makedirs(new_dir_path, exist_ok=False)

        self._updateMetaForNewFile(
            self._path,
            dir_name,
            self.PROJECT_DEFAULT_META + "CreatedBy::= " + self._user.id + "\n",
            self.PROJECT_DEFAULT_DESCRIPTION,
            isDir=True
        )

        return f'"{dir_name}" Created'
    

    def _upload(self, uploaded_files: Union[str, Path]) -> str:
        """
        Uploads and saves the specified files to the current QuetzalFile directory, updating
        the metadata for each uploaded file.
        
        The method should fail when called on QuetzalFile with "File" filetype.

        Args:
            uploaded_files (Union[str, Path]): The files to be uploaded.

        Returns:
            A confirmation message indicating the number of files successfully uploaded.
        """
        assert self._mode == AccessMode.OWNER or self._permission != Permission.READ_ONLY
        assert self._type == FileType.DIRECTORY
        
        for uploaded_file in uploaded_files:
            assert Path(uploaded_file.name).stem not in reserved_names
            debug(f"\n\n\t{self.name} called on upload {uploaded_file.name}\n")

            dest: Path = self._root_dir / self._path / uploaded_file.name
            orig_name = dest
            start_num = 0
            while dest.exists():
                dest = appendText(orig_name, f"_{start_num}")
                start_num += 1

            with open(dest, mode="wb") as w:
                w.write(uploaded_file.getvalue())

            self._updateMetaForNewFile(
                self._path,
                dest.name,
                self.FILE_DEFAULT_META + "CreatedBy::= " + self._user.id + "\n",
                self.FILE_DEFAULT_DESCRIPTION,
            )

        return f"{len(uploaded_files)} files successfully uploaded"


    def _updateMetaForShare(self, shared: Visibility, permission: Permission):
        """
        Updates the metadata for the QuetzalFile (file or directory) to reflect new sharing settings,
        including visibility and permissions.

        Args:
            shared (Visibility): The new visibility setting for the file or directory.
            permission (Permission): The new permission setting for the file or directory.
        """
        metadata_path = self.getMetaDataPath(self._path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r+") as file:
                metadata = file.read()
                metadata = metadata.replace(
                    f"Visibility::= {self._visibility.value}",
                    f"Visibility::= {shared.value}",
                )
                metadata = metadata.replace(
                    f"Permission::= {self._permission.value}",
                    f"Permission::= {permission.value}",
                )
                file.seek(0)
                file.write(metadata)
                file.truncate()

        self._visibility = shared
        self._permission = permission


    def _share(self, shared: Visibility, permission: Permission) -> str:
        """
        Applies new sharing settings to the file or directory, affecting its visibility
        and access permissions, and updates the metadata accordingly.

        Args:
            shared (Visibility): The new visibility setting.
            permission (Permission): The new access permission setting.

        Returns:
            A confirmation message indicating the updated sharing settings.
        """
        assert (
            self._mode == AccessMode.OWNER or self._permission == Permission.FULL_WRITE
        )

        if shared == self._visibility and permission == self._permission:
            return None

        debug(f"\n\n\t{self.name} called on share {shared}:{permission}\n")

        self._updateMetaForShare(shared, permission)

        # If the QuetzalFile is a directory, apply the changes to all sub-projects and video files
        if self._type == FileType.DIRECTORY:
            for subfile in self.iterdir():
                subfile._updateMetaForShare(shared, permission)

        return f'"{self.name}" Sharing Setting Updated'

    def _updateMetaForDelete(self):
        """
        Cleans up the metadata and description files associated with the file or directory
        that is being deleted.
        """
        orig_path_full = self._metadata_dir / self._path
        metadata_path = self._getMetaDataPath(orig_path_full)
        description_path = self._getDescriptionPath(orig_path_full)

        if metadata_path.exists():
            os.remove(metadata_path)
        if description_path.exists():
            os.remove(description_path)
        if orig_path_full.exists():
            shutil.rmtree(orig_path_full)

    def _delete(self) -> str:
        """
        Deletes the current file or directory and cleans up its associated metadata
        and description files.

        Returns:
            A confirmation message indicating the successful deletion.
        """
        assert (
            self._mode == AccessMode.OWNER or self._permission == Permission.FULL_WRITE
        )

        debug(f"\n\n\t{self.name} called on delete\n")

        if self._type == FileType.FILE:
            os.remove(self._root_dir / self._path)
        else:  # For directories
            shutil.rmtree(self._root_dir / self._path)

        self._updateMetaForDelete()

        return f'"{self.name}" Deleted'

    def _updateMetaForAnalyze(self, new_progress: AnalysisProgress):
        """
        Updates the metadata for the file to reflect a new state of analysis progress.

        Args:
            new_progress (AnalysisProgress): The new analysis progress state to update the metadata with.
        """
        
        orig_path_full = self._metadata_dir / self._path
        metadata_path = self._getMetaDataPath(orig_path_full)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r+") as file:
                metadata = file.read()
                metadata = metadata.replace(
                    f"AnalysisProgress::= {self._analysis_progress.value}",
                    f"AnalysisProgress::= {new_progress.value}",
                )
                file.seek(0)
                file.write(metadata)
                file.truncate()

    def _syncAnalysisState(self):
        """
        Synchronizes the analysis state of the file with the current status in the
        analysis engine or database, updating the metadata as necessary.
        """
        assert self._type == FileType.FILE
        
        return

    def _analyze(self, option: AnalysisProgress, engine=None, device=None) -> str:
        """
        Initiat es analysis of the video file with the specified options, using the
        given analysis engine and device.

        Args:
            option (AnalysisProgress): The desired level of analysis to be performed.
            engine: The analysis engine to use for the analysis. Defaults to None.
            device: The cuda device to perform the analysis on. Defaults to None.

        Returns:
            A confirmation message indicating the completion of the analysis.
        """     
        
        assert (
            self._mode == AccessMode.OWNER or self._permission != Permission.READ_ONLY
        )
        debug(f"\n\t{self.name} called on analyze {option}\n")

        if option == None:
            return None

        return f'"{self.name}" Analysis Done'


    def _editDescription(self, value: str) -> str:
        """
        Updates the description for the file or directory with the given value.

        Args:
            value (str): The new description to be applied to the file or directory.

        Returns:
            A confirmation message indicating the successful update of the description.
        """
        assert (
            self._mode == AccessMode.OWNER or self._permission == Permission.FULL_WRITE
        )

        debug(f"\n\t{self.name} called on editMetaData {value}\n")
        description_file_path = self.getDescriptionPath(self._path)

        # Overwrite the existing description with the new value
        with open(description_file_path, "w") as file:
            file.write(value)

        return f'"{self.name}" Edit Success'

    
    def _updateMetaForCopy(self, dest: Path):
        """
        Updates the metadata for a file or directory when it is copied to a new location.

        Args:
            dest: The destination path where the file or directory has been copied.
        """
        source_metadata = self._metadata_dir / self._path
        dest_metadata = self._metadata_dir / dest
        
        if self._type == FileType.DIRECTORY:
            shutil.copytree(source_metadata, dest_metadata)
        
        source_metadata_path = self._getMetaDataPath(source_metadata)
        dest_metadata_path = self._getMetaDataPath(dest_metadata)
        shutil.copy2(source_metadata_path, dest_metadata_path)

        source_desc_path = self._getDescriptionPath(source_metadata)
        dest_desc_path = self._getDescriptionPath(dest_metadata)
        shutil.copy2(source_desc_path, dest_desc_path)
        
    
    def _copy(self, dest_dir: "QuetzalFile") -> str:
        """
        Copies the current file or directory to a new destination within the Quetzal file system, updating metadata
        accordingly.

        This method copies the file/directory represented by this instance to the specified destination directory,
        also handling metadata and description files to reflect the new location.

        Args:
            dest_dir (QueztalFile): The destination `QuetzalFile` directory where this file/directory is to be copied.

        Returns:
            A string message indicating the successful copy operation.
        """
        debug(f"{self.name} called on copy to ", dest_dir)
        assert dest_dir._type == FileType.DIRECTORY

        # Adjust Destination relative Path
        destination_path = dest_dir._abs_path
        dest_dir: Path = destination_path.relative_to(self._root_dir.absolute())

        # Rename destination if it is duplicates
        dest = getCopyName(self._root_dir / dest_dir / self._path.name)
        
        # Copy target File
        source = self._root_dir / self._path
        if self._type == FileType.DIRECTORY:
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)
        
        # Update metaData
        self._updateMetaForCopy(dest.relative_to(self._root_dir))
        
        return f'"{self.name}" copied to "{dest_dir}"'
            

    def _updateMetaForMove(self, dest_dir: Path):
        """
        Updates the metadata for a file or directory when it is moved to a new location.

        This includes moving the metadata and description files to reflect the new file path and updating any
        directory-specific metadata if the moved entity is a directory.

        Args:
            dest_dir (Path): The destination path where the file or directory has been moved.
        """
        source_metadata = self._metadata_dir / self._path
        
        if self._type == FileType.DIRECTORY:
            dest_metadata = self._metadata_dir / dest_dir
            shutil.move(source_metadata, dest_metadata)
        else:
            dest_metadata = self._metadata_dir / dest_dir / self._path.name

        source_metadata_path = self._getMetaDataPath(source_metadata)
        dest_metadata_path = self._getMetaDataPath(dest_metadata)
        shutil.move(source_metadata_path, dest_metadata_path)

        source_desc_path = self._getDescriptionPath(source_metadata)
        dest_desc_path = self._getDescriptionPath(dest_metadata)
        shutil.move(source_desc_path, dest_desc_path)
        
    
    def _move(self, dest_dir: "QuetzalFile") -> str:
        """
        Moves the current file or directory to a new destination within the Quetzal file system, updating metadata
        accordingly.

        This method moves the file/directory represented by this instance to the specified destination directory,
        handling metadata and description files to reflect the new location. It also ensures that the move does not
        violate any system rules, such as moving into a subdirectory of itself.

        Args:
            dest_dir (QuetzalFile): The destination `QuetzalFile` directory where this file/directory is to be moved.

        Returns:
            A string message indicating the successful move operation.
        """
        assert (
            self._mode == AccessMode.OWNER or self._permission == Permission.FULL_WRITE
        )
        debug(f"{self.name} called on move to ", dest_dir)

         # Adjust Destination relative Path
        destination_path = dest_dir._abs_path
        dest_dir: Path = destination_path.relative_to(self._root_dir.absolute())

        # Varify destination is valid
        source = self._root_dir / self._path
        dest = self._root_dir / dest_dir

        if (dest / self._path.name).exists():
            raise FileExistsError(
                f'File/directory with name "{self._path.name}" already exist at the destination.'
            )

        if dest.is_relative_to(source):
            raise Exception(f"You can't move a directory into itself.")

        # Move file, metadata, and description
        if self._type == FileType.DIRECTORY:
            shutil.move(source, dest)
        else:
            dest = dest / self._path.name
            shutil.move(source, dest)
        
        # Update metaData
        self._updateMetaForMove(dest_dir)
        
        return f'"{self.name}" moved to "{dest_dir}"'
    

    @staticmethod
    def _getMetaDataPath(path: Path) -> Path:
        """
        Constructs and returns the full path to the metadata file for a given absolute file or directory path.

        Args:
            path (Path): The path to the file or directory whose metadata path is to be constructed.

        Returns:
            The full `Path` object pointing to the metadata file.
        """
        return path.with_name(path.name + META_SUFFIX)

    @staticmethod
    def _getDescriptionPath(path: Path) -> Path:
        """
        Constructs and returns the full path to the description file for a given file or directory path.

        Args:
            path (Path): The path to the file or directory whose description path is to be constructed.

        Returns:
            The full `Path` object pointing to the description file.
        """
        return path.with_name(path.name + INFO_SUFFIX)

    def getMetaDataPath(self, path: Path) -> Path:
        """
        Retrieves the full path to the metadata file associated with a given file or directory path, 
        based on current QuetzalFile configuration.

        Args:
            path (Path): The path to the file or directory whose metadata file path is to be retrieved.

        Returns:
            The full `Path` object pointing to the metadata file.
        """
        return self._getMetaDataPath(self._metadata_dir / path)

    def getDescriptionPath(self, path: Path):
        """
        Retrieves the full path to the description file associated with a given file or directory path, 
        based on current QuetzalFile configuration.

        Args:
            path (Path): The path to the file or directory whose description file path is to be retrieved.

        Returns:
            The full `Path` object pointing to the description file.
        """
        return self._getDescriptionPath(self._metadata_dir / path)

    def _makeDefaultDescription(self, path: Union[str, Path]):
        """
        Creates a default description file for a file or directory if one does not already exist.

        This method is typically called when a new file or directory is added to ensure that it has a basic description.

        Args:
            path: The path to the file or directory for which a default description is to be created.
        """
        if self._type == FileType.DIRECTORY:
            default_content = self.PROJECT_DEFAULT_DESCRIPTION
        else:
            default_content = self.FILE_DEFAULT_DESCRIPTION

        with open(path, "w") as file:
            file.write(default_content)

    def getDescription(self) -> str:
        """
        Retrieves the description of the current file or directory.

        If a description file does not exist, a default description is created and then retrieved.

        Returns:
            The description text of the file or directory.
        """
        description_file_path = self.getDescriptionPath(self._path)

        # if not os.path.exists(description_file_path):
        if not description_file_path.exists():
            self._makeDefaultDescription(description_file_path)

        with open(description_file_path, "r") as file:
            return file.read()

    def __format__(self, __format_spec: str) -> str:
        return (
            "\n".join(
                [
                    "<QueztalFile::=" + str(self._path) + ">",
                    "type::= " + str(self._type),
                    "createdby::= " + str(self._created_by),
                    "user::= " + str(self._owner),
                    "permission::= " + str(self._permission),
                    "visibility::= " + str(self._visibility),
                    "analysis_progress::= " + str(self._analysis_progress),
                ]
            )
            + "\n"
        )

    def __repr__(self) -> str:
        return (
            "\n".join(
                [
                    "<QueztalFile::=" + str(self._path) + ">",
                    "type::= " + str(self._type),
                    "createdby::= " + str(self._created_by),
                    "user::= " + str(self._owner),
                    "permission::= " + str(self._permission),
                    "visibility::= " + str(self._visibility),
                    "analysis_progress::= " + str(self._analysis_progress),
                ]
            )
            + "\n"
        )


## LET pdoc3 to generate documentation for private methods 
__pdoc__ = {name: True
            for name, klass in globals().items()
            if name.startswith('_') and isinstance(klass, type)}
__pdoc__.update({f'{name}.{member}': True
                 for name, klass in globals().items()
                 if isinstance(klass, type)
                 for member in klass.__dict__.keys()
                 if member not in {'__module__', '__dict__', 
                                   '__weakref__', '__doc__'}})