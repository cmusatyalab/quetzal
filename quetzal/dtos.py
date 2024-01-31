from enum import Enum
import os
import logging
from typing import Any, Literal, List
import shutil
import unittest
import os
from quetzal.video import *
from quetzal.engines.vpr_engine.anyloc_engine import AnyLocEngine
from quetzal.compute_vlad import generate_VLAD
import torch
import streamlit as st

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
debug = lambda *args: logger.debug(" ".join([str(arg) for arg in args]))

EXAMPLE_INFO = """Route: army-demo
Recorded Date (MM/DD/YYYY): 10/5/2023
Uploader: Admin
Weather Condition: Sunny, Green
Description: Recorded by Mihir and Thom"""

VIDEO_DEFAULT = "Uploader::= default\nRecorded Date (MM/DD/YYYY)::= default\nTime-of-day::= default\nWeather Condition::= default\nDescription::= default"
PROJECT_DEFAULT = "Route Location::= default\nLast Update (MM/DD/YYYY)::= default\nLast Edit by::= default\nDescription::= default"
VIDEO_DEFAULT_META = "FileType::= file\nVisibility::= private\nPermission::= full_write\nAnalysisProgress::= none\n"
PROJECT_DEFAULT_META = (
    "FileType::= directory\nVisibility::= private\nPermission::= full_write\n"
)
USER_ROOT_META = (
    "FileType::= directory\nVisibility::= shared\nPermission::= read_only\n"
)
USER_ROOT_DESCRIPTION = (
    "Description::=root directory for "
)


class Permission(Enum):
    READ_ONLY = "read_only"
    POST_ONLY = "post_only"
    FULL_WRITE = "full_write"


class Visibility(Enum):
    SHARED = "shared"
    PRIVATE = "private"


class FileType(Enum):
    FILE = "file"
    DIRECTORY = "directory"


class AnalysisProgress(Enum):
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


class Action(Enum):
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


def extractFilname(file_path):
    """
    Returns the file name without its extension.

    :param file_path: The full path to the file or just the file name.
    :return: File name without its extension.
    """
    return os.path.splitext(file_path)[0]

def extractExtension(file_path):
    return os.path.splitext(file_path)[1]

def replaceInitialSegment(path, new_initial_segment):
    path_parts = path.split(os.path.sep)
    path_parts[0] = new_initial_segment
    return os.path.join(*path_parts)

def getCopyName(path):
    return appendText(path, "_copy")

def appendText(path, text):
    return extractFilname(path) + text + extractExtension(path)

class QuetzalFile:
    
    def _renameAnalysisData(self, newName):
        for data_dir in ["database", "query"]:
            analysis_path = os.path.join(
                self.metadata_dir,
                os.path.dirname(self._path),
                data_dir,
                extractFilname(os.path.basename(self._path)),
            )
            if os.path.exists(analysis_path):
                new_path = os.path.join(
                    self.metadata_dir,
                    os.path.dirname(self._path),
                    data_dir,
                    extractFilname(os.path.basename(newName)),
                )
                os.rename(analysis_path, new_path)
                
    def updateMetaData(self, action, new_path=None, new_analysis_progress:AnalysisProgress=None):
        """Update metadata when a file or directory is modified."""
        orig_path_full = os.path.join(self.metadata_dir, self._path)
        metadata_path = self._getMetaDataPath(orig_path_full)
        description_path = self._getDescriptionPath(orig_path_full)

        if action == "delete":
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            if os.path.exists(description_path):
                os.remove(description_path)
            if os.path.exists(orig_path_full):
                shutil.rmtree(orig_path_full)
                
        elif action in ["rename", "move"]:
            ## Rename The Folder
            new_path_full = os.path.join(self.metadata_dir, new_path)
            if self.type == FileType.DIRECTORY:
                if os.path.exists(orig_path_full):
                    os.rename(orig_path_full, new_path_full)
            else:
                # Rename analysis data
                self._renameAnalysisData(new_path)

            ## Rename Metadata
            new_metadata_path = self._getMetaDataPath(new_path_full)
            if os.path.exists(metadata_path):
                os.rename(metadata_path, new_metadata_path)
            else:
                with open(new_metadata_path, "w") as file:
                    file.write(
                        PROJECT_DEFAULT_META
                        if self.type == FileType.DIRECTORY
                        else VIDEO_DEFAULT_META
                    )

            ## Rename Description
            new_description_path = self._getDescriptionPath(new_path_full)
            if os.path.exists(description_path):
                os.rename(description_path, new_description_path)
            else:
                with open(new_description_path, "w") as file:
                    file.write(
                        PROJECT_DEFAULT
                        if self.type == FileType.DIRECTORY
                        else VIDEO_DEFAULT
                    )

        elif action == "analyze":
            if os.path.exists(metadata_path):
                with open(metadata_path, "r+") as file:
                    metadata = file.read()
                    metadata = metadata.replace(
                        f"AnalysisProgress::= {self.analysis_progress.value}",
                        f"AnalysisProgress::= {new_analysis_progress.value}",
                    )
                    file.seek(0)
                    file.write(metadata)
                    file.truncate()

        if action == "reset_analysis":
            metadata_path = self.getMetaDataPath()
            if os.path.exists(metadata_path):
                with open(metadata_path, "r+") as file:
                    metadata = file.read()
                    metadata = metadata.replace(
                        f"AnalysisProgress::= {self.analysis_progress.value}",
                        "AnalysisProgress::= none",
                    )
                    file.seek(0)
                    file.write(metadata)
                    file.truncate()


    def _rename(self, new_file_name):
        assert self.mode == "user" or self.permission == Permission.FULL_WRITE
        assert extractFilname(new_file_name) not in ["database", "query"]
        
        debug(f"\n\n\t{self.getName()} called on rename {new_file_name}\n")

        new_file_name = extractFilname(new_file_name) + extractExtension(self._path)
        
        new_path_abs = os.path.join(
            self.root_dir, os.path.dirname(self._path), new_file_name
        )
        if os.path.exists(new_path_abs):
            raise FileExistsError(f'File/directory with name "{new_file_name}" already exist at the destination.')
        
        os.rename(os.path.join(self.root_dir, self._path), new_path_abs)

        new_path = os.path.join(os.path.dirname(self._path), new_file_name)
        self.updateMetaData("rename", new_path)
        org_name = self.getName()
        self._path = new_path
        
        return f'"{org_name}" renamed to "{self.getName()}"'


    def _newDir(self, target_path, dir_name, meta_data, description):
        
        new_dir_path = os.path.normpath(os.path.join(self.root_dir, target_path, dir_name))
        os.makedirs(new_dir_path, exist_ok=True)

        # Crate New Meta Data, Description, and Directory
        new_dir_metadata_path = os.path.normpath(os.path.join(self.metadata_dir, target_path, dir_name))
        os.makedirs(new_dir_metadata_path, exist_ok=True)

        new_dir_metadata_file = self._getMetaDataPath(new_dir_metadata_path)
        with open(new_dir_metadata_file, "w") as file:
            file.write(meta_data + "CreatedBy::= " + self.user + "\n")

        description_file_path = self._getDescriptionPath(new_dir_metadata_path)
        with open(description_file_path, "w") as file:
            file.write(description)
        

    def _newDirectory(self, dir_name):
        assert self.mode == "user" or self.permission != Permission.READ_ONLY
        assert self.type == FileType.DIRECTORY
        assert extractFilname(dir_name) not in ["database", "query"]

        debug(f"\n\n\t{self.getName()} called on New dir {dir_name}\n")
        
        self._newDir(self._path, dir_name, PROJECT_DEFAULT_META, PROJECT_DEFAULT)
        
        return f'"{dir_name}" Created'


    def _upload(self, uploaded_files):
        assert self.mode == "user" or self.permission != Permission.READ_ONLY
        assert self.type == FileType.DIRECTORY
        for uploaded_file in uploaded_files:  
            assert extractFilname(uploaded_file.name) not in ["database", "query"]
            debug(f"\n\n\t{self.getName()} called on upload {uploaded_file.name}\n")
            
            dest = os.path.join(self.root_dir, self._path, uploaded_file.name)
            destName = os.path.basename(dest)
            
            start_num = 0
            original_name = destName
            while os.path.exists(dest):
                # raise FileExistsError
                destName = appendText(original_name, "_" + str(start_num))
                dest = os.path.join(
                    self.root_dir, 
                    self._path, 
                    destName
                )
                start_num += 1
            
            with open(dest, mode='wb') as w:
                w.write(uploaded_file.getvalue())

            new_file_metadata = os.path.join(self.metadata_dir, self._path, destName)

            new_file_metadata_path = self._getMetaDataPath(new_file_metadata)
            with open(new_file_metadata_path, "w") as file:
                file.write(VIDEO_DEFAULT_META + "CreatedBy::= " + self.user + "\n")

            new_file_description_path = self._getDescriptionPath(new_file_metadata)
            with open(new_file_description_path, "w") as file:
                file.write(VIDEO_DEFAULT)
        
        return f'{len(uploaded_files)} files successfully uploaded'

            

    def _share(self, shared: Visibility, permission: Permission):
        assert self.mode == "user" or self.permission == Permission.FULL_WRITE
        # assert self.type == FileType.DIRECTORY
        
        if shared == self.visibility and permission == self.permission:
            return None
        
        debug(f"\n\n\t{self.getName()} called on share {shared}:{permission}\n")


        self._updateMetadataForShare(shared, permission)

        # If the QuetzalFile is a directory, apply the changes to all sub-projects and video files
        if self.type == FileType.DIRECTORY:
            for subfile in self.listFiles():
                subfile._updateMetadataForShare(shared, permission)
        
        return f'"{self.getName()}" Sharing Setting Updated'
        
    def _updateMetadataForShare(self, shared: Visibility, permission: Permission):
        metadata_path = self.getMetaDataPath()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r+") as file:
                metadata = file.read()
                metadata = metadata.replace(f"Visibility::= {self.visibility.value}", f"Visibility::= {shared.value}")
                metadata = metadata.replace(f"Permission::= {self.permission.value}", f"Permission::= {permission.value}")
                file.seek(0)
                file.write(metadata)
                file.truncate()

        self.visibility = shared
        self.permission = permission
    
    
    def _deleteAnalysisData(self):
        """Delete associated database and query analysis data."""
        for data_dir in ["database", "query"]:
            analysis_path = os.path.join(
                self.metadata_dir,
                os.path.dirname(self._path),
                data_dir,
                extractFilname(os.path.basename(self._path)),
            )
            if os.path.exists(analysis_path):
                shutil.rmtree(analysis_path)
                
    def _delete(self):
        assert self.mode == "user" or self.permission == Permission.FULL_WRITE

        debug(f"\n\n\t{self.getName()} called on delete\n")

        if self.type == FileType.FILE:
            self._deleteAnalysisData()
            os.remove(os.path.join(self.root_dir, self._path))
        else:  # For directories
            shutil.rmtree(os.path.join(self.root_dir, self._path))

        self.updateMetaData("delete")
        
        return f'"{self.getName()}" Deleted'

    # def _download(self):

    #     debug(f"\n\t{self.getName()} called on download\n")
    #     pass
    
    def _syncAnalysisState(self):
        assert self.type == FileType.FILE
        
        video = DatabaseVideo(
            os.path.abspath(self.root_dir), 
            os.path.dirname(self._path), 
            os.path.basename(self._path), 
            os.path.abspath(self.metadata_dir)
        )
        
        if AnyLocEngine._is_vlad_ready(video):
            self.updateMetaData("analyze", new_analysis_progress=AnalysisProgress.FULL)
            self.analysis_progress = AnalysisProgress.FULL
            return
        
        video = QueryVideo(os.path.abspath(self.root_dir), 
            os.path.dirname(self._path), 
            os.path.basename(self._path), 
            os.path.abspath(self.metadata_dir)
        ) 
        
        if AnyLocEngine._is_vlad_ready(video):
            self.updateMetaData("analyze", new_analysis_progress=AnalysisProgress.HALF)
            self.analysis_progress = AnalysisProgress.HALF
            return
        
        self.updateMetaData("analyze", new_analysis_progress=AnalysisProgress.NONE)
        self.analysis_progress = AnalysisProgress.NONE
        return

    def _analyze(self, option: AnalysisProgress):
        assert self.mode == "user" or self.permission != Permission.READ_ONLY
        debug(f"\n\t{self.getName()} called on analyze {option}\n")

        if option == None:
            return None
        
        self._syncAnalysisState()
        if self.analysis_progress >= option:
            return None
        
        db_video = None
        query_video = None
        if option == AnalysisProgress.FULL:
            db_video = DatabaseVideo(
                os.path.abspath(self.root_dir), 
                os.path.dirname(self._path), 
                os.path.basename(self._path), 
                os.path.abspath(self.metadata_dir)
            )
        if option == AnalysisProgress.HALF:
            query_video = QueryVideo(os.path.abspath(self.root_dir), 
                os.path.dirname(self._path), 
                os.path.basename(self._path), 
                os.path.abspath(self.metadata_dir)
            ) 
            
        generate_VLAD(db_video, query_video, torch.device("cuda:0"))

        self.updateMetaData("analyze", new_analysis_progress=option)
        self.analysis_progress = option
        
        return f'"{self.getName()}" Analysis Done'


    def _editDescription(self, value):
        assert self.mode == "user" or self.permission == Permission.FULL_WRITE

        debug(f"\n\t{self.getName()} called on editMetaData {value}\n")
        description_file_path = self.getDescriptionPath()

        # Overwrite the existing description with the new value
        with open(description_file_path, "w") as file:
            file.write(value)
        
        return f'"{self.getName()}" Edit Success'
    
    def _copyAnalysisData(self, newDir, destName, move=False):
        for data_dir in ["database", "query"]:
            analysis_path = os.path.join(
                self.metadata_dir,
                os.path.dirname(self._path),
                data_dir,
                extractFilname(os.path.basename(self._path)),
            )
            if os.path.exists(analysis_path):
                copy_path = os.path.join(
                    self.metadata_dir,
                    newDir,
                    data_dir,
                    extractFilname(destName),
                )
                if move:
                    shutil.move(analysis_path, copy_path)
                else: 
                    shutil.copytree(analysis_path, copy_path)
                
    def _copy(self, destination: 'QuetzalFile'):
        debug(f"{self.getName()} called on copy to ", destination)
        
        destination_path = destination._absPath
        destination = os.path.relpath(destination_path, os.path.abspath(self.root_dir))
        
        source = os.path.join(self.root_dir, self._path)
        
        dest = os.path.join(self.root_dir, destination, os.path.basename(self._path))
        destName = os.path.basename(self._path)
        while os.path.exists(dest):
            # raise FileExistsError
            dest = os.path.join(
                self.root_dir, 
                destination, 
                getCopyName(destName)
            )
            destName = getCopyName(destName)
        
        
        if self.type == FileType.DIRECTORY:
            # dest = os.path.join(dest, os.path.basename(self._path))
            shutil.copytree(source, dest)
            source_metadata = os.path.join(self.metadata_dir, self._path)
            dest_metadata = os.path.join(self.metadata_dir, destination, destName)
            shutil.copytree(source_metadata, dest_metadata)
        else:
            shutil.copy2(source, dest)
            source_metadata = os.path.join(self.metadata_dir, self._path)
            dest_metadata = os.path.join(self.metadata_dir, destination, destName)
            
        source_metadata_path = self._getMetaDataPath(source_metadata)
        dest_metadata_path = self._getMetaDataPath(dest_metadata)
        shutil.copy2(source_metadata_path, dest_metadata_path)

        source_desc_path = self._getDescriptionPath(source_metadata)
        dest_desc_path = self._getDescriptionPath(dest_metadata)
        shutil.copy2(source_desc_path, dest_desc_path)
        
        if self.type == FileType.FILE:
            self._copyAnalysisData(destination, destName)
            
        return f'"{self.getName()}" copised to "{destination}"'
        
        
    def _move(self, destination: 'QuetzalFile'):
        assert self.mode == "user" or self.permission == Permission.FULL_WRITE
        debug(f"{self.getName()} called on move to ", destination)
        
        destination_path = destination._absPath
        destination = os.path.relpath(destination_path, os.path.abspath(self.root_dir))
        
        source = os.path.normpath(os.path.join(self.root_dir, self._path))
        dest = os.path.join(self.root_dir, destination)
        # new_path = os.path.join(destination, os.path.basename(self._path))
        
        destName = os.path.basename(self._path)
        if os.path.exists(os.path.join(self.root_dir, destination, os.path.basename(self._path))):
            raise FileExistsError(f'File/directory with name "{destName}" already exist at the destination.')
        
        if os.path.commonpath([source, dest]) == source:
            raise Exception(f"You can't move a directory into itself.")

        
        if self.type == FileType.DIRECTORY:
            # dest = os.path.join(dest, os.path.basename(self._path))
            shutil.move(source, dest)
            source_metadata = os.path.join(self.metadata_dir, self._path)
            dest_metadata = os.path.join(self.metadata_dir, destination)
            shutil.move(source_metadata, dest_metadata)
        else:
            dest = os.path.join(self.root_dir, destination, os.path.basename(self._path))
            shutil.move(source, dest)
            
        source_metadata = os.path.join(self.metadata_dir, self._path)
        dest_metadata = os.path.join(self.metadata_dir, destination, os.path.basename(self._path))
                
        source_metadata_path = self._getMetaDataPath(source_metadata)
        dest_metadata_path = self._getMetaDataPath(dest_metadata)
        shutil.move(source_metadata_path, dest_metadata_path)
        
        source_desc_path = self._getDescriptionPath(source_metadata)
        dest_desc_path = self._getDescriptionPath(dest_metadata)
        shutil.move(source_desc_path, dest_desc_path)
        
        if self.type == FileType.FILE:
            self._copyAnalysisData(destination, destName, move=True)
        
        return f'"{self.getName()}" moved to "{destination}"'

    @staticmethod
    def _getMetaDataPath(path):
        return extractFilname(path) + "_meta.txt"
    
    def getMetaDataPath(self):
        return self._getMetaDataPath(os.path.normpath(os.path.join(self.metadata_dir, self._path)))
    
    @staticmethod
    def _getDescriptionPath(path):
        return extractFilname(path) + ".txt"
    
    def getDescriptionPath(self):
        return self._getDescriptionPath(os.path.join(self.metadata_dir, self._path))
    
    
    def getDescription(self):
        description_file_path = self.getDescriptionPath()

        if not os.path.exists(description_file_path):
            if self.type == FileType.DIRECTORY:  # Assuming directories are projects
                default_content = PROJECT_DEFAULT
            else:  # Assuming files are videos
                default_content = VIDEO_DEFAULT

            # Create a new file with the default content
            with open(description_file_path, "w") as file:
                file.write(default_content)

            return default_content

        with open(description_file_path, "r") as file:
            return file.read()

    def loadMetaData(self):
        # Load metadata from metadata file
        metadata_path = self.getMetaDataPath()
                
        if not os.path.exists(metadata_path):
            raise FileNotFoundError

        with open(metadata_path, "r") as file:
            data = file.read().splitlines()
        metadata = {line.split("::= ")[0]: line.split("::= ")[1] for line in data}

        return (
            metadata.get("CreatedBy", "unknown"),
            FileType(metadata.get("FileType", "file")),
            Visibility(metadata.get("Visibility", "private")),
            AnalysisProgress(metadata.get("AnalysisProgress", "none")),
            Permission(metadata.get("Permission", "full_write")),
        )

    _perform = {
        Action.NEW_DIR: _newDirectory,
        Action.UPLOAD_FILE: _upload,
        Action.RENAME: _rename,
        Action.SHARE: _share,
        Action.DELETE: _delete,
        # Action.DOWNLOAD: _download,
        Action.ANALYZE: _analyze,
        Action.EDIT: _editDescription,
        Action.COPY: _copy,
        Action.MOVE: _move,
    }

    # @staticmethod
    # def init_query_file_state
    
    @property
    def _absPath(self) -> str:
        return os.path.abspath(os.path.join(self.root_dir, self._path))
        
    def __init__(
        self,
        path: str,  # path to the dir/file from root_dir
        root_dir: str,
        metadata_dir: str,
        user: str = "temp",
        mode: Literal["user", "shared"] = "user"
    ):
        self._path = path
        self.root_dir = root_dir
        self.metadata_dir = metadata_dir
        self.user = user
        self.mode = mode
        self.path = replaceInitialSegment(path, "home")
                
        if mode=="user" and not os.path.exists(root_dir):
            self._newDir("../", user, USER_ROOT_META, USER_ROOT_DESCRIPTION + user)
            while(not self._getMetaDataPath(root_dir)): pass
        
        (
            self.owner,
            self.type,
            self.visibility,
            self.analysis_progress,
            self.permission,
        ) = self.loadMetaData()
                            
        if not os.path.exists(os.path.join(root_dir, self._path)):
            raise FileNotFoundError

    def perform(self, action: Action, input: dict):
        print(action, input, self)
        return QuetzalFile._perform[action](self, **input)

    def getName(self) -> str:
        return str(os.path.basename(self._path))

    def listFiles(
        self,
        sharedOnly=False,
        directoryOnly=False,
        excludeUser=False,
    ) -> List["QuetzalFile"]:
        assert self.type == FileType.DIRECTORY

        directories = []
        files = []
        directory_path = os.path.join(self.root_dir, self._path)

        sorted_items = sorted(os.listdir(directory_path), key=lambda x: x.lower())
        for item in sorted_items:
            item_path = os.path.join(self._path, item)
            item_full_path = os.path.join(directory_path, item)
            
            # Filter only directory
            is_dir = os.path.isdir(item_full_path)
            if directoryOnly and not is_dir:
                continue
            
            try:
                file = QuetzalFile(
                    path=item_path, 
                    root_dir=self.root_dir, 
                    metadata_dir=self.metadata_dir, 
                    user=self.user, 
                    mode=self.mode)
            except: # metadata not found
                continue
            
            # Filter only shared
            if sharedOnly and file.visibility != Visibility.SHARED:
                continue
                
            if os.path.isdir(item_full_path):
                directories.append(file)
            else:
                files.append(file)

        return directories + files

    def __format__(self, __format_spec: str) -> str:
        return "\n".join(
            ["<QueztalFile::=" + self._path + ">",
             "type::= " + self.type,
             "createdby::= " + self.owner,
             "permission::= " + self.permission,
             "visibility::= "+ self.visibility,
             "analysis_progress::= " + self.analysis_progress
            ]
        )

    def __repr__(self) -> str:
        return "\n".join(
            ["<QueztalFile::=" + self._path + ">",
             "type::= " + str(self.type),
             "createdby::= " + str(self.owner),
             "permission::= " + str(self.permission),
             "visibility::= "+ str(self.visibility),
             "analysis_progress::= " + str(self.analysis_progress)
            ]
        )
