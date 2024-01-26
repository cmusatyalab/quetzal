from enum import Enum
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
debug = lambda *args: logger.debug(" ".join([str(arg) for arg in args]))

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

class Action(Enum):
    NEW_DIR = "new_dir"
    UPLOAD_FILE = "upload_file"
    RENAME = "rename"
    SHARE = "share"
    DELETE = "delete"
    DOWNLOAD = "download"
    # MOVE = "move"
    
class QuetzalFile():
    
    def _new_dir(self, dir_name):
        assert self.permission != Permission.READ_ONLY
        
        debug(f"\n\n\t{self.get_name()} called on New dir {dir_name}\n")
        pass
    
    def _upload(self, *temp):
        assert self.permission != Permission.READ_ONLY
        
        debug(f"\n\n\t{self.get_name()} called on upload {temp}\n")
        pass
    
    def _rename(self, file_name):
        assert self.permission == Permission.FULL_WRITE
        
        debug(f"\n\n\t{self.get_name()} called on rename {file_name}\n")
        pass
    
    def _share(self, shared: Visibility, permission: Permission):
        assert self.permission == Permission.FULL_WRITE
        
        debug(f"\n\n\t{self.get_name()} called on share {shared}:{permission}\n")
        pass
    
    def _delete(self, *temp):
        assert self.permission == Permission.FULL_WRITE
        
        debug(f"\n\n\t{self.get_name()} called on delete {temp}\n")
        pass
    
    def _download(self, *temp):
        
        debug(f"\n\t{self.get_name()} called on download {temp}\n")
        pass
    
    _perform = {
        Action.NEW_DIR: _new_dir,
        Action.UPLOAD_FILE: _upload,
        Action.RENAME: _rename,
        Action.SHARE: _share,
        Action.DELETE: _delete,
        Action.DOWNLOAD: _download,
    }
    
    # @staticmethod
    # def init_query_file_state
    
    def __init__(
        self, 
        path: str, 
        type: FileType, 
        visibility: Visibility,  
        analysis_progress: AnalysisProgress,
        permission: Permission,
        owner: str = None,
    ):
        self.path: str = path
        self.owner: str = owner
        self.type: FileType = type
        self.visibility: Visibility = visibility
        self.analysis_progress: AnalysisProgress = analysis_progress
        self.permission: Permission = permission

    def perform(self, action: Action, input: dict):
        print(action, input, self)
        QuetzalFile._perform[action](self, **input)
        
    def get_name(self) -> str:
        return str(os.path.basename(self.path))
    
    def __format__(self, __format_spec: str) -> str:
        return "<QueztalFile:" + self.path + ">"
    
    def __repr__ (self) -> str:
        return "<QueztalFile:" + self.path + ">"