from enum import Enum
import os

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
    
class QuetzalFile():
    
    @staticmethod
    def _new_dir(file: "QuetzalFile", value):
        assert file.permission != Permission.READ_ONLY
        
        print(f"{file.get_name()} called on New dir")
        pass
    
    @staticmethod
    def _upload(file: "QuetzalFile", value):
        assert file.permission != Permission.READ_ONLY
        
        print(f"{file.get_name()} called on upload")
        pass
    
    @staticmethod
    def _rename(file: "QuetzalFile", value):
        assert file.permission == Permission.FULL_WRITE
        
        print(f"{file.get_name()} called on rename")
        pass
    
    @staticmethod
    def _share(file: "QuetzalFile", value):
        assert file.permission == Permission.FULL_WRITE
        
        print(f"{file.get_name()} called on share")
        pass
    
    @staticmethod
    def _delete(file: "QuetzalFile", value):
        assert file.permission == Permission.FULL_WRITE
        
        print(f"{file.get_name()} called on delete")
        pass
    
    @staticmethod
    def _download(file: "QuetzalFile", value):
        
        print(f"{file.get_name()} called on download")
        pass
    
    _perform = {
        Action.NEW_DIR: _new_dir.__func__,
        Action.UPLOAD_FILE: _upload.__func__,
        Action.RENAME: _rename.__func__,
        Action.SHARE: _share.__func__,
        Action.DELETE: _delete.__func__,
        Action.DOWNLOAD: _download.__func__,
    }
    
    # @staticmethod
    # def init_query_file_state
    
    def __init__(
        self, 
        path: str, 
        owner: str,
        type: FileType, 
        visibility: Visibility,  
        analysis_progress: AnalysisProgress,
        permission: Permission
    ):
        self.path: str = path
        self.owner: str = owner
        self.type: FileType = type
        self.visibility: Visibility = visibility
        self.analysis_progress: AnalysisProgress = analysis_progress
        self.permission: Permission = permission

    def perform(self, action: Action, input):
        print(action, input, self)
        QuetzalFile._perform[action](self, input)
        
    def get_name(self) -> str:
        return os.path.basename(self.path)