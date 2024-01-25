from streamlit_elements import elements, mui, lazy, sync
from typing import Literal, List, Dict
from typing import Union
from quetzal_app.dtos import *
import streamlit as st
from collections import defaultdict
from streamlit_extras.stylable_container import stylable_container
from streamlit_float import *
from quetzal_app.utils.utils import *
from functools import partial


PRIMARY_COLOR = "#c9e6fd"


class MuiToggleButton:
    def __init__(self, value, icon, label):
        self.value: str = value
        self.icon: str = icon
        self.label: str = label

    def render(self):
        with mui.ToggleButton(value=self.value):
            getattr(mui.icon, self.icon)()
            mui.Typography(
                self.label,
                sx={
                    "fontFamily": "sans-serif",
                    "textTransform": "capitalize",
                    "fontSize": "inherit",
                },
            )


class MuiSideBarMenu:
    toggle_buttons_style = {
        # "gap": "1rem",
        "width": "100%",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "mx": "1rem",
            "px": "1.0rem",
            "py": "0.2rem",
            "justifyContent": "start",
            "justifyItem": "start",
            "gap": "0.8rem",
            "&:not(:last-of-type)": {
                "borderRadius": "2rem",
            },
            "&:not(:first-of-type)": {
                "borderRadius": "2rem",
            },
        },
        "& .MuiToggleButton-root": {
            "&.Mui-selected": {
                # "color": "#c9e6fd",
                "bgcolor": PRIMARY_COLOR,
                "&:hover": {"bgcolor": PRIMARY_COLOR},
            },
            "&:hover": {"bgcolor": "grey.200"},
            "fontSize": 14,
        },
    }

    def __init__(self, toggle_buttons: List[MuiToggleButton], key="main"):
        self.toggle_buttons = toggle_buttons
        self.key = key

        if "MuiSideBarMenu" not in st.session_state:
            st.session_state.MuiSideBarMenu = {key: self.toggle_buttons[0].value}
        else:
            if key not in st.session_state.MuiSideBarMenu:
                st.session_state.MuiSideBarMenu[key] = self.toggle_buttons[0].value

    def render(self):
        def handleController(event, controller):
            if controller != None:
                st.session_state.MuiSideBarMenu[self.key] = controller

        with mui.ToggleButtonGroup(
            orientation="vertical",
            value=st.session_state.MuiSideBarMenu[self.key],
            onChange=handleController,
            exclusive=True,
            sx=MuiSideBarMenu.toggle_buttons_style,
        ):
            for button in self.toggle_buttons:
                button.render()


class MuiFileList:
    list_style = {
        "py": "0px",
        "& .MuiListItem-root": {
            "py": "6px",
            "&:hover": {"backgroundColor": "grey.200"},
            "&.Mui-selected": {
                "bgcolor": PRIMARY_COLOR,
                "&:hover": {"bgcolor": PRIMARY_COLOR},
            },
        },
    }

    sub_header_style = {
        "&:hover": {"backgroundColor": "white !important"},
    }

    @staticmethod
    def handleClickAway(self):
        st.session_state.MuiFileListState[self.key]["item"] = -1
        print(st.session_state.MuiFileListState)

    @staticmethod
    def openSymbolHelp(self, event):
        st.session_state.MuiFileListState[self.key]["help_anchor"] = {
            "top": event["clientY"],
            "left": event["clientX"],
        }
        print(st.session_state.MuiFileListState)

    @staticmethod
    def closeSymbolHelp(self, event):
        st.session_state.MuiFileListState[self.key]["help_anchor"] = None
        print(st.session_state.MuiFileListState)

    def __init__(self, file_list, max_height="50%", key="main"):
        self.file_list = file_list
        self.max_height = max_height
        self.key = key

        if "MuiFileListState" not in st.session_state:
            st.session_state.MuiFileListState = {key: {"item": -1, "help_anchor": None}}
        else:
            if key not in st.session_state.MuiFileListState:
                st.session_state.MuiFileListState[key] = {
                    "item": -1,
                    "help_anchor": None,
                }

    def symbolHelper(self):
        anchor = st.session_state.MuiFileListState[self.key]["help_anchor"]
        icon_style = {"fontSize": 14}
        text_style = {"fontSize": 14, "pr": 1}

        with mui.Popover(
            sx={"pointerEvents": "none"},
            open=bool(anchor),
            anchorReference="anchorPosition",
            anchorPosition=anchor,
            anchorOrigin={"vertical": "top", "horizontal": "left"},
            transformOrigin={"vertical": "top", "horizontal": "left"},
            onClose=lambda event: MuiFileList.closeSymbolHelp(self, event),
            disableRestoreFocus=True,
        ):
            with mui.Stack(
                spacing=0.2,
                direction="column",
                alignItems="center",
                justifyContent="center",
                sx={"mx": "0.5rem", "my": "0.2rem"},
            ):
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="center",
                    justifyContent="center",
                    sx={"py": "3px"},
                ):
                    getattr(
                        mui.icon, MuiFileListItem.visibility_icon[Visibility.SHARED]
                    )(sx=icon_style)
                    mui.Typography("Shared/Public", sx=text_style)
                    getattr(
                        mui.icon, MuiFileListItem.visibility_icon[Visibility.PRIVATE]
                    )(sx=icon_style)
                    mui.Typography("Private", sx=text_style)

                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="center",
                    justifyContent="center",
                    sx={"py": "3px"},
                ):
                    getattr(
                        mui.icon, MuiFileListItem.permission_icon[Permission.READ_ONLY]
                    )(sx=icon_style)
                    mui.Typography("Read Only", sx=text_style)
                    getattr(
                        mui.icon, MuiFileListItem.permission_icon[Permission.POST_ONLY]
                    )(sx=icon_style)
                    mui.Typography("Post Only/Can't Edit", sx=text_style)
                    getattr(
                        mui.icon, MuiFileListItem.permission_icon[Permission.FULL_WRITE]
                    )(sx=icon_style)
                    mui.Typography("Full Write", sx=text_style)

                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="center",
                    justifyContent="center",
                    # sx={"py": "1px"},
                ):
                    getattr(
                        mui.icon, MuiFileListItem.progress_icon[AnalysisProgress.HALF]
                    )(sx=icon_style)
                    mui.Typography("Analysis Only for the Query Done", sx=text_style)
                    getattr(
                        mui.icon, MuiFileListItem.progress_icon[AnalysisProgress.FULL]
                    )(sx=icon_style)
                    mui.Typography("Full Analysis for the Database Done", sx=text_style)

    def render(self):
        with mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0px",
                "border": "0px",
                "maxHeight": self.max_height,
                "overflow": "auto",
                "bgcolor": "white",
                "padding": "0px",
            },
        ):
            # assert st.session_state.has_key("MuiFileListState")

            with mui.ListSubheader(
                component="div",
                sx={"px": "0px"},
            ):
                with mui.ListItem(
                    divider=False,
                    sx=MuiFileList.sub_header_style,
                ):
                    mui.ListItemIcon(mui.icon.NoIcon())
                    mui.ListItemText(primary="Name")
                    mui.Typography(
                        "State",
                        sx={"px": "8px"},
                        onMouseEnter=lambda event: MuiFileList.openSymbolHelp(
                            self, event
                        ),
                        onMouseLeave=lambda event: MuiFileList.closeSymbolHelp(
                            self, event
                        ),
                    )
                    mui.ListItemIcon(mui.icon.NoIcon())
                mui.Divider()

            with mui.ClickAwayListener(
                mouseEvent="onMouseDown",
                touchEvent="onTouchStart",
                onClickAway=lambda: MuiFileList.handleClickAway(self),
            ):
                with mui.List(
                    dense=True,
                    sx=MuiFileList.list_style,
                ):
                    mui.Divider()
                    for i, items in enumerate(self.file_list):
                        MuiFileListItem(idx=i, key=self.key, **items).render()
            self.symbolHelper()


class MuiFileListItem:
    STATE_ICON_STYLE = {"fontSize": 16}
    visibility_icon = {Visibility.SHARED: "Group", Visibility.PRIVATE: "Lock"}
    permission_icon = {
        Permission.READ_ONLY: "EditOff",
        Permission.POST_ONLY: "UploadFile",
        Permission.FULL_WRITE: "DriveFileRenameOutline",
    }
    file_type_icon = {
        FileType.FILE: defaultdict(lambda: "Movie"),
        FileType.DIRECTORY: {
            Visibility.PRIVATE: "Folder",
            Visibility.SHARED: "FolderShared",
        },
    }
    progress_icon = {
        AnalysisProgress.FULL: "CheckCircle",
        AnalysisProgress.HALF: "CheckCircleOutline",
        AnalysisProgress.NONE: "NotInterestedOutlined",
    }

    @staticmethod
    def handleItemSelect(self):
        st.session_state.MuiFileListState[self.key]["item"] = self.idx
        print(st.session_state.MuiFileListState)
        if self.onClick:
            self.onClick()

    def __init__(
        self,
        name: str,
        type: FileType,
        state: Dict[str, Union[Visibility, AnalysisProgress, Permission]],
        idx,
        key,
        onClick=None,
    ):
        # assert st.session_state.has_key("MuiFileListState")

        self.name = name
        self.type = type
        self.visibility = state["visibility"]
        self.analysis_process = state["analyzed"]
        self.permission = state["permission"]
        self.idx = idx
        self.key = key
        self.onClick = onClick

    def render(self):
        secondary_action = mui.IconButton(edge="end", children=[mui.icon.MoreVert()])

        color_style = {"color": "#EA4335"} if self.type == FileType.FILE else {}
        list_item_icon = mui.ListItemIcon(
            getattr(
                mui.icon, MuiFileListItem.file_type_icon[self.type][self.visibility]
            )(sx=color_style)
        )

        list_item_text = mui.ListItemText(primary=self.name)

        state_icons = []
        state_icons.append(
            getattr(mui.icon, MuiFileListItem.visibility_icon[self.visibility])(
                sx=MuiFileListItem.STATE_ICON_STYLE
            )
        )
        if self.type == FileType.FILE:
            if self.analysis_process != AnalysisProgress.NONE:
                state_icons.append(
                    getattr(
                        mui.icon, MuiFileListItem.progress_icon[self.analysis_process]
                    )(sx=MuiFileListItem.STATE_ICON_STYLE)
                )
        elif self.type == FileType.DIRECTORY:
            if self.visibility == Visibility.SHARED:
                state_icons.append(
                    getattr(mui.icon, MuiFileListItem.permission_icon[self.permission])(
                        sx=MuiFileListItem.STATE_ICON_STYLE
                    )
                )

        states = mui.Stack(
            spacing=1,
            direction="row",
            alignItems="center",
            justifyContent="center",
            sx={"my": 0, "pr": "32px", "minWidth": "40px"},
            children=state_icons,
        )

        return mui.ListItem(
            secondaryAction=secondary_action,
            divider=True,
            button=True,
            selected=(self.idx == st.session_state.MuiFileListState[self.key]["item"]),
            children=[list_item_icon, list_item_text, states],
            onDoubleClick=lambda: print("DOUBLE CLICKED"),
            onClick=lambda: MuiFileListItem.handleItemSelect(self),
            disableRipple=True,
        )


class MuiActionMenu:
    menu_style = {
        "& .MuiPaper-root": {
            "marginTop": "0.5rem",
            "minWidth": 180,
            "boxShadow": "rgb(255, 255, 255) 0px 0px 0px 0px, rgba(0, 0, 0, 0.05) 0px 0px 0px 1px, rgba(0, 0, 0, 0.1) 0px 10px 15px -3px, rgba(0, 0, 0, 0.05) 0px 4px 6px -2px",
            "& .MuiMenu-list": {
                "padding": "4px 0",
            },
            "& .MuiMenuItem-root": {
                "hight": "148px",
                "min-height": "10px !important",
                "&:hover": {"backgroundColor": "grey.200"},
                "& .MuiTypography-root": {"fontSize": 14},
            },
        },
    }

    # @property
    # def file(self) -> QuetzalFile:
    #     return st.session_state.ActionMenuInput[self.key]["file"]

    # @file.setter
    # def file(self, value: QuetzalFile):
    #     st.session_state.ActionMenuInput[self.key]["file"] = value

    # @property
    # def anchor(self) -> Dict[str, int]:
    #     return st.session_state.ActionMenuInput[self.key]["anchor"]

    # @anchor.setter
    # def anchor(self, value: Dict[str, int]):
    #     st.session_state.ActionMenuInput[self.key]["anchor"] = value

    @staticmethod
    def resetAnchor():
        # call this at the end
        for k, v in st.session_state.ActionMenuInput.items():
            st.session_state.ActionMenuInput[k]["anchor"] = None

    @staticmethod
    def initActionMenuState(key):
        if "ActionMenuInput" not in st.session_state:
            st.session_state.ActionMenuInput = {key: {"anchor": None, "file": None}}
        else:
            if key not in st.session_state.ActionMenuInput:
                st.session_state.ActionMenuInput[key] = {"anchor": None, "file": None}

    # @staticmethod
    # def setFile(key, value: QuetzalFile):
    #     st.session_state.ActionMenuInput[key]["file"] = value

    # @staticmethod
    # def getFile(key) -> QuetzalFile:
    #     return st.session_state.ActionMenuInput[key]["file"]

    # @staticmethod
    # def setAnchor(key, value: Dict[str, int]):
    #     st.session_state.ActionMenuInput[key]["anchor"] = value

    # @staticmethod
    # def getAnchor(key) -> Dict[str, int]:
    #     return st.session_state.ActionMenuInput[key]["anchor"]

    def __init__(
        self,
        mode: List[Literal["upload", "edit", "delete", "download"]] = ["upload"],
        key="main",
        action_handler: callable = None
    ):
        self.onClick = action_handler
        self.key = key
        self.mode = mode

        self.initActionMenuState(key)

    def handleClose(self, event, action: Action):
        print("MuiActionMenu.handleClose: ", action)
        st.session_state.ActionMenuInput[self.key]["anchor"] = None
        # self.anchor = None
        if action != "backdropClick" and self.onClick != None:
            self.onClick(st.session_state.ActionMenuInput[self.key]["file"], action)
        print()

    def render(self):
        # anchor = self.anchor
        anchor = st.session_state.ActionMenuInput[self.key]["anchor"]

        with mui.Menu(
            anchorOrigin={
                "vertical": "top",
                "horizontal": "right",
            },
            transformOrigin={
                "vertical": "top",
                "horizontal": "right",
            },
            anchorReference="anchorPosition",
            anchorPosition=anchor,
            open=bool(anchor),
            onClose=self.handleClose,
            sx=MuiActionMenu.menu_style,
        ):
            targetFile: QuetzalFile = st.session_state.ActionMenuInput[self.key]["file"]
            # targetFile: QuetzalFile = self.file
            if not targetFile:
                return
            
            with mui.MenuList():
                ### PUT/UPLOAD SECTION
                if "upload" in self.mode:
                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(event, Action.NEW_DIR),
                        disabled=(targetFile.permission == Permission.READ_ONLY),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.CreateNewFolder(fontSize="small")
                        mui.ListItemText("New Project")

                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(
                            event, Action.UPLOAD_FILE
                        ),
                        disabled=(targetFile.permission == Permission.READ_ONLY),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.UploadFile(fontSize="small")
                        mui.ListItemText("Upload Video")

                    self.mode.remove("upload")

                    if self.mode:
                        mui.Divider()

                ### READ Section
                if "download" in self.mode and targetFile.type != FileType.DIRECTORY:
                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(event, Action.DOWNLOAD),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.FileDownloadOutlined(fontSize="small")
                        mui.ListItemText("Download Video")
                    self.mode.remove("download")

                    if self.mode:
                        mui.Divider()

                ### EDIT SECTION
                if "edit" in self.mode:
                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(event, Action.RENAME),
                        disabled=(targetFile.permission != Permission.FULL_WRITE),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.DriveFileRenameOutline(fontSize="small")
                        mui.ListItemText("Rename")

                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(event, Action.SHARE)
                    ):
                        with mui.ListItemIcon():
                            mui.icon.Share(fontSize="small")
                        mui.ListItemText("Share")
                    self.mode.remove("edit")

                    if self.mode:
                        mui.Divider()

                ### DELETE SECTION
                if "delete" in self.mode:
                    with mui.MenuItem(
                        onClick=lambda event: self.handleClose(event, Action.DELETE),
                        disabled=(targetFile.permission != Permission.FULL_WRITE),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.Delete(fontSize="small")
                        mui.ListItemText("Delete")
                    self.mode.remove("delete")


class MuiFilePathBreadcrumbs:
    visibility_icon = {Visibility.SHARED: "GroupOutlined", Visibility.PRIVATE: "Lock"}

    # def rename():
    #     # call backend rename

    #     st.page

    def __init__(self, file: QuetzalFile, key="main", onClick=None):
        self.file = file
        self.link_path = get_directory_list(file.path)
        self.onClick = onClick
        self.key = key
        self.action_menu = MuiActionMenu(
            mode=["upload", "edit", "delete"],
            key=self.key,
            action_handler=StreamlitDialog.initializeActionPrompt
        )

    def render(self):
        def handle_more(event, key):
            st.session_state.ActionMenuInput[self.key]["anchor"] = {
                "top": event["clientY"],
                "left": event["clientX"],
            }
            # self.action_menu.anchor = {
            #     "top": event["clientY"],
            #     "left": event["clientX"],
            # }
            st.session_state.ActionMenuInput[self.key]["file"] = self.file
            # self.action_menu.file = self.file
            print("You clicked a breadcrumb more")

        def make_click_handler(index):
            def click_handler(event):
                print("You clicked a breadcrumb")
                print("Index:", index)

            return click_handler

        chip_style = {
            "border": "0px",
            "& .MuiChip-label": {
                "fontSize": 18,
            },
        }

        broadcrumb_style = {
            # "px": "0.5rem",
            "py": "0.5rem",
            "& .MuiBreadcrumbs-separator": {"mx": "0rem"},
        }

        state_icon_style = {"padding": "0.5rem", "fontSize": 18}

        with mui.Stack(
            spacing=0,
            direction="row",
            alignItems="center",
            justifyContent="start",
            sx={"my": 0},
        ):
            with mui.Breadcrumbs(
                separator=mui.icon.NavigateNext(fontSize="medium"),
                sx=broadcrumb_style,
            ):
                for i, label in enumerate(
                    self.link_path[:-1]
                ):  # Ensure you're using enumerate() here
                    mui.Chip(
                        label=label,
                        onClick=make_click_handler(
                            i
                        ),  # Use the inner function as the event handler
                        variant="outlined",
                        clickable=True,
                        disableRipple=True,
                        sx=chip_style,
                    )
                mui.Chip(
                    label=self.link_path[-1],
                    onClick=lambda event: handle_more(event, len(self.link_path) - 1),
                    onDelete=lambda event: handle_more(event, len(self.link_path) - 1),
                    deleteIcon=mui.icon.ExpandMore(),
                    variant="outlined",
                    clickable=True,
                    disableRipple=True,
                    sx=chip_style,
                )

            getattr(
                mui.icon, MuiFilePathBreadcrumbs.visibility_icon[self.file.visibility]
            )(sx=state_icon_style)
            self.action_menu.render()


class UploadButton:
    def __init__(self, file: QuetzalFile, key="main", onClick=None):
        self.file = file
        self.onClick = onClick
        self.key = key
        self.action_menu = MuiActionMenu(
            mode=["upload"],
            key=self.key,
            action_handler=StreamlitDialog.initializeActionPrompt
        )

    def render(self):
        def handle_click(event):
            st.session_state.ActionMenuInput[self.key]["anchor"] = {
                "top": event["clientY"],
                "left": event["clientX"],
            }
            # self.action_menu.anchor = {
            #     "top": event["clientY"],
            #     "left": event["clientX"],
            # }
            st.session_state.ActionMenuInput[self.key]["file"] = self.file
            # self.action_menu.file = self.file
            print("UploadButton.handle_click\n")

        with mui.Button(
            variant="contained",
            startIcon=mui.icon.Add(),
            disableRipple=True,
            onClick=lambda event: handle_click(event),
            sx={
                "my": "1rem",
                "mx": "1rem",
                "px": "24px",
                "py": "17.5px",
                "borderRadius": "1rem",
                "bgcolor": "white",
                "color": "initial",
                "&:hover": {"bgcolor": "#edf1f9"},
            },
        ):
            mui.Typography("New", sx={"textTransform": "none", "fontSize": 14})

        self.action_menu.render()


## Action Handler
class StreamlitDialog:
    DIRECTORY_SHARE_INFO = "Sharing this directory will also apply the same settings to all its subdirectories and files."
    FILE_SHARE_INFO = "All other system users will have access to this file based on the assigned permissions."
    DELETE_INFO = " will be deleted forever and you won't be able to restore it."

    @staticmethod
    def initializeActionPrompt(file: QuetzalFile, action: Action):
        print("initializeActionPrompt:", file, action)
        # StreamlitDialog.file = file
        # StreamlitDialog.action = action
        st.session_state.MuiDialogState = {"file": file, "action": action}
        
    # @property
    # def file(self) -> QuetzalFile:
    #     return st.session_state.MuiDialogState["file"]

    # @file.setter
    # def file(self, value: QuetzalFile):
    #     st.session_state.MuiDialogState["file"] = value
        
    # @property
    # def action(self) -> Action:
    #     return st.session_state.MuiDialogState["action"]

    # @action.setter
    # def action(self, value: Action):
    #     st.session_state.MuiDialogState["action"] = value
        
    @staticmethod
    def initDialogState():
        if "MuiDialogState" not in st.session_state:
            st.session_state.MuiDialogState = {"action": None, "file": None}
    
    # @staticmethod
    # def setFile(value: QuetzalFile):
    #     st.session_state.MuiDialogState["file"] = value

    # @staticmethod
    # def getFile() -> QuetzalFile:
    #     return st.session_state.MuiDialogState["file"]

    # @staticmethod
    # def setAction(value: Action):
    #     st.session_state.MuiDialogState["action"] = value

    # @staticmethod
    # def getAction() -> Action:
    #     return st.session_state.MuiDialogState["action"]
    
    @staticmethod
    def resetDialog():
        print("RESET!")
        st.session_state.MuiDialogState = {"action": None, "file": None}
        # print(StreamlitDialog.file, StreamlitDialog.action)
        print(st.session_state.MuiDialogState)
        print(st.session_state.MuiDialogState["action"], st.session_state.MuiDialogState["file"])
        # StreamlitDialog.file = None
        # StreamlitDialog.action = None
        
    
    def __init__(
        self,
        action_handler: callable = None
    ):
        self.action_handler = action_handler
        
        self.initDialogState()

        share_info = lambda file: (
            StreamlitDialog.FILE_SHARE_INFO
            if file.type == FileType.FILE
            else StreamlitDialog.DIRECTORY_SHARE_INFO
        )
        
        dialog_dict = dict()
        dialog_dict[Action.NEW_DIR] = lambda file: StreamlitDialogItem(
            mode="input",
            title="New Project",
            submit_txt="Create",
            onSubmit=partial(file.perform, action=Action.NEW_DIR),
        )
        
        dialog_dict[Action.UPLOAD_FILE] = lambda file: StreamlitDialogItem("upload")
        
        dialog_dict[Action.RENAME] = lambda file: StreamlitDialogItem(
            mode="input",
            title="Rename",
            submit_txt="OK",
            onSubmit=partial(file.perform, action=Action.RENAME),
        )
        
        dialog_dict[Action.SHARE] = lambda file: StreamlitDialogItem(
            mode="share",
            title="Share: " + file.get_name(),
            submit_txt="Save",
            info_txt=share_info(file),
            onSubmit=partial(file.perform, action=Action.SHARE),
        )
        
        dialog_dict[Action.DELETE] = lambda file: StreamlitDialogItem(
            "info",
            "Delete forever?",
            submit_txt="Delete Forever",
            info_txt=file.get_name() + StreamlitDialog.DELETE_INFO,
            onSubmit=partial(file.perform, action=Action.DELETE),
        )
        
        
        dialog_dict[Action.DOWNLOAD] = None
        
        self.dialogs = dialog_dict
            
    def render(self):
        # file: QuetzalFile = self.file
        # action: Action = self.action
        file: QuetzalFile = st.session_state.MuiDialogState["file"]
        action: Action = st.session_state.MuiDialogState["action"]
        
        # print(action, action)
        if not file:
            return
        
        print("there is file")
        # print(self.dialogs)
        print(file)
        self.dialogs[action](file).render()
        

class StreamlitDialogItem:
    font_style = {"textTransform": "none", "fontSize": 14, "fontWeight": 500}
    cancel_button_style = {
        "height": "32px",
        "mx": "0.5rem",
        "px": "16px",
        "py": "16px",
        "borderRadius": "1rem",
        "color": "#4285F4",
        "bgcolor": "transparent",
        "&:hover": {
            "bgcolor": "#edf1f9",
        },
    }
    ok_button_style = {
        "height": "32px",
        "mx": "0.5rem",
        "px": "16px",
        "py": "16px",
        "borderRadius": "1rem",
        "color": "white",
        "bgcolor": "#1266F1",
        "&:hover": {
            "bgcolor": "#4285F4",
        },
    }
    text_field_style = {
        "width": "100%",
        "height": "42px",
        "& .MuiTextField-root": {
            "height": "42px",
        },
        "& .MuiOutlinedInput-root": {
            "height": "42px",
            "& > input": {
                "height": "9px",
            },
        },
    }
    select_permission_style = {
        "height": "42px",
        "& > div": {
            "pt": "14px",
            "pb": "14px",
        },
        "& .MuiOutlinedInput-root": {
            "height": "42px",
        },
    }
    toggle_buttons_style = {
        # "gap": "1rem",
        "width": "100%",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "mx": "1rem",
            "px": "1.0rem",
            "py": "0.2rem",
            "justifyContent": "start",
            "justifyItem": "start",
            "gap": "0.8rem",
            "&:not(:last-of-type)": {
                "borderRadius": "1rem",
            },
            "&:not(:first-of-type)": {
                "borderRadius": "1rem",
            },
        },
        "& .MuiToggleButton-root": {
            "&.Mui-selected": {
                # "color": "#c9e6fd",
                "bgcolor": PRIMARY_COLOR,
                "&:hover": {"bgcolor": PRIMARY_COLOR},
                "&.Mui-disabled": {
                    "bgcolor": "grey.200",
                },
            },
            "&:hover": {"bgcolor": "grey.200"},
            "fontSize": 14,
        },
    }

    DIALOG_WIDTH = 400
    DIALOG_HEIGHT = 197
    DIALOG_INFO_WIDTH = 512
    DIALOG_SHARE_HEIGHT = 251
    DIALOG_DOWNLOAD_WIDTH = 600
    
    @staticmethod
    def initActionMenuState(key):
        if "ActionMenuInput" not in st.session_state:
            st.session_state.ActionMenuInput = {key: {"anchor": None, "file": None}}
        else:
            if key not in st.session_state.ActionMenuInput:
                st.session_state.ActionMenuInput[key] = {"anchor": None, "file": None}


    def __init__(
        self,
        mode: Literal["info", "input", "share"],
        init_value = "",
        title: str = "Rename",
        submit_txt: str = "OK",
        info_txt: str = "",
        key="main",
        onSubmit=None,
    ):
        self.onSubmit = onSubmit
        self.key = key
        self.mode: Literal["info", "input", "share", "upload"] = mode
        self.title = title
        self.submit_txt = submit_txt
        self.info_txt = info_txt
        self.target_value = init_value
        
        self.dialog_width = StreamlitDialogItem.DIALOG_WIDTH
        self.dialog_height = StreamlitDialogItem.DIALOG_HEIGHT
        if mode != "input":
            self.dialog_width = StreamlitDialogItem.DIALOG_INFO_WIDTH
        if mode == "share":
            self.dialog_height = StreamlitDialogItem.DIALOG_SHARE_HEIGHT
        if mode == "upload":
            self.dialog_width = StreamlitDialogItem.DIALOG_DOWNLOAD_WIDTH

    def handleClose(self):
        print("Dialog closed")
        StreamlitDialog.resetDialog()
        st.experimental_rerun()

    def handleFormSubmit(self):
        
        if self.onSubmit != None:
            print("Submit!")
            print(self.target_value)
            self.onSubmit(input=self.target_value)
        self.handleClose()
        

    def takeUserInput(self, event):
        print("takeUserInput called")
        print(event)
        self.target_value = event["target"]["value"]

    def updateShare(self, event):
        self.target_value["shared"] = not self.target_value["shared"] 

    def handlePermission(self, event, controller):
        if controller != None:
            self.target_value["permission"] = controller

    def _renderUpload(self):
        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px", padding="2rem 2rem 2rem"
        )
        dialog_container = float_dialog(
            bool(st.session_state.MuiDialogState["file"]),
            background="var(--default-backgroundColor)",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:
            uploaded_files = st.file_uploader(
                "Choose a Video file", accept_multiple_files=True, key=self.key
            )
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                st.write("filename:", uploaded_file.name)
                st.write(bytes_data)

            with stylable_container(
                key="dialog_container_" + self.key,
                css_styles=f"""{{
                        display: block;
                        div {{
                                width: calc({self.dialog_width}px - 3rem);
                                height: auto;
                            }}
                        iframe {{
                            width: calc({self.dialog_width}px - 3rem);
                            # height: calc({self.dialog_height}px - 8.6px - 1.5rem);
                        }}
                    }}
                    """,
            ):
                with elements("dialog_elements_" + self.key):
                    with mui.Stack(
                        spacing=0,
                        direction="row",
                        alignItems="start",
                        justifyContent="end",
                        sx={"mt": "1.35rem", "mb": "1rem"},
                    ):
                        with mui.Button(
                            onClick=self.handleClose,
                            sx=StreamlitDialogItem.cancel_button_style,
                        ):
                            mui.Typography("Cancel", sx=StreamlitDialogItem.font_style)
                        with mui.Button(
                            type="submit",
                            onClick=self.handleFormSubmit,
                            sx=StreamlitDialogItem.ok_button_style,
                        ):
                            mui.Typography("Done", sx=StreamlitDialogItem.font_style)

    def render(self):
        if self.mode == "upload":
            self._renderUpload()
            return

        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px", padding="1.5rem 1.5rem 0rem"
        )
        dialog_container = float_dialog(
            bool(st.session_state.MuiDialogState["file"]),
            background="var(--default-backgroundColor)",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:
            with stylable_container(
                key="dialog_container_"+self.key,
                css_styles=f"""{{
                        display: block;
                        div {{
                                width: calc({self.dialog_width}px - 3rem);
                                height: auto;
                            }}
                        iframe {{
                            width: calc({self.dialog_width}px - 3rem);
                            height: calc({self.dialog_height}px - 8.6px - 1.5rem);
                        }}
                    }}
                    """,
            ):
                with elements("dialog_elements_" + self.key):
                    with mui.Paper(
                        variant="outlined",
                        sx={
                            "borderRadius": "0px",
                            "border": "0px",
                            "width": "100%",
                            "height": "calc(197px - 3rem)",
                            "m": "0px",
                            "p": "0px",
                            "position": "absolute",
                            "left": "0px",
                            "top": "0px",
                        },
                    ):
                        ## Title + Exit Button
                        with mui.Stack(
                            spacing=0,
                            direction="row",
                            alignItems="start",
                            justifyContent="space-between",
                            sx={"my": 0},
                        ):
                            mui.Typography(
                                variant="h5",
                                component="div",
                                children=[self.title],
                                sx={"margin-bottom": "0.5rem"},
                            )

                            with mui.IconButton(
                                onClick=self.handleClose, sx={"fontSize": 18, "p": 0}
                            ):
                                mui.icon.Close(fontSize="inherit")

                        ## Input Textfield
                        if self.mode == "input":
                            mui.TextField(
                                autoFocus=True,
                                required=True,
                                margin="dense",
                                fullWidth=True,
                                variant="outlined",
                                defaultValue=self.target_value,
                                onChange=lazy(lambda value: self.takeUserInput(value)),
                                sx=StreamlitDialogItem.text_field_style,
                            )

                        ## Info Text
                        if self.mode == "info" or self.mode == "share":
                            mui.Typography(
                                variant="body2",
                                gutterBottom=True,
                                children=self.info_txt,
                                sx={"height": "54px"},
                            )

                        ## Share Setting
                        if self.mode == "share":
                            with mui.Stack(
                                direction="row",
                                spacing=0,
                                alignItems="center",
                                justifyContent="center",
                                sx={"height": "54px"},
                            ):
                                # mui.icon.Compare(sx={"py": "7px"})
                                mui.Typography("Share", sx={"fontSize": 14})
                                switch_on = self.target_value["shared"]
                                mui.Switch(
                                    checked=switch_on,
                                    onChange=self.updateShare,
                                )

                                ## Share Permission
                                toggle_buttons = [
                                    MuiToggleButton(
                                        Permission.READ_ONLY.name,
                                        MuiFileListItem.permission_icon[
                                            Permission.READ_ONLY
                                        ],
                                        "Read Only",
                                    ),
                                    MuiToggleButton(
                                        Permission.POST_ONLY.name,
                                        MuiFileListItem.permission_icon[
                                            Permission.POST_ONLY
                                        ],
                                        "Post Only",
                                    ),
                                    MuiToggleButton(
                                        Permission.FULL_WRITE.name,
                                        MuiFileListItem.permission_icon[
                                            Permission.FULL_WRITE
                                        ],
                                        "Full Write",
                                    ),
                                ]

                                with mui.ToggleButtonGroup(
                                    value=self.target_value["permission"],
                                    onChange=self.handlePermission,
                                    exclusive=True,
                                    disabled=not switch_on,
                                    sx=StreamlitDialogItem.toggle_buttons_style,
                                ):
                                    for button in toggle_buttons:
                                        button.render()

                        ## Cancel + OK button
                        with mui.Stack(
                            spacing=0,
                            direction="row",
                            alignItems="start",
                            justifyContent="end",
                            sx={"mt": "1.35rem", "mb": "1rem"},
                        ):
                            with mui.Button(
                                onClick=self.handleClose,
                                sx=StreamlitDialogItem.cancel_button_style,
                            ):
                                mui.Typography(
                                    "Cancel", sx=StreamlitDialogItem.font_style
                                )
                            with mui.Button(
                                type="submit",
                                onClick=self.handleFormSubmit,
                                sx=StreamlitDialogItem.ok_button_style,
                            ):
                                mui.Typography(
                                    self.submit_txt, sx=StreamlitDialogItem.font_style
                                )
