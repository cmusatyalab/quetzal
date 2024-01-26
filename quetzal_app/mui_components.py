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
import logging
from threading import Lock


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
debug = lambda *args: logger.debug(" ".join([str(arg) for arg in args]))

PRIMARY_COLOR = "#c9e6fd"


class MuiOnFocusHandler:
    all_key = "all_scanners"

    def __init__(self):
        self.clickAwayHandlers = {self.all_key: []}

    def clickAwayHandler(self, key):
        if key in self.clickAwayHandlers:
            for handler in self.clickAwayHandlers[key]:
                handler()
        for handler in self.clickAwayHandlers[self.all_key]:
            handler()

    def registerHandler(self, handler, keys=None):
        # if key is missing, it register handlers to all existing scanner
        if keys == None:
            keys = [self.all_key]
        elif not isinstance(keys, list):
            keys = [keys]

        if isinstance(handler, list):
            for key in keys:
                self.clickAwayHandlers[key].extend(handler)
        else:
            assert callable(handler)
            for key in keys:
                self.clickAwayHandlers[key].append(handler)

    def setScanner(self, key):
        self.clickAwayHandlers[key] = self.clickAwayHandlers.setdefault(key, [])

        with mui.ClickAwayListener(
            mouseEvent="onMouseDown",
            touchEvent="onTouchStart",
            onClickAway=lambda: self.clickAwayHandler(key),
        ):
            mui.Typography(" ")


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

        return self


class MuiSideBarMenu:
    toggle_buttons_style = {
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

        return self


class MuiFileList:
    list_style = {
        "py": "0px",
        # "& .MuiListItem-root": {
        #     "py": "6px",
        #     "&.Mui-selected": {
        #         "bgcolor": "transparent",
        #     },
        # },
    }

    @property
    def selected_item(self) -> QuetzalFile:
        return st.session_state.MuiFileListState[self.key]["item"]

    @selected_item.setter
    def selected_item(self, value: QuetzalFile):
        st.session_state.MuiFileListState[self.key]["item"] = value

    def __init__(
        self, 
        file_list, 
        max_height="50%", 
        key="main",
        onClick=None,
        onDoubleClick=None,
        onClickMore=None,
        
    ):
        self.file_list = file_list
        self.max_height = max_height
        self.key = key
        self.onClick = onClick
        self.onDoubleClick = onDoubleClick
        self.onClickMore = onClickMore

        if "MuiFileListState" not in st.session_state:
            st.session_state.MuiFileListState = {key: {"item": -1, "help_anchor": None}}
        else:
            if key not in st.session_state.MuiFileListState:
                st.session_state.MuiFileListState[key] = {
                    "item": -1,
                    "help_anchor": None,
                }

    def openSymbolHelp(self, event):
        st.session_state.MuiFileListState[self.key]["help_anchor"] = {
            "top": event["clientY"],
            "left": event["clientX"],
        }
        debug(
            "MuiFileList.openSymbolHelp: state =",
            st.session_state.MuiFileListState,
            "\n",
        )

    def closeSymbolHelp(self, event):
        st.session_state.MuiFileListState[self.key]["help_anchor"] = None
        debug(
            "MuiFileList.closeSymbolHelp: state =",
            st.session_state.MuiFileListState,
            "\n",
        )

    def onFocusOut(self):
        st.session_state.MuiFileListState[self.key]["item"] = -1
        debug(
            "MuiFileList.handleClickAway: state =",
            st.session_state.MuiFileListState,
            "\n",
        )
        
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
        def buildItemClickHandler(id):
            def _onClick(event):
                self.selected_item = id
                if self.onClick: self.onClick(event)
            return _onClick

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
            with mui.ListSubheader(
                component="div",
                sx={"px": "0px"},
            ):
                with mui.ListItem(divider=False):
                    mui.ListItemIcon(mui.icon.NoIcon())
                    MuiFileListItem.listTextFormater(filename="Name", owner="Created by")
                    mui.Typography(
                        "State",
                        sx={"px": "8px"},
                        onMouseEnter=self.openSymbolHelp,
                        onMouseLeave=self.closeSymbolHelp,
                    )
                    mui.ListItemIcon(mui.icon.NoIcon())
                mui.Divider()

            with mui.List(
                dense=True,
                sx=MuiFileList.list_style,
            ):
                mui.Divider()
                for i, file in enumerate(self.file_list):
                    MuiFileListItem(
                        key=self.key,
                        onClick=buildItemClickHandler(i),
                        onClickMore=self.onClickMore,
                        file=file,
                        selected=(self.selected_item == i),
                    ).render()
                    mui.Divider()

            self.symbolHelper()
        return self


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
    
    # item_style = {
    #     "py": "6px",
    #     "&.Mui-selected": {
    #         "bgcolor": "transparent",
    #     },
    # }
    
    more_icon_style = {
        False: {
            "margin":"0px 7px 0px 0px !important", 
            "padding":0, 
        },
        True: {
            "margin":"0px 7px 0px 0px !important", 
            "padding":0, 
        }, 
    }
    
    list_style = {
        False: {
            "&:hover": {
                "bgcolor": "grey.200"
            },
            "& .MuiListItem-root": {
                "py": "6px",
                "pr": "0.7rem",
                "&.Mui-selected": {
                    "bgcolor": "transparent",
                },
            },
        },
        True: {
            "bgcolor": PRIMARY_COLOR,
            "&:hover": {
                "bgcolor": PRIMARY_COLOR
            },
            "& .MuiListItem-root": {
                "py": "6px",
                "pr": "0.7rem",
                "&.Mui-selected": {
                    "bgcolor": "transparent",
                },
            },
        }, 
    }
    
    # "&:hover": {"backgroundColor": "grey.200"},
            # "&.Mui-selected": {
                # "bgcolor": PRIMARY_COLOR,
                # "&:hover": {"bgcolor": PRIMARY_COLOR},
            # },
        
    @staticmethod
    def listTextFormater(filename, owner):
        grid = mui.Grid(
            container=True,
            children=[
                mui.Grid(mui.ListItemText(primary=filename), item=True, xs=8),
                mui.Grid(
                    mui.ListItemText(primary=owner),
                    item=True,
                    xs=4,
                ),
            ],
        )
        return grid
    
    def buildClickHandler(self, handler):
        def _handlerWithValue(event: dict):
            # self.lock.acquire()
            debug("before Call", st.session_state.ActionMenuInput)
            debug("MuiFileListItem.valuedHandler: ", handler)
            event["target"] = event.setdefault("target", dict())
            event["target"]["value"] = self.file
            if handler: handler(event)
            if self.moreClicked and self.onClickMore:
                debug("OnClickMOre!")
                self.onClickMore(event)
            debug("after call", st.session_state.ActionMenuInput)
            # self.lock.release()
            # self.lock.

        return _handlerWithValue

    def __init__(
        self,
        file: QuetzalFile,
        selected,
        key,
        onClick=None,
        onClickMore=None,
        onDoubleClick=lambda x: debug("DOUBLECLICKED"),
    ):
        self.file = file
        self.selected = selected
        self.key = key
        self.onClick = onClick
        self.onClickMore= onClickMore
        self.onDoubleClick = onDoubleClick
        self.moreClicked = False
        self.lock = Lock()
        
    def _signalMoreClicked(self):
        self.moreClicked = True

    def render(self):
        file = self.file
        

        color_style = {"color": "#EA4335"} if file.type == FileType.FILE else {}
        list_item_icon = mui.ListItemIcon(
            getattr(
                mui.icon, MuiFileListItem.file_type_icon[file.type][file.visibility]
            )(sx=color_style)
        )

        list_item_text = MuiFileListItem.listTextFormater(
            filename=file.get_name(), owner=file.owner if file.owner else " "
        )

        state_icons = []
        state_icons.append(
            getattr(mui.icon, MuiFileListItem.visibility_icon[file.visibility])(
                sx=MuiFileListItem.STATE_ICON_STYLE
            )
        )
        if file.type == FileType.FILE:
            if file.analysis_progress != AnalysisProgress.NONE:
                state_icons.append(
                    getattr(
                        mui.icon, MuiFileListItem.progress_icon[file.analysis_progress]
                    )(sx=MuiFileListItem.STATE_ICON_STYLE)
                )
        # elif file.type == FileType.DIRECTORY:
        if file.visibility == Visibility.SHARED:
            state_icons.append(
                getattr(mui.icon, MuiFileListItem.permission_icon[file.permission])(
                    sx=MuiFileListItem.STATE_ICON_STYLE
                )
            )

        states = mui.Stack(
            spacing=1,
            direction="row",
            alignItems="center",
            justifyContent="center",
            sx={"my": 0, "pr": "32px", "minWidth": "49px"},
            children=state_icons,
        )

        # secondary_action = mui.IconButton(
        #     edge="end", 
        #     children=[mui.icon.MoreVert()],
        #     onClick=self.buildClickHandler(self.onClickMore),
        #     sx={"p":0},
        # )
            
        with mui.Stack(
            spacing=0.01,
            direction="row",
            alignItems="center",
            justifyContent="center",
            sx=self.list_style[self.selected]
        ):
            mui.ListItem(
                # secondaryAction=secondary_action,
                # divider=True,
                button=True,
                selected=self.selected,
                children=[list_item_icon, list_item_text, states],
                onDoubleClick=self.buildClickHandler(self.onDoubleClick),
                onClick=self.buildClickHandler(self.onClick),
                disableRipple=True,
            )
            mui.IconButton(
                edge="end", 
                children=[mui.icon.MoreVert()],
                onClick=self.buildClickHandler(self.onClickMore),
                sx={
                    "margin":"0px 7px 0px 0px !important", 
                    "padding":0, 
                }, 
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

    @property
    def file(self) -> QuetzalFile:
        return st.session_state.ActionMenuInput[self.key]["file"]

    @file.setter
    def file(self, value: QuetzalFile):
        st.session_state.ActionMenuInput[self.key]["file"] = value

    @property
    def anchor(self) -> Dict[str, int]:
        return st.session_state.ActionMenuInput[self.key]["anchor"]

    @anchor.setter
    def anchor(self, value: Dict[str, int]):
        st.session_state.ActionMenuInput[self.key]["anchor"] = value

    @staticmethod
    def resetAnchor(exculde_keys=[]):
        if not isinstance(exculde_keys, list):
            exculde_keys = [exculde_keys]
        # call this at the end
        for k, v in st.session_state.ActionMenuInput.items():
            if k not in exculde_keys:
                st.session_state.ActionMenuInput[k]["anchor"] = None

        debug("MuiActionMenu.resetAnchor: Resetting")

    @staticmethod
    def initActionMenuState(key):
        if "ActionMenuInput" not in st.session_state:
            st.session_state.ActionMenuInput = {key: {"anchor": None, "file": None}}
        else:
            if key not in st.session_state.ActionMenuInput:
                st.session_state.ActionMenuInput[key] = {"anchor": None, "file": None}

    def __init__(
        self,
        mode: List[Literal["upload", "edit", "delete", "download"]] = ["upload"],
        key="main",
        onClick=None,
    ):
        self.onClick = onClick
        self.key = key
        self.mode = mode

        self.initActionMenuState(key)

    def handleClose(self, event, action: Action):
        debug("MuiActionMenu.handleClose: Action =", action, event, "\n")
        self.anchor = None
        if action != "backdropClick" and self.onClick != None:
            event["target"] = event.setdefault("target", dict())
            event["target"]["value"] = {"file": self.file, "action": action}
            self.onClick(event)

    def buildMenuOpener(self, file: QuetzalFile):
        def _openMenu(event):
            self.openMenu(
                file=file,
                anchor={
                    "top": event["clientY"],
                    "left": event["clientX"],
                },
            )


        return _openMenu

    def openMenu(self, file: QuetzalFile, anchor: Dict[str, int]):
        debug("ActionHandler.OpenMenu: ", file, anchor)
        self.file = file
        self.anchor = anchor

    def render(self):
        anchor = self.anchor
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
            targetFile: QuetzalFile = self.file
            if not targetFile:
                return self

            with mui.MenuList():
                ### PUT/UPLOAD SECTION
                if "upload" in self.mode and targetFile.type != FileType.FILE:
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
                        onClick=lambda event: self.handleClose(event, Action.SHARE),
                        disabled=(targetFile.permission != Permission.FULL_WRITE),
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

        return self


class MuiFilePathBreadcrumbs:
    visibility_icon = {Visibility.SHARED: "GroupOutlined", Visibility.PRIVATE: "Lock"}

    chip_style = {
        "border": "0px",
        "& .MuiChip-label": {
            "fontSize": 18,
        },
    }

    broadcrumb_style = {
        "py": "0.5rem",
        "& .MuiBreadcrumbs-separator": {"mx": "0rem"},
    }

    def __init__(self, file: QuetzalFile, key="main", onClick=None):
        self.file = file
        self.link_path = get_directory_list(file.path)
        self.onClick = onClick
        self.key = key

    def render(self):
        def buildClickHandler(clicked_path):
            def handleClick(event: dict):
                debug("MuiFilePathBreadcrumbs.handleClick: ", clicked_path)
                event["target"] = event.setdefault("target", dict())
                event["target"]["value"] = clicked_path
                if self.onClick:
                    self.onClick(event)

            return handleClick

        state_icon_style = {"padding": "0.5rem", "fontSize": 18}

        with mui.Stack(
            spacing=0,
            direction="row",
            alignItems="center",
            justifyContent="start",
            sx={"my": 0},
        ):
            clip_common_attr = {
                "variant": "outlined",
                "clickable": True,
                "disableRipple": True,
                "sx": MuiFilePathBreadcrumbs.chip_style,
            }

            with mui.Breadcrumbs(
                separator=mui.icon.NavigateNext(fontSize="medium"),
                sx=MuiFilePathBreadcrumbs.broadcrumb_style,
            ):
                for i, label in enumerate(self.link_path[:-1]):
                    mui.Chip(
                        label=label,
                        onClick=buildClickHandler(
                            os.path.join(*self.link_path[: i + 1])
                        ),
                        **clip_common_attr,
                    )
                mui.Chip(
                    label=self.link_path[-1],
                    onClick=buildClickHandler(self.file.path),
                    onDelete=buildClickHandler(self.file.path),
                    deleteIcon=mui.icon.ExpandMore(),
                    **clip_common_attr,
                )

            getattr(
                mui.icon, MuiFilePathBreadcrumbs.visibility_icon[self.file.visibility]
            )(sx=state_icon_style)

        return self


class UploadButton:
    def __init__(self, key="main", onClick=None):
        self.onClick = onClick
        self.key = key

    def render(self):
        def handleClick(event):
            debug("UploadButton.handleClick: Clicked\n")
            if self.onClick:
                self.onClick(event)

        with mui.Button(
            variant="contained",
            startIcon=mui.icon.Add(),
            disableRipple=True,
            onClick=lambda event: handleClick(event),
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

        return self


## Action Handler
class FileActionDialog:
    DIRECTORY_SHARE_INFO = "Sharing this directory will also apply the same settings to all its subdirectories and files."
    FILE_SHARE_INFO = "All other system users will have access to this file based on the assigned permissions."
    DELETE_INFO = " will be deleted forever and you won't be able to restore it."

    @property
    def file(self) -> QuetzalFile:
        return st.session_state.DialogState["file"]

    @file.setter
    def file(self, value: QuetzalFile):
        st.session_state.DialogState["file"] = value

    @property
    def action(self) -> Action:
        return st.session_state.DialogState["action"]

    @action.setter
    def action(self, value: Action):
        st.session_state.DialogState["action"] = value

    @staticmethod
    def initDialogState():
        if "DialogState" not in st.session_state:
            st.session_state.DialogState = {"action": None, "file": None}

    @staticmethod
    def buildDialogOpener():
        def _openDialog(event):
            debug("FileActionDialog.openDialog:", event["target"], "\n")
            st.session_state.DialogState = event["target"]["value"]

        return _openDialog

    @staticmethod
    def closeDialog():
        debug("FileActionDialog.closeDialog\n")
        st.session_state.DialogState = {"action": None, "file": None}
        st.session_state.do_rerun = True

    def _postProcessResult(self, value, action):
        debug("FileActionDialog._postProcessResult\n")
        print(value, action)
        if action == Action.NEW_DIR:
            value = {"dir_name": value}
        if action == Action.UPLOAD_FILE:
            value = {}
        if action == Action.RENAME:
            value = {"file_name": value}
        if action == Action.SHARE:
            value = {
                "permission": getattr(Permission, value["permission"]),
                "shared": Visibility.SHARED if value["shared"] else Visibility.PRIVATE,
            }
        if action == Action.DELETE:
            value = {}
        if action == Action.DOWNLOAD:
            value = {}

        return value

    def handleClose(self, event):
        print(event)
        value = event["target"].get("value", None)
        if value != None:
            value = self._postProcessResult(value, self.action)
            self.file.perform(self.action, value)
        self.closeDialog()

    def __init__(self, key="main", onSubmit=None):
        self.onSubmit = onSubmit

        self.initDialogState()

        share_info = lambda file: (
            FileActionDialog.FILE_SHARE_INFO
            if file.type == FileType.FILE
            else FileActionDialog.DIRECTORY_SHARE_INFO
        )

        dialog_dict = dict()
        dialog_dict[Action.NEW_DIR] = lambda file: MuiDialogItem(
            mode="input",
            init_value="",
            title="New Project",
            submit_txt="Create",
            onSubmit=self.handleClose,
        )
        dialog_dict[Action.UPLOAD_FILE] = lambda file: MuiDialogItem(
            mode="upload",
            submit_txt="Done",
            onSubmit=self.handleClose, 
        )
        dialog_dict[Action.RENAME] = lambda file: MuiDialogItem(
            mode="input",
            init_value=file.get_name(),
            title="Rename",
            submit_txt="OK",
            onSubmit=self.handleClose,
        )
        dialog_dict[Action.SHARE] = lambda file: MuiDialogItem(
            mode="share",
            init_value={
                "shared": file.visibility == Visibility.SHARED,
                "permission": file.permission.name,
            },
            title="Share: " + file.get_name(),
            submit_txt="Save",
            info_txt=share_info(file),
            onSubmit=self.handleClose,
        )
        dialog_dict[Action.DELETE] = lambda file: MuiDialogItem(
            mode="info",
            title="Delete forever?",
            submit_txt="Delete Forever",
            info_txt= f'"{file.get_name()}"' + FileActionDialog.DELETE_INFO,
            onSubmit=self.handleClose,
        )
        dialog_dict[Action.DOWNLOAD] = None

        self.dialogs = dialog_dict

    def openDialog(self, file: QuetzalFile, action: Action):
        self.file = file
        self.action = action

    def render(self):
        debug("FileActionDialog.render: file =", self.file, "\n")

        if not self.file:
            return self

        self.dialogs[self.action](self.file).render()

        return self


class MuiDialogItem:
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
        "width": "100%",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "mx": "0.3rem",
            "py": "0.5rem",
            "justifyContent": "start",
            "justifyItem": "start",
            "gap": "0.3rem",
            "&:not(:last-of-type)": {
                "borderRadius": "1rem",
            },
            "&:not(:first-of-type)": {
                "borderRadius": "1rem",
            },
        },
        "& .MuiToggleButton-root": {
            "&.Mui-selected": {
                "bgcolor": PRIMARY_COLOR,
                "&:hover": {"bgcolor": PRIMARY_COLOR},
                "&.Mui-disabled": {
                    "bgcolor": "grey.200",
                },
            },
            "&:hover": {"bgcolor": "grey.200"},
            "fontSize": 14,
            "& > p": {"lineHeight": 1.2},
        },
    }

    DIALOG_WIDTH = 400
    DIALOG_HEIGHT = 197
    DIALOG_INFO_WIDTH = 512
    DIALOG_SHARE_HEIGHT = 251
    DIALOG_DOWNLOAD_WIDTH = 600
    
    share_option_buttons = [
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

    @staticmethod
    def initDialogItemData():
        if "DialogItemState" not in st.session_state:
            st.session_state.DialogItemState = None

    def __init__(
        self,
        mode: Literal["info", "input", "share"],
        init_value="",
        title: str = "Rename",
        submit_txt: str = "OK",
        info_txt: str = "",
        key="main",
        open=True,
        onSubmit=None,
        onClose=None,
    ):
        self.onSubmit = onSubmit
        self.onClose = onSubmit if onClose == None else onClose
        self.key = key
        self.mode: Literal["info", "input", "share", "upload"] = mode
        self.title = title
        self.submit_txt = submit_txt
        self.info_txt = info_txt
        self.target_value = init_value
        self.open = open

        self.initDialogItemData()
        if st.session_state.DialogItemState == None:
            st.session_state.DialogItemState = init_value

        self.dialog_width = MuiDialogItem.DIALOG_WIDTH
        self.dialog_height = MuiDialogItem.DIALOG_HEIGHT
        if mode != "input":
            self.dialog_width = MuiDialogItem.DIALOG_INFO_WIDTH
        if mode == "share":
            self.dialog_height = MuiDialogItem.DIALOG_SHARE_HEIGHT
        if mode == "upload":
            self.dialog_width = MuiDialogItem.DIALOG_DOWNLOAD_WIDTH

    def handleClose(self, event):
        debug("MuiDialogItem.handleClose: Dialog closed\n")
        if self.onClose:
            self.onClose(event)
        st.session_state.DialogItemState = None

    def handleFormSubmit(self, event):
        debug("MuiDialogItem.handleFormSubmit: onSubmit =", self.onSubmit)
        if self.onSubmit != None:
            event["target"] = event.setdefault("target", dict())
            event["target"]["value"] = st.session_state.DialogItemState
            debug(
                "MuiDialogItem.handleFormSubmit: result = ",
                st.session_state.DialogItemState,
            )

            self.onSubmit(event)
        st.session_state.DialogItemState = None

    def takeUserInput(self, event):
        debug("MuiDialogItem.takeUserInput: event['target'] = ", event["target"])
        st.session_state.DialogItemState = event["target"]["value"]

    def updateShare(self, event):
        st.session_state.DialogItemState[
            "shared"
        ] = not st.session_state.DialogItemState["shared"]

    def handlePermission(self, event, controller):
        if controller != None:
            st.session_state.DialogItemState["permission"] = controller

    def render_title(self):
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
                
    def render_input_textfield(self):
        mui.TextField(
            defaultValue=st.session_state.DialogItemState,
            autoFocus=True,
            required=True,
            fullWidth=True,
            margin="dense",
            variant="outlined",
            onChange=lazy(lambda value: self.takeUserInput(value)),
            sx=MuiDialogItem.text_field_style,
        )
        
    def render_info_text(self):
        mui.Typography(
            variant="body2",
            gutterBottom=True,
            children=self.info_txt,
            sx={"height": "54px"},
        )
        
    def render_share_options(self):
        with mui.Stack(
            direction="row",
            spacing=0,
            alignItems="center",
            justifyContent="center",
            sx={"height": "54px"},
        ):
            # mui.icon.Compare(sx={"py": "7px"})
            mui.Typography("Share", sx={"fontSize": 14})
            switch_on = st.session_state.DialogItemState["shared"]
            mui.Switch(
                checked=switch_on,
                onChange=self.updateShare,
            )

            ## Share Permission
            with mui.ToggleButtonGroup(
                value=st.session_state.DialogItemState[
                    "permission"
                ],
                onChange=self.handlePermission,
                exclusive=True,
                disabled=not switch_on,
                sx=MuiDialogItem.toggle_buttons_style,
            ):
                for button in self.share_option_buttons:
                    button.render()
            
    def render_submit_button(self):
        with mui.Stack(
            spacing=0,
            direction="row",
            alignItems="start",
            justifyContent="end",
            sx={"mt": "1.35rem", "mb": "1rem"},
        ):
            with mui.Button(
                onClick=self.handleClose,
                sx=MuiDialogItem.cancel_button_style,
            ):
                mui.Typography("Cancel", sx=MuiDialogItem.font_style)
            with mui.Button(
                type="submit",
                onClick=self.handleFormSubmit,
                sx=MuiDialogItem.ok_button_style,
            ):
                mui.Typography(
                    self.submit_txt, sx=MuiDialogItem.font_style
                )
                    
    def renderUpload(self):
        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px", padding="2rem 2rem 2rem"
        )
        dialog_container = float_dialog(
            self.open,
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
                    self.render_submit_button()
                            
    def render(self):
        if self.mode == "upload":
            self.renderUpload()
            return self

        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px", padding="1.5rem 1.5rem 0rem"
        )
        dialog_container = float_dialog(
            self.open,
            background="var(--default-backgroundColor)",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:
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
                        self.render_title()

                        ## Input Textfield
                        if self.mode == "input":
                            self.render_input_textfield()

                        ## Info Text
                        if self.mode == "info" or self.mode == "share":
                            self.render_info_text()

                        ## Share Setting
                        if self.mode == "share":
                            self.render_share_options()

                        ## Cancel + OK button
                        self.render_submit_button()
        return self
