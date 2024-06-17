from collections import defaultdict
from pathlib import Path
import datetime
import logging

import torch
import streamlit as st
from streamlit_elements import elements, mui, lazy
from streamlit_extras.stylable_container import stylable_container
from streamlit_float import float_css_helper, float_dialog

from typing import Dict, List, Literal

from quetzal.dtos.dtos import (
    QuetzalFile, FileType, Permission, Visibility,
    AnalysisProgress, Action, AccessMode
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
debug = lambda *args: logger.debug(" ".join([str(arg) for arg in args]))

ELEMENT_BOTTOM_MARGIN = "7.594px"
PRIMARY_COLOR = "#c9e6fd"
GOOGLE_RED = "#EA4335"
GOOGLE_DARK_RED = "#d33a2e"
GOOGLE_DARK_BLUE = "#1266F1"
GOOGLE_DEEP_BLUE = "#e9eef6"
GOOGLE_BLUE = "#4285F4"
GOOGLE_LIGHT_BLUE = "#edf1f9"

def css_to_dict(css_str):
    """
    Converts CSS string to a Python dictionary
    """
    css_dict = {}
    # Split the string into blocks
    blocks = css_str.split('},')
    for block in blocks:
        if block.strip():  # Check if the block is not just whitespace
            selector, styles = block.split('{', 1)
            selector = selector.strip().replace('\n', '')
            styles_dict = {}
            for line in styles.split(';'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    styles_dict[key.strip()] = value.strip()
            css_dict[selector] = styles_dict
    return css_dict

def dict_to_css(css_dict):
    """
    Converts a Python dictionary to a CSS string
    """
    css_str = ""
    for selector, styles in css_dict.items():
        css_str += selector + " {\n"
        for key, value in styles.items():
            css_str += f"    {key}: {value};\n"
        css_str += "}\n\n"
    return css_str.strip()

scroll_style = {
    "&::-webkit-scrollbar" : {
        "background-color": "transparent",
        "width": "8px",
        "position": "absolute",
        "left": 0,
    },

    "&::-webkit-scrollbar-track": {
        "background-color": "transparent",
    },

    "&::-webkit-scrollbar-thumb": {
        "background-color": "#babac0",
        "border-radius": "4px",
    },
    
    "&::-webkit-scrollbar-thumb:hover": {
        "background-color":"#757575",
    },

    "&::-webkit-scrollbar-button": {
        "display": "none",
    },
}

scroll_style_css = dict_to_css(scroll_style)


TEST_FILE_LIST = [
    {
        "name": "Project Name 1",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 1",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 2",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Project Name 2",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 3",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 4",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 3",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Video Name 5",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 4",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 6",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 5",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 7",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Project Name 6",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 8",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 7",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Video Name 9",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Project Name 8",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Video Name 10",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.FULL,
            "permission": Permission.POST_ONLY,
        },
    },
    {
        "name": "Project Name 9",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Very Long Video Name 11",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Very Very Long Project Name 10",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.READ_ONLY,
        },
    },
    {
        "name": "Video Name 12",
        "type": FileType.FILE,
        "state": {
            "visibility": Visibility.PRIVATE,
            "analyzed": AnalysisProgress.HALF,
            "permission": Permission.FULL_WRITE,
        },
    },
    {
        "name": "Project Name 11",
        "type": FileType.DIRECTORY,
        "state": {
            "visibility": Visibility.SHARED,
            "analyzed": AnalysisProgress.NONE,
            "permission": Permission.FULL_WRITE,
        },
    },
]


def convert_to_quetzal_files(file_info):
    name = file_info["name"]
    type = file_info["type"]
    state = file_info["state"]
    visibility = state["visibility"]
    analysis_progress = state["analyzed"]
    permission = state["permission"]

    # Assuming the owner is the same for all files, using a placeholder
    owner = "jinho yi"

    quetzal_file = QuetzalFile(
        path=name,  # Using 'name' as 'path' for simplicity
        owner=owner,
        type=type,
        visibility=visibility,
        analysis_progress=analysis_progress,
        permission=permission,
    )

    return quetzal_file


outlined_button_style = {
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

contained_button_style = {
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
    "&.Mui-disabled": {"bgcolor": "grey.200"},
}


def setEventValue(event, value):
    event["target"] = event.setdefault("target", dict())
    event["target"]["value"] = value
    return event


def getEventValue(event):
    return event["target"]["value"]


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
        "py": "0.5rem",
        "& .MuiToggleButtonGroup-grouped": {
            "border": 0,
            "mx": "0.5rem",
            "px": "1rem",
            "py": "0.25rem",
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

    def __init__(
        self,
        toggle_buttons: List[MuiToggleButton],
        key="main",
        onChange=None,
    ):
        self.toggle_buttons = toggle_buttons
        self.key = key
        self.onChange = onChange

        if "MuiSideBarMenu" not in st.session_state:
            st.session_state.MuiSideBarMenu = {key: self.toggle_buttons[0].value}
        else:
            if key not in st.session_state.MuiSideBarMenu:
                st.session_state.MuiSideBarMenu[key] = self.toggle_buttons[0].value

    def render(self):
        def handleController(event, controller):
            if controller != None:
                st.session_state.MuiSideBarMenu[self.key] = controller
                setEventValue(event, controller)
                if self.onChange:
                    self.onChange(event)

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


class ClickedFiles:
    def __init__(self, session_state_key):
        self.session_state_key = session_state_key

    def __getitem__(self, key):
        return st.session_state.MuiFileListState[self.session_state_key][
            "last_clicked"
        ].get(key, None)

    def __setitem__(self, key, value):
        st.session_state.MuiFileListState[self.session_state_key]["last_clicked"][
            key
        ] = value
        
    def clear(self):
        st.session_state.MuiFileListState[self.session_state_key]["last_clicked"].clear()


class MuiFileList:
    list_style = {
        "py": "0px",
    }

    @property
    def selected_item(self) -> QuetzalFile:
        return st.session_state.MuiFileListState[self.key]["item"]

    @selected_item.setter
    def selected_item(self, value: QuetzalFile):
        st.session_state.MuiFileListState[self.key]["item"] = value

    @property
    def last_clicked(self):
        return self._last_clicked

    def __init__(
        self,
        file_list,
        max_height="50%",
        key="main",
        tightDisplay=False,
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
        self._last_clicked = ClickedFiles(key)
        self.tightDisplay = tightDisplay

        if "MuiFileListState" not in st.session_state:
            st.session_state.MuiFileListState = {
                key: {"item": None, "help_anchor": None, "last_clicked": dict()}
            }
        else:
            if key not in st.session_state.MuiFileListState:
                st.session_state.MuiFileListState[key] = {
                    "item": None,
                    "help_anchor": None,
                    "last_clicked": dict(),
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
        st.session_state.MuiFileListState[self.key]["item"] = None
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
                    mui.Typography("Shallow Analysis Done", sx=text_style)
                    getattr(
                        mui.icon, MuiFileListItem.progress_icon[AnalysisProgress.FULL]
                    )(sx=icon_style)
                    mui.Typography("Deep Analysis Done", sx=text_style)

    def render(self):
        def buildItemClickHandler(file):
            def _onClick(event):
                if self.selected_item == file:
                    self.selected_item = None
                else:
                    self.selected_item = file

                last_click = self.last_clicked[file]
                # if (
                #     last_click
                #     and last_click + datetime.timedelta(seconds=0.7)
                #     > datetime.datetime.now()
                # ):
                if last_click:
                    if self.onDoubleClick:
                        if getEventValue(event).type == FileType.DIRECTORY:
                            self.selected_item = None
                        self.onDoubleClick(event)
                    self.last_clicked[file] = None
                    return
                
                else:
                    self.last_clicked.clear()
                    self.last_clicked[file] = datetime.datetime.now()

                if self.onClick:
                    self.onClick(event)

                debug("Clicked:", self.last_clicked[file])

            return _onClick
 
        scroll_style.update()
        with mui.Paper(
            variant="outlined",
            sx={
                "width": f"calc(100% - {1.0 if self.tightDisplay else 0.5}rem)",
                "borderRadius": "0px",
                "border": "0px",
                "height": self.max_height,
                "bgcolor": "transparent",
                "padding": "0px",
                "overflow-y": "scroll", ## adds extra padding of 0.5rem 
                "padding-left": "0.5rem",
                "padding-right": "0.5rem" if self.tightDisplay else "1rem",
                **scroll_style
            }
        ):
            with mui.ListSubheader(
                component="div",
                sx={"px": "0px"},
            ):
                with mui.ListItem(divider=False):
                    MuiFileListItem.listTextFormater(
                        filename="Name", owner="Created by"
                    )
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
                if len(self.file_list) == 0:
                    with mui.Stack(
                        alignItems="center",
                        justifyContent="center",
                        sx={
                            "borderRadius": "0.3rem",
                            "border": "0px",
                            "bgcolor": "transparent",
                            "padding": "0",
                            "height": f"18rem",
                            "maxHeight": f"calc({self.max_height} - 48px)",
                        },
                    ):
                        mui.icon.FolderOff(
                            sx={"fontSize": "5rem", "color": GOOGLE_BLUE}
                        )
                        mui.Typography(
                            "Add new project or upload video", sx={"fontSize": "0.8rem"}
                        )

                for i, file in enumerate(self.file_list):
                    MuiFileListItem(
                        key=self.key,
                        onClick=buildItemClickHandler(file),
                        onClickMore=self.onClickMore,
                        # onDoubleClick=self.onDoubleClick,
                        file=file,
                        selected=(self.selected_item == file),
                    ).render()
                    mui.Divider()

            self.symbolHelper()
        return self


class MuiFileListItem:
    ## Onclick event.target.value will contain the target File

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

    more_icon_style = {
        False: {
            "margin": "0px 7px 0px 0px !important",
            "padding": 0,
        },
        True: {
            "margin": "0px 7px 0px 0px !important",
            "padding": 0,
        },
    }

    list_style = {
        False: {
            "&:hover": {"bgcolor": "grey.200"},
            "& .MuiListItem-root": {
                "py": "6px",
                "pr": "0.7rem",
                "&.Mui-selected": {
                    "bgcolor": "transparent",
                },
                "&:hover": {"bgcolor": "transparent"},
            },
        },
        True: {
            "bgcolor": PRIMARY_COLOR,
            "&:hover": {"bgcolor": PRIMARY_COLOR},
            "& .MuiListItem-root": {
                "py": "6px",
                "pr": "0.7rem",
                "&.Mui-selected": {
                    "bgcolor": "transparent",
                },
                "&:hover": {"bgcolor": "transparent"},
            },
        },
    }

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
            debug("MuiFileListItem.valuedHandler: ", handler)
            setEventValue(event, self.file)
            if handler:
                handler(event)

        return _handlerWithValue

    def __init__(
        self,
        file: QuetzalFile,
        selected,
        key,
        onClick=None,
        onClickMore=None,
        # onDoubleClick=lambda x: debug("DOUBLECLICKED"),
    ):
        self.file = file
        self.selected = selected
        self.key = key
        self.onClick = onClick
        self.onClickMore = onClickMore
        # self.onDoubleClick = onDoubleClick

    def render(self):
        file = self.file

        color_style = {"color": "#d33a2e"} if file.type == FileType.FILE else {}
        list_item_icon = mui.ListItemIcon(
            getattr(
                mui.icon, MuiFileListItem.file_type_icon[file.type][file.visibility]
            )(sx=color_style)
        )

        list_item_text = MuiFileListItem.listTextFormater(
            filename=file.name, owner=file.createdBy if file.createdBy else " "
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
            sx={"my": 0, "pr": "32px", "minWidth": "58px"},
            children=state_icons,
        )

        with mui.Stack(
            spacing=0.01,
            direction="row",
            alignItems="center",
            justifyContent="center",
            sx=self.list_style[self.selected],
        ):
            mui.ListItem(
                button=True,
                selected=self.selected,
                children=[list_item_icon, list_item_text, states],
                # onDoubleClick=self.buildClickHandler(self.onDoubleClick),
                onClick=self.buildClickHandler(self.onClick),
                disableRipple=True,
            )
            mui.IconButton(
                edge="end",
                children=[mui.icon.MoreVert()],
                onClick=self.buildClickHandler(self.onClickMore),
                sx={
                    "margin": "0px 7px 0px 0px !important",
                    "padding": 0,
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

    @staticmethod
    def initActionMenuState(key):
        if "ActionMenuInput" not in st.session_state:
            st.session_state.ActionMenuInput = {key: {"anchor": None, "file": None}}
        else:
            if key not in st.session_state.ActionMenuInput:
                st.session_state.ActionMenuInput[key] = {"anchor": None, "file": None}

    def __init__(
        self,
        mode: List[Literal["upload", "edit", "delete", "download", "move"]] = [
            "upload"
        ],
        onlyNew: bool = False,
        key="main",
        onClick=None,
    ):
        self.onClick = onClick
        self.key = key
        self.mode = mode
        self.onlyNew = onlyNew

        self.initActionMenuState(key)

    def buildHandleClose(self, action):
        return lambda event: self.handleClose(event, action)

    def handleClose(self, event, action: Action):
        debug("MuiActionMenu.handleClose: Action =", action, event, "\n")
        self.anchor = None
        if action != "backdropClick" and self.onClick != None:
            setEventValue(event, {"file": self.file, "action": action})
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
                        onClick=self.buildHandleClose(Action.NEW_DIR),
                        disabled=(
                            targetFile.permission == Permission.READ_ONLY
                            and targetFile.mode != AccessMode.OWNER
                        ),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.CreateNewFolder(fontSize="small")
                        mui.ListItemText("New Project")

                    if not self.onlyNew:
                        with mui.MenuItem(
                            onClick=self.buildHandleClose(Action.UPLOAD_FILE),
                            disabled=(
                                targetFile.permission == Permission.READ_ONLY
                                and targetFile.mode != AccessMode.OWNER
                            ),
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
                        onClick=self.buildHandleClose(Action.DOWNLOAD),
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
                        onClick=self.buildHandleClose(Action.RENAME),
                        disabled=(
                            targetFile.permission != Permission.FULL_WRITE
                            and targetFile.mode != AccessMode.OWNER
                            or targetFile.path == Path()
                        ),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.DriveFileRenameOutline(fontSize="small")
                        mui.ListItemText("Rename")

                    with mui.MenuItem(
                        onClick=self.buildHandleClose(Action.SHARE),
                        disabled=(
                            targetFile.permission != Permission.FULL_WRITE
                            and targetFile.mode != AccessMode.OWNER
                            or targetFile.path == Path()
                        ),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.Share(fontSize="small")
                        mui.ListItemText("Share")
                    self.mode.remove("edit")

                    if self.mode:
                        mui.Divider()

                if "move" in self.mode:
                    if targetFile.mode == AccessMode.OWNER:
                        with mui.MenuItem(
                            onClick=self.buildHandleClose(Action.MOVE),
                            disabled=(
                                targetFile.permission != Permission.FULL_WRITE
                                and targetFile.mode != AccessMode.OWNER
                                or targetFile.path == Path()
                            ),
                        ):
                            with mui.ListItemIcon():
                                mui.icon.DriveFileMoveOutlined(fontSize="small")
                            mui.ListItemText("Move")

                    with mui.MenuItem(
                        onClick=self.buildHandleClose(Action.COPY),
                        # disabled=(targetFile.permission != Permission.FULL_WRITE),
                    ):
                        with mui.ListItemIcon():
                            mui.icon.ContentCopyOutlined(fontSize="small")
                        mui.ListItemText("Copy")
                    self.mode.remove("move")

                    if self.mode:
                        mui.Divider()

                ### DELETE SECTION
                if "delete" in self.mode:
                    with mui.MenuItem(
                        onClick=self.buildHandleClose(Action.DELETE),
                        disabled=(
                            targetFile.permission != Permission.FULL_WRITE
                            and targetFile.mode != AccessMode.OWNER
                            or targetFile.path == Path()
                        ),
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

    chip_style_small = {
        "border": "0px",
        "& .MuiChip-label": {
            "fontSize": 12,
        },
    }

    broadcrumb_style = {
        "py": "0.5rem",
        "& .MuiBreadcrumbs-separator": {"mx": "0rem"},
    }

    def __init__(
        self,
        file: QuetzalFile,
        key="main",
        onClick=None,
        size: Literal["small", "medium"] = "medium",
    ):
        self.file = file
        self.link_path = (Path("home") / file.path).parts
        self.onClick = onClick
        self.key = key
        self.size: Literal["small", "medium"] = size

    def render(self):
        def buildClickHandler(clicked_path):
            def handleClick(event: dict):
                debug("MuiFilePathBreadcrumbs.handleClick: ", clicked_path)
                setEventValue(event, clicked_path)
                if self.onClick:
                    self.onClick(event)

            return handleClick

        state_icon_style = {"padding": "0.5rem", "fontSize": 18}

        chip_style = {"small": self.chip_style_small, "medium": self.chip_style}

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
                "sx": chip_style[self.size],
            }

            with mui.Breadcrumbs(
                separator=mui.icon.NavigateNext(fontSize=self.size),
                sx=MuiFilePathBreadcrumbs.broadcrumb_style,
            ):
                for i, label in enumerate(self.link_path[:-1]):
                    mui.Chip(
                        label=label,
                        onClick=buildClickHandler(Path(*self.link_path[1 : i + 1])),
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


class MuiEditButton:
    font_style = {"fontSize": "0.8rem", "color": GOOGLE_BLUE}
    disabled_style = {"fontSize": "0.8rem", "color": "grey.400"}

    def __init__(
        self,
        mode: Literal["edit", "share", "analysis"] = "edit",
        key="main",
        onClick=None,
        disabled=False,
    ):
        self.onClick = onClick
        self.key = key
        self.mode = mode
        self.disabled = disabled
        self.style = self.disabled_style if disabled else self.font_style

        match mode:
            case "edit":
                self.icon = "Edit"
                self.text = "Edit"
            case "share":
                self.icon = "Share"
                self.text = "Setting"
            case "analysis":
                self.icon = "AutoAwesome"
                self.text = "Analyze"
            case _:
                raise "No Such mode: " + mode

    def onClickHander(self, event):
        debug("EditButton.onClickHandler:", "help")
        if self.onClick:
            self.onClick(event)

    def render(self):
        with mui.Button(
            variant="text",
            startIcon=getattr(mui.icon, self.icon)(sx=self.style),
            sx={
                "height": "min-content",
                "margin": 0,
                "padding": 0,
            },
            onClick=self.onClickHander,
            disabled=self.disabled,
        ):
            mui.Typography(self.text, sx=self.style)

        return self


class MuiInfoDisplay:
    @staticmethod
    def parse_info_text(info_text):
        """
        filetext should have following format

        <section header>::= <section info no new line>

        for example

        Route::= Hot-metal-bridge
        Created by::= Admin
        """
        lines = info_text.strip().split("\n")
        return [line.split("::=") for line in lines if line]

    @property
    def expand(self) -> bool:
        return st.session_state.MuiInfoDisplayState[self.key]["expand"]

    @expand.setter
    def expand(self, value: bool):
        st.session_state.MuiInfoDisplayState[self.key]["expand"] = value

    def initInfoDisplayState(self, closed):
        if "MuiInfoDisplayState" not in st.session_state:
            st.session_state.MuiInfoDisplayState = {self.key: {"expand": not closed}}
        elif self.key not in st.session_state.MuiInfoDisplayState:
            st.session_state.MuiInfoDisplayState[self.key] = {"expand": not closed}

    def __init__(
        self,
        title="File Metadata",
        expendable=False,
        closed=False,
        divider=True,
        info_items=None,
        key="main",
        secondaryItem: MuiEditButton = None,
    ):
        """
        Info should be a list of (<sectoin header>,<section info>) pairs
        """

        self.title = title
        self.divier = divider
        self.expendable = expendable
        self.key = key
        self.secondaryItem = secondaryItem
        self.info_items = info_items

        self.initInfoDisplayState(closed)

    def render(self):
        def toggle_expand():
            self.expand = not self.expand

        with mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "1rem",
                "border": "0px",
                "bgcolor": "transparent",
                "padding": "0.5rem 1rem",
            },
        ):
            with mui.Stack(
                spacing=1,
                direction="row",
                alignItems="end",
                justifyContent="space-between",
                sx={"padding": 0, "margin": 0, "py": "0.5rem"},
            ):
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="center",
                    justifyContent="center",
                    sx={"padding": 0, "margin": 0, "height": "min-content"},
                ):
                    mui.Typography(
                        self.title,
                        variant="subtitle1",
                        sx={"lineHeight": 1.2, "fontWeight": 400, "fontSize": "1.0rem"},
                    )

                    if self.expendable and self.info_items:
                        with mui.IconButton(
                            sx={"fontSize": 16, "p": 0}, onClick=toggle_expand
                        ):
                            if self.expand:
                                mui.icon.ExpandLess(fontSize="inherit")
                            else:
                                mui.icon.ExpandMore(fontSize="inherit")

                if self.secondaryItem:
                    self.secondaryItem.render()

            if self.expand and self.info_items:
                for key, value in self.info_items:
                    if key and value:
                        mui.Typography(
                            variant="body2",
                            children=key.strip(),
                            sx={
                                "lineHeight": 1.1,
                                "fontWeight": 400,
                                "fontSize": "0.75rem",
                            },
                        )
                        mui.Typography(
                            variant="body2",
                            children=value.strip(),
                            sx={"lineHeight": 1.2, "pb": "1rem", "fontSize": "0.9rem"},
                    )
        if self.divier:
            mui.Divider()

        return self


class MuiUploadButton:
    def __init__(self, key="main", onClick=None):
        self.onClick = onClick
        self.key = key

    def render(self):
        def handleClick(event):
            debug("MuiUploadButton.handleClick: Clicked\n")
            if self.onClick:
                self.onClick(event)

        with mui.Button(
            variant="contained",
            startIcon=mui.icon.Add(),
            disableRipple=True,
            onClick=lambda event: handleClick(event),
            sx={
                "my": "0.5rem",
                "mx": "0.5rem",
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
    FILE_SHARE_INFO = "All other system users will have access to this file based on the assigned permissions. The file won't be accessible by others if any of its parent directory is not shared"
    DELETE_INFO = " will be deleted forever and you won't be able to restore it."
    ANALYZE_INFO = """Choose the analysis option that best suits your needs:

1. Deep Analysis (Database + Query): Opt for this to register the video for both database and query usage. This comprehensive method takes longer to complete.
2. Shallow Analysis (Query-Only): Select this for a quicker analysis, suitable for registering the video solely as a query video. It generally takes about one-third of the time needed for deep analysis.

Note: To enable comparisons within your project, at least one video must be set up as a Database video."""

    option_convert = {
        "Deep Analysis": AnalysisProgress.FULL,
        "Shallow Analysis": AnalysisProgress.HALF,
    }
    analyze_options = ["Deep Analysis", "Shallow Analysis"]

    @property
    def file(self) -> QuetzalFile:
        return st.session_state.DialogState[self.key]["file"]

    @file.setter
    def file(self, value: QuetzalFile):
        st.session_state.DialogState[self.key]["file"] = value

    @property
    def action(self) -> Action:
        return st.session_state.DialogState[self.key]["action"]

    @action.setter
    def action(self, value: Action):
        st.session_state.DialogState[self.key]["action"] = value

    @property
    def onRender(self) -> Action:
        return st.session_state.DialogState[self.key]["onRender"]

    @staticmethod
    def initDialogState(key):
        if "DialogState" not in st.session_state:
            st.session_state.DialogState = {key: {"action": None, "file": None}}
        elif key not in st.session_state.DialogState:
            st.session_state.DialogState[key] = {"action": None, "file": None}

    @staticmethod
    def buildDialogOpener(key):
        def _openDialog(event):
            debug("FileActionDialog.openDialog:", key, "\n")
            st.session_state.DialogState[key] = getEventValue(event)
            print(st.session_state.DialogState[key])
        return _openDialog

    @staticmethod
    def closeDialog(key):
        debug("FileActionDialog.closeDialog\n")
        st.session_state.DialogState[key] = {"action": None, "file": None}

    def _postProcessResult(self, value, action):
        debug("FileActionDialog._postProcessResult\n")

        match action:
            case Action.NEW_DIR:
                value = {"dir_name": value}
            case Action.UPLOAD_FILE:
                value = {"uploaded_files": value}
            case Action.RENAME:
                value = {"new_file_name": value}
            case Action.SHARE:
                value = {
                    "permission": getattr(Permission, value["permission"]),
                    "shared": (
                        Visibility.SHARED if value["shared"] else Visibility.PRIVATE
                    ),
                }
            case Action.DELETE:
                value = {}
            case Action.DOWNLOAD:
                value = {}
            case Action.ANALYZE:
                value = {
                    "option": self.option_convert[value],
                    "engine": "vpr_engine.anyloc_engine.AnyLocEngine",
                    "device": self.device
                }
            case Action.EDIT:
                value = {"value": value}
            case Action.COPY | Action.MOVE:
                value = {"dest_dir": value["target_directory"]}

        return value

    def handleClose(self, event):
        if self.action != Action.ANALYZE:
            self.handleSubmit(event)
        else:
            self.handleAnalyze(event)

        self.closeDialog(self.key)

    def _handleSubmit(self, value, action=None):
        if action == None:
            action = self.action
        try:
            result = self.file.perform(action, value)
            if result != None:
                # st.session_state.page_states["info"] = MuiInfo(result, "success")
                MuiInfo(result, "success")
        except Exception as e:
            # raise e
            # st.session_state.page_states["info"] = MuiInfo(f"{e}", "error")
            MuiInfo(f"{e}", "error")

    def handleSubmit(self, event):
        value = event["target"].get("value", None)
        if value != None:
            value = self._postProcessResult(value, self.action)
            self._handleSubmit(value)

    def handleAnalyze(self, event):
        value = event["target"].get("value", None)
        debug("FileActionDialog.handleAnalyze: ", event)
        if value == None:
            return

        debug("FileActionDialog.handleAnalyze: doing p0stProcess")
        value = self._postProcessResult(value, Action.ANALYZE)
        if self.file.analysis_progress >= value["option"]:
            debug("FileActionDialog.handleAnalyze: already done")
            # st.session_state.page_states["info"] = MuiInfo(
            MuiInfo(
                f'"{self.file.name}" has been already analyzed with the given option'
            )
            print(st.session_state.page_states)
            setEventValue(
                event,
                {
                    "file": self.file,
                    "action": "process",
                    "onRender": None,
                },
            )
            FileActionDialog.buildDialogOpener(self.key)(event)
        else:
            setEventValue(
                event,
                {
                    "file": self.file,
                    "action": "process",
                    "onRender": lambda: FileActionDialog._handleSubmit(
                        self, value, Action.ANALYZE
                    ),
                },
            )
            FileActionDialog.buildDialogOpener(self.key)(event)

    def __init__(self, key="main", onSubmit=None, device=torch.device("cuda:0")):
        self.onSubmit = onSubmit
        self.key = key
        self.device = device

        self.initDialogState(key)

    def openDialog(self, file: QuetzalFile, action: Action):
        self.file = file
        self.action = action

    def render(self):
        if not self.file:
            return self

        share_info = lambda file: (
            FileActionDialog.FILE_SHARE_INFO
            if file.type == FileType.FILE
            else FileActionDialog.DIRECTORY_SHARE_INFO
        )

        match self.action:
            case Action.NEW_DIR:
                MuiDialogItem(
                    mode="input",
                    initValue="New Project",
                    title="New Project",
                    submitText="Create",
                    size="small",
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.UPLOAD_FILE:
                MuiDialogItem(
                    mode="upload",
                    submitText="Done",
                    size="large",
                    width="large",
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.RENAME:
                MuiDialogItem(
                    mode="input",
                    initValue=self.file.name,
                    title="Rename",
                    submitText="OK",
                    size="small",
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.SHARE:
                MuiDialogItem(
                    mode="share",
                    initValue={
                        "shared": self.file.visibility == Visibility.SHARED,
                        "permission": self.file.permission.name,
                    },
                    title="Share: " + self.file.name,
                    submitText="Save",
                    infoText=share_info(self.file),
                    size="medium",
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.DELETE:
                MuiDialogItem(
                    mode="info",
                    title="Delete forever?",
                    submitText="Delete Forever",
                    infoText=f'"{self.file.name}"' + FileActionDialog.DELETE_INFO,
                    size="small",
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.DOWNLOAD:
                pass
            case Action.ANALYZE:
                MuiDialogItem(
                    mode="options",
                    title=f'Analyze "{self.file.name}" For Comparison',
                    infoText=FileActionDialog.ANALYZE_INFO,
                    infoSize="large",
                    options=self.analyze_options,
                    size="large",
                    key=self.key,
                    onSubmit=self.handleAnalyze,
                    onClose=lambda x: self.closeDialog(self.key),
                ).render()
            case Action.EDIT:
                MuiDialogItem(
                    mode="input",
                    title=f"Edit Metadata: " + f"{self.file.name}",
                    size="large",
                    submitText="Save",
                    paragraph=True,
                    initValue=self.file.getDescription(),
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.MOVE:
                MuiDialogItem(
                    mode="files",
                    initValue={
                        "current_directory": QuetzalFile.fromFile(self.file, "./", self.file.user),
                        "selected": None,
                        "target_directory": None,
                    },
                    title="Move " + f'"{self.file.name}"',
                    infoText="Current location: " + "parentDir",
                    infoSize="small",
                    size="large_large",
                    submitText="Move",
                    paragraph=True,
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case Action.COPY:
                MuiDialogItem(
                    mode="files",
                    initValue={
                        "current_directory": QuetzalFile.fromFile(self.file, "./", self.file.user),
                        "selected": None,
                        "target_directory": None,
                    },
                    title="Copy " + f'"{self.file.name}"',
                    infoText="Select one of your projects as the destination to copy the selected video or project",
                    infoSize="small",
                    size="large_large",
                    submitText="Copy",
                    paragraph=True,
                    key=self.key,
                    onSubmit=self.handleClose,
                ).render()
            case "process":
                MuiDialogItem(
                    mode="process",
                    title=f"Processing: " + f"{self.file.name if isinstance(self.file, QuetzalFile) else ' '}",
                    size="large_large",
                    submitText="Done",
                    key=self.key,
                    onSubmit=self.onRender,
                    onClose=lambda x: self.closeDialog(self.key),
                ).render()
            case _:
                return

        if self.key != "sub_dialog_uniuqe_key":
            FileActionDialog("sub_dialog_uniuqe_key", device=self.device).render()

        return self


class MuiInfo:
    
    def setInfoState(self):
        st.session_state.MuiInfoState = self
        
    def __init__(self, text, serverity="info"):
        self.text = text
        self.open = True
        self.severity = serverity
        self.setInfoState()
        
    @staticmethod
    def closeAlert():
        st.session_state.MuiInfoState = None
        
    @staticmethod
    def getAlert() -> 'MuiInfo':
        if "MuiInfoState" not in st.session_state:
            st.session_state.MuiInfoState = None
        return st.session_state.MuiInfoState
    

    def render(self, margin=False):
        def alertClose():
            self.open = False

        if self.open:
            mui.Alert(
                severity=self.severity,
                children=[self.text],
                # onClose=alertClose,
                sx={
                    "margin": "0.5rem" if margin else "0rem",
                    "border-radius": "2rem",
                    "my": "4px",
                    "mx": "-0.5rem",
                },
            )


class MuiDialogItem:
    font_style = {"textTransform": "none", "fontSize": 14, "fontWeight": 500}

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

    text_field_paragraph_style = {}

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
                "&.Mui-disabled": {
                    "border": "1px",
                },
            },
            "&:not(:first-of-type)": {
                "borderRadius": "1rem",
                "&.Mui-disabled": {
                    "border": "1px",
                },
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

    DIALOG_SMALL_WIDTH = 400
    DIALOG_MEDIUM_WIDTH = 512
    DIALOG_LARGE_WIDTH = 600
    DIALOG_LARGE_LARGE_WIDTH = 800

    DIALOG_SMALL_HEIGHT = 197
    DIALOG_MEDIUM_HEIGHT = 267
    DIALOG_LARGE_HEIGHT = 350
    DIALOG_LARGE_LARGE_HEIGHT = 550

    share_option_buttons = [
        MuiToggleButton(
            Permission.READ_ONLY.name,
            MuiFileListItem.permission_icon[Permission.READ_ONLY],
            "Read Only",
        ),
        MuiToggleButton(
            Permission.POST_ONLY.name,
            MuiFileListItem.permission_icon[Permission.POST_ONLY],
            "Post Only",
        ),
        MuiToggleButton(
            Permission.FULL_WRITE.name,
            MuiFileListItem.permission_icon[Permission.FULL_WRITE],
            "Full Write",
        ),
    ]

    def initDialogItemData(self):
        if "DialogItemState" not in st.session_state:
            st.session_state.DialogItemState = {self.key: None}
        elif self.key not in st.session_state.DialogItemState:
            st.session_state.DialogItemState[self.key] = None

        if st.session_state.DialogItemState[self.key] == None:
            st.session_state.DialogItemState[self.key] = self.targetValue

    def _set_size(self, size, height, width):
        width_size = {
            "small": self.DIALOG_SMALL_WIDTH,
            "medium": self.DIALOG_MEDIUM_WIDTH,
            "large": self.DIALOG_LARGE_WIDTH,
            "large_large": self.DIALOG_LARGE_LARGE_WIDTH,
        }

        height_size = {
            "small": self.DIALOG_SMALL_HEIGHT,
            "medium": self.DIALOG_MEDIUM_HEIGHT,
            "large": self.DIALOG_LARGE_HEIGHT,
            "large_large": self.DIALOG_LARGE_LARGE_HEIGHT,
        }

        self.dialog_height = height_size[size]
        self.dialog_width = width_size[size]
        if height:
            self.dialog_height = height_size[height]
        if width:
            self.dialog_width = width_size[width]

    def __init__(
        self,
        mode: Literal[
            "info", "input", "share", "upload", "options", "files", "process"
        ],
        initValue="",
        title: str = "Rename",
        submitText: str = "OK",
        infoText: str = "",
        infoSize: Literal["small", "medium", "large"] = "medium",
        paragraph: bool = False,
        options: list[str] = None,
        size: Literal["small", "medium", "large"] = "small",
        height: Literal["small", "medium", "large"] = None,
        width: Literal["small", "medium", "large"] = None,
        key="main",
        open=True,
        onSubmit=None,
        onClose=None,
    ):
        self.onSubmit = onSubmit
        self.onClose = onSubmit if onClose == None else onClose
        self.key = key
        self.mode: Literal[
            "info", "input", "share", "upload", "options", "files", "process"
        ] = mode
        self.title = title
        self.submitText = submitText
        self.infoSize = infoSize
        self.info_txt = infoText
        self.targetValue = initValue
        self.open = open
        self.options = options
        self.paragraph = paragraph
        self.submitDisable = False

        self.initDialogItemData()
        self._set_size(size, height, width)

    ## TITLE
    def renderTitle(self):
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

            with mui.IconButton(onClick=self.handleClose, sx={"fontSize": 18, "p": 0}):
                mui.icon.Close(fontSize="inherit")

    ## INPUT TEXTFIELD
    def takeUserInput(self, event):
        debug("MuiDialogItem.takeUserInput: event['target'] = ", event["target"])
        st.session_state.DialogItemState[self.key] = getEventValue(event)

    def renderInputTextField(self):
        debug("REDNERING PARAGRPHA: ", st.session_state.DialogItemState)
        debug("PARAGRPHA:", self.paragraph)
        mui.TextField(
            defaultValue=st.session_state.DialogItemState[self.key],
            autoFocus=True,
            required=True,
            fullWidth=True,
            multiline=self.paragraph,
            rows=7 if self.paragraph else 1,
            margin="dense",
            variant="outlined",
            onChange=lazy(lambda value: self.takeUserInput(value)),
            sx=(
                MuiDialogItem.text_field_paragraph_style
                if self.paragraph
                else MuiDialogItem.text_field_style
            ),
        )

    ## INFO
    def renderInfoText(self):
        info_height = {"small": "16px", "medium": "56px", "large": "210px"}

        with mui.Box(
            component="pre",  # Use pre tag to preserve formatting
            sx={
                "fontFamily": "inherit",  # To match the Typography styling
                "whiteSpace": "pre-wrap",  # To allow line breaks
                "margin": 0,  # Remove default margin of pre
            },
        ):
            mui.Typography(
                paragraph=True,
                variant="body2",
                gutterBottom=True,
                children=self.info_txt,
                sx={"height": info_height[self.infoSize]},
            )

    ## OPTIONS
    def buildOptionHandler(self, option):
        def handleOption(event):
            if self.onSubmit != None:
                setEventValue(event, option)
                debug(
                    "MuiDialogItem.handleOption: result = ",
                    option,
                )
                self.onSubmit(event)

        return handleOption

    def renderOptions(self):
        with mui.Stack(
            direction="row",
            spacing=0,
            alignItems="center",
            justifyContent="center",
        ):
            for option in self.options:
                with mui.Button(
                    type="submit",
                    onClick=self.buildOptionHandler(option),
                    sx=contained_button_style,
                ):
                    mui.Typography(option, sx=MuiDialogItem.font_style)

    ## OPTION SHARE
    def updateShare(self, event):
        st.session_state.DialogItemState[self.key]["shared"] = (
            not st.session_state.DialogItemState[self.key]["shared"]
        )

    def handlePermission(self, event, controller):
        if controller != None:
            st.session_state.DialogItemState[self.key]["permission"] = controller

    def renderShareOptions(self):
        with mui.Stack(
            direction="row",
            spacing=0,
            alignItems="center",
            justifyContent="center",
            sx={"height": "54px"},
        ):
            # mui.icon.Compare(sx={"py": "7px"})
            mui.Typography("Share", sx={"fontSize": 14})
            switch_on = st.session_state.DialogItemState[self.key]["shared"]
            mui.Switch(
                checked=switch_on,
                onChange=self.updateShare,
            )

            ## Share Permission
            with mui.ToggleButtonGroup(
                value=st.session_state.DialogItemState[self.key]["permission"],
                onChange=self.handlePermission,
                exclusive=True,
                disabled=not switch_on,
                sx=MuiDialogItem.toggle_buttons_style,
            ):
                for button in self.share_option_buttons:
                    button.render()

    ## SUBMIT BUTTON
    def handleClose(self, event):
        debug("MuiDialogItem.handleClose: Dialog closed\n")
        if self.onClose:
            self.onClose(event)
        st.session_state.DialogItemState[self.key] = None

    def handleFormSubmit(self, event):
        debug("MuiDialogItem.handleFormSubmit: onSubmit =", self.onSubmit)
        if self.onSubmit != None:
            setEventValue(event, st.session_state.DialogItemState[self.key])
            debug(
                "MuiDialogItem.handleFormSubmit: result = ",
                st.session_state.DialogItemState,
            )

            self.onSubmit(event)
        st.session_state.DialogItemState[self.key] = None

    def renderSubmitButton(self):
        style = {
            "mt": "1.35rem" if not self.mode == "files" else "0rem",
            "mb": "1rem",
        }

        with mui.Stack(
            spacing=0,
            direction="row",
            alignItems="start",
            justifyContent="end",
            sx=style,
        ):
            with mui.Button(
                onClick=self.handleClose,
                sx=outlined_button_style,
            ):
                mui.Typography("Cancel", sx=MuiDialogItem.font_style)
            with mui.Button(
                type="submit",
                onClick=self.handleFormSubmit,
                sx=contained_button_style,
                disabled=self.submitDisable,
            ):
                mui.Typography(self.submitText, sx=MuiDialogItem.font_style)

    ## FILE SYSTEM
    def fileSelectHandler(self, event):
        with st.session_state.lock:
            file = getEventValue(event)
            if file != None:
                st.session_state.DialogItemState[self.key]["selected"] = file
                st.session_state.DialogItemState[self.key]["target_directory"] = file

    def fileDoubleClickHandler(self, event):
        with st.session_state.lock:
            file = getEventValue(event)
            if file != None:
                st.session_state.DialogItemState[self.key]["current_directory"] = file
                st.session_state.DialogItemState[self.key]["target_directory"] = None

    def renderFileSystem(self):
        if "selected" not in st.session_state.DialogItemState[self.key]:
            st.session_state.DialogItemState[self.key]["selected"] = None

        if "current_directory" not in st.session_state.DialogItemState[self.key]:
            st.session_state.DialogItemState[self.key]["current_directory"] = None

        if "target_directory" not in st.session_state.DialogItemState[self.key]:
            st.session_state.DialogItemState[self.key]["target_directory"] = None
        elif st.session_state.DialogItemState[self.key]["target_directory"] == None:
            self.submitDisable = True

        if st.session_state.DialogItemState[self.key]["target_directory"]:
            self.submitDisable = False

        action_menu = MuiActionMenu(
            mode=["upload"],
            key="dialog_upload",
            onlyNew=True,
            onClick=FileActionDialog.buildDialogOpener("sub_dialog_uniuqe_key"),
        ).render()

        no_upload = MuiActionMenu(
            mode=["edit"],
            key="dialog_no_upload_menu",
            onClick=FileActionDialog.buildDialogOpener("sub_dialog_uniuqe_key"),
        ).render()

        mui.Divider()

        def fileListMoreHandler(event: dict):
            with st.session_state.lock:
                file = getEventValue(event)
                if file != None:
                    logger.debug("Calling MenuOpener (Dialog)")
                    no_upload.buildMenuOpener(file=file)(event)

        curr_dir: QuetzalFile = st.session_state.DialogItemState[self.key][
            "current_directory"
        ]

        MuiFileList(
            file_list=curr_dir.iterdir(directoryOnly=True),
            max_height=f"calc({self.DIALOG_LARGE_LARGE_HEIGHT}px - 13rem)",
            key=self.key,
            tightDisplay=True,
            onClick=self.fileSelectHandler,
            onDoubleClick=self.fileDoubleClickHandler,
            onClickMore=fileListMoreHandler,
        ).render()

        mui.Divider()

        def breadCrumbClickHandler(event: dict):
            clicked_path = getEventValue(event)

            with st.session_state.lock:
                if (
                    clicked_path
                    == st.session_state.DialogItemState[self.key][
                        "current_directory"
                    ].path
                ):
                    action_menu.buildMenuOpener(
                        file=st.session_state.DialogItemState[self.key][
                            "current_directory"
                        ]
                    )(event)
                else:
                    # clicked_path = replaceInitialSegment(clicked_path, "./")
                    new_dir = QuetzalFile.fromFile(curr_dir, clicked_path)
                    st.session_state.DialogItemState[self.key][
                        "current_directory"
                    ] = new_dir
                    st.session_state.DialogItemState[self.key][
                        "target_directory"
                    ] = None

        with mui.Stack(
            spacing=1,
            direction="row",
            alignItems="center",
            justifyContent="space-between",
            sx={"m": "0px", "p": "0px"},
        ):
            MuiFilePathBreadcrumbs(
                file=curr_dir,
                key="dialog_breadcrumb",
                onClick=breadCrumbClickHandler,
                size="small",
            ).render()

            def setToCurrentFile(event):
                st.session_state.DialogItemState[self.key]["target_directory"] = (
                    st.session_state.DialogItemState[self.key]["current_directory"]
                )

            disabled = (
                st.session_state.DialogItemState[self.key]["target_directory"]
                == st.session_state.DialogItemState[self.key]["current_directory"]
            )

            mui.Button(
                "Select current directory",
                variant="text",
                sx={"color": GOOGLE_BLUE},
                onClick=setToCurrentFile,
                disabled=disabled,
            )

        # mui.Divider()

    ## UPLOAD
    def renderUpload(self):
        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px", padding="2rem 2rem 2rem", border="2ps"
        )
        dialog_container = float_dialog(
            self.open,
            background="white",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:
            st.session_state.DialogItemState[self.key] = st.file_uploader(
                "Choose a Video file", accept_multiple_files=True
            )

            with stylable_container(
                key="dialog_container_" + self.key,
                css_styles=f"""{{
                        display: block;
                        & div {{
                                width: calc({self.dialog_width}px - 3rem);
                                height: auto;
                            }}
                        & iframe {{
                            width: calc({self.dialog_width}px - 3rem);
                        }}
                    }}
                    """,
            ):
                with elements("dialog_elements_" + self.key):
                    self.renderSubmitButton()

    def renderProcess(self):
        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px",
            padding="1.5rem 1.5rem 1.5rem",
            border="1px solid #e0e0e0 !important",
        )
        dialog_container = float_dialog(
            self.open,
            background="white",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:
            with st.spinner(self.title):
                if self.onSubmit:
                    self.onSubmit()

            self.handleClose(None)
            st.rerun()

    ## Main Render
    def render(self, info: MuiInfo = None):
        
        if self.mode == "upload":
            self.renderUpload()
            return self

        if self.mode == "process":
            self.renderProcess()
            return self

        float_dialog_css = float_css_helper(
            width=f"{self.dialog_width}px",
            padding="1.5rem 1.5rem 0rem",
            border="1px solid #e0e0e0 !important",
        )
        dialog_container = float_dialog(
            self.open,
            background="white",
            transition_from="center",
            css=float_dialog_css,
        )

        with dialog_container:

            with stylable_container(
                key="dialog_container_" + self.key,
                css_styles=f"""{{
                        display: block;
                        & div {{
                                width: calc({self.dialog_width}px - 3rem);
                                height: auto;
                            }}
                        & iframe {{
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
                        if info:
                            info.render(margin=True)

                        ## Title + Exit Button
                        self.renderTitle()

                        ## Input Textfield
                        if self.mode == "input":
                            self.renderInputTextField()

                        ## Info Text
                        if self.mode in ["info", "share", "options", "files"]:
                            self.renderInfoText()

                        ## Analysis Text
                        if self.mode in ["options"]:
                            MuiOptionButton(
                                variant="contained",
                                options=[
                                    [option, self.buildOptionHandler(option)]
                                    for option in self.options
                                ],
                            ).render()

                        ## Share Setting
                        if self.mode == "share":
                            self.renderShareOptions()

                        if self.mode in ["files"]:
                            self.renderFileSystem()

                        ## Cancel + OK button
                        if self.mode not in ["options"]:
                            self.renderSubmitButton()

        return self


class MuiFileDetails:
    @property
    def showVideo(self) -> QuetzalFile:
        return st.session_state.FileDetailState[self.key]["show_video"]

    @showVideo.setter
    def showVideo(self, value: QuetzalFile):
        st.session_state.FileDetailState[self.key]["show_video"] = value

    @staticmethod
    def initFileDetails(key):
        if "FileDetailState" not in st.session_state:
            st.session_state.FileDetailState = {key: {"show_video": False}}
        else:
            if key not in st.session_state.FileDetailState:
                st.session_state.FileDetailState[key] = {"show_video": False}

    def __init__(
        self,
        file: QuetzalFile,
        width=340,
        key="file_detail_main",
        top_margin=0,
        onClick=None,
        onClose=None,
    ):
        self.file: QuetzalFile = file
        self.width = width
        self.video_placeholder_height = int(width / 16 * 9 + 0.5)
        self.key = key
        self.onClick = onClick
        self.onClose = onClose
        self.top_margin = top_margin

        if isinstance(file, QuetzalFile) and file.type == FileType.FILE:
            file._syncAnalysisState()
        self.initFileDetails(self.key)

    def renderInfo(self):
        def buildActionClickHandler(action):
            def _onClick(event):

                debug("renderInfo.BuildActionClicker:", action, event)
                setEventValue(
                    event,
                    {
                        "file": self.file,
                        "action": action,
                    },
                )
                FileActionDialog.buildDialogOpener(self.key)(event)

            return _onClick

        def buildHandler(video_type):
            def _onClick(event):
                setEventValue(event, video_type)
                if self.onClick:
                    self.onClick(event)

            return _onClick

        if self.file.type == FileType.FILE:
            MuiOptionButton(
                variant="outlined",
                options=[
                    ["Use as Query", buildHandler("query")],
                    ["Use as Database", buildHandler("database")],
                ],
                disabled=[
                    (self.file.analysis_progress == AnalysisProgress.NONE),
                    (self.file.analysis_progress != AnalysisProgress.FULL),
                ],
                padding="0.5rem 0rem 1rem 0rem",
            ).render()

        with mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0px",
                "border": "0px",
                "bgcolor": "white",
                "padding": "0px",
                "mx": "-0.5rem",
            },
        ):
            mui.Divider()
            with mui.List(sx={"py": "0rem"}):
                info = {
                    AnalysisProgress.NONE: "No Analysis Done",
                    AnalysisProgress.HALF: "Shallow Analysis Done",
                    AnalysisProgress.FULL: "Deep Analysis Done",
                }
                if self.file.type == FileType.FILE:
                    MuiInfoDisplay(
                        title="Video Analysis",
                        info_items=[["State", info[self.file.analysis_progress]]],
                        expendable=False,
                        divider=True,
                        secondaryItem=MuiEditButton(
                            "analysis",
                            onClick=buildActionClickHandler(Action.ANALYZE),
                            disabled=(
                                self.file.permission == Permission.READ_ONLY
                                and self.file.mode != AccessMode.OWNER
                            ),
                        ),
                        key="video_analysis",
                    ).render()

                info = {
                    Visibility.SHARED: "Shared",
                    Visibility.PRIVATE: "Private",
                    Permission.READ_ONLY: "Read Only",
                    Permission.POST_ONLY: "Post Only",
                    Permission.FULL_WRITE: "Full Write",
                }

                MuiInfoDisplay(
                    title="Sharing",
                    info_items=[
                        ["Sharing", info[self.file.visibility]],
                        ["Permission", info[self.file.permission]],
                        ["Created By", self.file.createdBy],
                    ],
                    expendable=True,
                    divider=True,
                    secondaryItem=MuiEditButton(
                        "share",
                        onClick=buildActionClickHandler(Action.SHARE),
                        disabled=(self.file.mode != AccessMode.OWNER),
                    ),
                    key="sharing",
                ).render()

                MuiInfoDisplay(
                    title="File Metadata",
                    info_items=MuiInfoDisplay.parse_info_text(
                        self.file.getDescription()
                    ),
                    expendable=True,
                    divider=True,
                    secondaryItem=MuiEditButton(
                        "edit",
                        onClick=buildActionClickHandler(Action.EDIT),
                        disabled=(
                            self.file.permission != Permission.FULL_WRITE
                            and self.file.mode != AccessMode.OWNER
                        ),
                    ),
                    key="file_metadata",
                ).render()

    def renderInsturction(self):
        with stylable_container(
            key="info_title_container_" + self.key,
            css_styles=f"""{{
                display: block;
                & div {{
                    width: 100%;
                    height: auto;
                }}
                & iframe {{
                    width: 100%;
                    margin-top: 1rem;
                    height: calc(25rem + {ELEMENT_BOTTOM_MARGIN});
                    max-height: calc(100vh - {self.top_margin} - 3rem);
                }}
            }}
            """
        ):
            with elements("info_instruction_" + self.key):
                with mui.Stack(
                    alignItems="center",
                    justifyContent="center",
                    sx={
                        "borderRadius": "0.3rem",
                        "border": "0px",
                        "bgcolor": "white",
                        "padding": "0",
                        "height": f"25rem",
                        "maxHeight": f"calc(100vh - {ELEMENT_BOTTOM_MARGIN})"
                    },
                ):
                    mui.icon.Info(sx={"fontSize": "5rem", "color": GOOGLE_BLUE})
                    mui.Typography("Select an item to see details", sx={"fontSize": "0.8rem"})

    def renderTitle(self):
        with stylable_container(
            key="info_title_container_" + self.key,
            css_styles=f"""{{
                display: block;
                & div {{
                    width: 100%;
                    height: auto;
                }}
                & iframe {{
                    width: 100%;
                    height: 50px;
                    margin-top: 1rem;
                }}
            }}
            """
        ):
            with elements("info_title_" + self.key):
                ## Title
                with mui.Stack(
                    spacing=1,
                    direction="row",
                    alignItems="start",
                    justifyContent="space-between",
                    sx={
                        "padding": "0rem 1rem",
                    },
                ):
                    with mui.Stack(
                        spacing=1,
                        direction="row",
                        alignItems="center",
                        justifyContent="start",
                        sx={
                            "padding": "0rem 0rem",
                        },
                    ):
                        color_style = (
                            {"color": "#d33a2e"}
                            if self.file.type == FileType.FILE
                            else {}
                        )
                        getattr(
                            mui.icon,
                            MuiFileListItem.file_type_icon[self.file.type][
                                self.file.visibility
                            ],
                        )(sx=color_style)
                        mui.Typography(
                            variant="subtitle1",
                            component="div",
                            children=[self.file.name],
                        )
                        
                    mui.IconButton(
                        mui.icon.Close(sx={"fontSize": "1.5rem"}),
                        onClick=self.onClose,
                        sx={"p": "0.1rem"},
                    )


    def render(self):
        if self.file == None:
            self.renderInsturction()
            return
        else:
            self.renderTitle()
        
        with stylable_container(
            key="info_main_divier",
            css_styles=f"""{{
                & .stMarkdown {{
                    width: 100%;
                }}
                & hr {{
                    margin-bottom: 1rem;
                    margin-top: 1rem;
                }}
            }}
            """,
        ):
            st.divider()
            
        title_height = 73.59 #px
        with stylable_container(
                key="info_main_display",
                css_styles=f"""{{
                    display: block;
                    border-radius: 0;
                    border-bottom-right-radius: 1rem !important;
                    border-bottom-left-radius: 1rem !important;
                    & > div:nth-child(2) {{
                        height: calc(100vh - {title_height}px - 1rem - {self.top_margin}) !important;
                        overflow-y: scroll;
                        overflow-x: clip;
                        border-radius: 0;
                        border-bottom-right-radius: 1rem !important;
                        border-bottom-left-radius: 1rem !important;
                        {scroll_style_css}
                    }};
                    background-color: white;
                }}
                """,
            ):
                with st.container():
                    ## Video Placeholder
                    if self.showVideo:
                        
                        st.video(str(self.file._abs_path), format="video/mp4", start_time=0)
                        self.showVideo = False

                        with elements("show_info_" + self.key):
                            self.renderInfo()
                    else:
                        with elements("show_info_" + self.key):
                            with mui.Stack(
                                # variant="outlined",
                                alignItems="center",
                                justifyContent="center",
                                sx={
                                    "borderRadius": "0.3rem",
                                    "border": "0px",
                                    "bgcolor": "grey.100",
                                    "margin-bottom": "0.5rem",
                                    "height": f"{self.video_placeholder_height}px",
                                },
                            ):

                                def video_show():
                                    self.showVideo = True
                                
                                match self.file.type:
                                    case FileType.FILE:
                                        with mui.Button(
                                            variant="text",
                                            sx={
                                                "height": "min-content",
                                                "margin": 0,
                                                "padding": 0,
                                            },
                                            onClick=video_show,
                                        ):
                                            mui.Typography(
                                                "Display Video",
                                                sx={"fontSize": "0.8rem", "color": GOOGLE_BLUE},
                                            )
                                        mui.Typography(
                                            "+ show download option", sx={"fontSize": "0.8rem"}
                                        )
                                    case FileType.DIRECTORY:
                                        getattr(
                                            mui.icon,
                                            MuiFileListItem.file_type_icon[self.file.type][
                                                self.file.visibility
                                            ],
                                        )(sx={"fontSize": "5rem"})

                            self.renderInfo()

        return self

class MuiOptionButton:
    button_style = {
        "outlined": outlined_button_style,
        "contained": contained_button_style,
    }

    def __init__(
        self,
        key="main",
        padding="0px",
        margin="0px",
        options=[],
        variant: Literal["outlined", "contained"] = "outlined",
        disabled=None,
    ):
        self.margin = margin
        self.padding = padding
        self.variant = variant
        self.key = key
        if disabled == None:
            disabled = [False] * len(options)
        self.options = [
            (*option, disable) for option, disable in zip(options, disabled)
        ]

    def render(self):
        with mui.Stack(
            spacing=0,
            direction="row",
            alignItems="center",
            justifyContent="center",
            sx={"m": self.margin, "p": self.padding},
        ):
            for option, onClick, disabled in self.options:
                with mui.Button(
                    variant=self.variant,
                    sx=self.button_style[self.variant],
                    onClick=onClick,
                    disabled=disabled,
                ):
                    mui.Typography(option, sx=MuiDialogItem.font_style)
        return self


class MuiComparePrompt:
    def __init__(
        self,
        project: QuetzalFile = None,
        query: QuetzalFile = None,
        database: QuetzalFile = None,
        onClicks: QuetzalFile = None,
    ):
        self.project = project.name if project else "Not Selected"
        self.query = query.name if query else "Not Selected"
        self.database = database.name if database else "Not Selected"
        self.onClicks = onClicks
        self.disabled = not (database and query)

    def render(self):
        with mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "1rem",
                "border": "0px",
                "borderColor": "grey.500",
                "overflow": "auto",
                "my": "1rem",
                "mx": "0.5rem",
                "bgcolor": GOOGLE_DEEP_BLUE,
                "padding-left": "0.2rem",
                "padding-bottom": "0.5rem",
            },
        ):
            MuiInfoDisplay(
                title="Selected for Comparison",
                info_items=[
                    # ["Project", self.project],
                    ["Database Video", self.database],
                    ["Query Video", self.query],
                ],
                expendable=False,
                divider=False,
                key="selected_for_comaprison",
            ).render()

            with mui.Button(
                startIcon=mui.icon.Compare(),
                sx=outlined_button_style,
                onClick=self.onClicks[0],
                disabled=self.disabled,
            ):
                mui.Typography("RUN COMPARISON", sx=MuiDialogItem.font_style)

            with mui.Button(
                startIcon=mui.icon.CastConnected(),
                sx=outlined_button_style,
                onClick=self.onClicks[1],
                disabled=self.disabled,
            ):
                mui.Typography("REALTIME MATCHING", sx=MuiDialogItem.font_style)
                
            with mui.Button(
                startIcon=mui.icon.Podcasts(),
                sx=outlined_button_style,
                onClick=self.onClicks[2],
                disabled=self.disabled,
            ):
                mui.Typography("STREAM MATCHING", sx=MuiDialogItem.font_style)
                
        return self
