import logging
import os

import streamlit as st
from streamlit import session_state as ss
from streamlit_elements import elements, mui
from streamlit_extras.stylable_container import stylable_container
from quetzal.engines.align_engine.dtw_engine import DTWEngine
from quetzal.engines.align_engine.realtime_engine import RealtimeAlignmentEngine
from quetzal.engines.engine import AlignmentEngine
import torch
import pickle, shelve, json

from quetzal_app.elements.mui_components import (
    ELEMENT_BOTTOM_MARGIN,
    FileActionDialog,
    getEventValue,
    MuiActionMenu,
    MuiComparePrompt,
    MuiFileDetails,
    MuiFileList,
    MuiFilePathBreadcrumbs,
    MuiInfo,
    MuiOnFocusHandler,
    MuiSideBarMenu,
    MuiToggleButton,
    MuiUploadButton,
    setEventValue,
)
from quetzal_app.page.page_state import Page, PageState
from quetzal_app.utils.utils import get_base64
from quetzal.dtos.dtos import FileType, QuetzalFile
from quetzal.dtos.video import DatabaseVideo, QueryVideo
from quetzal.align_frames import align_frame_pairs, align_video_frames
from pathlib import Path

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

BORDER_RADIUS = "0.8rem"
BACKGROUND_COLOR = "#f8fafd"
MENU_WIDTH = 256
INFO_WIDTH = 340
TOP_MARGIN = "64px"

TITLE = "Quetzal"

LOGO_FILE = os.path.normpath(Path(__file__).parent.joinpath("../quetzal_logo_trans.png"))
LOGO_FILE = f"data:image/jpeg;base64,{get_base64(LOGO_FILE)}"
# ELEMENT_BOTTOM_MARGIN = "7.594px"
PAGE_NAME = "file_explorer"


class FileExplorerPage(Page):
    name = PAGE_NAME

    def __init__(self, root_state: PageState, to_page: list[callable]):
        self.root_state = root_state
        self.page_state = self.init_page_state(root_state)
        self.to_page = to_page

    def init_page_state(self, root_state: PageState) -> PageState:
        user_default = {
            "root_dir": root_state.root_dir,
            "metadata_dir": root_state.metadata_dir,
            "user": root_state.user,
            "path": root_state.user,
            "home": root_state.user,
        }

        shared_default = {
            "root_dir": root_state.root_dir,
            "metadata_dir": root_state.metadata_dir,
            "user": root_state.user,
            "path": "./",
        }

        init_dir = QuetzalFile(**user_default)

        init_state = PageState(
            curr_dir=init_dir,
            info_file=None,
            query=None,
            database=None,
            last_dir=init_dir,
            menu="user",
            user_default=user_default,
            shared_default=shared_default,
        )

        return init_state

    def render(self):
        st.markdown(
            f"""
                <style>
                    # @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap'); 

                    # html, body, [class*="css"] {{
                    #     font-family: 'Roboto', sans-serif;
                    # }}
                    .block-container {{ /* Removes streamlit default white spaces in the main window*/
                        padding: 0rem;
                        background: {BACKGROUND_COLOR};
                        
                        & > div {{
                            height: calc(100vh - 42px);
                        }}
                    }}
                    .stSlider {{ /* Removes white spaces in the main window*/
                        padding-left: 1rem;
                        padding-right: 1rem;
                    }}
                    [class^="st-emotion-cache-"] {{ /* Removes streamlit default gap between containers, elements*/
                        gap: 0rem;
                        border-radius: {BORDER_RADIUS};
                        background-color: transparent
                    }}
                </style>
            """,
            unsafe_allow_html=True,
        )

        # Initialize Variable
        curr_dir: QuetzalFile = self.page_state["curr_dir"]
        if curr_dir != self.page_state["last_dir"]:
            self.page_state["info_file"] = None
        self.page_state["last_dir"] = curr_dir

        on_focus_handler = MuiOnFocusHandler()

        menu_content = MenuContent(
            on_focus_handler=on_focus_handler,
            page_state=self.page_state,
            to_video_comparison=self.open_video_comparison,
            to_real_time_comparison=self.open_realtime_comparison,
            to_stream_comparison=self.open_stream_comparison
        )

        file_list_content = FileListContent(
            on_focus_handler=on_focus_handler, page_state=self.page_state
        )

        info_content = InfoContent(
            on_focus_handler=on_focus_handler,
            page_state=self.page_state,
            file_list_content=file_list_content,
        )

        with stylable_container(
            key="full_container",
            css_styles=f"""{{
                    display: block;
                    min-height: calc(100vh);
                    max-height: calc(100vh);
                    width: 100%;
                    background-color: {BACKGROUND_COLOR};
                    border: 0px;
                    border-radius: 0px;
                }}
                """,
        ):
            with ss.lock:
                menu_c, files_c, info_c, gap = st.columns([2, 100, 2, 1])
                with menu_c:
                    MenuContainer(
                        width=MENU_WIDTH,
                    ).render(menu_content)

                with files_c:
                    AlertDisplay(self.root_state).render()

                    FileListContainer(
                        left_column_width=MENU_WIDTH,
                        right_column_width=INFO_WIDTH,
                    ).render(file_list_content)

                with info_c:
                    InfoContainer(
                        width=INFO_WIDTH,
                    ).render(info_content)

                FileActionDialog(
                    "main_dialog", device=self.root_state.torch_device
                ).render()

    def open_video_comparison(self, event):
        setEventValue(
            event,
            {
                "file": True,
                "action": "process",
                "onRender": lambda: self._align_video(
                    torch_device=self.root_state.torch_device
                ),
            },
        )
        FileActionDialog.buildDialogOpener("main_dialog")(event)
        
    def open_realtime_comparison(self, event):
        setEventValue(
            event,
            {
                "file": True,
                "action": "process",
                "onRender": lambda: self._align_realtime(
                    torch_device=self.root_state.torch_device
                ),
            },
        )
        FileActionDialog.buildDialogOpener("main_dialog")(event)
        
    def open_stream_comparison(self, event):
        setEventValue(
            event,
            {
                "file": True,
                "action": "process",
                "onRender": lambda: self._align_stream(
                    torch_device=self.root_state.torch_device
                ),
            },
        )
        FileActionDialog.buildDialogOpener("main_dialog")(event)

    def _align_video(
        self,
        overlay: bool = True,
        torch_device: torch.device = torch.device("cuda:0"),
    ):
        
        ## Load DTW and VLAD Features ##
        database_video = DatabaseVideo.from_quetzal_file(self.page_state.database)
        query_video = QueryVideo.from_quetzal_file(self.page_state.query)

        db_frame_list = database_video.get_frames()
        query_frame_list = query_video.get_frames()
        warp_query_frame_list = query_frame_list
        
        alignemnt_engine: AlignmentEngine = DTWEngine(torch_device)
        # alignemnt_engine: AlignmentEngine = RealtimeAlignmentEngine(torch_device)
        matches, warp_query_frame_list = alignemnt_engine.align_frame_list(database_video, query_video, overlay)

        # if not overlay:
        #     matches = align_video_frames(
        #         database_video=database_video,
        #         query_video=query_video,
        #         torch_device=torch_device,
        #     )
        # else:
        #     matches, warp_query_frame_list = align_frame_pairs(
        #         database_video=database_video,
        #         query_video=query_video,
        #         torch_device=torch_device,
        #     )
        
        comparison_matches = {
            "query": query_video,
            "database": database_video,
            "matches": matches,
            "query_frames": query_frame_list,
            "db_frames": db_frame_list,
            "warp_query_frames": warp_query_frame_list,
        }
        self.root_state["comparison_matches"] = comparison_matches
        
        # for development
        
        with open('query.pkl', 'wb') as f:                      
            pickle.dump(str(self.page_state.query._path), f)
            f.close()
        with open('db.pkl', 'wb') as f:
            pickle.dump(str(self.page_state.database._path), f)           
            f.close()
        with open('matches.pkl', 'wb') as f:
            pickle.dump(matches, f)
            f.close()
        with open('warp_query_frame_list.pkl', 'wb') as f:
            pickle.dump(warp_query_frame_list, f)
            f.close()
        with open('query_frame_list.pkl', 'wb') as f:
            pickle.dump(query_frame_list, f)
            f.close()
        with open('db_frame_list.pkl', 'wb') as f:
            pickle.dump(db_frame_list, f)
            f.close()
        
        self.to_page[1]()
        
    def _align_realtime(
        self,
        overlay: bool = True,
        torch_device: torch.device = torch.device("cuda:0"),
    ):
        
        ## Load DTW and VLAD Features ##
        database_video = DatabaseVideo.from_quetzal_file(self.page_state.database)
        query_video = QueryVideo.from_quetzal_file(self.page_state.query)

        db_frame_list = database_video.get_frames()
        query_frame_list = query_video.get_frames()
        warp_query_frame_list = query_frame_list
        
        # alignemnt_engine: AlignmentEngine = DTWEngine(torch_device)
        # matches, warp_query_frame_list = alignemnt_engine.align_frame_list(database_video, query_video, False)

        comparison_matches = {
            "query": query_video,
            "database": database_video,
            "matches": [[0, 0], [1, 1]],
            "query_frames": query_frame_list,
            "db_frames": db_frame_list,
            "warp_query_frames": warp_query_frame_list,
        }
        self.root_state["comparison_matches"] = comparison_matches
        
        self.to_page[2]()
        
    def _align_stream(
        self,
        overlay: bool = True,
        torch_device: torch.device = torch.device("cuda:0"),
    ):
        
        ## Load DTW and VLAD Features ##
        database_video = DatabaseVideo.from_quetzal_file(self.page_state.database)
        query_video = QueryVideo.from_quetzal_file(self.page_state.query)

        db_frame_list = database_video.get_frames()
        query_frame_list = query_video.get_frames()

        comparison_matches = {
            "query": query_video,
            "database": database_video,
            "matches": [[0, 0], [1, 1]],
            "query_frames": query_frame_list,
            "db_frames": db_frame_list,
            "warp_query_frames": [],
        }
        self.root_state["comparison_matches"] = comparison_matches
        
        self.to_page[3]()


class AlertDisplay:
    def __init__(self, page_state):
        self.page_state = page_state

    def draw_background(self):
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0",
                "border": "0px",
                "width": "100%",
                "maxWidth": "720px",
                "height": "56px",
                "bgcolor": BACKGROUND_COLOR,
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -1,
            },
        )

    def render(self):
        with stylable_container(
            key="alert_container",
            css_styles=f"""{{
                display: block;
                & div {{
                    width: 100%;
                    height: auto;
                    z-index: 22;
                }}
                & iframe {{
                    width: 100%;
                    max-width: 720px;
                    height: 56px;
                    z-index: 22;
                }}
            }}
            """,
        ):
            with stylable_container(
                key="stylable_container_sub",
                css_styles=f"""{{
                    display: block;
                    border-radius: 0;
                    height: 70px;
                    position: absolute;
                    right: calc({200}px - 2%) !important;
                    width: calc(104% - {200}px - {MENU_WIDTH}px) !important;
                    background-color: transparent;
                }}
                """,
            ):
                with elements("info_dialog"):
                    self.draw_background()
                    alert = MuiInfo.getAlert()
                    if alert:
                        alert.render()
                    # if self.page_state["info"]:
                    #     self.page_state["info"].render()


class MenuContent:

    def __init__(
        self, 
        on_focus_handler, 
        page_state, 
        to_video_comparison, 
        to_real_time_comparison,
        to_stream_comparison,
    ):
        self.on_focus_handler: MuiOnFocusHandler = on_focus_handler
        self.page_state = page_state
        self.to_video_comparison = to_video_comparison
        self.to_real_time_comparison = to_real_time_comparison
        self.to_stream_comparison = to_stream_comparison

        self.upload_menu = MuiActionMenu(
            mode=["upload"],
            key="upload_action_menu",
            onClick=FileActionDialog.buildDialogOpener("main_dialog"),
        )

    def draw_background(self):
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0",
                "border": "0px",
                "width": "100%",
                "height": "100vh",
                "bgcolor": BACKGROUND_COLOR,
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -2,
            },
        )

    def onChangeHandler(self, event):
        logger.debug("onChangeHandler: Side bar Menu")
        with ss.lock:
            value = getEventValue(event)
            self.page_state["menu"] = value
            if value == "user":
                curr_dir = QuetzalFile(**self.page_state["user_default"])
            elif value == "shared":
                curr_dir = QuetzalFile(**self.page_state["shared_default"])
            self.page_state["curr_dir"] = curr_dir

    def render(self):

        with elements("menu"):
            self.draw_background()

            with mui.Paper(
                variant="outlined",
                sx={
                    "borderRadius": "0px",
                    "border": "0px",
                    "width": "100%",
                    "height": f"calc(100vh - {ELEMENT_BOTTOM_MARGIN})",
                    "bgcolor": BACKGROUND_COLOR,
                },
            ):
                ## Title
                with mui.Stack(
                    spacing=0.5,
                    direction="row",
                    alignItems="center",
                    justifyContent="start",
                    sx={"height": 55, "pb": "3px"}
                ):
                    mui.Avatar(
                        alt="Quetzal",
                        src=LOGO_FILE,
                        sx={"width": 52, "height": 52 }
                    )
                    mui.Typography(
                        TITLE,
                        sx={
                            "fontSize": "h5.fontSize",
                            # /* top | left and right | bottom */
                            "margin": "0.5rem 1rem 0.25rem",
                        },
                    )

                ## Upload Button
                MuiUploadButton(
                    key="upload_button",
                    onClick=self.upload_menu.buildMenuOpener(
                        self.page_state["curr_dir"]
                    ),
                ).render()

                ## Side bar Menu
                toggle_buttons = [
                    MuiToggleButton("user", "Home", "My Projects"),
                    MuiToggleButton("shared", "FolderShared", "Shared by Others"),
                ]
                MuiSideBarMenu(
                    toggle_buttons=toggle_buttons,
                    key="main_menu",
                    onChange=self.onChangeHandler,
                ).render()

                ## Compare Prompt
                MuiComparePrompt(
                    database=self.page_state.database,
                    query=self.page_state.query,
                    project=None,
                    onClicks=[self.to_video_comparison, self.to_real_time_comparison, self.to_stream_comparison],
                ).render()

                ## Action Menu + Handlers
                self.upload_menu.render()
                self.on_focus_handler.setScanner(key="menu_col")
                self.on_focus_handler.registerHandler(
                    keys="menu_col",
                    handler=lambda: MuiActionMenu.resetAnchor(
                        exculde_keys=["upload_action_menu"]
                    ),
                )
        return self


class FileListContent:

    def __init__(self, on_focus_handler, page_state):
        self.on_focus_handler: MuiOnFocusHandler = on_focus_handler
        self.page_state = page_state

        self.action_menu = MuiActionMenu(
            mode=["upload", "edit", "delete", "move"],
            key="full_menu",
            onClick=FileActionDialog.buildDialogOpener("main_dialog"),
        )

        self.no_upload = MuiActionMenu(
            mode=["edit", "delete", "move"],
            key="no_upload_menu",
            onClick=FileActionDialog.buildDialogOpener("main_dialog"),
        )

    def draw_background(self, mx, bm):
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "0",
                "border": "0px",
                "width": "100%",
                "height": "100vh",
                "bgcolor": BACKGROUND_COLOR,
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -2,
            },
        )
        mui.Paper(
            variant="outlined",
            sx={
                "borderRadius": "1rem",
                "border": "0px",
                "width": f"calc(100% - {mx}rem)",
                "height": f"calc(100vh - {bm}rem)",
                "bgcolor": "white",
                "m": "0px",
                "p": "0px",
                "position": "absolute",
                "left": "0px",
                "top": "0px",
                "zIndex": -1,
            },
        )

    def breadCrumbClickHandler(self, event: dict):
        clicked_path = getEventValue(event)
        logger.debug(f"breadCrumbClickHandler: {clicked_path}")

        with ss.lock:
            curr_dir = self.page_state["curr_dir"]
            clicked_path = getEventValue(event)
            if clicked_path == curr_dir.path:
                self.action_menu.buildMenuOpener(file=curr_dir)(event)
            else:
                self.page_state["curr_dir"] = QuetzalFile.fromFile(
                    curr_dir, clicked_path
                )
                # self.page_state["info"] = None
                MuiInfo.closeAlert()

    def fileListMoreHandler(self, event: dict):
        file: QuetzalFile = getEventValue(event)
        if file is not None:
            logger.debug("Calling MenuOpener")
            self.no_upload.buildMenuOpener(file=file)(event)

    def fileClickHandler(self, event: dict):
        with ss.lock:
            file: QuetzalFile = getEventValue(event)
            if self.page_state["info_file"] and self.page_state["info_file"] == file:
                self.page_state["info_file"] = None
            else:
                self.page_state["info_file"] = file

    def fileDoubleClickHandler(self, event: dict):
        with ss.lock:
            file: QuetzalFile = getEventValue(event)
            if file.type == FileType.DIRECTORY:
                self.page_state["curr_dir"] = file
                # self.page_state["info"] = None
                MuiInfo.closeAlert()
            else:  # filtype = FILE
                self.page_state["info_file"] = None

    def render(self):

        with elements("files"):
            # Set Background
            inner_padding = 0.5
            default_margin = 0.5
            right_margin = 1
            bottom_margin = 0.5

            self.draw_background(mx=right_margin, bm=bottom_margin)

            with mui.Paper(
                variant="outlined",
                sx={
                    "box-sizing": "border-box",
                    "borderRadius": "1rem",
                    "border": "0px",
                    "width": f"calc(100% - {right_margin}rem)",
                    "height": f"calc(100vh - {default_margin + bottom_margin}rem)",
                    "bgcolor": "white",
                    "padding": f"{inner_padding}rem",
                    "zIndex": 2,
                },
            ):
                ## BreadCrumb
                MuiFilePathBreadcrumbs(
                    file=self.page_state["curr_dir"],
                    key="curr_dir_breadcrumb",
                    onClick=self.breadCrumbClickHandler,
                ).render()
                breadcrumbs_height = 3  # rem == 48px

                ## File List
                filter_content = self.page_state["menu"] == "shared"
                self.file_list = MuiFileList(
                    file_list=self.page_state["curr_dir"].iterdir(
                        sharedOnly=filter_content,
                        # excludeUser=filter_content
                    ),
                    max_height=f"calc(100vh - {breadcrumbs_height + default_margin + inner_padding + bottom_margin}rem)",
                    key="main",
                    onClickMore=self.fileListMoreHandler,
                    onClick=self.fileClickHandler,
                    onDoubleClick=self.fileDoubleClickHandler,
                ).render()

                ## Action Menus + Click Focus Handler
                self.action_menu.render()
                self.no_upload.render()
                self.on_focus_handler.setScanner(key="file_list")
                self.on_focus_handler.registerHandler(
                    keys="file_list",
                    handler=lambda: MuiActionMenu.resetAnchor(
                        exculde_keys=["full_menu", "no_upload_menu"]
                    ),
                )
        return self


class InfoContent:

    def __init__(self, on_focus_handler, page_state, file_list_content):
        self.on_focus_handler: MuiOnFocusHandler = on_focus_handler
        self.page_state = page_state
        self.file_list_content: FileListContent = file_list_content

    def onVideoSelect(self, event):
        with ss.lock:
            video_type = getEventValue(event)
            self.page_state[video_type] = self.page_state["info_file"]

    def closeDetail(self, event):
        self.file_list_content.file_list.onFocusOut()
        self.page_state["info_file"] = None

    def render(self):
        info_file = self.page_state["info_file"]

        MuiFileDetails(
            file=info_file,
            width=INFO_WIDTH,
            top_margin=TOP_MARGIN,
            key="main_dialog",
            onClick=self.onVideoSelect,
            onClose=self.closeDetail,
        ).render()
        return self


class MenuContainer:

    def __init__(
        self,
        width=MENU_WIDTH,
    ):
        self.width = width

    def render(self, content):
        with stylable_container(
            key="menu_container",
            css_styles=f"""{{
                    display: block;
                    & div {{
                            width: {self.width}px;
                            height: auto;
                        }}
                    & iframe {{
                        width: {self.width}px;
                        height: calc(100vh - {ELEMENT_BOTTOM_MARGIN});
                    }}
                    # width: {self.width}px;
                }}
                """,
        ):
            content.render()
        return self


class FileListContainer:

    def __init__(
        self,
        left_column_width,
        right_column_width,
    ):
        self.left_column_width = left_column_width
        self.right_column_width = right_column_width

    def render(self, content):
        with stylable_container(
            key="filelist_container",
            css_styles=f"""{{
                display: block;
                & div {{
                    width: 100%;
                    height: auto;
                }}
                & iframe {{
                    width: calc(100%) !important;
                    margin-top: {TOP_MARGIN};
                    height: calc(100vh - {ELEMENT_BOTTOM_MARGIN} - {TOP_MARGIN});
                }}
            }}
            """,
        ):
            with stylable_container(
                key="filelist_container_sub",
                css_styles=f"""{{
                    display: block;
                    border-radius: 0;
                    position: absolute;
                    right: calc({self.right_column_width}px - 2%) !important;
                    width: calc(104% - {self.left_column_width}px - {self.right_column_width}px) !important;
                    background-color: {BACKGROUND_COLOR};
                }}
                """,
            ):
                content.render()
        return self


class InfoContainer:

    def __init__(
        self,
        width=INFO_WIDTH,
    ):
        self.width = width

    def render(self, content):
        with stylable_container(
            key="file_info_container",
            css_styles=f"""{{
                display: block;
                & div {{
                    width: 100%;
                    height: auto;
                }}
                & iframe {{
                    width: {self.width}px !important;
                }}
                    
                & video {{
                    right: 0px !important;
                    width: {self.width}px !important;
                    margin-bottom: -1rem;
                }}
            }}
            """,
        ):

            with stylable_container(
                key="file_info_container_sub",
                css_styles=f"""{{
                    display: block;
                    margin-top: {TOP_MARGIN};
                    border-radius: 1rem 1rem 1rem 1rem;
                    height: calc(100vh - {ELEMENT_BOTTOM_MARGIN} - {TOP_MARGIN}) !important;
                    max-height: calc(100vh - 1rem - {TOP_MARGIN}) !important;
                    position: absolute;
                    right: 0px !important;
                    width: {self.width}px !important;
                    background-color: white;
                }}
                """,
            ):
                content.render()

        return self
