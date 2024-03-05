from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.runtime.app_session import AppSession
from streamlit.runtime import Runtime
import asyncio
import threading
import gc


def get_browser_session_id() -> str:
   # Get the session_id for the current running script 
    try:
        ctx = get_script_run_ctx()
        return ctx.session_id
    except Exception as e:
        raise Exception("Could not get browser session id") from e
        
# def find_streamlit_main_loop() -> asyncio.BaseEventLoop:
#     loops = []
#     for obj in gc.get_objects():
#         try:
#             if isinstance(obj, asyncio.BaseEventLoop):
#                 loops.append(obj)
#         except ReferenceError:
#             ...
        
#     main_thread = next((t for t in threading.enumerate() if t.name == 'MainThread'), None)
#     if main_thread is None:
#         raise Exception("No main thread")
#     main_loop = next((lp for lp in loops if lp._thread_id == main_thread.ident), None) # type: ignore
#     if main_loop is None:
#         raise Exception("No event loop on 'MainThread'")
    
#     return main_loop
    
def get_streamlit_session(session_id: str) -> AppSession:
    runtime: Runtime = Runtime.instance()
    session = next((
        s.session
        for s in runtime._session_mgr.list_sessions()
        if s.session.id == session_id
    ), None)
    if session is None:
        raise Exception(f"Streamlit session not found for {session_id}")
    return session

def get_streamlit_sessions() -> list[AppSession]:
    runtime: Runtime = Runtime.instance()
    return [s.session for s in runtime._session_mgr.list_sessions()]

# This is it!
# get_browser_session_id needs to be run on the relevant script thread,
# then you can call the rest of this on other threads.
# streamlit_loop = find_streamlit_main_loop()
streamlit_session = get_streamlit_session(get_browser_session_id())

def notify() -> None:
    streamlit_session._handle_rerun_script_request()
    # for session in get_streamlit_sessions():
    #     session._handle_rerun_script_request()