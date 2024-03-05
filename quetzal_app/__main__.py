# SPDX-FileCopyrightText: 2024 Carnegie Mellon University
#
# SPDX-License-Identifier: GPL-2.0-only
import configparser
import argparse
from pathlib import Path

import streamlit.web.bootstrap as bootstrap

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--listen", default="localhost", help="address to listen on")
parser.add_argument("-p", "--port", type=int, default=7860, help="port to listen on")

def load_config_from_file(filepath):
    """
    Load and parse the configuration file, then update the `config` dictionary
    with values from the file.
    """
    cfgparser = configparser.ConfigParser()
    cfgparser.optionxform = str
    cfgparser.read(filepath)

    config = dict()
    for section in cfgparser.sections():
        for option in cfgparser.options(section):
            raw_value = cfgparser.get(section, option)
            try:
                value = int(raw_value)
            except ValueError:
                value = raw_value.strip('"')  # Remove quotes if present
            key_name = f"{section}_{option}"
            config[key_name] = value
                
    return config

def main():
    # args = parser.parse_args()
    args, input_arg = parser.parse_known_args()

    entrypoint = Path(__file__).parent.joinpath("test_app.py")
    config = dict(
        browser_gatherUsageStats=False,
        client_toolbarMode="viewer",
        server_address=args.listen,
        server_port=args.port,
        server_headless=True,
        server_fileWatcherType="none",
        runner_fastReruns=True,
    )
    
    st_config_file= Path(__file__).parent.joinpath(".streamlit/config.toml")
    st_config = load_config_from_file(st_config_file)
    config.update(st_config)
        
    bootstrap.load_config_options(flag_options=config)
    bootstrap.run(str(entrypoint), False, input_arg, config)

import threading
import time
from streamlit.runtime import Runtime


def watcher():
    time.sleep(2)
    while True:
        
        main_thread = [t for t in threading.enumerate()]
        runtime: Runtime = Runtime.instance()
        sessions = [(s.session.id, s.session._state, s.session._session_state) for s in runtime._session_mgr.list_sessions()]
        print(main_thread)
        print(sessions)
        time.sleep(2)

if __name__ == "__main__":
    t = threading.Thread(target=watcher)
    t.start()
    main()