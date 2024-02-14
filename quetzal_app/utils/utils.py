from base64 import b64encode
import os

def format_time(seconds, show_hours=True, final_time=False):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60

    if final_time:
        if hours == 0:
            return f"{minutes:02d}:{seconds:05.2f}", False
    else:
        if not show_hours:
            return f"{minutes:02d}:{seconds:05.2f}", False

    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}", True

def get_base64(svg_file):
    with open(svg_file, "rb") as file:
        return b64encode(file.read()).decode()
    
def get_icon(file_path):
    with open(file_path, "r") as svg_file:
        svg_content = svg_file.read()
        encoded_svg = b64encode(svg_content.encode("utf-8")).decode("utf-8")
    svg_image = f'<img src="data:image/svg+xml;base64,{encoded_svg}" alt="Icon" style="width: 24px; height: 24px;">'
    return svg_image

def get_directory_list(path):
    directories = []
    while True:
        path, directory = os.path.split(path)

        if directory != "":
            directories.append(directory)
        else:
            # If the path has been completely reduced to its root, break the loop
            if path != "":
                directories.append(path)
            break

    directories.reverse()
    return directories
