import os
import re
import bcrypt
import shutil

def hash_password(plain_text_password):
    hashed = bcrypt.hashpw(plain_text_password.encode('utf-8'), bcrypt.gensalt())
    return hashed

def check_password(hashed_password, user_password):
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password)

def delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' has been deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")

def get_directories(target_directory):
    """
    Retrieves all directory names inside the given target directory.

    Args:
    target_directory (str): The path of the target directory.

    Returns:
    list: A list of names of all directories inside the target directory.
    """

    directories = [d for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]
    return directories

def is_valid_directory_name(name):
    """
    Checks if the given name is appropriate for a filesystem directory.

    Args:
    name (str): The directory name to check.

    Returns:
    bool: True if the name is valid, False otherwise.
    """

    # Check for null or empty string
    if not name or name.isspace():
        return False

    # Define illegal characters for directory names
    # This may vary between different operating systems and file systems
    illegal_chars = r'<>:"/\\|?* '

    # Check for illegal characters
    if any(char in illegal_chars for char in name):
        return False

    # Check for reserved names in Windows (e.g., CON, PRN, AUX, NUL, COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, LPT1, LPT2, LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, and LPT9)
    if os.name == 'nt' and re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$', name, re.IGNORECASE):
        return False

    # Check for length constraints (generally 255 characters, but can be filesystem-dependent)
    if len(name) > 100:
        return False

    return True

