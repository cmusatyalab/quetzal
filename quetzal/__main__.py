import argparse
import os
from pathlib import Path
from quetzal.dtos.dtos import QuetzalFile

def generate_metadata_files(root_dir, metadata_dir, creator="admin"):
    """
    Generate missing metadata files for each file and directory in root_dir
    using the Path module from pathlib for improved path handling.
    """
    root_dir = Path(root_dir)
    metadata_dir = Path(metadata_dir)

    for item in root_dir.rglob("*"):
        relative_path = item.relative_to(root_dir)

        user_folder = False
        if item.parent == root_dir:
            user_folder = True

        # Generate paths for .info.txt and .meta.txt files in metadata_dir
        meta_path = metadata_dir / relative_path.as_posix()
        info_path = QuetzalFile._getDescriptionPath(meta_path)
        meta_path = QuetzalFile._getMetaDataPath(meta_path)

        # Ensure the directory structure exists in metadata_dir
        info_path.parent.mkdir(parents=True, exist_ok=True)

        if item.is_dir():
            # os.makedirs(meta_path, exist_ok=True)
            if not meta_path.exists():
                meta_path.write_text(
                    QuetzalFile.PROJECT_DEFAULT_META + "CreatedBy::= " + creator + "\n"
                )
            if not info_path.exists():
                info_path.write_text(QuetzalFile.PROJECT_DEFAULT_DESCRIPTION)

        else:
            if not meta_path.exists():
                if user_folder:
                    meta_path.write_text(QuetzalFile.USER_ROOT_META)
                else:
                    meta_path.write_text(QuetzalFile.FILE_DEFAULT_META)
            if not info_path.exists():
                if user_folder:
                    info_path.write_text(
                        QuetzalFile.USER_ROOT_META + "CreatedBy::= " + creator + "\n"
                    )
                else:
                    info_path.write_text(QuetzalFile.FILE_DEFAULT_DESCRIPTION)


def generate_root_meta(root_dir, metadata_dir, creator="admin"):
    root_dir = Path(root_dir).parent
    metadata_dir = Path(metadata_dir).parent

    for item in root_dir.iterdir():
        relative_path = item.relative_to(root_dir)

        info_path = QuetzalFile._getDescriptionPath(
            metadata_dir / relative_path.as_posix()
        )
        meta_path = QuetzalFile._getMetaDataPath(
            metadata_dir / relative_path.as_posix()
        )

        if not meta_path.exists():
            meta_path.write_text(
                QuetzalFile.ROOT_META + "CreatedBy::= " + creator + "\n"
            )
        if not info_path.exists():
            info_path.write_text(QuetzalFile.ROOT_DESCRIPTOIN)


def init_app(database_root, metadata_root, creator="admin"):
    """Initialize the application with given database and metadata roots."""
    os.makedirs(database_root, exist_ok=True)
    os.makedirs(metadata_root, exist_ok=True)

    generate_root_meta(database_root, metadata_root, creator)

    print(
        f"Initialized app with database root: {database_root} and metadata root: {metadata_root}"
    )


def add_user(username, database_root, metadata_root, creator="admin"):
    """Add a new user directory."""
    user_dir = os.path.join(database_root, username)
    os.makedirs(user_dir, exist_ok=True)
    generate_metadata_files(database_root, metadata_root, creator)
    print(f"Added user {username} with directory: {user_dir}")


def import_data(database_root, metadata_root, creator="admin"):
    """Placeholder function for data import."""
    print(
        f"Importing data to database root: {database_root} and metadata root: {metadata_root}"
    )
    if not Path(database_root).exists() or not Path(metadata_root).exists():
        raise FileNotFoundError(
            f"Either {database_root} or {metadata_root} do not Exist"
        )
    generate_root_meta(database_root, metadata_root, creator)
    generate_metadata_files(database_root, metadata_root, creator)


def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Quetzal CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Parser for "init" command
    parser_init = subparsers.add_parser(
        "init", help="Initialize the database at given route"
    )
    parser_init.add_argument(
        "-d",
        "--database_root",
        default="./data/home/root",
        help="Database root directory",
    )
    parser_init.add_argument(
        "-m",
        "--metadata_root",
        default="./data/meta_data/root",
        help="Metadata root directory",
    )

    # Parser for "user" command
    parser_user = subparsers.add_parser("user", help="Add a new user to the database")
    parser_user.add_argument(
        "-n", "--username", required=True, help="Username for the new user"
    )
    parser_user.add_argument(
        "-d",
        "--database_root",
        default="./data/home/root",
        help="Database root directory",
    )
    parser_user.add_argument(
        "-m",
        "--metadata_root",
        default="./data/meta_data/root",
        help="Metadata root directory",
    )

    # Parser for "import" command
    parser_import = subparsers.add_parser(
        "import", help="Use data at database_root to generate metadata at metadata_root"
    )
    parser_import.add_argument(
        "-d",
        "--database_root",
        default="./data/home/root",
        help="Database root directory",
    )
    parser_import.add_argument(
        "-m",
        "--metadata_root",
        default="./data/meta_data/root",
        help="Metadata root directory",
    )

    # Parse the args
    args = parser.parse_args()

    # Use match-case for command dispatch
    match args.command:
        case "init":
            init_app(args.database_root, args.metadata_root)
        case "user":
            add_user(args.username, args.database_root, args.metadata_root)
        case "import":
            import_data(args.database_root, args.metadata_root)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
