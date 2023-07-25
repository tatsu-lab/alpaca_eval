import re
import sys


def update_version(file_path, new_version):
    # Read in the file
    with open(file_path, "r") as file:
        filedata = file.read()

    # Replace the target string
    version_regex = r"__version__ = ['\"]([^'\"]*)['\"]"
    filedata = re.sub(version_regex, f'__version__ = "{new_version}"', filedata)

    # Write the file out again
    with open(file_path, "w") as file:
        file.write(filedata)


if __name__ == "__main__":
    # Get the version from command line arguments
    new_version = sys.argv[1]

    # Update the version
    update_version("src/alpaca_eval/__init__.py", new_version)
