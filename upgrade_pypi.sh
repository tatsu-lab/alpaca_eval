# Helper to push a new version to pypi.
# Remember to run `rm -rf ./dist` to remove previous versions; otherwise, those will get re-uploaded.
python3 -m pip install --upgrade build
python3 -m build # creates dist/ folder
python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/* # your repository is NOT testpypi!!!
