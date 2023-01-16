echo "Deploy addons/mmocc/pythonocc-utils ..."
cd mmcore/addons/mmocc && git submodule add git@github.com:contextmachine/pythonocc-utils.git && python -m pip install ./pythonocc-utils
git commit -m "Added pythonocc-utils submodule"
echo "Success! Use 'import OCC.Utils' to module import."