#
echo "[install_to_rhinocode.sh] Сlearing previous cmmcore versions"
"$HOME/.rhinocode/py39-rh8/python3.9" -m pip uninstall mmcore -y


if [ ! -e $PWD/build ]
then
  echo "[install_to_rhinocode.sh] Going to the root of the cmmcore repository root"
  cd ..
fi
echo "[install_to_rhinocode.sh]" "Starting in" "$(pwd)"
"$HOME/.rhinocode/py39-rh8/python3.9" -m pip install -e "$(pwd)"
echo "[install_to_rhinocode.sh] Running tests ... "
"$HOME/.rhinocode/py39-rh8/python3.9" "$(pwd)/tests/test_intersections.py"
"$HOME/.rhinocode/py39-rh8/python3.9" "$(pwd)/tests/test_vec_speedups.py"