cd Python-3.12.*/
make altinstall
printf "checks:\n"
python3.12 --version
pip3.12 --version
rm -r /tmp/build-python


#printf "checks:\n"
#python3.12 --version
#pip3.12 --version
#printf "cleanup:\n"
#cd /
#rm -r tmp/build-python
#printf "done"
# ./configure --enable-optimizations