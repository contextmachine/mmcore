mkdir /tmp/build-python
cd /tmp/build-python
apt update -y
apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libpython3-dev
wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz
tar -xf Python-3.12.2.tgz
cd Python-3.12.*/
bash ./configure --enable-optimizations

export CPU_COUNT=$(cat /proc/cpuinfo | grep processor | wc -l)
make -j $CPU_COUNT


#
#make altinstall
#printf "checks:\n"
#python3.12 --version
#pip3.12 --version
#printf "cleanup:\n"
#cd /
#rm -r tmp/build-python
#printf "done"
# ./configure --enable-optimizations