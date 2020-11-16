## [Federal-Learning-Based-On-PySyft](https://github.com/lfalive/Federal-Learning-Based-On-PySyft)

### 参考

[联邦学习小系统搭建和测试（PySyft + Raspberry Pi 4）](https://zhuanlan.zhihu.com/p/181733116)

[OpenMined](https://github.com/OpenMined)/[PySyft](https://github.com/OpenMined/PySyft)

### 安装树莓派系统

[官方镜像下载链接](https://www.raspberrypi.org/downloads/raspberry-pi-os/)，选择Raspberry Pi OS (32-bit) with desktop and recommended software。

使用读卡器将树莓派的内存卡插入电脑，安装[Etcher](https://www.balena.io/etcher/)。选择zip镜像文件和内存卡对应盘符进行flash，十几分钟可以完成。

### PC与树莓派连接

系统镜像写入完成后，重新插拔一下读卡器，再将配置好的`wpa_supplicant.conf`文件和空白`ssh`文件复制到Boot盘符里，用于树莓派连接电脑热点和ssh连接。本项目根目录有`wpa_supplicant.conf`模板文件。
将SD卡插入到树莓派主板背面的micro SD插槽上即可开机。打开电脑热点，很快就可以看到一个raspberrypi设备连接到了电脑热点，此时可以记录下树莓派的局域网IP，现在就可以通过`ssh`工具连接该IP。一般初始登录账号为`pi`，密码`raspberry`。

为了能使用树莓派图形化界面，需要`xrdp`远程桌面服务。

```bash
sudo apt-get install xrdp 	//安装xrdp远程桌面服务
sudo /etc/init.d/xrdp start //开启xrdp服务
sudo update-rc.d xrdp defaults 	//将xrdp服务加入到默认系统启动列表
```

### 基本配置

然后PC端可用远程桌面连接访问树莓派图形界面，可以改登录密码、[改时区和中文显示](https://blog.csdn.net/qq_41204464/article/details/82941496)等。

系统自带了python3.7.3，换个源就好（如清华源）。先注释掉`/etc/pip.conf`中的内容，再运行：

```bash
sudo python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

再换系统源，[参考链接](https://mirrors.tuna.tsinghua.edu.cn/help/raspbian/)。然后更新一下系统，应该会更新很多包。

```bash
sudo apt-get update
sudo apt-get upgrade
```

### 配置项目环境

#### 树莓派端

##### 安装pytorch

采用知乎教程第三种方法安装`pytorch`，直接安装别人编译好的`Pytorch`和`torchvision`的包，本项目根目录有两个所需whl文件。

```bash
python3 -m pip install torch-1.4.0a0+f43194e-cp37-cp37m-linux_armv7l.whl
python3 -m pip install torchvision-0.5.0a0+9cdc814-cp37-cp37m-linux_armv7l.whl
```

##### 安装syft0.2.4

```bash
sudo apt-get install libopenblas-dev m4 libblas-dev cmake libatlas-base-dev gfortran libffi-dev libavformat-dev libavdevice-dev libjpeg-dev
python3 -m pip install -U pip setuptools
python3 -m pip install syft==0.2.4 --no-dependencies
python3 -m pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 requests~=2.22.0 numpy~=1.18.1
```

#### PC端

##### 安装pytorch

命令安装即可，装`cuda`版本也可以。

```bash
python -m pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

##### 安装syft0.2.4

如果`zstd`模块安装不上，就删掉`zstd`，正常安装其他模块。

```bash
python -m pip install -U pip setuptools
python -m pip install syft==0.2.4 --no-dependencies
python -m pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 requests~=2.22.0 numpy~=1.18.1
```

### 本地测试

进入项目目录，开三个终端分别运行`run_websocket_server.py`，仅显示一句`Serving. Press CTRL-C to stop.`即成功，不会显示其他内容。

```bash
python run_websocket_server.py --port 8777 --id alice
python run_websocket_server.py --port 8778 --id bob
python run_websocket_server.py --port 8779 --id charlie
```

再运行`main.py`即可。

### 正式运行

将`run_websocket_server.py`文件copy到树莓派中，并用三个树莓派分别运行即可，终端显示内容和本地测试一样。

```bash
python3 run_websocket_server.py --host 'xxx.xxx.xxx.xxx' --port 8777 --id alice
python3 run_websocket_server.py --host 'xxx.xxx.xxx.xxx' --port 8778 --id bob
python3 run_websocket_server.py --host 'xxx.xxx.xxx.xxx' --port 8779 --id charlie
```

树莓派端运行时要加入各自的IP参数，PC端也要修改`main.py`内`WebsocketClientWorker`函数对应的host参数再运行。

训练的epoch和batch_size等具体参数，命令行运行的时候加对应参数即可，例如：

```bash
python main.py --batch_size 256
```