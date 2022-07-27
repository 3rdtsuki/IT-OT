#### 模拟ot靶场

安装修改版的modbus-tk

```sh
cd modbus-tk-fingerprint
sudo python3 setup.py install
```

mtu运行`python3 master.py`，里面的ip改为两个rtu的ip

两个rtu运行`sudo python3 slave.py`

listen.py抓包