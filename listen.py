from scapy.all import *
cnt=0
def monitor(packet):
    global cnt
    cnt+=1
    print(cnt)
packet = sniff(prn=monitor,filter="tcp",store=1,timeout=5)
wrpcap(r'C:\Users\Mika\Desktop\组会\202205\output.pcap', packet)
