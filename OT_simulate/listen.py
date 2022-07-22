from scapy.all import *
cnt=0
def monitor(packet):
    global cnt
    cnt+=1
    print(cnt)
packet = sniff(prn=monitor,filter="tcp",store=1,timeout=180)
wrpcap(r'output.pcap', packet)
