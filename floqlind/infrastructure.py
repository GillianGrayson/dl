from enum import Enum
import socket

class DataPath(Enum):
    local_ah = 'E:/YandexDisk/Work'
    local_aa = 'E:/YandexDisk'

def get_path():
    host_name = socket.gethostname()
    path = ''
    if host_name == 'DESKTOP-K9VO2TI':
        path = DataPath.local_ah.value
    elif host_name == 'DESKTOP-7H2CNDR':
        path = DataPath.local_aa.value
    return path