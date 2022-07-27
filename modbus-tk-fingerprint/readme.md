#### modbus-tk-fingerprint

这是一个修改版的modbus-tk库，给响应modbus报文增加了指纹字段

##### 1.在slave端，给modbus报文添加指纹字段

`modbus-tk/modbus.py` Slave类，并修改每个add_slave构造函数

```python
class Slave(object):
    def __init__(self, slave_id, unsigned=True, memory=None, fingerprint=None):
        """Constructor"""
        self._id = slave_id
        self._fingerprint = fingerprint  # 增加一个指纹字段
```

Line 756：

```python
# 根据不同功能码，构建响应报文的pdu
response_pdu = self._fn_code_map[function_code](request_pdu)
# 增加指纹，uft-8编码
response_pdu += self._fingerprint.encode("utf-8")
```

##### 2.在master端，实现读取modbus报文标识字段功能

`modbus.py` Master类：excute中解析响应的pdu

line167:对于读线圈和离散输入，注释掉原来的逻辑，复制读保持寄存器的代码

```python
        if function_code == defines.READ_COILS or function_code == defines.READ_DISCRETE_INPUTS:
            # is_read_function = True
            # pdu = struct.pack(">BHH", function_code, starting_address, quantity_of_x)
            # byte_count = quantity_of_x // 8
            # if (quantity_of_x % 8) > 0:
            #     byte_count += 1
            # nb_of_digits = quantity_of_x
            # if not data_format:
            #     data_format = ">" + (byte_count * "B")
            # if expected_length < 0:
            #     # No length was specified and calculated length can be used:
            #     # slave + func + bytcodeLen + bytecode + crc1 + crc2
            #     expected_length = byte_count + 5
            is_read_function = True
            pdu = struct.pack(">BHH", function_code, starting_address, quantity_of_x)
            if not data_format:
                data_format = ">" + (quantity_of_x * "H")
            if expected_length < 0:
                # No length was specified and calculated length can be used:
                # slave + func + bytcodeLen + bytecode x 2 + crc1 + crc2
                expected_length = 2 * quantity_of_x + 5
```

Line 381：注释掉长度异常检查

```python
if is_read_function:
    # get the values returned by the reading function
    byte_count = byte_2
    data = response_pdu[2:]  # 数据段是pdu的第三个字段，现在又加上了标识字段
    # if byte_count != len(data):
    #     # the byte count in the pdu is invalid
    #     raise ModbusInvalidResponseError(
    #         "Byte count is {0} while actual number of bytes is {1}. ".format(byte_count, len(data))
    #     )
```

Line 419：excute要返回实际的数据长度，便于获取数据

```python
return result, byte_2  # 返回数据段，以及不包含标识部分的长度
```

然后重新安装modbus-tk

```sh
pip uninstall modbus-tk
python setup.py install
```
