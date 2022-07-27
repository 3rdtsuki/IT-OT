#!/usr/bin/env python
# -*- coding: utf_8 -*-
"""
 Modbus TestKit: Implementation of Modbus protocol in python
 (C)2009 - Luc Jean - luc.jean@gmail.com
 (C)2009 - Apidev - http://www.apidev.fr
 This is distributed under GNU LGPL license, see license.txt
"""

from __future__ import print_function
from cmath import log

import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp, hooks
import logging
import time

import pdb

logger = modbus_tk.utils.create_logger("console", level=logging.DEBUG)

query_interval = 10
slave_hosts = []


def query_one_slave(slave: modbus_tcp.TcpMaster):
    global logger
    logger.debug("on slave: {} {}".format(slave._host, slave._port))

    logger.debug("LC:// 1")
    co_data,data_len = slave.execute(slave=1, function_code=cst.READ_COILS, starting_address=0, quantity_of_x=10)
    logger.info("co_data: {}".format(co_data))
    slave.close()
    
    logger.debug("LC:// 2")
    di_data,data_len = slave.execute(2, cst.READ_DISCRETE_INPUTS, 0, 10)
    logger.info("di_data: {}".format(di_data))
    slave.close()

    logger.debug("LC:// 3")
    hr_data,data_len = slave.execute(3, cst.READ_HOLDING_REGISTERS, 0, 10)
    logger.info("hr_data: {}".format(hr_data))
    slave.close()

    # 可选
    # logger.info("set coil with {:x}".format(~co_data & 1))
    # slave.execute(1, cst.WRITE_SINGLE_COIL, 0, output_value=(~co_data & 1))
    # slave.close()
    print()


def main():
    """main"""
    global query_interval
    global slave_hosts

    def on_after_recv(data):
        master, bytes_data = data
        logger.info(bytes_data)

    hooks.install_hook('modbus.Master.after_recv', on_after_recv)

    try:

        def on_before_connect(args):
            master = args[0]
            logger.debug("on_before_connect {0} {1}".format(master._host, master._port))

        hooks.install_hook("modbus_tcp.TcpMaster.before_connect", on_before_connect)

        def on_after_recv(args):
            response = args[1]
            logger.debug("on_after_recv {0} bytes received".format(len(response)))

        hooks.install_hook("modbus_tcp.TcpMaster.after_recv", on_after_recv)

        # Connect to the slave

        rtu1 = modbus_tcp.TcpMaster(host="rtu1", port=502, timeout_in_sec=2.0)
        rtu2 = modbus_tcp.TcpMaster(host="localhost", port=502, timeout_in_sec=2.0)
        # slave_hosts.append(rtu1)
        logger.info("rtu1 connected")
        slave_hosts.append(rtu2)
        logger.info("rtu2 connected")

        cnt = 0
        while True:
            print('cnt = %d' % (cnt))
            cnt += 1
            list(map(query_one_slave, slave_hosts))
            time.sleep(query_interval)

    except modbus_tk.modbus.ModbusError as exc:
        logger.error("%s- Code=%d", exc, exc.get_exception_code())


if __name__ == "__main__":
    main()
