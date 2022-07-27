#!/usr/bin/env python
# -*- coding: utf_8 -*-
"""
 Modbus TestKit: Implementation of Modbus protocol in python
 (C)2009 - Luc Jean - luc.jean@gmail.com
 (C)2009 - Apidev - http://www.apidev.fr
 This is distributed under GNU LGPL license, see license.txt
"""

import sys

import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import random
import time


def main():
    """main"""

    logger = modbus_tk.utils.create_logger(name="console", record_format="%(message)s")

    try:
        # Create the server
        server = modbus_tcp.TcpServer(port=502)
        logger.info("running...")
        logger.info("enter 'quit' for closing the server")

        server.start()

        slave_3 = server.add_slave(slave_id=3, fingerprint='HOLDING_REGISTERS')
        slave_3.add_block('A', cst.HOLDING_REGISTERS, 0, 100)

        slave_2 = server.add_slave(slave_id=2, fingerprint='READ_DISCRETE_INPUTS')
        slave_2.add_block('A', cst.READ_DISCRETE_INPUTS, 0, 100)

        slave_1 = server.add_slave(slave_id=1, fingerprint='READ_COILS')
        slave_1.add_block('A', cst.READ_COILS, 0, 100)

        slave_3.set_values('A', 0, 100)
        slave_2.set_values('A', 0, 1)
        slave_1.set_values('A', 0, 1)
        while True:
            # cmd = sys.stdin.readline()
            relays = slave_1.get_values('A', 0, 4)[0]
            status = slave_2.get_values('A', 0, 4)[0]
            if relays and status:
                voltage = random.randint(0, 128)  # 每5秒随机设定电压值，不能超过127，否则不能被utf8编码
                slave_3.set_values('A', 0, voltage)
            else:
                slave_3.set_values('A', 0, 0)
            time.sleep(5.0)


        while True:
            cmd = sys.stdin.readline()
            args = cmd.split(' ')

            if cmd.find('quit') == 0:
                sys.stdout.write('bye-bye\r\n')
                break

            elif args[0] == 'add_slave':
                slave_id = int(args[1])
                server.add_slave(slave_id)
                sys.stdout.write('done: slave %d added\r\n' % slave_id)

            elif args[0] == 'add_block':
                slave_id = int(args[1])
                name = args[2]
                block_type = int(args[3])
                starting_address = int(args[4])
                length = int(args[5])
                slave = server.get_slave(slave_id)
                slave.add_block(name, block_type, starting_address, length)
                sys.stdout.write('done: block %s added\r\n' % name)

            elif args[0] == 'set_values':
                slave_id = int(args[1])
                name = args[2]
                address = int(args[3])
                values = []
                for val in args[4:]:
                    values.append(int(val))
                slave = server.get_slave(slave_id)
                slave.set_values(name, address, values)
                values = slave.get_values(name, address, len(values))
                sys.stdout.write('done: values written: %s\r\n' % str(values))

            elif args[0] == 'get_values':
                slave_id = int(args[1])
                name = args[2]
                address = int(args[3])
                length = int(args[4])
                slave = server.get_slave(slave_id)
                values = slave.get_values(name, address, length)
                sys.stdout.write('done: values read: %s\r\n' % str(values))

            else:
                sys.stdout.write("unknown command %s\r\n" % args[0])
    finally:
        server.stop()


if __name__ == "__main__":
    main()
