#!/usr/bin/env python

# SERIAL_PORT_PATH="/dev/cu.usbmodem14434401"  # your actual path to the Arduino Native serial port device goes here
SERIAL_PORT_PATH = "/dev/ttyACM0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/serial0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/tty0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/ttyS0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/ttyUSB0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/ttyS0"  # your actual path to the Arduino Native serial port device goes here
# SERIAL_PORT_PATH = "/dev/ttyS0"  # your actual path to the Arduino Native serial port device goes here
import sys, os

os.system("clear")
import hackeeg
from hackeeg import ads1299

import serial, io, msgpack, time

count = 0
print(f"  Connecting to \"{SERIAL_PORT_PATH}\"...")
# ser = serial.Serial(SERIAL_PORT_PATH, 2000000, timeout=1)
# ser = serial.serial_for_url(SERIAL_PORT_PATH, baudrate=2000000, timeout=1)
# ser = serial.serial_for_url(SERIAL_PORT_PATH, baudrate=2000000)
# ser = serial.serial_for_url(SERIAL_PORT_PATH, baudrate=115200)
ser = serial.serial_for_url(SERIAL_PORT_PATH, baudrate=2000000, timeout=1, write_timeout=0.1)
ser.reset_input_buffer()
ser.reset_output_buffer()
sp = io.TextIOWrapper(io.BufferedRWPair(ser, ser))


def send(msg, ser=ser):
    print(f"  sending {msg}...", end='', flush=True)

    # msg += "\r\n"

    # msg = msg.encode()
    # ser.write(msg)
    sp.write(msg)

    print(f",  receiving...", end='', flush=True)
    x = ser.read()
    print(f",  received: {x}", flush=True)
    return ser.readline()


def listen(ser=ser):
    return ser.read()


print(f"  Connected to {ser}!")
print("\n\n")

send("stop")
x = True




try:
    while x or True:
        count += 1
        # x = ser.readline()
        # x=send("BOARDLEDON")
        # x = send("TEXT")
        # send("text")
        x=send("help")

        # x = listen(ser)
        # print(f"#{count}: {x}")
        time.sleep(0.1)
        # print()
finally:
    ser.close()
exit()
# ser = serial.Serial(SERIAL_PORT_PATH, baudrate=112500, timeout=0.1)
# ser = serial.serial_for_url(SERIAL_PORT_PATH, baudrate=112500, timeout=0.1)
# ser.reset_input_buffer()
# raw_serial_port = io.TextIOWrapper(io.BufferedRWPair(ser, ser))
# message_pack_unpacker = msgpack.Unpacker(raw_serial_port, raw=False, use_list=False)
#
# print(f"ser: {ser}")
# print(f"raw_serial_port: {raw_serial_port}")
# print(f"message_pack_unpacker: {message_pack_unpacker}")
# exit()

import hackeeg
from hackeeg import ads1299

hackeegB = hackeeg.HackEEGBoard(SERIAL_PORT_PATH)
hackeegB.connect()
exit()
# os.system("chmod 777 /dev/ttyACM0")
prefix = "  ----  "
print(prefix, "Connecting to Arduino")
weiter = True
while weiter:
    import time

    # time.sleep(3)
    # time.sleep(10)
    try:
        import hackeeg
        from hackeeg import ads1299

        hackeegB = hackeeg.HackEEGBoard(SERIAL_PORT_PATH)
        hackeegB.connect()
        weiter = False
    except Exception as e:
        print(f" e: {e}")
print(prefix, "A")
hackeegB.sdatac()
print(prefix, "B")
hackeegB.reset()
print(prefix, "C")
hackeegB.blink_board_led()
print(prefix, "D")
hackeegB.disable_all_channels()
print(prefix, "E")
sample_mode = ads1299.HIGH_RES_250_SPS | ads1299.CONFIG1_const
print(prefix, "F")
hackeegB.wreg(ads1299.CONFIG1, sample_mode)
print(prefix, "G")
test_signal_mode = ads1299.INT_TEST_4HZ | ads1299.CONFIG2_const
print(prefix, "H")
hackeegB.wreg(ads1299.CONFIG2, test_signal_mode)
print(prefix, "I")
hackeegB.enable_channel(0)
print(prefix, "J")
hackeegB.wreg(ads1299.CH7SET, ads1299.TEST_SIGNAL | ads1299.GAIN_1X)
hackeegB.rreg(ads1299.CH5SET)

# Unipolar mode - setting SRB1 bit sends mid-supply voltage to the N inputs
hackeegB.wreg(ads1299.MISC1, ads1299.SRB1)
# add channels into bias generation
hackeegB.wreg(ads1299.BIAS_SENSP, ads1299.BIAS8P)
hackeegB.rdatac()
hackeegB.start()

while True:
    result = hackeegB.read_response()
    status_code = result.get('STATUS_CODE')
    status_text = result.get('STATUS_TEXT')
    data = result.get(hackeegB.DataKey)
    if data:
        decoded_data = result.get(hackeegB.DecodedDataKey)
        if decoded_data:
            timestamp = decoded_data.get('timestamp')
            ads_gpio = decoded_data.get('ads_gpio')
            loff_statp = decoded_data.get('loff_statp')
            loff_statn = decoded_data.get('loff_statn')
            channel_data = decoded_data.get('channel_data')
            print(f"timestamp:{timestamp} | gpio:{ads_gpio} loff_statp:{loff_statp} loff_statn:{loff_statn} |   ",
                  end='')
            for channel_number, sample in enumerate(channel_data):
                print(f"{channel_number + 1}:{sample} ", end='')
            print()
        else:
            print(data)
        sys.stdout.flush()
