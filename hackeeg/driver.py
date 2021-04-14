import base64
import binascii
import io
import json
import sys
from json import JSONDecodeError

import msgpack
import serial
import time

from . import ads1299

# TODO
# - MessagePack
# - MessagePack / Json Lines testing convenience functions

NUMBER_OF_SAMPLES = 10000
DEFAULT_BAUDRATE = 115200
# DEFAULT_BAUDRATE = 2000000
# DEFAULT_BAUDRATE = 9600
SAMPLE_LENGTH_IN_BYTES = 38  # 216 bits encoded with base64 + '\r\n\'

SPEEDS = {250: ads1299.HIGH_RES_250_SPS,
          500: ads1299.HIGH_RES_500_SPS,
          1000: ads1299.HIGH_RES_1k_SPS,
          2000: ads1299.HIGH_RES_2k_SPS,
          4000: ads1299.HIGH_RES_4k_SPS,
          8000: ads1299.HIGH_RES_8k_SPS,
          16000: ads1299.HIGH_RES_16k_SPS}

GAINS = {1: ads1299.GAIN_1X,
         2: ads1299.GAIN_2X,
         4: ads1299.GAIN_4X,
         6: ads1299.GAIN_6X,
         8: ads1299.GAIN_8X,
         12: ads1299.GAIN_12X,
         24: ads1299.GAIN_24X}


class Status:
    Ok = 200
    BadRequest = 400
    Error = 500


class HackEEGException(Exception):
    pass


class HackEEGBoard:
    TextMode = 0
    JsonLinesMode = 1
    MessagePackMode = 2

    CommandKey = "COMMAND"
    ParametersKey = "PARAMETERS"
    HeadersKey = "HEADERS"
    DataKey = "DATA"
    DecodedDataKey = "DECODED_DATA"
    StatusCodeKey = "STATUS_CODE"
    StatusTextKey = "STATUS_TEXT"

    MpCommandKey = "C"
    MpParametersKey = "P"
    MpHeadersKey = "H"
    MpDataKey = "D"
    MpStatusCodeKey = "C"
    MpStatusTextKey = "T"
    MaxConnectionAttempts = 2
    ConnectionSleepTime = 0.1

    # def __init__(self, serial_port_path=None, baudrate=DEFAULT_BAUDRATE, debug=True):
    def __init__(self, serial_port_path=None, baudrate=DEFAULT_BAUDRATE, debug=False):
        self.mode = None
        self.message_pack_unpacker = None
        self.debug = debug
        self.baudrate = baudrate
        self.rdatac_mode = False
        self.serial_port_path = serial_port_path

        if serial_port_path:
            pass
            print(f"step a", flush=True)
            self.raw_serial_port = serial.serial_for_url(serial_port_path, baudrate=self.baudrate, timeout=0.1)
            # self.raw_serial_port = serial.serial_for_url(serial_port_path, 2000000, timeout=0.1)
            print(f"step b", flush=True)
            self.raw_serial_port.reset_input_buffer()
            print(f"step c", flush=True)
            self.serial_port = io.TextIOWrapper(io.BufferedRWPair(self.raw_serial_port, self.raw_serial_port))
            # self.serial_port = io.BufferedRWPair(self.raw_serial_port, self.raw_serial_port, 1)
            # self.serial_port = self.raw_serial_port
            print(f"step d", flush=True)
            self.message_pack_unpacker = msgpack.Unpacker(self.raw_serial_port, raw=False, use_list=False)
            print(f"step e", flush=True)

    def connect(self):
        print(f"step f", flush=True)
        self.mode = self._sense_protocol_mode()
        print(f"step g", flush=True)
        if self.mode == self.TextMode:
            print(f"step h")
            attempts = 0
            connected = False
            while attempts < self.MaxConnectionAttempts:
                print("\n\n", )
                print("*" * 40)
                print("*" * 40)
                try:
                    print(" -- trying jsonlines_mode()...")
                    self.jsonlines_mode()
                    connected = True
                    break
                except JSONDecodeError:
                    print("&& ... Connecting...")
                    # if attempts == 0:
                    #     print("Connecting...", end='')
                    # elif attempts > 0:
                    #     pass
                    #     print('.', end='')
                    sys.stdout.flush()
                    attempts += 1
                    time.sleep(self.ConnectionSleepTime)
            if attempts > 0:
                print()
            if not connected:
                raise HackEEGException("Can't connect to Arduino")
        self.sdatac()
        line = self.serial_port.readline()
        while line:
            line = self.serial_port.readline()

    def _serial_write(self, command):
        print(f"    _serial_write(..): writing command: \n      \"{command}\"", end='', flush=True)
        command = command.encode()
        print("...")
        self.raw_serial_port.write(command)
        # self.serial_port.write(command)
        print(f"    _serial_write(..): Attempting to flush...", flush=True)
        self.serial_port.flush()
        print(f"    _serial_write(..): Flushed!", flush=True)

    def _serial_readline(self, serial_port=None):
        if serial_port is None:
            line = self.serial_port.readline()
        elif serial_port == "raw":
            line = self.raw_serial_port.readline()
        else:
            raise HackEEGException('Unknown serial port designator; must be either None or "raw"')
        return line

    def _serial_read_messagepack_message(self):
        message = self.message_pack_unpacker.unpack()
        if self.debug:
            print(f"message: {message}")
        return message

    def _decode_data(self, response):
        """decode ADS1299 sample status bits - datasheet, p36
        The format is:
        1100 + LOFF_STATP[0:7] + LOFF_STATN[0:7] + bits[4:7] of the GPIOregister"""
        error = False
        if response:
            data = response.get(self.DataKey)
            if data is None:
                data = response.get(self.MpDataKey)
                if type(data) is str:
                    try:
                        data = base64.b64decode(data)
                    except binascii.Error:
                        print(f"incorrect padding: {data}")

            if data and (type(data) is list or type(data) is bytes):
                data_hex = ":".join("{:02x}".format(c) for c in data)
                if error:
                    print(data_hex)
                timestamp = int.from_bytes(data[0:4], byteorder='little')
                sample_number = int.from_bytes(data[4:8], byteorder='little')
                ads_status = int.from_bytes(data[8:11], byteorder='big')
                ads_gpio = ads_status & 0x0f
                loff_statn = (ads_status >> 4) & 0xff
                loff_statp = (ads_status >> 12) & 0xff
                extra = (ads_status >> 20) & 0xff

                channel_data = []
                for channel in range(0, 8):
                    channel_offset = 11 + (channel * 3)
                    sample = int.from_bytes(data[channel_offset:channel_offset + 3], byteorder='big', signed=True)
                    channel_data.append(sample)

                response['timestamp'] = timestamp
                response['sample_number'] = sample_number
                response['ads_status'] = ads_status
                response['ads_gpio'] = ads_gpio
                response['loff_statn'] = loff_statn
                response['loff_statp'] = loff_statp
                response['extra'] = extra
                response['channel_data'] = channel_data
                response['data_hex'] = data_hex
                response['data_raw'] = data
        return response

    def set_debug(self, debug):
        self.debug = debug

    def read_response(self, serial_port=None, force_debug=False):
        """read a response from the Arduino– must be in JSON Lines mode"""
        # print(f"Listening on {serial_port}, {self.serial_port}")
        # serial_port="raw"
        message = self._serial_readline(serial_port=serial_port)
        print(f"      message: {message}")
        try:
            response_obj = json.loads(message)
        except UnicodeDecodeError:
            response_obj = None
        # except JSONDecodeError:
        #     response_obj = None
        if self.debug or force_debug:
            print(f"read_response line: {message}")
        if self.debug or force_debug:
            print("json response:")
            print(self.format_json(response_obj))
        return self._decode_data(response_obj)

    def read_rdatac_response(self):
        """read a response from the Arduino– JSON Lines or MessagePack mode are ok"""
        if self.mode == self.MessagePackMode:
            response_obj = self._serial_read_messagepack_message()
        else:
            message = self._serial_readline()
            try:
                response_obj = json.loads(message)
            except JSONDecodeError:
                response_obj = {}
                print()
                print(f"json decode error: {message}")
        if self.debug:
            print(f"read_response obj: {response_obj}")
        result = None
        try:
            result = self._decode_data(response_obj)
        except AttributeError:
            pass
        return result

    def format_json(self, json_obj):
        return json.dumps(json_obj, indent=4, sort_keys=True)

    def send_command(self, command, parameters=None, force_debug=True):
        if self.debug:
            print(f"command: {command}  parameters: {parameters}")
        # commands are only sent in JSON Lines mode
        new_command_obj = {self.CommandKey: command, self.ParametersKey: parameters}
        new_command = json.dumps(new_command_obj)

        if self.debug:
            print("json command:")
            print(self.format_json(new_command_obj))

        if self.debug or force_debug:
            print(f"   send_command(..) at _serial_write...")  # \n  new_command: {new_command}  \n\n parameters: {parameters}")

        self._serial_write(new_command)

        if self.debug or force_debug:
            print(f"   send_command(..) at _serial_write... \"\n\"")

        self._serial_write('\n')

    def send_text_command(self, command):
        self._serial_write(command + '\n')

    def execute_command(self, command, parameters=None, serial_port=None):
        if parameters is None:
            parameters = []
        print(f" execute_command():: sending {command},  {parameters}")
        self.send_command(command, parameters)
        print(f"  **  Response: WAITING")
        response = self.read_response(serial_port=serial_port)
        print(f"  **  Response: {response}")
        return response

    def _sense_protocol_mode(self):
        try:
            print(f" _sense_protocol_mode: step a", flush=True)
            self.send_command("stop")
            print(f" _sense_protocol_mode: step b", flush=True)
            self.send_command("sdatac")
            print(f" _sense_protocol_mode: step c", flush=True)
            result = self.execute_command("nop")
            print(f" _sense_protocol_mode: step d", flush=True)
            return self.JsonLinesMode
        except Exception as e:
            print(f" _sense_protocol_mode: exception: {e}", flush=True)
            # exit()
            return self.TextMode

    def ok(self, response):
        return response.get(self.StatusCodeKey) == Status.Ok

    def wreg(self, register, value):
        command = "wreg"
        parameters = [register, value]
        return self.execute_command(command, parameters)

    def rreg(self, register):
        command = "rreg"
        parameters = [register]
        response = self.execute_command(command, parameters)
        return response

    def nop(self):
        return self.execute_command("nop")

    def boardledon(self):
        return self.execute_command("boardledon")

    def boardledoff(self):
        return self.execute_command("boardledoff")

    def ledon(self):
        return self.execute_command("ledon")

    def ledoff(self):
        return self.execute_command("ledoff")

    def micros(self):
        return self.execute_command("micros")

    def text_mode(self):
        return self.send_command("text")

    def reset(self):
        return self.execute_command("reset")

    def start(self):
        return self.execute_command("start")

    def stop(self):
        return self.execute_command("stop")

    def rdata(self):
        return self.execute_command("rdata")

    def version(self):
        result = self.execute_command("version")
        return result

    def status(self):
        return self.execute_command("status")

    def jsonlines_mode(self):
        old_mode = self.mode
        self.mode = self.JsonLinesMode
        print("\n\n")
        print(f"      jsonlines_mode(): entered!")
        time.sleep(0.5)
        if old_mode == self.TextMode and True:
            print(f"      jsonlines_mode(): running textmode")
            self.send_text_command("jsonlines")
            print(f"      jsonlines_mode(): attempting to get a response...")
            response = self.read_response()
            print(f"      jsonlines_mode(): response: {response}")
            return self.read_response()
        if old_mode == self.JsonLinesMode:
            print(f"      jsonlines_mode(): running jsonlinesmode")
            self.execute_command("jsonlines")

    def messagepack_mode(self):
        old_mode = self.mode
        self.mode = self.MessagePackMode
        if old_mode == self.TextMode:
            self.send_text_command("jsonlines")
            response = self.read_response()
            self.execute_command("messagepack")
            return response
        elif old_mode == self.JsonLinesMode:
            response = self.execute_command("messagepack")
            return response

    def rdatac(self):
        result = self.execute_command("rdatac", serial_port="raw")
        if self.ok(result):
            self.rdatac_mode = True
        return result

    def sdatac(self):
        if self.mode == self.JsonLinesMode:
            result = self.execute_command("sdatac")
        else:
            self.send_command("sdatac")
            result = self.read_response(serial_port="raw")
        self.rdatac_mode = False
        return result

    def stop_and_sdatac_messagepack(self):
        """used to smoothly stop data transmission while in MessagePack mode–
        mostly avoids exceptions and other hiccups"""
        self.send_command("stop")
        self.send_command("sdatac")
        self.send_command("nop")
        try:
            line = self.serial_port.read()
        except UnicodeDecodeError:
            line = self.raw_serial_port.read()

    def enable_channel(self, channel, gain=None):
        if gain is None:
            gain = ads1299.GAIN_1X
        temp_rdatac_mode = self.rdatac_mode
        if self.rdatac_mode:
            self.sdatac()
        command = "wreg"
        parameters = [ads1299.CHnSET + channel, ads1299.ELECTRODE_INPUT | gain]
        self.execute_command(command, parameters)
        if temp_rdatac_mode:
            self.rdatac()

    def disable_channel(self, channel):
        command = "wreg"
        parameters = [ads1299.CHnSET + channel, ads1299.PDn | ads1299.SHORTED]
        self.execute_command(command, parameters)

    def enable_all_channels(self):
        for channel in range(1, 9):
            self.enable_channel(channel)

    def disable_all_channels(self):
        for channel in range(1, 9):
            self.disable_channel(channel)

    def blink_board_led(self):
        self.execute_command("boardledon")
        time.sleep(0.3)
        self.execute_command("boardledoff")
