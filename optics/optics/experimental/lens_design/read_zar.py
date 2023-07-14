from pathlib import Path
import zipfile
from typing import Union, Generator, Tuple

import argparse
import logging

from examples import log

__all__ = ['extract', 'repack', 'read']


def decompress_lzw(compressed: bytes) -> bytes:
    """
    Decompresses bytes using the variable LZW algorithm, starting with code strings of length 9.
    https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch

    Adapted from https://gist.github.com/BertrandBordage/611a915e034c47aa5d38911fc0bc7df9

    :param compressed: The compressed bytes without header.
    :return: The decompressed bytes.
    """
    # Convert input to bits
    compressed_bits: str = bin(int.from_bytes(compressed, 'big'))[2:].zfill(len(compressed) * 8)  # convert to binary string and pad to 8-fold length

    code_word_length = 8
    words: list = [_.to_bytes(1, 'big') for _ in range(2**code_word_length)]  # integer codes refer to a words in an expanding dictionary

    bit_index = 0
    previous_word: bytes = b''
    decompressed: list = []

    while True:
        if 2**code_word_length <= len(words):  # If the dictionary is full
            code_word_length += 1              # increase the code word length
        if bit_index + code_word_length > len(compressed_bits):
            break  # stop when the bits run out
        # Get the next code word from the data bit string
        code = int(compressed_bits[bit_index:bit_index + code_word_length], 2)
        bit_index += code_word_length

        # If word in dictionary, use it; else add it as a new word
        latest_word: bytes = words[code] if code < len(words) else previous_word + previous_word[:1]
        decompressed.append(latest_word)  # Update result
        if len(previous_word) > 0:  # Skip first iteration
            words.append(previous_word + latest_word[:1])  # Add as new encoding

        previous_word: bytes = latest_word

    return b''.join(decompressed)  # convert to bytes


def read(input_file_path: Union[Path, str]) -> Generator[Tuple, None, None]:
    """
    Generates a series of (unpacked file name, unpacked file contents) tuples in the order found in the archive.

    :param input_file_path: The archive or the path to the archive.
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    with open(input_file_path, 'rb') as input_file:
        version_length = 2
        while True:
            version = input_file.read(version_length)
            if len(version) < version_length:
                break  # end of file
            if version[0] == 0xec:
                header_length = 0x288 - version_length
            elif version[0] == 0xea:
                header_length = 0x14c - version_length
            else:
                log.warning(f'Unknown ZAR header "{version.hex()}"!')
                header_length = 0x288 - version_length
                version = 0xec03.to_bytes(2, 'big')  # override and cross fingers

            header = input_file.read(header_length)

            # flag1 = int.from_bytes(header[0x04-version_length:0x08-version_length], byteorder='little', signed=False)
            # flag2 = int.from_bytes(header[0x08-version_length:0x10-version_length], byteorder='little', signed=False)
            # log.info(f'Header {flag1} {flag2} {header[0x20-version_length:0x30-version_length].hex()}')
            if version[0] == 0xec:
                packed_file_size = int.from_bytes(header[0x10-version_length:0x18-version_length], byteorder='little', signed=False)
                unpacked_file_size = int.from_bytes(header[0x18-version_length:0x20-version_length], byteorder='little', signed=False)
                packed_file_name = header[0x30-version_length:].decode('utf-16-le')
                packed_file_name = packed_file_name[:packed_file_name.find('\0')]  # ignore all 0's on the right
            else:
                packed_file_size = int.from_bytes(header[0x0c-version_length:0x10-version_length], byteorder='little', signed=False)
                unpacked_file_size = int.from_bytes(header[0x10-version_length:0x14-version_length], byteorder='little', signed=False)
                packed_file_name = header[0x20-version_length:]
                packed_file_name = packed_file_name[:packed_file_name.find(0x00)]
                packed_file_name = packed_file_name.decode('utf-8')
            log.debug(f'Version {version.hex()}. Packed file {packed_file_name} has size {unpacked_file_size} ({packed_file_size}) bytes.')

            # Read and process data
            data = input_file.read(packed_file_size)
            if packed_file_name[-4:].upper() == '.LZW':
                data = decompress_lzw(data)
                packed_file_name = packed_file_name[:-4]

            # Yield a series of tuples from the Generator
            yield packed_file_name, data


def extract(input_file_path: Union[Path, str], output_path: Union[Path, str, None] = None):
    """
    Imports the data from a zar archive file and writes it as a regular directory.

    :param input_file_path: The path to zar-file.
    :param output_path: The path where the files should be saved. Default: the same as the input_file_path but
    without the extension.
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    # By default, just drop the .zar extension
    if output_path is None:
        output_path = input_file_path.name
        if output_path.lower().endswith('.zar'):
            output_path = output_path[:-4]
        output_path = input_file_path.parent / output_path
    else:
        if isinstance(output_path, str):
            output_file_path = Path(output_path)
    Path.mkdir(output_path, exist_ok=True)
    log.debug(f'Extracting {input_file_path} to directory {output_path}...')

    # Unpack and store the recovered data
    for unpacked_file_name, unpacked_data in read(input_file_path):
        with open(output_path / unpacked_file_name, 'wb') as unpacked_file:
            unpacked_file.write(unpacked_data)


def repack(input_file_path: Union[Path, str], output_file_path: Union[Path, str, None] = None):
    """
    Imports the data from a zar archive file and writes it as a regular zip file.

    :param input_file_path: The path to zar-file.
    :param output_file_path: The path to the zip file. Default: the same as the input_file_path but with the extension
    changed to 'zip'
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    # By default, just change .zar to .zip
    if output_file_path is None:
        output_file_name = input_file_path.name
        if output_file_name.lower().endswith('.zar'):
            output_file_name = output_file_name[:-4]
        output_file_name += '.zip'
        output_file_path = input_file_path.parent / output_file_name
    else:
        if isinstance(output_file_path, str):
            if not output_file_path.lower().endswith('.zip'):
                output_file_path += '.zip'
            output_file_path = Path(output_file_path)
        Path.mkdir(output_file_path.parent, exist_ok=True)
    log.debug(f'Converting {input_file_path} to zip archive {output_file_path}...')

    # Open the output archive and start storing unpacked files
    with zipfile.ZipFile(output_file_path, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=False,
                         compresslevel=9) as archive_file:
        # Unpack and store the recovered data
        for unpacked_file_name, unpacked_data in read(input_file_path):
            archive_file.writestr(f'{output_file_path.name}/{unpacked_file_name}', unpacked_data)


if __name__ == '__main__':
    # input_file_path = Path('/home/tom/Downloads/zars/LA1116-Zemax.zar')
    # input_file_path = Path('/home/tom/Downloads/zars/LB1901-A-ML-Zemax(ZAR).zar')
    # input_file_path = Path('/home/tom/Downloads/zars/LA1024-Zemax(ZAR).zar')
    # input_file_path = Path('/home/tom/Downloads/zars/LB1761-A-ML-Zemax(ZAR).zar')

    # Initialize parser
    input_parser = argparse.ArgumentParser(description='''
 Zemax archive files (.zar) unpacker and zipper.
 
 Examples:
 > unzar -i filename.zar
 > unzar -i filename.zar -z
 > unzar -i filename.zar -o filename.zip
 > unzar -i filename.zar -o subfolder/filename.zip
 > unzar -i filename.zar -v debug
 ''')
    input_parser.add_argument('-v', '--verbosity', choices=['debug', 'info', 'warning', 'error'],
                              help='the path to the PNG image file describing the simulation structure', default='debug')
    input_parser.add_argument('-i', '--input', type=str, nargs='*', help='the input archive', default=['/home/tom/Downloads/zars/LB1761-A-ML-Zemax(ZAR).zar'])
    input_parser.add_argument('-o', '--output', type=str, help='the output archive or directory')
    input_parser.add_argument('-z', '--zip', type=bool, help='create an archive instead of a directory', default=False)
    input_args = input_parser.parse_args()
    if input_args.verbosity.lower().startswith('deb'):
        log.setLevel(logging.DEBUG)
    elif input_args.verbosity.lower().startswith('war'):
        log.setLevel(logging.WARNING)
    elif input_args.verbosity.lower().startswith('err'):
        log.setLevel(logging.ERROR)
    for _ in log.handlers:
        _.setLevel(log.level)
    log.debug('Parsed input arguments: ' + str(input_args))

    if input_args.input is None:
        input_parser.print_help()
        exit(1)

    for input_file in input_args.input:
        log.info(f'Loading {input_file}...')
        input_file_path = Path(input_file)

        if input_args.zip:
            repack(input_file_path, input_args.output)
            log.info(f'Converted {input_file_path} to zip archive.')
        else:
            extract(input_file_path, input_args.output)
            log.info(f'Extracted {input_file_path} to directory.')
