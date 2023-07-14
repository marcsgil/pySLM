import sys
import os
import platform
import re

from optics import log


# Must install the SDK from: https://www.alpao.com/Download/AlpaoSDK

# Tell it where the BAX240.cfg or equivalent are.
os.environ['ACECFG'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

# Load the platform-specific library
arch = platform.architecture()
if re.match('^Windows', arch[1]):
    if arch[0] == '64bit':
        # Win64
        dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'win')
        sys.path.append(dll_path)
        os.environ['PYTHONPATH'] = ';'.join([os.environ['PYTHONPATH'], dll_path])

        import optics.external.alpao_dm.win.Lib64.asdk as asdk
    else:
        # Win32
        import optics.external.alpao_dm.win.Lib.asdk as asdk
else:
    # Linux?
    if arch[0] == '64bit':
        import optics.external.alpao_dm.linux.x64.asdk as asdk
    else:
        import optics.external.alpao_dm.linux.x86.asdk as asdk

