"""Original Maxwell fixtures; no game-derived inputs."""
import struct

EXIT = 0xe30000000007000f


def graphics(instructions, *, color=False):
    header = bytearray(0x50)
    if color:
        struct.pack_into('<I', header, 0x48, 1)
    code = bytearray(header)
    for instruction in instructions:
        # Bundles begin with a scheduling word, relative to the end of SPH.
        if (len(code) - len(header)) % 32 == 0:
            code += bytes(8)
        code += struct.pack('<Q', instruction)
    return code


def mov32(value, mask=0xf):
    return 0x0100000000000000 | (value << 20) | (7 << 16) | (mask << 12)
