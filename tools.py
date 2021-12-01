import struct
from io import SEEK_END
from types import SimpleNamespace

import numpy as np

TRAC_HDR_SIZE = 240
HDR_FORMAT = '<7i4h8i2h4i13h2H31h6f'
HDR_KEYS = ('tracl', 'tracr', 'fldr', 'tracf', 'ep', 'cdp', 'cdpt', 'trid',
            'nvs', 'nhs', 'duse', 'offset', 'gelev', 'selev', 'sdepth', 'gdel',
            'sdel', 'swdep', 'gwdep', 'scalel', 'scalco', 'sx', 'sy', 'gx',
            'gy', 'counit', 'wevel', 'swevel', 'sut', 'gut', 'sstat', 'gstat',
            'tstat', 'laga', 'lagb', 'delrt', 'muts', 'mute', 'ns', 'dt',
            'gain', 'igc', 'igi', 'corr', 'sfs', 'sfe', 'slen', 'styp', 'stas',
            'stae', 'tatyp', 'afilf', 'afils', 'nofilf', 'nofils', 'lcf',
            'hcf', 'lcs', 'hcs', 'year', 'day', 'hour', 'minute', 'sec',
            'timbas', 'trwf', 'grnors', 'grnofr', 'grnlof', 'gaps', 'otrav',
            'd1', 'f1', 'd2', 'f2', 'ungpow', 'unscale')


def readsu(file):
    with open(file, 'rb') as f:
        # Read number of samples
        f.seek(114)
        ns = struct.unpack('<H', f.read(2))[0]

        # Find out file size
        f.seek(0, SEEK_END)
        file_size = f.tell()

        # Compute number of traces
        trac_data_size = ns * 4
        ntrac = file_size // (trac_data_size + TRAC_HDR_SIZE)

        print(f'Number of samples: {ns}')
        print(f'Number of traces: {ntrac}')

        data_format = f'{ns}f'

        d = np.zeros(shape=(ns, ntrac), dtype=np.float32)
        hdr = []
        f.seek(0)
        for i in range(ntrac):
            # Read trace header
            hdr_bytes = f.read(TRAC_HDR_SIZE)
            hdr_values = struct.unpack_from(HDR_FORMAT, hdr_bytes)
            hdr.append(SimpleNamespace(**dict(zip(HDR_KEYS, hdr_values))))

            # Read data trace
            data_bytes = f.read(trac_data_size)
            d[:, i] = np.array(struct.unpack(data_format, data_bytes),
                               dtype=np.float32)

        return d, hdr


def writesu(file, d, hdr):
    with open(file, 'wb') as f:
        ns, ntrac = d.shape
        trac_data_size = ns * 4
        data_format = f'{ns}f'

        print(f'Number of samples: {ns}')
        print(f'Number of traces: {ntrac}')

        def get_hdr_pos(i):
            return (TRAC_HDR_SIZE + trac_data_size) * i

        def get_data_pos(i):
            return (TRAC_HDR_SIZE + trac_data_size) * i + TRAC_HDR_SIZE

        for i in range(ntrac):
            # Write trace header
            f.seek(get_hdr_pos(i))
            hdr_bytes = struct.pack(
                HDR_FORMAT, hdr[i].tracl, hdr[i].tracr, hdr[i].fldr,
                hdr[i].tracf, hdr[i].ep, hdr[i].cdp, hdr[i].cdpt, hdr[i].trid,
                hdr[i].nvs, hdr[i].nhs, hdr[i].duse, hdr[i].offset,
                hdr[i].gelev, hdr[i].selev, hdr[i].sdepth, hdr[i].gdel,
                hdr[i].sdel, hdr[i].swdep, hdr[i].gwdep, hdr[i].scalel,
                hdr[i].scalco, hdr[i].sx, hdr[i].sy, hdr[i].gx, hdr[i].gy,
                hdr[i].counit, hdr[i].wevel, hdr[i].swevel, hdr[i].sut,
                hdr[i].gut, hdr[i].sstat, hdr[i].gstat, hdr[i].tstat,
                hdr[i].laga, hdr[i].lagb, hdr[i].delrt, hdr[i].muts,
                hdr[i].mute, hdr[i].ns, hdr[i].dt, hdr[i].gain, hdr[i].igc,
                hdr[i].igi, hdr[i].corr, hdr[i].sfs, hdr[i].sfe, hdr[i].slen,
                hdr[i].styp, hdr[i].stas, hdr[i].stae, hdr[i].tatyp,
                hdr[i].afilf, hdr[i].afils, hdr[i].nofilf, hdr[i].nofils,
                hdr[i].lcf, hdr[i].hcf, hdr[i].lcs, hdr[i].hcs, hdr[i].year,
                hdr[i].day, hdr[i].hour, hdr[i].minute, hdr[i].sec,
                hdr[i].timbas, hdr[i].trwf, hdr[i].grnors, hdr[i].grnofr,
                hdr[i].grnlof, hdr[i].gaps, hdr[i].otrav, hdr[i].d1, hdr[i].f1,
                hdr[i].d2, hdr[i].f2, hdr[i].ungpow, hdr[i].unscale)
            f.write(hdr_bytes)
            # Write trace data
            f.seek(get_data_pos(i))
            data_bytes = struct.pack(data_format, *d[:, i])
            f.write(data_bytes)
