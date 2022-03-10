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
        hdr = new_empty_hdr(ntrac)
        f.seek(0)
        for i in range(ntrac):
            # Read trace header
            hdr_bytes = f.read(TRAC_HDR_SIZE)
            hdr_values = struct.unpack_from(HDR_FORMAT, hdr_bytes)
            for key, value in zip(HDR_KEYS, hdr_values):
                hdr[key][i] = value

            # Read data trace
            data_bytes = f.read(trac_data_size)
            d[:, i] = np.array(struct.unpack(data_format, data_bytes),
                               dtype=np.float32)

        return d, SimpleNamespace(**hdr)


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
                HDR_FORMAT, hdr.tracl[i], hdr.tracr[i], hdr.fldr[i],
                hdr.tracf[i], hdr.ep[i], hdr.cdp[i], hdr.cdpt[i], hdr.trid[i],
                hdr.nvs[i], hdr.nhs[i], hdr.duse[i], hdr.offset[i],
                hdr.gelev[i], hdr.selev[i], hdr.sdepth[i], hdr.gdel[i],
                hdr.sdel[i], hdr.swdep[i], hdr.gwdep[i], hdr.scalel[i],
                hdr.scalco[i], hdr.sx[i], hdr.sy[i], hdr.gx[i], hdr.gy[i],
                hdr.counit[i], hdr.wevel[i], hdr.swevel[i], hdr.sut[i],
                hdr.gut[i], hdr.sstat[i], hdr.gstat[i], hdr.tstat[i],
                hdr.laga[i], hdr.lagb[i], hdr.delrt[i], hdr.muts[i],
                hdr.mute[i], hdr.ns[i], hdr.dt[i], hdr.gain[i], hdr.igc[i],
                hdr.igi[i], hdr.corr[i], hdr.sfs[i], hdr.sfe[i], hdr.slen[i],
                hdr.styp[i], hdr.stas[i], hdr.stae[i], hdr.tatyp[i],
                hdr.afilf[i], hdr.afils[i], hdr.nofilf[i], hdr.nofils[i],
                hdr.lcf[i], hdr.hcf[i], hdr.lcs[i], hdr.hcs[i], hdr.year[i],
                hdr.day[i], hdr.hour[i], hdr.minute[i], hdr.sec[i],
                hdr.timbas[i], hdr.trwf[i], hdr.grnors[i], hdr.grnofr[i],
                hdr.grnlof[i], hdr.gaps[i], hdr.otrav[i], hdr.d1[i], hdr.f1[i],
                hdr.d2[i], hdr.f2[i], hdr.ungpow[i], hdr.unscale[i])
            f.write(hdr_bytes)
            # Write trace data
            f.seek(get_data_pos(i))
            data_bytes = struct.pack(data_format, *d[:, i])
            f.write(data_bytes)


def new_empty_hdr(ntrac):
    hdr = {
        'tracl': np.zeros(ntrac, dtype=np.int32),
        'tracr': np.zeros(ntrac, dtype=np.int32),
        'fldr': np.zeros(ntrac, dtype=np.int32),
        'tracf': np.zeros(ntrac, dtype=np.int32),
        'ep': np.zeros(ntrac, dtype=np.int32),
        'cdp': np.zeros(ntrac, dtype=np.int32),
        'cdpt': np.zeros(ntrac, dtype=np.int32),
        'trid': np.zeros(ntrac, dtype=np.int16),
        'nvs': np.zeros(ntrac, dtype=np.int16),
        'nhs': np.zeros(ntrac, dtype=np.int16),
        'duse': np.zeros(ntrac, dtype=np.int16),
        'offset': np.zeros(ntrac, dtype=np.int32),
        'gelev': np.zeros(ntrac, dtype=np.int32),
        'selev': np.zeros(ntrac, dtype=np.int32),
        'sdepth': np.zeros(ntrac, dtype=np.int32),
        'gdel': np.zeros(ntrac, dtype=np.int32),
        'sdel': np.zeros(ntrac, dtype=np.int32),
        'swdep': np.zeros(ntrac, dtype=np.int32),
        'gwdep': np.zeros(ntrac, dtype=np.int32),
        'scalel': np.zeros(ntrac, dtype=np.int16),
        'scalco': np.zeros(ntrac, dtype=np.int16),
        'sx': np.zeros(ntrac, dtype=np.int32),
        'sy': np.zeros(ntrac, dtype=np.int32),
        'gx': np.zeros(ntrac, dtype=np.int32),
        'gy': np.zeros(ntrac, dtype=np.int32),
        'counit': np.zeros(ntrac, dtype=np.int16),
        'wevel': np.zeros(ntrac, dtype=np.int16),
        'swevel': np.zeros(ntrac, dtype=np.int16),
        'sut': np.zeros(ntrac, dtype=np.int16),
        'gut': np.zeros(ntrac, dtype=np.int16),
        'sstat': np.zeros(ntrac, dtype=np.int16),
        'gstat': np.zeros(ntrac, dtype=np.int16),
        'tstat': np.zeros(ntrac, dtype=np.int16),
        'laga': np.zeros(ntrac, dtype=np.int16),
        'lagb': np.zeros(ntrac, dtype=np.int16),
        'delrt': np.zeros(ntrac, dtype=np.int16),
        'muts': np.zeros(ntrac, dtype=np.int16),
        'mute': np.zeros(ntrac, dtype=np.int16),
        'ns': np.zeros(ntrac, dtype=np.uint16),
        'dt': np.zeros(ntrac, dtype=np.uint16),
        'gain': np.zeros(ntrac, dtype=np.int16),
        'igc': np.zeros(ntrac, dtype=np.int16),
        'igi': np.zeros(ntrac, dtype=np.int16),
        'corr': np.zeros(ntrac, dtype=np.int16),
        'sfs': np.zeros(ntrac, dtype=np.int16),
        'sfe': np.zeros(ntrac, dtype=np.int16),
        'slen': np.zeros(ntrac, dtype=np.int16),
        'styp': np.zeros(ntrac, dtype=np.int16),
        'stas': np.zeros(ntrac, dtype=np.int16),
        'stae': np.zeros(ntrac, dtype=np.int16),
        'tatyp': np.zeros(ntrac, dtype=np.int16),
        'afilf': np.zeros(ntrac, dtype=np.int16),
        'afils': np.zeros(ntrac, dtype=np.int16),
        'nofilf': np.zeros(ntrac, dtype=np.int16),
        'nofils': np.zeros(ntrac, dtype=np.int16),
        'lcf': np.zeros(ntrac, dtype=np.int16),
        'hcf': np.zeros(ntrac, dtype=np.int16),
        'lcs': np.zeros(ntrac, dtype=np.int16),
        'hcs': np.zeros(ntrac, dtype=np.int16),
        'year': np.zeros(ntrac, dtype=np.int16),
        'day': np.zeros(ntrac, dtype=np.int16),
        'hour': np.zeros(ntrac, dtype=np.int16),
        'minute': np.zeros(ntrac, dtype=np.int16),
        'sec': np.zeros(ntrac, dtype=np.int16),
        'timbas': np.zeros(ntrac, dtype=np.int16),
        'trwf': np.zeros(ntrac, dtype=np.int16),
        'grnors': np.zeros(ntrac, dtype=np.int16),
        'grnofr': np.zeros(ntrac, dtype=np.int16),
        'grnlof': np.zeros(ntrac, dtype=np.int16),
        'gaps': np.zeros(ntrac, dtype=np.int16),
        'otrav': np.zeros(ntrac, dtype=np.int16),
        'd1': np.zeros(ntrac, dtype=np.float32),
        'f1': np.zeros(ntrac, dtype=np.float32),
        'd2': np.zeros(ntrac, dtype=np.float32),
        'f2': np.zeros(ntrac, dtype=np.float32),
        'ungpow': np.zeros(ntrac, dtype=np.float32),
        'unscale': np.zeros(ntrac, dtype=np.float32),
        'ntr': np.zeros(ntrac, dtype=np.int32),
        'mark': np.zeros(ntrac, dtype=np.int16),
        'shortpad': np.zeros(ntrac, dtype=np.int16)
    }

    return hdr
