import struct
from io import SEEK_END
from types import SimpleNamespace
import numpy as np

# https://docs.python.org/3/library/struct.html#format-strings

TRACE_HEADER_SIZE = 240  # in bytes
HEADER_FORMAT_STRING = '<7i4h8i2h4i13h2H31h6f'
HEADER_KEYS = ('tracl', 'tracr', 'fldr', 'tracf', 'ep', 'cdp', 'cdpt', 'trid',
               'nvs', 'nhs', 'duse', 'offset', 'gelev', 'selev', 'sdepth', 'gdel',
               'sdel', 'swdep', 'gwdep', 'scalel', 'scalco', 'sx', 'sy', 'gx',
               'gy', 'counit', 'wevel', 'swevel', 'sut', 'gut', 'sstat', 'gstat',
               'tstat', 'laga', 'lagb', 'delrt', 'muts', 'mute', 'ns', 'dt',
               'gain', 'igc', 'igi', 'corr', 'sfs', 'sfe', 'slen', 'styp', 'stas',
               'stae', 'tatyp', 'afilf', 'afils', 'nofilf', 'nofils', 'lcf',
               'hcf', 'lcs', 'hcs', 'year', 'day', 'hour', 'minute', 'sec',
               'timbas', 'trwf', 'grnors', 'grnofr', 'grnlof', 'gaps', 'otrav',
               'd1', 'f1', 'd2', 'f2', 'ungpow', 'unscale')


# read a binary filepath in .su format
def readsu(file_path):
    with open(file_path, 'rb') as file:

        # Read number of samples (how many values a trace has)
        file.seek(114)  # change stream position to byte 114
        bytes_to_unpack = file.read(2)  # read 2 bytes
        trace_samples_amount = struct.unpack('<H', bytes_to_unpack)[0]

        file_size = _get_file_size(file)

        # Compute number of traces
        trace_data_size = trace_samples_amount * 4
        traces_amount = file_size // (trace_data_size + TRACE_HEADER_SIZE)

        print(f'Number of samples: {trace_samples_amount}')
        print(f'Number of traces: {traces_amount}')

        traces_data = np.zeros(shape=(trace_samples_amount, traces_amount), dtype=np.float32)
        headers = _new_empty_header(traces_amount)

        data_format_string = f'{trace_samples_amount}f'
        file.seek(0)
        for index in range(traces_amount):
            # Read trace header
            header_bytes = file.read(TRACE_HEADER_SIZE)
            header_values = struct.unpack_from(HEADER_FORMAT_STRING, header_bytes)

            for key, value in zip(HEADER_KEYS, header_values):
                headers[key][index] = value

            # Read data trace
            data_bytes = file.read(trace_data_size)
            traces_data[:, index] = np.array(struct.unpack(data_format_string, data_bytes),
                                             dtype=np.float32)

        return traces_data, SimpleNamespace(**headers)


def writesu(file_path, traces_data, hdr):
    with open(file_path, 'wb') as f:
        n_samples, n_traces = traces_data.shape
        trace_data_size = n_samples * 4
        data_format = f'{n_samples}f'

        print(f'Number of samples: {n_samples}')
        print(f'Number of traces: {n_traces}')

        def get_header_position(index):
            return (TRACE_HEADER_SIZE + trace_data_size) * index

        def get_data_position(index):
            return (TRACE_HEADER_SIZE + trace_data_size) * index + TRACE_HEADER_SIZE

        for i in range(n_traces):
            # Write trace header
            f.seek(get_header_position(i))
            header_bytes = struct.pack(
                HEADER_FORMAT_STRING, hdr.tracl[i], hdr.tracr[i], hdr.fldr[i],
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
            f.write(header_bytes)
            # Write trace data
            f.seek(get_data_position(i))
            data_bytes = struct.pack(data_format, *traces_data[:, i])
            f.write(data_bytes)


def _get_file_size(file):
    file.seek(0, SEEK_END)
    return file.tell()


def _new_empty_header(traces_amount):
    header = {
        'tracl': np.zeros(traces_amount, dtype=np.int32),
        'tracr': np.zeros(traces_amount, dtype=np.int32),
        'fldr': np.zeros(traces_amount, dtype=np.int32),
        'tracf': np.zeros(traces_amount, dtype=np.int32),
        'ep': np.zeros(traces_amount, dtype=np.int32),
        'cdp': np.zeros(traces_amount, dtype=np.int32),
        'cdpt': np.zeros(traces_amount, dtype=np.int32),
        'trid': np.zeros(traces_amount, dtype=np.int16),
        'nvs': np.zeros(traces_amount, dtype=np.int16),
        'nhs': np.zeros(traces_amount, dtype=np.int16),
        'duse': np.zeros(traces_amount, dtype=np.int16),
        'offset': np.zeros(traces_amount, dtype=np.int32),
        'gelev': np.zeros(traces_amount, dtype=np.int32),
        'selev': np.zeros(traces_amount, dtype=np.int32),
        'sdepth': np.zeros(traces_amount, dtype=np.int32),
        'gdel': np.zeros(traces_amount, dtype=np.int32),
        'sdel': np.zeros(traces_amount, dtype=np.int32),
        'swdep': np.zeros(traces_amount, dtype=np.int32),
        'gwdep': np.zeros(traces_amount, dtype=np.int32),
        'scalel': np.zeros(traces_amount, dtype=np.int16),
        'scalco': np.zeros(traces_amount, dtype=np.int16),
        'sx': np.zeros(traces_amount, dtype=np.int32),
        'sy': np.zeros(traces_amount, dtype=np.int32),
        'gx': np.zeros(traces_amount, dtype=np.int32),
        'gy': np.zeros(traces_amount, dtype=np.int32),
        'counit': np.zeros(traces_amount, dtype=np.int16),
        'wevel': np.zeros(traces_amount, dtype=np.int16),
        'swevel': np.zeros(traces_amount, dtype=np.int16),
        'sut': np.zeros(traces_amount, dtype=np.int16),
        'gut': np.zeros(traces_amount, dtype=np.int16),
        'sstat': np.zeros(traces_amount, dtype=np.int16),
        'gstat': np.zeros(traces_amount, dtype=np.int16),
        'tstat': np.zeros(traces_amount, dtype=np.int16),
        'laga': np.zeros(traces_amount, dtype=np.int16),
        'lagb': np.zeros(traces_amount, dtype=np.int16),
        'delrt': np.zeros(traces_amount, dtype=np.int16),
        'muts': np.zeros(traces_amount, dtype=np.int16),
        'mute': np.zeros(traces_amount, dtype=np.int16),
        'ns': np.zeros(traces_amount, dtype=np.uint16),
        'dt': np.zeros(traces_amount, dtype=np.uint16),
        'gain': np.zeros(traces_amount, dtype=np.int16),
        'igc': np.zeros(traces_amount, dtype=np.int16),
        'igi': np.zeros(traces_amount, dtype=np.int16),
        'corr': np.zeros(traces_amount, dtype=np.int16),
        'sfs': np.zeros(traces_amount, dtype=np.int16),
        'sfe': np.zeros(traces_amount, dtype=np.int16),
        'slen': np.zeros(traces_amount, dtype=np.int16),
        'styp': np.zeros(traces_amount, dtype=np.int16),
        'stas': np.zeros(traces_amount, dtype=np.int16),
        'stae': np.zeros(traces_amount, dtype=np.int16),
        'tatyp': np.zeros(traces_amount, dtype=np.int16),
        'afilf': np.zeros(traces_amount, dtype=np.int16),
        'afils': np.zeros(traces_amount, dtype=np.int16),
        'nofilf': np.zeros(traces_amount, dtype=np.int16),
        'nofils': np.zeros(traces_amount, dtype=np.int16),
        'lcf': np.zeros(traces_amount, dtype=np.int16),
        'hcf': np.zeros(traces_amount, dtype=np.int16),
        'lcs': np.zeros(traces_amount, dtype=np.int16),
        'hcs': np.zeros(traces_amount, dtype=np.int16),
        'year': np.zeros(traces_amount, dtype=np.int16),
        'day': np.zeros(traces_amount, dtype=np.int16),
        'hour': np.zeros(traces_amount, dtype=np.int16),
        'minute': np.zeros(traces_amount, dtype=np.int16),
        'sec': np.zeros(traces_amount, dtype=np.int16),
        'timbas': np.zeros(traces_amount, dtype=np.int16),
        'trwf': np.zeros(traces_amount, dtype=np.int16),
        'grnors': np.zeros(traces_amount, dtype=np.int16),
        'grnofr': np.zeros(traces_amount, dtype=np.int16),
        'grnlof': np.zeros(traces_amount, dtype=np.int16),
        'gaps': np.zeros(traces_amount, dtype=np.int16),
        'otrav': np.zeros(traces_amount, dtype=np.int16),
        'd1': np.zeros(traces_amount, dtype=np.float32),
        'f1': np.zeros(traces_amount, dtype=np.float32),
        'd2': np.zeros(traces_amount, dtype=np.float32),
        'f2': np.zeros(traces_amount, dtype=np.float32),
        'ungpow': np.zeros(traces_amount, dtype=np.float32),
        'unscale': np.zeros(traces_amount, dtype=np.float32),
        'ntr': np.zeros(traces_amount, dtype=np.int32),
        'mark': np.zeros(traces_amount, dtype=np.int16),
        'shortpad': np.zeros(traces_amount, dtype=np.int16)
    }

    return header
