from types import SimpleNamespace
import numpy as np
import numpy.typing as npt
from .UtilsModel import Utils


class Header(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class SuData:
    """Store the SU seismic data file.

    Attributes:
        traces (ndarray): Data traces from the entire file.
        headers (Header): Trace headers from the entire file.
        gather_count (int): Number of gathers. None if gather_keyword was not
          specified at creation.
    """

    def __init__(self, traces: npt.NDArray[np.float_], headers: Header, gather_keyword=None):
        """Initialize the SuData

        Args:
            gather_keyword: The header keyword that comprises the gathers.

        """
        self.traces = traces
        self.headers = headers
        self.num_traces = traces.shape[1]
        self._gather_separation_indices = None
        self.num_gathers = None
        self.gather_keyword = gather_keyword
        if gather_keyword != None:
            self._gather_separation_indices = self._compute_gather_separation_indices(
                gather_keyword
            )
            self.num_gathers = len(self._gather_separation_indices) - 1

    @staticmethod
    def new_empty_gathers(
        num_samples_per_trace: int,
        gather_keyword: str,
        gather_values: list,
        num_traces_per_gather: int,
    ):
        num_traces = num_traces_per_gather * len(gather_values)
        traces = np.zeros(shape=(num_samples_per_trace, num_traces), dtype=float)
        headers = Utils.new_empty_header(num_traces)
        for i, value in enumerate(gather_values):
            itrace_start = i * num_traces_per_gather
            itrace_end = itrace_start + num_traces_per_gather
            headers[gather_keyword][itrace_start:itrace_end] = value

        return SuData(traces, Header(**headers), gather_keyword)

    @property
    def num_samples(self) -> int:
        """Number of samples per data trace."""
        return self.headers.ns[0]

    def _compute_gather_separation_indices(self, keyword):
        # Header key word | SU keywords | SEGY trace header fields
        separation_indices = [0]
        separation_key = self.headers[keyword]
        for trace_index in range(1, self.num_traces):
            if separation_key[trace_index] != separation_key[trace_index - 1]:
                separation_indices.append(trace_index)
        separation_indices.append(self.num_traces)
        return separation_indices

    def traces_from_gather(self, gather_index: int):
        """Get all the data traces from the index-specified gather.

        In order to work correctly, this function needs two conditions met:
        - gather_keyword is set to a valid keyword when creating the object;
        - The traces in the file are already sorted by the specified keyword.

        Args:
            gather_index (int): The index of the gather. Check the num_gathers
              property to find out how many gathers are there.

        Returns:
            All traces from the specified gather.

        """
        start_index = self._gather_separation_indices[gather_index]
        stop_index = self._gather_separation_indices[gather_index + 1]

        return self.traces[:, start_index:stop_index]

    def _gather_value_to_index(self, gather_value: int):
        for gather_index in range(self.num_gathers):
            if self.headers_from_gather(gather_index, self.gather_keyword)[0] == gather_value:
                return gather_index

    def traces_from_gather_value(self, gather_value: int):
        return self.traces_from_gather(self._gather_value_to_index(gather_value))

    def headers_from_gather_value(self, gather_value: int, keyword: str):
        return self.headers_from_gather(self._gather_value_to_index(gather_value), keyword)

    def headers_from_gather(self, gather_index: int, keyword: str):
        """Get all the trace headers from the index-specified gather.

        Args:
            gather_index (int): The index of the gather.
            keyword (str): The header keyword to be obtained.

        Returns:
            All headers from the specified gather.

        """
        start_index = self._gather_separation_indices[gather_index]
        stop_index = self._gather_separation_indices[gather_index + 1]

        return self.headers[keyword][start_index:stop_index]
