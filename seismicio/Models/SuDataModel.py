from types import SimpleNamespace
import numpy as np
import numpy.typing as npt
from .UtilsModel import Utils
import pandas as pd


class Header(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class SuFile:
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

        self.gather_indices = None

        if gather_keyword is None:
            print("NO GAHTER KEYWORD AADASPODUWAFNU")
            return

        # Compute gather index database

        separation_indices = [0]
        separation_key = self.headers[gather_keyword]

        gather_values = [separation_key[0]]

        for trace_index in range(1, self.num_traces):
            if separation_key[trace_index] != separation_key[trace_index - 1]:
                gather_values.append(separation_key[trace_index])
                separation_indices.append(trace_index)
        separation_indices.append(self.num_traces)

        print(f"gather values len {len(gather_values)}")

        self.gather_indices = pd.DataFrame(
            {"start": separation_indices[:-1], "stop": separation_indices[1:]},
            index=gather_values,
        )

        self._gather_separation_indices = separation_indices
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

        return SuFile(traces, Header(**headers), gather_keyword)

    @property
    def num_samples(self) -> int:
        """Number of samples per data trace."""
        return self.headers.ns[0]

    def traces_from_gather_index(self, gather_index: int):
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
        start_index = self.gather_indices["start"].iat[gather_index]
        stop_index = self.gather_indices["stop"].iat[gather_index]
        return self.traces[:, start_index:stop_index]

    def traces_from_gather_value(self, gather_value: int):
        start_index = self.gather_indices["start"].at[gather_value]
        stop_index = self.gather_indices["stop"].at[gather_value]
        return self.traces[:, start_index:stop_index]

    def headers_from_gather_index(self, gather_index: int, keyword: str):
        """Get all the trace headers from the index-specified gather.

        Args:
            gather_index (int): The index of the gather.
            keyword (str): The header keyword to be obtained.

        Returns:
            All headers from the specified gather.

        """
        start_index = self.gather_indices["start"].iat[gather_index]
        stop_index = self.gather_indices["stop"].iat[gather_index]

        return self.headers[keyword][start_index:stop_index]

    def headers_from_gather_value(self, gather_value: int, keyword: str):
        """Get all `keyword` headers from the value-specified gather.

        Args:
          gather_value (int): Value in the keyword that specifies the gather.
          keyword (str): Which header to get, specified by its keyword.

        Returns:
          Within the specified gather, an array containing all values of the specified gather.
        """
        start_index = self.gather_indices["start"].at[gather_value]
        stop_index = self.gather_indices["stop"].at[gather_value]

        return self.headers[keyword][start_index:stop_index]

    def gather_value_to_index(self, gather_value: int):
        """Find out the integer position index of the gather with the given value"""
        gather_values: pd.Index = self.gather_indices.index
        gather_index = gather_values.get_loc(gather_value)
        return gather_index

    def gather_index_to_value(self, gather_index: int):
        """Find out the value of the gather with the given integer position index"""
        gather_values: pd.Index = self.gather_indices.index
        gather_value = gather_values[gather_index]
        return gather_value
