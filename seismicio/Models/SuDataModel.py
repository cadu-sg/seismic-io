class SuData:
    def __init__(self, traces, headers):
        self.traces = traces
        self.headers = headers
        self.traces_amount = traces.shape[1]

    def _shot_separation_indices(self):
        """
        _shot_separation_indices comes from ep (energy point),
        a header on seismic data usefull to separate shots

        Returns:
            list(int): separation indices for each shot gather
        """
        separation_indices = []
        shot_group_index = self.headers.ep
        for trace_index in range(1, self.traces_amount):
            if shot_group_index[trace_index] != shot_group_index[trace_index - 1]:
                separation_indices.append(trace_index)
        return separation_indices


    def get_shot_gather(self, shot_index):
        """
        Obtém um shot gather (conjunto de traços que pertencem a um mesmo shot),
        dados os traços e os headers de um arquivo SU.

        Traços pertencem mesmo shot se eles possuírem o mesmo número ep em seu header.

        Args:
            shot_index (int): Indice do shot
            traces (ndarray): Sismograma dos traços.
            headers (SimpleNameSpace): Headers dos traços.
        
        Returns:
            ndarray: Shot gather selecionado
        """
        slicing_indices = (
            [0] + self._shot_separation_indices() + [self.traces_amount]
        )

        start_index = slicing_indices[shot_index]
        stop_index = slicing_indices[shot_index + 1]

        return self.traces[:, start_index:stop_index]
