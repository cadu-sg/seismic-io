from types import SimpleNamespace


class SuData:
    def __init__(self, traces, headers):
        self.traces = traces
        self.headers = headers
        self.traces_amount = traces.shape[1]

    def _get_separation_indices(self, key):
        separation_indices = [0]
        separation_key = self.headers[key]
        for trace_index in range(1, self.traces_amount):
            if separation_key[trace_index] != separation_key[trace_index - 1]:
                separation_indices.append(trace_index)
        separation_indices.append(self.traces_amount)
        return separation_indices

    def get_shot_gather(self, shot_index):
        """Get the common shot gather at the specified index.
         
        the specified 
        Obtém um common shot gather (conjunto de traços que pertencem a um
          mesmo shot).

        Para essa função funcionar corretamente, os traços devem ter o keyword
        'ep' ordenado em ordem ascendente. Isso porque o fatiamento de shots é
        baseado nesse header.

        Args:
            shot_index: Índice do shot gather a ser obtido.

        Returns:
            Shot gather selecionado.
        """
        separation_indices = self._get_separation_indices("ep")

        start_index = separation_indices[shot_index]
        stop_index = separation_indices[shot_index + 1]

        return self.traces[:, start_index:stop_index]


class Header(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
