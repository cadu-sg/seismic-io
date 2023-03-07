def _ep_separation_indices(headers, traces_amount):
    separation_indices = []
    ep = headers.ep
    for trace_i in range(1, traces_amount):
        if ep[trace_i] != ep[trace_i - 1]:
            separation_indices.append(trace_i)
    return separation_indices


def get_shot_gather(shot_index, traces, headers):
    """Obtém um shot gather (conjunto de traços que pertencem a um mesmo shot),
    dados os traços e os headers de um arquivo SU.

    Traços pertencem mesmo shot se eles possuírem o mesmo número ep em seu header.

    Args:
        traces (ndarray): Sismograma dos traços.
        headers (SimpleNameSpace): Headers dos traços.
    
    Returns:
        ndarray: Shot gather selecionado
    """
    traces_amount = traces.shape[1]
    slicing_indices = (
        [0] + _ep_separation_indices(headers, traces_amount) + [traces_amount]
    )

    start_index = slicing_indices[shot_index]
    stop_index = slicing_indices[shot_index + 1]

    return traces[:, start_index:stop_index]
