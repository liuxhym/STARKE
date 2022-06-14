import json


def write_file(L, trace_path):
    final_results_str = json.dumps(L)
    with open(trace_path, "a") as f:
        f.write(final_results_str + '\n')


def read_file(trace_path):
    f = open(trace_path)
    L = json.load(f)
    return L
