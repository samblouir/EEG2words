import importlib

def run_it(input=None):
    import new_path
    importlib.reload(new_path)
    del new_path
