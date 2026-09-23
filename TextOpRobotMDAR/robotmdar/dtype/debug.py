
def pdb_decorator(func):
    def wrapped(*args,**kwargs):
        try:
            func(*args,**kwargs)
        except Exception as e:
            print(">>>BUG:",e)
            import pdb;pdb.post_mortem()
    return wrapped


def notused(func):

    def wrapper(*args, **kwargs):
        print(f"Not used: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper
