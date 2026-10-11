class CallableExample:
    def __init__(self):
        print("Instance created (inside __init__)")

    def __call__(self, *args, **kwargs):
        print(f"Instance called like a function (inside __call__) with args: {args}")

obj = CallableExample() # Output: Instance created (inside __init__)
obj(1, 2, kwarg="value") # Output: Instance called like a function (inside __call__) with args: (1, 2)
