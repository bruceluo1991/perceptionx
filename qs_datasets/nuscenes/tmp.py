class A:
    def __init__(self, foo: list) -> None:
        self.foo = foo


l = [1, 2, 3]
a = A(l)
l = []
print(a.foo)
