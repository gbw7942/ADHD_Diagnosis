x=10 #global
def outer_function():
    y=20 #local variable
    def inner_function():
        nonlocal y
        global x
        y=30
        x=50

        print(x,y)
    inner_function()

outer_function()