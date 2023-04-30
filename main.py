# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch

def torch_hi():
    # Use a breakpoint in the code line below to debug your script.
    if torch.cuda.is_available():
        print("hello cuda")
    else:
        print("hello cpu")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch_hi()
    for _ in range(10):
        print(_)
