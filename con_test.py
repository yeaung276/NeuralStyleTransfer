# import time
# from multiprocessing import Process, Pipe
from background.pipeLine import Channel
from background.backgroundTask import BackgroundTask

def gg():
    Channel.get_parent_end().send(1)

if __name__ == '__main__':
    back = BackgroundTask(Channel.get_child_end())
    back.start()
    gg()
    gg()

