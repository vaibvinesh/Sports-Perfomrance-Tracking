import os
import threading


class CombinedThread(threading.Thread):

    def __init__(self, camNo):
        threading.Thread.__init__(self)
        self.camNo = camNo

    def run(self):
        run_cam(self.camNo)


def run_cam(camNo):
    if camNo == 0:
        os.system("python Feed_Camera1.py")
    elif camNo == 1:
        os.system("python Feed_Camera2.py")


thread1 = CombinedThread(0)
thread2 = CombinedThread(1)
thread1.start()
thread2.start()
