from threading import Timer#,Thread,Event
import time

class perpetualTimer():

    def __init__(self,t,hFunction,name):
        self.t=t
        self.hFunction = hFunction
        self.name = name
        self.thread = Timer(self.t,self.handle_function)
        self.time_last = 0

    def handle_function(self):
        self.hFunction()
        delta=(time.monotonic() - self.time_last) - self.t
        #if delta > 0.1:
        #    print(f"hF[{self.name}]={delta}")
        if delta > self.t:
            delta = self.t
        self.thread = Timer(self.t,self.handle_function)
        self.thread.start()
        self.time_last = time.monotonic()


    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

    def reset(self):
        self.cancel()
        self.start()
