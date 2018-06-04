from MainAppView import MainAppView

class MainAppController(object):
    def init_view(self,root):
        self.view = MainAppView(master=root)
        self.view.start_gui()
        

