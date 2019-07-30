class DataStructure:
    def __init__(self, dom, non, overlap, ehistory, history, middle, motion):
        self.dom = dom
        self.non = non
        self.edge_history = ehistory
        self.history = history
        self.middle = middle
        self.motion = motion
        self.overlap = overlap

    def get_dom(self):
        return self.dom
    def get_non(self):
        return self.non
    def get_overlap(self):
        return self.overlap
    def get_ehistory(self):
        return self.edge_history
    def get_history(self):
        return self.history
    def get_middle(self):
        return self.middle
    def get_motion(self):
        return self.motion