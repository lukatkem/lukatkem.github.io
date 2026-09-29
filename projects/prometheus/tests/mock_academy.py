class MockAcademy:
    """Records every call; simulates the Academy's lifecycle."""
    def __init__(self, refuse: bool = False):
        self.calls = []
        self.refuse = refuse
    def start_distill(self, domain, stories, out):
        self.calls.append(("distill", domain, stories, out))
        if self.refuse:
            return {"started": False, "reason": "job already running"}
        return {"started": True}
    def start_train(self, steps, size):
        self.calls.append(("train", steps, size))
        return {"started": True}
    def status(self):
        return {"active": False}
    def wait_for_completion(self, **kwargs):
        return {"active": False}
