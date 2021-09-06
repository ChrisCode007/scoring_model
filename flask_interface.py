class FlaskInterface:
    def __init__(self, ls, lr):
        self.ls = ls
        self.lr = lr
        
    def get_result_ls(self, x_test):
        return self.ls.predict_proba(x_test)
    
    def get_result_lr(self, x_test):
        return self.lr.predict_proba(x_test)