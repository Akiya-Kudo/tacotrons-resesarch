class EarlyStopping:
    def __init__(self, patience=10, verbose=1):
        '''
        Parameters:
            patience(int): 監視するエポック数(デフォルトは10)
            verbose(int): 早期終了の出力フラグ。 出力(1),出力しない(0)        
        '''
        self.epochs = 0
        self.loss_memory = float("inf")
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        '''
        Parameters:
            current_loss(float): 1エポック終了後の検証データの損失
        Return:
            Boolean:監視回数の上限までに前エポックの損失を超えたらTrue
        '''
        if current_loss > self.loss_memory:
            self.epochs += 1
            if self.epochs > self.patience:
                if self.verbose:
                    print(f"Early Stopping! loss value exceed consecutively {str(self.patience + 1)} times")
                return True
        else:
            self.epochs = 0
            self.loss_memory = current_loss
            return False
    def get_current_loss(self):
        return self.loss_memory
    def get_epochs(self):
        return self.epochs


