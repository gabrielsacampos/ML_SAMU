from imblearn.under_sampling import RandomUnderSampler

class Resampler:
    def __init__(self, X_train_default, y_train_default):
        self.X_train_default = X_train_default
        self.y_train_default = y_train_default
        self.X_train_resampled = None
        self.y_train_resampled = None

    def execute_undersampler(self):
        rus = RandomUnderSampler(random_state=42)
        self.X_train_resampled, self.y_train_resampled = rus.fit_resample(self.X_train_default, self.y_train_default)
        return self.X_train_resampled, self.y_train_resampled
        