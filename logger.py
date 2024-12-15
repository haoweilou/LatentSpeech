import pandas as pd 

class Log:
    def __init__(self,**kwargs):
        self.metrics = {key: value for key, value in kwargs.items()}
        self.metrics["total_loss"] = 0
        self.log_df = pd.DataFrame(columns=self.metrics.keys())
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key] = value
            else:
                raise KeyError(f"Metric '{key}' does not exist in Log. Add it first by initializing or updating.")
        
        self.metrics['total_loss'] = sum(
            value for key, value in self.metrics.items() if key != 'total_loss'
        )
        self.log_df = pd.concat([self.log_df, pd.DataFrame([self.metrics])], ignore_index=True)

    def get_log(self):
        return self.log_df

    def __repr__(self):
        return f"Step {len(self.log_df)} "+ " ".join([f"{key}: {value:.03f}" for key, value in self.metrics.items()])
    
    def save(self, path):
        self.log_df.to_csv(path, index=True)
        
    def load(self, path):
        try:
            self.log_df = pd.read_csv(path,index_col=0)
            if not self.log_df.empty:
                self.metrics = self.log_df.iloc[-1].to_dict()
        except Exception as e:
            print(f"Error loading log from {path}: {e}")