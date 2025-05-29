from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

class SaveAndLogCallback(BaseCallback):
    def __init__(self, save_freqs, save_path, log_file=None, verbose=1):
        """
        Callback to save the model and record logs.

        :param save_freqs: List of save frequencies (in steps).
        :param save_path: Path to the folder where models will be saved.
        :param log_file: Path to the log file (optional).
        :param verbose: Verbosity level (0 = silent, 1 = display logs).
        """
        super().__init__(verbose)
        self.save_freqs = set(save_freqs)
        self.save_path = save_path
        self.log_file = log_file
        self.last_checkpoint_time = None  # To calculate time between checkpoints

    def _on_step(self) -> bool:
        """
        Method called at each training step.
        """
        # Save the model at specified steps
        if self.num_timesteps in self.save_freqs:
            self._save_model(self.num_timesteps)
        return True

    def _save_model(self, steps):
        """
        Saves the model and writes logs.

        :param steps: Number of training steps.
        """
        path = f"{self.save_path}/model_{steps}_steps"
        self.model.save(path)

        # Calculate time elapsed since last checkpoint
        now = datetime.now()
        time_since_last = 0
        if self.last_checkpoint_time is not None:
            time_since_last = (now - self.last_checkpoint_time).total_seconds()
        self.last_checkpoint_time = now

        # Display in console if verbose > 0
        if self.verbose > 0:
            print(f"Model saved at {path}")
            print(f"Time since last checkpoint: {time_since_last:.2f} seconds")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"Model saved at {path}, Time since last checkpoint: {time_since_last:.2f} seconds\n")