from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

class SaveAndLogCallback(BaseCallback):
    def __init__(self, save_freqs, save_path, log_file=None, verbose=1):
        """
        Callback pour sauvegarder le modèle et enregistrer les logs.

        :param save_freqs: Liste des fréquences de sauvegarde (en steps).
        :param save_path: Chemin du dossier où sauvegarder les modèles.
        :param log_file: Chemin du fichier de log (facultatif).
        :param verbose: Niveau de verbosité (0 = silencieux, 1 = affichage des logs).
        """
        super().__init__(verbose)
        self.save_freqs = set(save_freqs)
        self.save_path = save_path
        self.log_file = log_file
        self.last_checkpoint_time = None  # Pour calculer le temps entre checkpoints

    def _on_step(self) -> bool:
        """
        Méthode appelée à chaque étape d'entraînement.
        """
        # Sauvegarder le modèle aux steps spécifiés
        if self.num_timesteps in self.save_freqs:
            self._save_model(self.num_timesteps)
        return True

    def _save_model(self, steps):
        """
        Sauvegarde le modèle et écrit les logs.

        :param steps: Nombre d'étapes d'entraînement.
        """
        path = f"{self.save_path}/model_{steps}_steps"
        self.model.save(path)

        # Calculer le temps écoulé depuis le dernier checkpoint
        now = datetime.now()
        time_since_last = 0
        if self.last_checkpoint_time is not None:
            time_since_last = (now - self.last_checkpoint_time).total_seconds()
        self.last_checkpoint_time = now

        # Afficher dans la console si verbose > 0
        if self.verbose > 0:
            print(f"Model saved at {path}")
            print(f"Time since last checkpoint: {time_since_last:.2f} seconds")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"Model saved at {path}, Time since last checkpoint: {time_since_last:.2f} seconds\n")