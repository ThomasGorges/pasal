from concurrent.futures import ThreadPoolExecutor
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import os
from time import sleep
import math
from utility.parameters import generate_new_parameters
from tempfile import TemporaryDirectory
import json
import shutil


class Study:
    def __init__(self, name="") -> None:
        self._name = name
        self._study = None
        self._uri = JournalStorage(JournalFileStorage('../output/studies.log'))
        self._stored_study_names = []
        self._task_worker_dict = dict()
        self._task_tempdir_dict = dict()

    @property
    def best_params(self):
        return self._study.best_params

    def create_or_resume(self):
        self.create(resume_if_exists=True)

    def create(self, resume_if_exists=False):
        self.fetch_all_studies()

        if not resume_if_exists and self._name in self._stored_study_names:
            raise RuntimeError(f"Study with the name {self._name} already exists.")

        self._study = optuna.create_study(
            storage=self._uri, study_name=self._name, load_if_exists=resume_if_exists, direction='minimize'
        )
    
    def export_results(self):
        trial_results = {}
        
        trials = self._study.trials
        num_trials = len(trials)

        for trial_idx in range(num_trials):
            trial = trials[trial_idx]

            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            trial_results[trial_idx] = trial.params
            trial_results[trial_idx]['objective_value'] = trial.value
            trial_results[trial_idx]['user_attrs'] = trial.user_attrs
        
        if not os.path.isdir('study_results'):
            os.mkdir('study_results')

        filepath = f"../output/study_results/{self._name}.json"

        with open(filepath, "w") as f:
            json.dump(trial_results, f, indent=4)

        print(f"Results of study \"{self._name}\" saved to {filepath}")


    def optimize(self, objective_fn, fold_id, search_space: dict, num_trials: int, num_parallel_jobs=1, exclude_list=None):
        num_finished_trials = 0
        num_active_workers = 0
        finished = False

        with ThreadPoolExecutor(max_workers=num_parallel_jobs) as executer:
            while not finished:
                # Queue workers if possible
                while num_active_workers != num_parallel_jobs:
                    if num_finished_trials + num_active_workers == num_trials:
                        break
                    
                    # Generate specific hyperparameters on demand
                    # Parameters are not generated before, because Optuna might change the search space over time
                    trial = self._study.ask()
                    generate_new_parameters(trial, search_space)
                    temp_dir = TemporaryDirectory(prefix='universalscent_')

                    # Create worker
                    future_result = executer.submit(objective_fn, fold_id, trial.params, temp_dir.name, True, exclude_list)

                    # Track worker
                    self._task_worker_dict[trial] = future_result
                    self._task_tempdir_dict[trial] = temp_dir
                    num_active_workers += 1
                
                # Check if workers finished
                finished_trials = []
                for trial, worker in self._task_worker_dict.items():
                    if worker.done():
                        # Worker finished, save results
                        results = worker.result()

                        mse_loss = results['mse_loss']

                        num_active_workers -= 1
                        num_finished_trials += 1

                        finished_trials.append(trial)

                        if math.isnan(mse_loss) or math.isinf(mse_loss) or mse_loss >= 1e6:
                            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
                        else:
                            for k, v in results.items():
                                trial.set_user_attr(k, v)
                            self._study.tell(trial, mse_loss)

                        self._task_tempdir_dict[trial].cleanup()
                
                for trial in finished_trials:
                    del self._task_worker_dict[trial]
                    del self._task_tempdir_dict[trial]
                
                if num_finished_trials == num_trials:
                    finished = True
                
                sleep(1)

    def fetch_all_studies(self):
        all_studies = optuna.study.get_all_study_summaries(self._uri)

        for study in all_studies:
            study_name = study.study_name

            if study_name not in self._stored_study_names:
                self._stored_study_names.append(study_name)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def stored_study_names(self):
        return self._stored_study_names

    @property
    def study(self):
        return self._study