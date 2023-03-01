import os
import numpy as np
import torch
from transrate import transrate_eig, transrate_eig_proj


def build_experiment_folder(experiment_name):
    experiment_path = os.path.abspath(experiment_name)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    logs_filepath = experiment_path
    return logs_filepath


class ExperimentBuilder(object):
    def __init__(self, args, data, model, device):
        self.args, self.device = args, device

        self.model = model
        self.logs_filepath = build_experiment_folder(experiment_name=self.args.experiment_name)

        self.total_losses = dict()
        self.state = dict()

        self.data = data(args.target, args=args)


    def estimation_iteration(self, source_encdoer):
        source_encdoer = source_encdoer.to(self.device)

        train_loader = self.data.loader_tr
        device = self.device

        source_encdoer.eval()
        torch.manual_seed(0)

        z = []
        y = []
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device=device)
                feat = source_encdoer.extract_feature(x_batch)
                if not self.args.use_proj_Z:
                    feat = torch.nn.functional.normalize(feat)
                z.extend(feat.cpu().numpy())
                y.extend(y_batch.cpu().numpy())
        z = np.stack(z).astype(np.float64) # size: n x d
        y = np.array(y)

        n, d = z.shape

        if self.args.use_proj_Z:
            transrate_eig_proj(np.copy(z), np.copy(y), self.args.experiment_name)
        else:
            z_centralized = z - np.mean(z, axis=0, keepdims=True).repeat(n, axis=0)
            transrate_eig(np.copy(z_centralized), np.copy(y), self.args.experiment_name)

    def run_experiment(self):
        self.estimation_iteration(source_encdoer=self.model.get_model_w_encoder())

