import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# ========================
# Wifi Parameter Setting.
# ========================
params_wifi = AttrDict(
    task_id=0,
    log_dir='./log/1128/all-scos2',
    model_dir='./model/1128/all-scos2',
    data_dir=['/data/Widar3.0/20181128-all/'],
    out_dir='./dataset/widar/output',
    cond_dir=['/data/Widar3.0/cond1128-all/'],
    fid_pred_dir = './dataset/widar/img_matric/pred',
    fid_data_dir = './dataset/widar/img_matric/data',
    # Training params
    max_iter=None, # Unlimited number of iterations.
    batch_size=64,
    learning_rate=1e-3,
    max_grad_norm=None,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=64,
    input_dim=90,
    extra_dim=[90],
    cond_dim=1,
    # Model params
    embed_dim=256,
    spatial_hidden_dim=128,
    tf_hidden_dim=128,
    hidden_dim=128,
    num_heads=8,
    num_spatial_block=8,
    num_tf_block=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    
    # linear schedule
    # noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),    
    # simple cosine schedule
    noise_schedule=(1e-3 * (2 + 2*np.cos(np.linspace(0, np.pi, 100)))).tolist()
    # power schedule
    # noise_schedule=(2.5e-3 * (1-np.linspace(0, 1, 100)**2)).tolist(),
    # tanh schedule
    # noise_schedule=(2e-3 * (1 + np.tanh(-np.linspace(0, 4, 100) + 2))).tolist(),
    # standard cosine schedule
    # noise_schedule=(3e-2 * np.clip(1 - (np.cos(0.5 * np.pi * ((np.arange(1, 100+1)/(100))+0.008)/(1+0.008)) ** 2) / (np.cos(0.5 * np.pi * ((np.arange(0, 100)/(100))+0.008)/(1+0.008)) ** 2), 0, 0.999)).tolist()
)

all_params = [params_wifi]