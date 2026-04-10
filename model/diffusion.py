import copy
import os
import pickle
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from vis import skeleton_render

from .utils import extract, make_beta_schedule

# Per-joint mass fractions derived from SEGMENT_JOINT_MAPPING (Winter, 2009).
# Matches eval_fcs.py and fcs_predictor.py: each segment's mass is split
# evenly across its constituent joints, then accumulated per joint.
# Trunk (49.7%) across 5 joints (0,3,6,9,12); collars (13,14) get 0.
# Total sums to 1.0.
_SMPL_JOINT_MASSES = [
    0.09940,  # 0:  Pelvis      (trunk)
    0.05000,  # 1:  L_Hip       (thigh_l)
    0.05000,  # 2:  R_Hip       (thigh_r)
    0.09940,  # 3:  Spine1      (trunk)
    0.07325,  # 4:  L_Knee      (thigh_l + shank_l)
    0.07325,  # 5:  R_Knee      (thigh_r + shank_r)
    0.09940,  # 6:  Spine2      (trunk)
    0.03050,  # 7:  L_Ankle     (shank_l + foot_l)
    0.03050,  # 8:  R_Ankle     (shank_r + foot_r)
    0.09940,  # 9:  Spine3      (trunk)
    0.00725,  # 10: L_Toe       (foot_l)
    0.00725,  # 11: R_Toe       (foot_r)
    0.09940,  # 12: Neck        (trunk)
    0.00000,  # 13: L_Collar    (not in segment mapping)
    0.00000,  # 14: R_Collar    (not in segment mapping)
    0.08100,  # 15: Head
    0.01400,  # 16: L_Shoulder  (upper_arm_l)
    0.01400,  # 17: R_Shoulder  (upper_arm_r)
    0.02200,  # 18: L_Elbow     (upper_arm_l + forearm_l)
    0.02200,  # 19: R_Elbow     (upper_arm_r + forearm_r)
    0.00800,  # 20: L_Wrist     (forearm_l)
    0.00800,  # 21: R_Wrist     (forearm_r)
    0.00600,  # 22: L_Palm      (hand_l)
    0.00600,  # 23: R_Palm      (hand_r)
]


def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        n_timestep=1000,
        schedule="linear",
        loss_type="l1",
        clip_denoised=True,
        predict_epsilon=True,
        guidance_weight=3,
        use_p2=False,
        cond_drop_prob=0.2,
        com_loss_weight=0.0,
        bilateral_loss_weight=0.0,
        foot_height_loss_weight=0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

        self.cond_drop_prob = cond_drop_prob
        self.com_loss_weight = com_loss_weight
        self.bilateral_loss_weight = bilateral_loss_weight
        self.foot_height_loss_weight = foot_height_loss_weight
        self.register_buffer(
            "_joint_masses",
            torch.tensor(_SMPL_JOINT_MASSES, dtype=torch.float32),
        )

        # make a SMPL instance for FK module
        self.smpl = smpl

        # FCS predictor for physics-aware training (set externally)
        self.fcs_predictor = None
        self.fcs_loss_weight = 0.0

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x
        
    def attach_normalizer(self, normalizer):
        """
        Cache the dataset normalizer's MinMaxScaler params as buffers so we can
        differentiably unnormalize during inference-time physics guidance. The
        FCS predictor was trained on unnormalized joint positions, so guidance
        must feed it unnormalized x_start to get calibrated outputs.

        The dataset's MinMaxScaler.inverse_transform is in-place and would
        break autograd, so we apply the affine inverse manually instead.
        """
        scaler = normalizer.scaler
        device = self.betas.device
        # MinMaxScaler.transform: x_scaled = x * scale_ + min_
        # MinMaxScaler.inverse_transform: x = (x_scaled - min_) / scale_
        scale = scaler.scale_.detach().clone().to(device).float()
        min_ = scaler.min_.detach().clone().to(device).float()
        self.register_buffer("_unnorm_scale", scale, persistent=False)
        self.register_buffer("_unnorm_min", min_, persistent=False)

    def _apply_physics_guidance(self, x_start, x_next, guidance_scale):
        """
        Inference-time physics guidance via FCS predictor gradient.

        Computes ∇_{x_start} FCS_predictor(FK(unnormalize(x_start))) and
        subtracts guidance_scale * grad from x_next. Operates on a detached
        copy of x_start so no gradient flows back through the diffusion model.
        """
        x_in = x_start.detach().requires_grad_(True)
        with torch.enable_grad():
            # Unnormalize: predictor was trained on unnormalized joint positions
            if hasattr(self, "_unnorm_scale"):
                x_clip = torch.clamp(x_in, -1.0, 1.0)
                x_unnorm = (x_clip - self._unnorm_min) / self._unnorm_scale
            else:
                x_unnorm = x_in
            b_, s_, _ = x_unnorm.shape
            # 151D layout: [0:4] contact, [4:7] root pos, [7:151] 24×6 rotations
            pos = x_unnorm[:, :, 4:7]
            q = ax_from_6v(x_unnorm[:, :, 7:].reshape(b_, s_, -1, 6))
            joints = self.smpl.forward(q, pos)  # (b, s, 24, 3)
            fcs_score = self.fcs_predictor(joints).mean()
            grad = torch.autograd.grad(fcs_score, x_in)[0]
        return x_next - guidance_scale * grad.detach()

    def ddim_sample(self, shape, cond, guidance_scale=0.0, guidance_start_step=25, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        use_guidance = guidance_scale > 0 and self.fcs_predictor is not None

        for step_idx, (time, time_next) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if use_guidance and step_idx >= guidance_start_step:
                x = self._apply_physics_guidance(x_start, x, guidance_scale)
        return x

    def long_ddim_sample(self, shape, cond, guidance_scale=0.0, guidance_start_step=25, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        if batch == 1:
            return self.ddim_sample(
                shape, cond,
                guidance_scale=guidance_scale,
                guidance_start_step=guidance_start_step,
            )

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        use_guidance = guidance_scale > 0 and self.fcs_predictor is not None

        for step_idx, (time, time_next, weight) in enumerate(tqdm(time_pairs, desc = 'sampling loop time step')):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if use_guidance and step_idx >= guidance_start_step:
                x = self._apply_physics_guidance(x_start, x, guidance_scale)

            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
        self,
        shape,
        cond,
        noise=None,
        constraint=None,
        return_diffusion=False,
        start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:] 

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
        self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # split off contact from the rest
        model_contact, model_out = torch.split(
            model_out, (4, model_out.shape[2] - 4), dim=2
        )
        target_contact, target = torch.split(target, (4, target.shape[2] - 4), dim=2)

        # velocity loss
        target_v = target[:, 1:] - target[:, :-1]
        model_out_v = model_out[:, 1:] - model_out[:, :-1]
        v_loss = self.loss_fn(model_out_v, target_v, reduction="none")
        v_loss = reduce(v_loss, "b ... -> b (...)", "mean")
        v_loss = v_loss * extract(self.p2_loss_weight, t, v_loss.shape)

        # FK loss
        b, s, c = model_out.shape
        # unnormalize
        # model_out = self.normalizer.unnormalize(model_out)
        # target = self.normalizer.unnormalize(target)
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        target_x = target[:, :, :3]
        target_q = ax_from_6v(target[:, :, 3:].reshape(b, s, -1, 6))

        # perform FK
        model_xp = self.smpl.forward(model_q, model_x)
        target_xp = self.smpl.forward(target_q, target_x)

        fk_loss = self.loss_fn(model_xp, target_xp, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b (...)", "mean")
        fk_loss = fk_loss * extract(self.p2_loss_weight, t, fk_loss.shape)

        # foot skate loss
        foot_idx = [7, 8, 10, 11]

        # find static indices consistent with model's own predictions
        static_idx = model_contact > 0.95  # N x S x 4
        model_feet = model_xp[:, :, foot_idx]  # foot positions (N, S, 4, 3)
        model_foot_v = torch.zeros_like(model_feet)
        model_foot_v[:, :-1] = (
            model_feet[:, 1:, :, :] - model_feet[:, :-1, :, :]
        )  # (N, S-1, 4, 3)
        model_foot_v[~static_idx] = 0
        foot_loss = self.loss_fn(
            model_foot_v, torch.zeros_like(model_foot_v), reduction="none"
        )
        foot_loss = reduce(foot_loss, "b ... -> b (...)", "mean")

        # FCS predictor loss on model's predicted joint positions
        fcs_loss = torch.tensor(0.0, device=model_xp.device)
        if self.fcs_predictor is not None and self.fcs_loss_weight > 0:
            fcs_pred = self.fcs_predictor(model_xp)  # (batch,)
            fcs_loss = fcs_pred.mean()

        # CoM / Base-of-Support balance loss
        # Penalises horizontal CoM being far from the mean position of
        # active foot contacts. Forces weight shift over standing foot.
        # Only applied on low-acceleration frames (static/slow poses) to
        # avoid suppressing dynamic dance moves (jumps, spins, lunges).
        if self.com_loss_weight > 0:
            masses = self._joint_masses.view(1, 1, 24, 1)            # (1,1,24,1)
            com = (model_xp * masses).sum(dim=2)                      # (B,S,3)
            com_h = com[:, :, :2]                                     # XY plane (B,S,2)
            # Mask out high-acceleration frames where CoM outside support
            # base is expected (jumps, explosive moves, weight transfers)
            com_acc = torch.norm(
                com[:, 2:, :] - 2 * com[:, 1:-1, :] + com[:, :-2, :], dim=-1  # (B,S-2)
            )
            # Pad to match sequence length (first and last frames get 0 acc)
            com_acc = F.pad(com_acc, (1, 1), value=0.0)              # (B,S)
            is_static = (com_acc < 0.01).float()                      # (B,S)
            # Detach foot positions for support center — prevents circular
            # gradient loop (moving feet changes target, which moves feet)
            foot_h = model_xp[:, :, foot_idx, :2].detach()           # (B,S,4,2)
            contact_w = (model_contact > 0.95).float()                # (B,S,4)
            denom = contact_w.sum(dim=-1, keepdim=True).clamp(min=1.) # (B,S,1)
            support_center = (foot_h * contact_w.unsqueeze(-1)).sum(dim=2) / denom  # (B,S,2)
            has_contact = (contact_w.sum(dim=-1) > 0).float()         # (B,S)
            com_dist_sq = ((com_h - support_center) ** 2).sum(dim=-1)  # (B,S)
            com_loss = (com_dist_sq * has_contact * is_static).mean()
        else:
            com_loss = torch.tensor(0.0, device=model_xp.device)

        # Bilateral foot exclusivity loss
        # Penalises both feet moving fast simultaneously (product of
        # left-foot and right-foot velocities). High product = airborne sliding.
        if self.bilateral_loss_weight > 0:
            left_v = torch.norm(
                model_feet[:, 1:, [0, 2], :] - model_feet[:, :-1, [0, 2], :], dim=-1
            ).mean(dim=-1)   # (B,S-1)
            right_v = torch.norm(
                model_feet[:, 1:, [1, 3], :] - model_feet[:, :-1, [1, 3], :], dim=-1
            ).mean(dim=-1)   # (B,S-1)
            # Only penalize when at least one foot is in contact — allow
            # legitimate airborne phases (jumps, hops, spins)
            any_contact = (model_contact[:, 1:] > 0.95).any(dim=-1).float()  # (B,S-1)
            bilateral_loss = (left_v * right_v * any_contact).mean()
        else:
            bilateral_loss = torch.tensor(0.0, device=model_xp.device)

        # Foot height during contact loss
        # Penalises feet hovering above ground while model predicts contact.
        # Ground reference = per-sequence min foot height, detached.
        if self.foot_height_loss_weight > 0:
            min_h = model_feet[:, :, :, 2].min(dim=1, keepdim=True)[0].detach()  # (B,1,4)
            adj_h = model_feet[:, :, :, 2] - min_h                   # (B,S,4)
            contact_w2 = (model_contact > 0.95).float()
            height_loss = (adj_h * contact_w2).mean()
        else:
            height_loss = torch.tensor(0.0, device=model_xp.device)

        losses = (
            0.636 * loss.mean(),
            2.964 * v_loss.mean(),
            0.646 * fk_loss.mean(),
            10.942 * foot_loss.mean(),
            self.fcs_loss_weight * fcs_loss,
            self.com_loss_weight * com_loss,
            self.bilateral_loss_weight * bilateral_loss,
            self.foot_height_loss_weight * height_loss,
        )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x)
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x.device).long()
        else:
            t = torch.full((batch_size,), t_override, device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def render_sample(
        self,
        shape,
        cond,
        normalizer,
        epoch,
        render_out,
        fk_out=None,
        name=None,
        sound=True,
        mode="normal",
        noise=None,
        constraint=None,
        sound_folder="ood_sliced",
        start_point=None,
        render=True,
        guidance_scale=0.0,
        guidance_start_step=25,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                    guidance_scale=guidance_scale,
                    guidance_start_step=guidance_start_step,
                )
                .detach()
                .cpu()
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples)

        if samples.shape[2] == 151:
            sample_contact, samples = torch.split(
                samples, (4, samples.shape[2] - 4), dim=2
            )
        else:
            sample_contact = None
        # do the FK all at once
        b, s, c = samples.shape
        pos = samples[:, :, :3].to(cond.device)  # np.zeros((sample.shape[0], 3))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        # go 6d to ax
        q = ax_from_6v(q).to(cond.device)

        if mode == "long":
            b, s, c1, c2 = q.shape
            assert s % 2 == 0
            half = s // 2
            if b > 1:
                # if long mode, stitch position using linear interp

                fade_out = torch.ones((1, s, 1)).to(pos.device)
                fade_in = torch.ones((1, s, 1)).to(pos.device)
                fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                    pos.device
                )
                fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                    pos.device
                )

                pos[:-1] *= fade_out
                pos[1:] *= fade_in

                full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                idx = 0
                for pos_slice in pos:
                    full_pos[idx : idx + s] += pos_slice
                    idx += half

                # stitch joint angles with slerp
                slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                left, right = q[:-1, half:], q[1:, :half]
                # convert to quat
                left, right = (
                    axis_angle_to_quaternion(left),
                    axis_angle_to_quaternion(right),
                )
                merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                # convert back
                merged = quaternion_to_axis_angle(merged)

                full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                full_q[:half] += q[0, :half]
                idx = half
                for q_slice in merged:
                    full_q[idx : idx + half] += q_slice
                    idx += half
                full_q[idx : idx + half] += q[-1, half:]

                # unsqueeze for fk
                full_pos = full_pos.unsqueeze(0)
                full_q = full_q.unsqueeze(0)
            else:
                full_pos = pos
                full_q = q
            full_pose = (
                self.smpl.forward(full_q, full_pos).detach().cpu().numpy()
            )  # b, s, 24, 3
            # squeeze the batch dimension away and render
            skeleton_render(
                full_pose[0],
                epoch=f"{epoch}",
                out=render_out,
                name=name,
                sound=sound,
                stitch=True,
                sound_folder=sound_folder,
                render=render
            )
            if fk_out is not None:
                outname = f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
                Path(fk_out).mkdir(parents=True, exist_ok=True)
                pickle.dump(
                    {
                        "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                        "full_pose": full_pose[0],
                    },
                    open(os.path.join(fk_out, outname), "wb"),
                )
            return

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = (
            sample_contact.detach().cpu().numpy()
            if sample_contact is not None
            else None
        )

        def inner(xx):
            num, pose = xx
            filename = name[num] if name is not None else None
            contact = sample_contact[num] if sample_contact is not None else None
            skeleton_render(
                pose,
                epoch=f"e{epoch}_b{num}",
                out=render_out,
                name=filename,
                sound=sound,
                contact=contact,
            )

        p_map(inner, enumerate(poses))

        if fk_out is not None and mode != "long":
            Path(fk_out).mkdir(parents=True, exist_ok=True)
            for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
                path = os.path.normpath(filename)
                pathparts = path.split(os.sep)
                pathparts[-1] = pathparts[-1].replace("npy", "wav")
                # path is like "data/train/features/name"
                pathparts[2] = "wav_sliced"
                audioname = os.path.join(*pathparts)
                outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
                pickle.dump(
                    {
                        "smpl_poses": qq.reshape((-1, 72)).cpu().numpy(),
                        "smpl_trans": pos_.cpu().numpy(),
                        "full_pose": pose,
                    },
                    open(f"{fk_out}/{outname}", "wb"),
                )
