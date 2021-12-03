import os
import math
import torch
import torchvision
from torch import optim
from tqdm import tqdm
from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from models.stylegan2.model import Generator
from models.utils import ensure_checkpoint_exists
from models.mapper.train_utils import STYLESPACE_DIMENSIONS

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(
    STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


# original code


def main(args):
    ensure_checkpoint_exists(args.ckpt)
    # text_inputs = torch.cat([clip.tokenize(args.description)]).cuda() # multilingual에서는 삭제
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                           truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init],
                            input_is_latent=True, randomize_noise=False)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema(
                [latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    clip_loss = CLIPLoss()
    id_loss = IDLoss(args)

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))
    # pbar = tqdm(range(100))

    for i in pbar:
        # Adjust the learning rate
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        # Generate an image using the latent vector
        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False,
                           input_is_stylespace=args.work_in_stylespace)

        # Loss 계산
        # CLIP
        c_loss = clip_loss(img_gen, args.description)

        # ID
        if args.id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)[0]
        else:
            i_loss = 0

        # L2
        if args.mode == "edit":
            if args.work_in_stylespace:
                l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum()
                              for c in range(len(latent_code_init))])
            else:
                l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
            # edit 모드일 경우 논문에 나온 full term loss
        else:
            loss = c_loss
            # 아닐 경우, clip loss 만

        # gradient descent -> update latent vector
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log 출력
        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False,
                                   input_is_stylespace=args.work_in_stylespace)

            torchvision.utils.save_image(
                img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))

        if args.mode == "edit":
            final_result = torch.cat([img_orig, img_gen])
        else:
            final_result = img_gen

    return final_result
