"""Training entrypoint configuration utilities for VEGA."""
from argparse import ArgumentParser

from configs.iemocap_config import IEMOCAP_CONFIG
from vega_utils.common import seed_everything


def parse_arguments():
    """Build and parse command line arguments."""
    parser = ArgumentParser(description='VEGA training runner')
    cfg = IEMOCAP_CONFIG

    def add_bool_flag(name: str, default: bool, help_text: str):
        parser.add_argument(f'--{name}', dest=name, action='store_true', help=help_text)
        parser.add_argument(f'--no_{name}', dest=name, action='store_false', help=f'Disable {help_text.lower()}')
        parser.set_defaults(**{name: default})

    # Basic settings
    parser.add_argument('--name', type=str, default=cfg['name'], help='Experiment identifier used in output paths.')
    parser.add_argument('--seed', type=int, default=cfg['seed'], help='Global random seed for reproducibility.')
    parser.add_argument('--Dataset', default=cfg['Dataset'], help='Target dataset: IEMOCAP or MELD.')
    parser.add_argument('--num_workers', type=int, default=cfg['num_workers'], help='DataLoader worker process count.')
    parser.add_argument('--epochs', type=int, default=cfg['epochs'], help='Total number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=cfg['batch_size'], help='Mini-batch size.')
    add_bool_flag('scheduler', cfg['scheduler'], 'Enable cosine learning-rate scheduling.')
    parser.add_argument('--optimizer', type=str, default=cfg['optimizer'], help='Optimizer name (e.g., AdamW, Adam, SGD).')
    parser.add_argument('--lr', type=float, default=cfg['lr'], help='Initial learning rate.')
    parser.add_argument('--l2', type=float, default=cfg['l2'], help='Weight decay coefficient.')

    """Loss"""
    parser.add_argument('--cls_lambda', type=float, default=cfg['cls_lambda'], help='Global weight for CLS loss.')
    parser.add_argument('--cls_all_lambda', type=float, default=cfg['cls_all_lambda'], help='Weight for all-modal CLS branch.')
    parser.add_argument('--cls_v_lambda', type=float, default=cfg['cls_v_lambda'], help='Weight for visual CLS branch.')
    parser.add_argument('--cls_t_lambda', type=float, default=cfg['cls_t_lambda'], help='Weight for text CLS branch.')
    parser.add_argument('--cls_a_lambda', type=float, default=cfg['cls_a_lambda'], help='Weight for audio CLS branch.')

    parser.add_argument('--cls_all_cls_kl_lambda', type=float, default=cfg['cls_all_cls_kl_lambda'], help='Global weight for CLS-to-CLS KL loss.')
    parser.add_argument('--a_cls_all_cls_kl_lambda', type=float, default=cfg['a_cls_all_cls_kl_lambda'], help='Audio branch weight in CLS KL loss.')
    parser.add_argument('--t_cls_all_cls_kl_lambda', type=float, default=cfg['t_cls_all_cls_kl_lambda'], help='Text branch weight in CLS KL loss.')
    parser.add_argument('--v_cls_all_cls_kl_lambda', type=float, default=cfg['v_cls_all_cls_kl_lambda'], help='Visual branch weight in CLS KL loss.')

    parser.add_argument('--clip_lambda', type=float, default=cfg['clip_lambda'], help='Global weight for CLIP loss.')
    parser.add_argument('--a_clip_lambda', type=float, default=cfg['a_clip_lambda'], help='Weight for audio CLIP branch.')
    parser.add_argument('--v_clip_lambda', type=float, default=cfg['v_clip_lambda'], help='Weight for visual CLIP branch.')
    parser.add_argument('--t_clip_lambda', type=float, default=cfg['t_clip_lambda'], help='Weight for text CLIP branch.')
    parser.add_argument('--all_clip_lambda', type=float, default=cfg['all_clip_lambda'], help='Weight for all-modal CLIP branch.')

    parser.add_argument('--clip_all_clip_kl_lambda', type=float, default=cfg['clip_all_clip_kl_lambda'], help='Global weight for CLIP-to-CLIP KL loss.')
    parser.add_argument('--a_clip_all_clip_kl_lambda', type=float, default=cfg['a_clip_all_clip_kl_lambda'], help='Audio branch weight in CLIP KL loss.')
    parser.add_argument('--v_clip_all_clip_kl_lambda', type=float, default=cfg['v_clip_all_clip_kl_lambda'], help='Visual branch weight in CLIP KL loss.')
    parser.add_argument('--t_clip_all_clip_kl_lambda', type=float, default=cfg['t_clip_all_clip_kl_lambda'], help='Text branch weight in CLIP KL loss.')

    """CLIP"""
    parser.add_argument('--CLIP_Model', type=str, default=cfg['CLIP_Model'],
                        choices=["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14",
                                 "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14-336"],
                        help='Pretrained CLIP backbone.')
    parser.add_argument('--clip_proj_layer_num', type=int, default=cfg['clip_proj_layer_num'], help='Number of layers in CLIP projection head.')
    parser.add_argument('--clip_proj_activation_fn', type=str, default=cfg['clip_proj_activation_fn'], help='Activation function in CLIP projection head.')
    parser.add_argument('--clip_proj_drop', type=float, default=cfg['clip_proj_drop'], help='Dropout rate in CLIP projection head.')

    parser.add_argument('--expr_img_folder', type=str, default=cfg['expr_img_folder'], help='Anchor expression folder name under anchor/.')
    parser.add_argument('--rand', type=float, default=cfg['rand'], help='Sampling ratio for stochastic anchor selection.')

    """Transformer Backbone"""
    parser.add_argument('--hidden_dim', type=int, default=cfg['hidden_dim'], help='Hidden size of multimodal backbone.')
    parser.add_argument('--n_head', type=int, default=cfg['n_head'], help='Attention head count per transformer.')
    parser.add_argument('--dropout', type=float, default=cfg['dropout'], help='Backbone dropout rate.')

    parser.add_argument('--outlayer_drop', type=float, default=cfg['outlayer_drop'], help='Classifier head dropout rate.')
    parser.add_argument('--outlayer_num', type=int, default=cfg['outlayer_num'], help='Number of layers in classifier head.')
    parser.add_argument('--outlayer_activation_fn', type=str, default=cfg['outlayer_activation_fn'], help='Activation function in classifier head.')

    # Loss function configurations
    add_bool_flag('clip_loss', cfg['clip_loss'], 'Enable CLIP supervision branch.')
    add_bool_flag('cls_loss', cfg['cls_loss'], 'Enable backbone classification supervision.')
    add_bool_flag('clip_all_clip_kl_loss', cfg['clip_all_clip_kl_loss'], 'Enable KL distillation among CLIP branches.')
    add_bool_flag('cls_all_cls_kl_loss', cfg['cls_all_cls_kl_loss'], 'Enable KL distillation among backbone branches.')

    return parser.parse_args()


def setup_environment(args):
    """Populate runtime-derived fields from parsed arguments."""
    from datetime import datetime
    from pathlib import Path
    import pytz
    import torch

    now_uk = datetime.now(pytz.utc).astimezone(pytz.timezone('Europe/London'))
    args.now = now_uk.strftime("[%m%d-%H%M]")

    args.checkpoint_root = Path(r'output') / args.name / args.now
    expr_img_root = Path('anchor') / str(args.expr_img_folder)
    fallback_expr_img_root = Path('anchor') / f'{args.expr_img_folder}_anchor'
    if not expr_img_root.exists() and fallback_expr_img_root.exists():
        expr_img_root = fallback_expr_img_root
    args.expr_img_root = str(expr_img_root)

    args.cuda = torch.cuda.is_available()
    print('Running on', 'GPU' if args.cuda else 'CPU')

    # Set the CLIP dimension based on model
    if args.CLIP_Model in ['openai/clip-vit-large-patch14', 'openai/clip-vit-large-patch14-336']:
        print('Using CLIP-ViT-Large, with dimension 768')
        args.clip_dim = 768
    else:
        print('Using CLIP-ViT-Base, with dimension 512')
        args.clip_dim = 512

    args.audio_dim = 1582 if args.Dataset == 'IEMOCAP' else 300
    args.visual_dim = 342
    args.text_dim = 1024
    args.n_speakers = 9 if args.Dataset == 'MELD' else 2
    args.n_classes = 7 if args.Dataset == 'MELD' else 6 if args.Dataset == 'IEMOCAP' else 1

    return args


def run() -> None:
    from main import main

    args = parse_arguments()
    seed_everything(args.seed)
    args = setup_environment(args)
    main(args)


if __name__ == '__main__':
    run()
