import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.config import configs
from pyutils.general import ensure_dir, logger

dataset = "fdtd"
model = "cnn"
exp_name = "train_PICCO"
root = f"log/{dataset}/{model}/{exp_name}"
script = 'train_multi_step.py'
config_file = f'configs/{dataset}/{model}/{exp_name}/train.yml'
checkpoint_dir = f'{dataset}/{model}/{exp_name}'
configs.load(config_file, recursive=True)


def task_launcher(args):
    dim, alg, pac, input_mode, eps_lap, field_norm_mode, stem, se, in_frames, out_frames, out_channels, offset_frames, kernel_size, backbone_dilation, if_pre_dwconv, r, id, if_pass_history, share_encoder, share_decoder, share_backbone, share_history_encoder, if_pass_grad, description, gpu_id, epochs, lr, criterion, if_spatial_mask, weighted_frames, checkpt, bs = args
    '''
    brief description of these handles:
    dim: hidden dimension in the model, 72 is used in our experiments
    alg: algorithm used in the backbone, options are "Conv2d" and "PacConv2d" representing the standard convolution and PAConv2d respectively
    pac: whether to use PAConv2d, if the alg is "PacConv2d", pac should be True, otherwise False
    input_mode: input mode of the model, options are "E0_Ji" and "eps_E0_Ji", if the alg is "PacConv2d", input_mode should be "E0_Ji", otherwise "eps_E0_Ji" 
    eps_lap: whether concate the laplacian of the last input field to the input, default is False
    field_norm_mode: field normalization mode, default is "max"
    stem: stem of the model, options are "FNO_lifting" and "NO2", FNO_lifting is the one used in FNO, NO2 is the one used in NeurLOight
    se: whether to use squeeze and excitation block in the model
    in_frames: number of input frames, 10 is recommended
    out_frames: number of output frames (# of frames to predict), default is 160
    out_channels: number of output channels in each sub model, default is 80, which means by default, the model will predict 80 frames in each sub model, and there are 2 sub models in total
    offset_frames: number of frames to skip between the input and output, default is the same as in_frames so that the input field and output field are continuous
    kernel_size: kernel size of the backbone, when dilated convolution is used, this should be set to required receptive field, for instance, if the receptive field is 5, and the dilation is 2, kernel_size should be set to 5 but not 3
    backbone_dilation: dilation of the backbone, 4 is recommended
    if_pre_dwconv: whether to use depthwise convolution before the dilated convolution in the backbone, set to true when the backbone dilation is larger than 1
    r: number of layers in the backbone, 8 is recommended
    id: id of the experiment
    if_pass_history: whether to pass the history field to the downstream sub-model, True is recommended
    share_encoder: whether to share the encoder in the sub-models, False is recommended
    share_decoder: whether to share the decoder in the sub-models, False is recommended
    share_backbone: whether to share the backbone in the sub-models, False is recommended
    share_history_encoder: whether to share the history encoder in the sub-models, False is recommended
    if_pass_grad: whether to pass the gradient to the downstream sub-model, False is recommended
    description: description of the model, should include the dataset device name, "mmi" or "mrr" or "metaline" should be included in the description
    gpu_id: gpu id to run the experiment
    epochs: number of epochs to train the model
    lr: initial learning rate
    criterion: loss function, "nL2norm" is recommended, this represents the normalized L2 norm
    if_spatial_mask: whether to use spatial mask in the loss function, False is recommended
    weighted_frames: whether to assign more weight to the last few frames in the loss function, if set to 0 the weight is uniform, if set an integer larger that 0, the weight will be assigned to the last few frames
    checkpt: checkpoint to resume the training, if set to "none", the training will start from scratch
    bs: batch size, 1 is recommended

    saved checkpoints are also provided:
    for MMI dataset:
    ./checkpoint/fdtd/cnn/train_ours_final/MultiStepDynamicCNN_alg-PacConv2d_of-160_oc-80_ks-17_r-8_hty-True_grad-False_se-0_sd-0_sh-0_des-ours_final_MMI_id-18_err-0.0494_epoch-94.pt
    for MRR dataset:
    ./checkpoint/fdtd/cnn/train_ours_final/MultiStepDynamicCNN_alg-PacConv2d_of-160_oc-80_ks-21_r-8_hty-True_grad-False_se-0_sd-0_sh-0_des-ours_final_MRR_KS21_id-18_err-0.0888_epoch-100.pt
    for Metaline dataset:
    ./checkpoint/fdtd/cnn/train_ours_final/MultiStepDynamicCNN_alg-PacConv2d_of-160_oc-80_ks-17_r-8_hty-True_grad-False_se-0_sd-0_sh-0_des-NO_MASKED_ours_final_META_REALIGN_RES_id-18_err-0.0944_epoch-98.pt
    '''
    assert pac == True if alg == "PacConv2d" else pac == False
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    pres = [
            'python3',
            script,
            config_file
            ]
    suffix = f"alg-{alg}_of-{out_frames}_oc-{out_channels}_ks-{kernel_size}_r-{r}_hty-{if_pass_history}_grad-{if_pass_grad}_se-{int(share_encoder)}_sb-{int(share_backbone)}_sd-{int(share_decoder)}_sh-{int(share_history_encoder)}_des-{description}_id-{id}"
    with open(os.path.join(root, f'{suffix}.log'), 'w') as wfid:
        if 'mrr' in description.lower():
            dataset_dir = 'processed_small_mrr_160'
            device_list = ['mrr_random']
            guidance_kernel_size_list = [5,5,5,5]
            guidance_padding_list = [2,2,2,2]
        elif 'mmi' in description.lower():
            dataset_dir = 'processed_small_mmi_160'
            device_list = ['mmi_3x3_L_random']
            guidance_kernel_size_list = [3,3,5,5]
            guidance_padding_list = [1,1,2,2]
        elif 'meta' in description.lower():
            dataset_dir = 'processed_small_metaline_160'
            device_list = ['metaline_3x3']
            guidance_kernel_size_list = [3,3,5,5]
            guidance_padding_list = [1,1,2,2]
        else:
            raise ValueError(f"dataset {description} not recognized")

        if stem == "FNO_lifting":
            kernel_size_list = [1]
            kernel_list = [96]
            stride_list = [1]
            padding_list = [0]
            dilation_list = [1]
            groups_list = [1]
            residual = [False]
            norm_list = [False]
            act_list = [False]
        elif stem == "NO2":
            kernel_size_list = [1, 3, 1, 3]
            kernel_list = [dim, dim, dim, dim]
            stride_list = [1, 1, 1, 1]
            padding_list = [0, 1, 0, 1]
            dilation_list = [1, 1, 1, 1]
            groups_list = [1, dim, 1, dim]
            residual = [False, True, False, True]
            norm_list = [False, True, False, True]
            act_list = [False, True, False, True]
            if se:
                encoder_se = [True, True, True, True]
            else:
                encoder_se = [False, False, False, False]
        else:
            raise ValueError(f"stem {stem} not recognized")
        exp = [
            f"--dataset.device_list={device_list}",
            f"--dataset.processed_dir={dataset_dir}",
            f"--dataset.in_frames={in_frames}",
            f"--dataset.offset_frames={offset_frames}",
            f"--dataset.out_frames={out_frames}",
            f"--dataset.out_channels={out_channels}",

            f"--run.n_epochs={epochs}",
            f"--run.batch_size={bs}",
            f"--run.gpu_id={gpu_id}",
            f"--run.log_interval=200",
            f"--run.random_state={59}",
            f"--run.multi_train_schedule={[i for i in range(1, out_frames//out_channels)]}",

            f"--criterion.name={criterion}",
            f"--criterion.weighted_frames={weighted_frames}",
            f"--criterion.if_spatial_mask={if_spatial_mask}",

            f"--scheduler.lr_min={lr*5e-3}",

            f"--plot.dir_name={model}_{exp_name}_des-{description}_id-{id}",
            f"--plot.autoreg={True}",

            f"--optimizer.lr={lr}",

            f"--model.dim={dim}",
            f"--model.field_norm_mode={field_norm_mode}",
            f"--model.input_cfg.input_mode={input_mode}",
            f"--model.input_cfg.eps_lap={eps_lap}",
            f"--model.out_channels={out_channels}",
            f"--model.num_iters={out_frames//out_channels}",
            f"--model.share_encoder={share_encoder}",
            f"--model.share_backbone={share_backbone}",
            f"--model.share_decoder={share_decoder}",
            f"--model.share_history_encoder={share_history_encoder}",
            f"--model.if_pass_history={if_pass_history}",
            f"--model.if_pass_grad={if_pass_grad}",

            f"--model.guidance_generator_cfg.kernel_size_list={guidance_kernel_size_list}",
            f"--model.guidance_generator_cfg.padding_list={guidance_padding_list}",

            f"--model.encoder_cfg.kernel_size_list={kernel_size_list}",
            f"--model.encoder_cfg.kernel_list={kernel_list}",
            f"--model.encoder_cfg.stride_list={stride_list}",
            f"--model.encoder_cfg.padding_list={padding_list}",
            f"--model.encoder_cfg.dilation_list={dilation_list}",
            f"--model.encoder_cfg.groups_list={groups_list}",
            f"--model.encoder_cfg.residual={residual}",
            f"--model.encoder_cfg.norm_list={norm_list}",
            f"--model.encoder_cfg.act_list={act_list}",
            f"--model.encoder_cfg.se={encoder_se}",
            f"--model.encoder_cfg.pac={False}",
            f"--model.encoder_cfg.if_pre_dwconv={False}",
            f"--model.encoder_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.encoder_cfg.conv_cfg.padding_mode={'replicate'}",

            f"--model.backbone_cfg.conv_cfg.type={alg}", 
            f"--model.backbone_cfg.kernel_size_list={[kernel_size]*r}",
            f"--model.backbone_cfg.kernel_list={[dim]*r}" if "2d" in alg else f"--model.backbone_cfg.conv_cfg.kernel_list={[1]*r}",
            f"--model.backbone_cfg.stride_list={[1]*r}",
            f"--model.backbone_cfg.padding_list={[kernel_size//2]*r}",
            f"--model.backbone_cfg.dilation_list={[backbone_dilation]*r}",
            f"--model.backbone_cfg.groups_list={[1]*r}",
            f"--model.backbone_cfg.norm_list={[True]*r}",
            f"--model.backbone_cfg.act_list={[True]*r}",
            f"--model.backbone_cfg.residual={[True]*r}",
            f"--model.backbone_cfg.conv_cfg.r={r}",
            f"--model.backbone_cfg.se={[se]*r}",
            f"--model.backbone_cfg.pac={pac}",
            f"--model.backbone_cfg.if_pre_dwconv={if_pre_dwconv}",
            f"--model.backbone_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.backbone_cfg.conv_cfg.padding_mode={'replicate'}",

            f"--model.decoder_cfg.conv_cfg.type={'Conv2d'}",
            f"--model.decoder_cfg.residual={[False, False]}",
            f"--model.decoder_cfg.kernel_list={[512, out_channels]}",
            f"--model.decoder_cfg.se={[se]*2}",
            f"--model.decoder_cfg.conv_cfg.padding_mode={'zeros'}" if pac else f"--model.decoder_cfg.conv_cfg.padding_mode={'replicate'}",

            f"--checkpoint.model_comment={suffix}",
            f"--checkpoint.resume={False}" if checkpt == "none" else f"--checkpoint.resume={True}",
            f"--checkpoint.restore_checkpoint={checkpt}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
            ]
        logger.info(f"running command {pres + exp}")
        subprocess.call(pres + exp, stderr=wfid, stdout=wfid)


if __name__ == '__main__':
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        [72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", False, 10, 160, 80, 10, 17, 4, True, 8, 18, True, False, False, False, False, False, "Test_reorgnaized_PICCO_MMI", 0, 100, 0.002, "nL2norm", False, 0, "none", 1], # uncomment this to train the model on mmi dataset
        # [72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", False, 10, 160, 80, 10, 17, 4, True, 8, 18, True, False, False, False, False, False, "Test_reorgnaized_PICCO_MRR", 1, 100, 0.002, "nL2norm", False, 0, "none", 1], # uncomment this to train the model on mrr dataset
        # [72, "PacConv2d", True, "E0_Ji", False, "max", "NO2", False, 10, 160, 80, 10, 17, 4, True, 8, 18, True, False, False, False, False, False, "Test_reorgnaized_PICCO_META", 2, 100, 0.002, "nL2norm", False, 0, "none", 1], # uncomment this to train the model on metaline dataset
        ]

    with Pool(8) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
