import torch
import torch.nn.functional as F 
import math
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, Attention


##### AM mechanisms hyperparameters #####
# you can change these to experiment with transforms and their effect on generation
scales_map = { # const values, used in scale-power
    "pow": {
        "up": 1.3,
        "mid": 1.,
        "down": 1.3,
    },
    "scale": {
        "up": 1.45,
        "mid": 1.,
        "down": 1.55,
    },
    "softmask": {
        "up": 12.5,
        "mid": 1.,
        "down": 12.5,
    },
    "softmask_qa": {
        "up": 0.6,
        "mid": 1.0,
        "down": 0.6,
    },
}


softmask_scale_schedule = {
    "up": [1.0, 1.55, 1.55, 1.55],
    "mid": None,
    "down": [1.0, 1.55, 1.55, 1.55],
}

softmask_s_schedule = {
    "up": [7.5, 5., 5., 5.],
    "mid": None,
    "down": [7.5, 5., 5., 5.],
}

softmask_qa_schedule = {
    "up": [0.65, 0.65, 0.65, 0.65],
    "mid": None,
    "down": [0.65, 0.65, 0.65, 0.65],
}
#########################################


def norm_attn_map(x):
    x_normed = (x - x.amin(dim=(0, 1, 2), keepdim=True)) / (x.amax(dim=(0, 1, 2), keepdim=True) - x.amin(dim=(0, 1, 2), keepdim=True))
    return x_normed


def softmask_transform(x, unet_part):
    s = scales_map["softmask"][unet_part]
    m = 0.5 # baseline value
    res = 1 / (1 + torch.exp(-s*(norm_attn_map(x) - m)))
    res = norm_attn_map(res)
    return res


def adaptive_softmask_transform(x, unet_part):
    s = scales_map["softmask"][unet_part]
    qa = scales_map["softmask_qa"][unet_part]
    qv = torch.quantile(x.float(), q=qa, dim=-2, keepdim=True).half().expand_as(x)
    eps = 1e-6 
    qv = torch.clamp(qv, min=eps)  # ensure quantile values aren't exactly zero
    res = 1 / (1 + torch.exp(-s*(norm_attn_map(x) - qv)))
    res = norm_attn_map(res)
    return res


UPPER_BOUND = 10.
def scale_transform(am, unet_part):
    s = scales_map["scale"][unet_part]
    return torch.clamp(am * s, max=UPPER_BOUND)


def pow_transform(am, unet_part):
    am = torch.pow(am, scales_map["pow"][unet_part])
    return am


def softshift_transform(am, unet_part):
    s = scales_map["softshift"][unet_part]    
    side = int(math.sqrt(am.shape[2]))
    k = int(0.75 * (side*side))  # 75th percentile index
    threshold, _ = torch.kthvalue(am, k, dim=2, keepdim=True)    
    mask = (am > threshold).to(am.dtype)
    am_shifted = am * (1 + (s-1) * mask)
    
    return am_shifted


def scaled_dot_product_attention(
    query, 
    key, 
    value, 
    attn_mask=None, 
    dropout_p=0.0, 
    is_causal=False, 
    scale=None, 
    am_transform=None,
    unet_part=None,
    target_tokens=None,
    inverse_neg_transforms=False,
    apply_adain=False,
    adain_weight=0.5,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if apply_adain:
        attn_weight_orig = attn_weight.clone()

    if am_transform is not None:
        target_tokens = [0, 1, 2, 3] if target_tokens is None else target_tokens
        A_targ = attn_weight[:, :, :, target_tokens]
        if inverse_neg_transforms:
            idx_to_inverse = []
            if target_tokens is not None:
                idx_to_inverse = [target_tokens.index(t) for t in [0] if t in target_tokens]
            else:                 
                idx_to_inverse = [0]
            A_targ[:, :, :, idx_to_inverse] = 1 - A_targ[:, :, :, idx_to_inverse]
            A_targ = am_transform(A_targ, unet_part)
            max_vals = torch.max(A_targ[:, :, :, idx_to_inverse], dim=-2, keepdim=True).values.expand_as(A_targ[:, :, :, idx_to_inverse])
            A_targ[:, :, :, idx_to_inverse] = max_vals - A_targ[:, :, :, idx_to_inverse]
            if not torch.all(A_targ >= 0):
                print('A_targ is not non-negative')
                print(A_targ.min(dim=-2, keepdim=True).values)
        else:
            A_targ = am_transform(A_targ, unet_part)
        attn_weight[:, :, :, target_tokens] = A_targ

    eps = 1e-6
    w = adain_weight
    if apply_adain:
        mean_orig = attn_weight_orig.mean(dim=-1, keepdim=True)
        std_orig = attn_weight_orig.std(dim=-1, keepdim=True)
        mean_transformed = attn_weight.mean(dim=-1, keepdim=True)
        std_transformed = attn_weight.std(dim=-1, keepdim=True)

        attn_weight_rescaled = (attn_weight - mean_transformed) * (std_orig / (std_transformed + eps)) + mean_orig
        attn_weight = w * attn_weight + (1 - w) * attn_weight_rescaled
        

    hs = torch.dropout(attn_weight, dropout_p, train=True) @ value

    if apply_adain:
        hs_orig = torch.dropout(attn_weight_orig, dropout_p, train=True) @ value
        mean_orig = hs_orig.mean(dim=-1, keepdim=True)
        std_orig = hs_orig.std(dim=-1, keepdim=True)
        mean_transformed = hs.mean(dim=-1, keepdim=True)
        std_transformed = hs.std(dim=-1, keepdim=True)
        hs_rescaled = (hs - mean_transformed) * (std_orig / (std_transformed + eps)) + mean_orig
        if torch.isnan(hs_rescaled).any(): # safe measure in case of numerical instability
            return hs
        hs = w * hs + (1 - w) * hs_rescaled

    return hs


def patched_ip_attn20(
    self,
    attn: Attention, 
    hidden_states, 
    encoder_hidden_states=None, 
    attention_mask=None, 
    temb=None, 
    scale=1.0, 
    ip_adapter_masks=None,
    am_transform=None,
):
    residual = hidden_states

    # separate ip_hidden_states from encoder_hidden_states
    if encoder_hidden_states is not None:
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        else:
            deprecation_message = (
                "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
            )
            deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                [encoder_hidden_states[:, end_pos:, :]],
            )

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if ip_adapter_masks is not None:
        if not isinstance(ip_adapter_masks, List):
            # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
            ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
        if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
            raise ValueError(
                f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                f"({len(ip_hidden_states)})"
            )
        else:
            for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                if mask is None:
                    continue
                if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                    raise ValueError(
                        "Each element of the ip_adapter_masks array should be a tensor with shape "
                        "[1, num_images_for_ip_adapter, height, width]."
                        " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                    )
                if mask.shape[1] != ip_state.shape[1]:
                    raise ValueError(
                        f"Number of masks ({mask.shape[1]}) does not match "
                        f"number of ip images ({ip_state.shape[1]}) at index {index}"
                    )
                if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                    raise ValueError(
                        f"Number of masks ({mask.shape[1]}) does not match "
                        f"number of scales ({len(scale)}) at index {index}"
                    )
    else:
        ip_adapter_masks = [None] * len(self.scale)

    # for ip-adapter
    for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
        ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
    ):
        skip = False
        if isinstance(scale, list):
            if all(s == 0 for s in scale):
                skip = True
        elif scale == 0:
            skip = True
        if not skip:
            if mask is not None:
                if not isinstance(scale, list):
                    scale = [scale] * mask.shape[1]

                current_num_images = mask.shape[1]
                for i in range(current_num_images):
                    ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                    ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1

                    if hasattr(self, 'is_target_block') and self.is_target_block:
                        am_transforms = self.am_transforms

                        def chain_transforms(am, unet_part):
                            for t in am_transforms:
                                am = t(am, unet_part)
                            return am

                        if self.current_tstep in self.target_tsteps:
                            adain_weight_sch = {0: 0.7, 1: 0.7, 2: 0.7, 3: 0.7}
                            _current_ip_hidden_states = scaled_dot_product_attention(
                                query, 
                                ip_key, 
                                ip_value, 
                                attn_mask=None, 
                                dropout_p=0.0, 
                                is_causal=False, 
                                am_transform=chain_transforms, 
                                unet_part=self.unet_part,
                                target_tokens=self.target_tokens,
                                inverse_neg_transforms=self.inverse_neg_transforms,
                                apply_adain=self.apply_adain,
                                adain_weight=adain_weight_sch[self.current_tstep]
                            )
                        else:
                            _current_ip_hidden_states = F.scaled_dot_product_attention(
                                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False,
                            )
                            self.current_tstep += 1
                    else: 
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                    _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                    mask_downsample = IPAdapterMaskProcessor.downsample(
                        mask[:, i, :, :],
                        batch_size,
                        _current_ip_hidden_states.shape[1],
                        _current_ip_hidden_states.shape[2],
                    )

                    mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                    hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
            else:
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                
                if hasattr(self, 'is_target_block') and self.is_target_block:
                    am_transforms = self.am_transforms
                    # if self.current_tstep > 0 and blend_transform in am_transforms:
                    #     print('removed blend at tstep', self.current_tstep)
                    #     am_transforms.remove(blend_transform) # don't ever want to do after first step

                    if self.apply_schedule:
                        try:
                            scales_map["softmask_qa"][self.unet_part] = softmask_qa_schedule[self.unet_part][self.current_tstep]
                            scales_map["softmask"][self.unet_part] = softmask_s_schedule[self.unet_part][self.current_tstep]
                            scales_map["scale"][self.unet_part] = softmask_scale_schedule[self.unet_part][self.current_tstep]
                        except IndexError:
                            print('IndexError at tstep', self.current_tstep)
                            raise
                    def chain_transforms(am, unet_part):
                        for t in am_transforms:
                            am = t(am, unet_part)
                        return am

                    if self.current_tstep in self.target_tsteps:
                        adain_weight_sch = {0: 0.7, 1: 0.7, 2: 0.7, 3: 0.7}
                        current_ip_hidden_states = scaled_dot_product_attention(
                            query, 
                            ip_key, 
                            ip_value, 
                            attn_mask=None, 
                            dropout_p=0.0, 
                            is_causal=False, 
                            am_transform=chain_transforms, 
                            unet_part=self.unet_part,
                            target_tokens=self.target_tokens,
                            inverse_neg_transforms=self.inverse_neg_transforms,
                            apply_adain=self.apply_adain,
                            adain_weight=adain_weight_sch[self.current_tstep]
                        )
                    else:
                        current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )
                    self.current_tstep += 1
                    # current_ip_hidden_states = scaled_dot_product_attention(
                    #     query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False, am_transform=self.am_transform, unet_part=self.unet_part
                    # )
                else: 
                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + scale * current_ip_hidden_states

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def unpatched_ip_attn20(
    self,
    attn: Attention, 
    hidden_states, 
    encoder_hidden_states=None, 
    attention_mask=None, 
    temb=None, 
    scale=1.0, 
    ip_adapter_masks=None,
):
    residual = hidden_states

    # separate ip_hidden_states from encoder_hidden_states
    if encoder_hidden_states is not None:
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        else:
            deprecation_message = (
                "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
            )
            deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                [encoder_hidden_states[:, end_pos:, :]],
            )

    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if ip_adapter_masks is not None:
        if not isinstance(ip_adapter_masks, List):
            # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
            ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
        if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
            raise ValueError(
                f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                f"({len(ip_hidden_states)})"
            )
        else:
            for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                if mask is None:
                    continue
                if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                    raise ValueError(
                        "Each element of the ip_adapter_masks array should be a tensor with shape "
                        "[1, num_images_for_ip_adapter, height, width]."
                        " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                    )
                if mask.shape[1] != ip_state.shape[1]:
                    raise ValueError(
                        f"Number of masks ({mask.shape[1]}) does not match "
                        f"number of ip images ({ip_state.shape[1]}) at index {index}"
                    )
                if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                    raise ValueError(
                        f"Number of masks ({mask.shape[1]}) does not match "
                        f"number of scales ({len(scale)}) at index {index}"
                    )
    else:
        ip_adapter_masks = [None] * len(self.scale)

    # for ip-adapter
    for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
        ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
    ):
        skip = False
        if isinstance(scale, list):
            if all(s == 0 for s in scale):
                skip = True
        elif scale == 0:
            skip = True
        if not skip:
            if mask is not None:
                if not isinstance(scale, list):
                    scale = [scale] * mask.shape[1]

                current_num_images = mask.shape[1]
                for i in range(current_num_images):
                    ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                    ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                        
                    _current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )
                    

                    _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                    mask_downsample = IPAdapterMaskProcessor.downsample(
                        mask[:, i, :, :],
                        batch_size,
                        _current_ip_hidden_states.shape[1],
                        _current_ip_hidden_states.shape[2],
                    )

                    mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                    hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
            else:
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                current_ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + scale * current_ip_hidden_states

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states


def patch_unet(
    unet, 
    target_parts, 
    am_transforms=None, 
    target_tsteps=[], 
    target_tokens=[], 
    inverse_neg_transforms=False, 
    apply_schedule=False,
    apply_adain=False
):
    cnt = 0
    assert am_transforms is not None
    
    for name, net in unet.named_modules():
        if not any(part_name in name for part_name in target_parts):
            continue
        if not name.endswith("attn2"):
            continue
        if net.processor.__class__.__name__ == "IPAdapterAttnProcessor2_0":
            net.processor.is_target_block = True
            net.processor.current_tstep = 0
            net.processor.unet_part = next(part for part in target_parts if part in name)
            net.processor.am_transforms = am_transforms
            net.processor.target_tsteps = target_tsteps
            net.processor.target_tokens = target_tokens
            net.processor.inverse_neg_transforms = inverse_neg_transforms
            net.processor.apply_schedule = apply_schedule
            net.processor.apply_adain = apply_adain
            cnt += 1

    # we can't patch __call__ since it is called from type()
    # so we just patch processor class itself (but it will trigger only in target blocks)
    IPAdapterAttnProcessor2_0.__call__ = patched_ip_attn20    
    if cnt == 0:
        print("WARNING: No IPAdapterAttnProcessor2_0 found/patched, check if load_ip_adapter was called")    
    
    return unet


def unpatch_unet(unet):
    for name, net in unet.named_modules():
        if not name.endswith("attn2"):
            continue
        if net.processor.__class__.__name__ == "IPAdapterAttnProcessor2_0":
            if hasattr(net.processor, 'is_target_block'):
                del net.processor.is_target_block
            if hasattr(net.processor, 'am_transform'):
                del net.processor.am_transform
            if hasattr(net.processor, 'unet_part'):
                del net.processor.unet_part
            if hasattr(net.processor, 'target_tsteps'):
                del net.processor.target_tsteps
            if hasattr(net.processor, 'current_tstep'):
                del net.processor.current_tstep
            if hasattr(net.processor, 'target_tokens'):
                del net.processor.target_tokens
            if hasattr(net.processor, 'inverse_neg_transforms'):
                del net.processor.inverse_neg_transforms
            if hasattr(net.processor, 'am_transforms'):
                del net.processor.am_transforms
            if hasattr(net.processor, 'apply_schedule'):
                del net.processor.apply_schedule
            if hasattr(net.processor, 'apply_adain'):
                del net.processor.apply_adain
    IPAdapterAttnProcessor2_0.__call__ = unpatched_ip_attn20
    return unet


def reset_patched_unet(pipe, output=None):
    """needs to be executed after pipeline execution"""
    unet = pipe.unet
    for name, net in unet.named_modules():
        if not name.endswith("attn2"):
            continue
        if net.processor.__class__.__name__ == "IPAdapterAttnProcessor2_0" and \
            hasattr(net.processor, 'current_tstep'):
                net.processor.current_tstep = 0 


# use this one for patching
default_transforms = [
    pow_transform,
    softmask_transform,
]
default_target_blocks = ["down", "up"]
default_target_tokens = [0, 1, 2, 3]
default_target_tsteps = [0]
default_inverse_neg_transforms = True
tnames_to_transforms = {
    "softmask": softmask_transform,
    "adaptive_softmask": adaptive_softmask_transform,
    "pow": pow_transform,
    "scale": scale_transform,
}

def patch_pipe(
    pipe, 
    target_parts, 
    target_tokens, 
    target_tsteps, 
    am_transforms, 
    inverse_neg_transforms,
    apply_schedule,
    apply_adain=False
):
    am_transforms = [tnames_to_transforms[t] for t in am_transforms]
    target_parts = target_parts or ["up", "mid", "down"]
    pipe.unet = patch_unet(pipe.unet, target_parts, am_transforms, target_tsteps, target_tokens, inverse_neg_transforms, apply_schedule, apply_adain)
    
    return pipe