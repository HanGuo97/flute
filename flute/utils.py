import math
import torch
import warnings
from typing import List, Dict, Optional
from . import packbits_utils
from . import NUM_SMS
from . import TEMPLATE_CONFIGS
from . import TEMPLATE_TUNED_WITHOUT_M_CONFIGS
from . import QGEMM_SIMPLE_DICT
from . import qgemm_simple

_WORKSPACES = {}


# 2 / 4
def make_qmap2_from_qmap(qmap: torch.Tensor) -> torch.Tensor:
    if qmap.ndim != 1:
        raise ValueError
    if qmap.dtype not in [torch.float16, torch.bfloat16]:
        raise TypeError
    qmap_size = qmap.shape[0]
    qmap2 = torch.empty(
        qmap_size,
        qmap_size,
        2,
        dtype=qmap.dtype,
        device=qmap.device)
    for i in range(qmap_size):
        for j in range(qmap_size):
            qmap2[i, j, 0] = qmap[i]
            qmap2[i, j, 1] = qmap[j]
    # the type can be any type with 32-bits,
    # since we `reinterpret_cast` it in the kernel
    return qmap2.view(dtype=torch.float32)


def make_workspace_streamk(device: torch.device) -> torch.Tensor:
    # currently, this function over-allocates the workspace for convenience
    blocks_max     = NUM_SMS * 4
    threads_max    = 256
    barrier_size   = 4
    accum_size_max = 4 * 64 * 8
    workspace_size_partials = blocks_max * threads_max * accum_size_max
    workspace_size_barriers = barrier_size * blocks_max
    return torch.zeros(workspace_size_partials + workspace_size_barriers, dtype=torch.uint8, device=device)


# this function 4/4
def get_workspace_streamk(device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        warnings.warn(f"Only CUDA devices are supported, but got: {device} ({device.type})")

    if device not in _WORKSPACES.keys():
        _WORKSPACES[device] = make_workspace_streamk(device)

    return _WORKSPACES[device]


def _pack_4bit(W: torch.Tensor, tile_P: int) -> torch.Tensor:
    num_bits = 4
    chunk_size_0 = 2
    chunk_size_1 = tile_P * 4
    chunk_size_2 = tile_P

    W_chunks = (
        W
        .view(
            int(W.shape[0]   / chunk_size_0),
            chunk_size_0,
            int(W.shape[1]   / chunk_size_1),
            int(chunk_size_1 / chunk_size_2),
            chunk_size_2,
        )
    )
    W_chunks = W_chunks.transpose(-1, -2)
    W_chunks_ = torch.zeros_like(W_chunks)
    W_chunks_[:, 0, :, :, 0] = W_chunks[:, 1, :, :, 0]
    W_chunks_[:, 0, :, :, 1] = W_chunks[:, 0, :, :, 0]
    W_chunks_[:, 0, :, :, 2] = W_chunks[:, 1, :, :, 1]
    W_chunks_[:, 0, :, :, 3] = W_chunks[:, 0, :, :, 1]
    W_chunks_[:, 1, :, :, 0] = W_chunks[:, 1, :, :, 2]
    W_chunks_[:, 1, :, :, 1] = W_chunks[:, 0, :, :, 2]
    W_chunks_[:, 1, :, :, 2] = W_chunks[:, 1, :, :, 3]
    W_chunks_[:, 1, :, :, 3] = W_chunks[:, 0, :, :, 3]
    Q = W_chunks_.reshape(W.shape)
    Q = packbits_utils.pack_integer_tensors(
        Q.to(dtype=torch.uint8),
        num_bits=num_bits)
    Q = Q.view(W.shape[0], -1)  # W.shape[1] // (16 // b)
    Q = Q.T.contiguous()
    return Q


def _pack_2bit(W: torch.Tensor, tile_P: int) -> torch.Tensor:
    num_bits = 2
    chunk_size_0 = 2
    chunk_size_1 = tile_P * 8
    chunk_size_2 = tile_P

    W_chunks = (
        W
        .view(
            int(W.shape[0]   / chunk_size_0),
            chunk_size_0,
            int(W.shape[1]   / chunk_size_1),
            int(chunk_size_1 / chunk_size_2),
            chunk_size_2,
        )
    )
    W_chunks = W_chunks.transpose(-1, -2)
    W_chunks_ = torch.zeros_like(W_chunks)
    W_chunks_[:, 0, :, :, 0] = W_chunks[:, 1, :, :, 0]
    W_chunks_[:, 0, :, :, 1] = W_chunks[:, 0, :, :, 0]
    W_chunks_[:, 0, :, :, 2] = W_chunks[:, 1, :, :, 1]
    W_chunks_[:, 0, :, :, 3] = W_chunks[:, 0, :, :, 1]
    W_chunks_[:, 0, :, :, 4] = W_chunks[:, 1, :, :, 2]
    W_chunks_[:, 0, :, :, 5] = W_chunks[:, 0, :, :, 2]
    W_chunks_[:, 0, :, :, 6] = W_chunks[:, 1, :, :, 3]
    W_chunks_[:, 0, :, :, 7] = W_chunks[:, 0, :, :, 3]
    W_chunks_[:, 1, :, :, 0] = W_chunks[:, 1, :, :, 4]
    W_chunks_[:, 1, :, :, 1] = W_chunks[:, 0, :, :, 4]
    W_chunks_[:, 1, :, :, 2] = W_chunks[:, 1, :, :, 5]
    W_chunks_[:, 1, :, :, 3] = W_chunks[:, 0, :, :, 5]
    W_chunks_[:, 1, :, :, 4] = W_chunks[:, 1, :, :, 6]
    W_chunks_[:, 1, :, :, 5] = W_chunks[:, 0, :, :, 6]
    W_chunks_[:, 1, :, :, 6] = W_chunks[:, 1, :, :, 7]
    W_chunks_[:, 1, :, :, 7] = W_chunks[:, 0, :, :, 7]
    Q = W_chunks_.reshape(W.shape)
    Q = packbits_utils.pack_integer_tensors(
        Q.to(dtype=torch.uint8),
        num_bits=num_bits)
    Q = Q.view(W.shape[0], -1)  # W.shape[1] // (16 // b)
    Q = Q.T.contiguous()
    return Q


def _pack_3bit(W: torch.Tensor, tile_P: int) -> torch.Tensor:
    if tile_P != 32:
        raise NotImplementedError

    num_bits = 3
    chunk_size_0 = 2
    chunk_size_1 = tile_P * 16
    chunk_size_2 = tile_P

    W_chunks = (
        W
        .view(
            int(W.shape[0]   / chunk_size_0),
            chunk_size_0,
            int(W.shape[1]   / chunk_size_1),
            int(chunk_size_1 / chunk_size_2),
            chunk_size_2,
        )
    )
    W_chunks = W_chunks.transpose(-1, -2)
    W_chunks_ = torch.zeros((
        int(W.shape[0]   / chunk_size_0),
        1,
        int(W.shape[1]   / chunk_size_1),
        int(chunk_size_1 / chunk_size_2) * chunk_size_0,
        chunk_size_2),
        dtype=W.dtype,
        device=W.device)

    W_chunks_[:, 0, :, :,  0] = W_chunks[:, 1, :, :,  0]
    W_chunks_[:, 0, :, :,  1] = W_chunks[:, 0, :, :,  0]
    W_chunks_[:, 0, :, :,  2] = W_chunks[:, 1, :, :,  3]
    W_chunks_[:, 0, :, :,  3] = W_chunks[:, 0, :, :,  3]
    W_chunks_[:, 0, :, :,  4] = W_chunks[:, 1, :, :,  6]
    W_chunks_[:, 0, :, :,  5] = W_chunks[:, 0, :, :,  6]
    W_chunks_[:, 0, :, :,  6] = W_chunks[:, 1, :, :,  9]
    W_chunks_[:, 0, :, :,  7] = W_chunks[:, 0, :, :,  9]
    W_chunks_[:, 0, :, :,  8] = W_chunks[:, 1, :, :, 12]
    W_chunks_[:, 0, :, :,  9] = W_chunks[:, 0, :, :, 12]
    W_chunks_[:, 0, :, :, 10] = W_chunks[:, 1, :, :,  1]
    W_chunks_[:, 0, :, :, 11] = W_chunks[:, 0, :, :,  1]
    W_chunks_[:, 0, :, :, 12] = W_chunks[:, 1, :, :,  4]
    W_chunks_[:, 0, :, :, 13] = W_chunks[:, 0, :, :,  4]
    W_chunks_[:, 0, :, :, 14] = W_chunks[:, 1, :, :,  7]
    W_chunks_[:, 0, :, :, 15] = W_chunks[:, 0, :, :,  7]
    W_chunks_[:, 0, :, :, 16] = W_chunks[:, 1, :, :, 10]
    W_chunks_[:, 0, :, :, 17] = W_chunks[:, 0, :, :, 10]
    W_chunks_[:, 0, :, :, 18] = W_chunks[:, 1, :, :, 13]
    W_chunks_[:, 0, :, :, 19] = W_chunks[:, 0, :, :, 13]
    W_chunks_[:, 0, :, :, 20] = W_chunks[:, 1, :, :,  2]
    W_chunks_[:, 0, :, :, 21] = W_chunks[:, 0, :, :,  2]
    W_chunks_[:, 0, :, :, 22] = W_chunks[:, 1, :, :,  5]
    W_chunks_[:, 0, :, :, 23] = W_chunks[:, 0, :, :,  5]
    W_chunks_[:, 0, :, :, 24] = W_chunks[:, 1, :, :,  8]
    W_chunks_[:, 0, :, :, 25] = W_chunks[:, 0, :, :,  8]
    W_chunks_[:, 0, :, :, 26] = W_chunks[:, 1, :, :, 11]
    W_chunks_[:, 0, :, :, 27] = W_chunks[:, 0, :, :, 11]
    W_chunks_[:, 0, :, :, 28] = W_chunks[:, 1, :, :, 14]
    W_chunks_[:, 0, :, :, 29] = W_chunks[:, 0, :, :, 14]
    W_chunks_[:, 0, :, :, 30] = W_chunks[:, 1, :, :, 15]
    W_chunks_[:, 0, :, :, 31] = W_chunks[:, 0, :, :, 15]

    binary_tensor = packbits_utils.to_binary(
        tensor=W_chunks_.to(dtype=torch.uint8),
        num_bits=num_bits,
        legacy=False)

    binary_tensor = binary_tensor.view(
        binary_tensor.shape[0],
        binary_tensor.shape[1],
        binary_tensor.shape[2],
        binary_tensor.shape[3],
        1,
        chunk_size_2 * num_bits)

    binary_tensor_ = torch.zeros((
        binary_tensor.shape[0],
        chunk_size_0,
        binary_tensor.shape[2],
        binary_tensor.shape[3],
        num_bits,
        packbits_utils.PackedNumBits),
        dtype=binary_tensor.dtype,
        device=binary_tensor.device)

    binary_tensor_[:, 0, :, :, 0,  0:16] = binary_tensor[:, 0, :, :, 0,  0:16]
    binary_tensor_[:, 1, :, :, 0,  0:14] = binary_tensor[:, 0, :, :, 0, 16:30]
    binary_tensor_[:, 0, :, :, 1,  0:16] = binary_tensor[:, 0, :, :, 0, 30:46]
    binary_tensor_[:, 1, :, :, 1,  0:14] = binary_tensor[:, 0, :, :, 0, 46:60]
    binary_tensor_[:, 0, :, :, 2,  0:16] = binary_tensor[:, 0, :, :, 0, 60:76]
    binary_tensor_[:, 1, :, :, 2,  0:14] = binary_tensor[:, 0, :, :, 0, 76:90]
    binary_tensor_[:, 1, :, :, 0, 14:16] = binary_tensor[:, 0, :, :, 0, 90:92]
    binary_tensor_[:, 1, :, :, 1, 14:16] = binary_tensor[:, 0, :, :, 0, 92:94]
    binary_tensor_[:, 1, :, :, 2, 14:16] = binary_tensor[:, 0, :, :, 0, 94:96]

    binary_tensor_0 = binary_tensor_[:, :, :, :, 0 , :]
    binary_tensor_1 = binary_tensor_[:, :, :, :, 1:, :].transpose(-3, -2)
    binary_tensor_0 = binary_tensor_0.reshape(-1).contiguous()
    binary_tensor_1 = binary_tensor_1.reshape(-1).contiguous()

    packed_tensor_0, padding_length_0 = packbits_utils.pack_bools_into_integers(
        tensor=binary_tensor_0,
        packed_dtype=packbits_utils.PackedDType)
    packed_tensor_1, padding_length_1 = packbits_utils.pack_bools_into_integers(
        tensor=binary_tensor_1,
        packed_dtype=packbits_utils.PackedDType)

    if padding_length_0 != 0:
        raise ValueError
    if padding_length_1 != 0:
        raise ValueError

    packed_tensor_0 = packed_tensor_0.view(W.shape[0], -1)
    packed_tensor_1 = packed_tensor_1.view(W.shape[0], -1)
    Q = torch.cat([packed_tensor_0, packed_tensor_1], dim=-1)
    Q = Q.T.contiguous()
    return Q


# (this and later) 4/4
def pack(
    W: torch.Tensor,
    num_bits: int,
    group_size: Optional[int] = None,
    template_ids: Optional[List[int]] = None,
) -> torch.Tensor:

    if W.ndim != 2:
        raise NotImplementedError

    if template_ids is None:
        if group_size is None:
            raise ValueError("Either `group_size` or `template_ids` must be provided")

        K, N = W.shape
        template_ids = []
        for dtype in [torch.float16, torch.bfloat16]:
            template_id = TEMPLATE_TUNED_WITHOUT_M_CONFIGS[(
                NUM_SMS,
                num_bits,
                group_size,
                N, K,
                str(dtype))]
            template_ids.append(template_id)

    # the packing is specialized to `tile_P`, which could
    # be different for different templates. We check that
    # all the templates have the same `tile_P`.
    tile_Ps = []
    for template_id in template_ids:
        template_config = get_template_config(
            num_bits=num_bits,
            template_id=template_id)
        tile_Ps.append(template_config["tileP"])
    if len(set(tile_Ps)) != 1:
        raise ValueError
    tile_P = tile_Ps[0]

    if num_bits == 4:
        return _pack_4bit(W, tile_P=tile_P)
    if num_bits == 2:
        return _pack_2bit(W, tile_P=tile_P)
    if num_bits == 3:
        return _pack_3bit(W, tile_P=tile_P)
    raise ValueError


def get_template_config(num_bits: int, template_id: int) -> Dict:
    config = TEMPLATE_CONFIGS[(num_bits, template_id)]
    return {
        "tileM": config["TileM"],
        "tileK": config["TileK"],
        "tileP": config["TileP"],
        "blocks": config["SMs"] * NUM_SMS,
    }


def get_template_ids(num_bits: int) -> List[int]:
    return [
        i for b, i in
        TEMPLATE_CONFIGS.keys()
        if b == num_bits]


# 4/4
def is_template_supported(
    M: int,
    N: int,
    K: int,
    num_bits: int,
    template_id: int,
) -> bool:

    template_config = get_template_config(
        num_bits=num_bits,
        template_id=template_id)

    P = math.ceil(N / 16) * num_bits
    tiles_M = math.ceil(M / template_config["tileM"])
    tiles_K = math.ceil(K / template_config["tileK"])
    tiles_P = math.ceil(P / template_config["tileP"])
    if num_bits == 3:
        tiles_P = math.ceil(math.ceil(N / 16) / template_config["tileP"])

    tiles = tiles_M * tiles_P * tiles_K
    if (tiles < template_config["blocks"]):
        return False
    return True


def reconstruct(
    weight: torch.Tensor,
    scales: torch.Tensor,
    tables: torch.Tensor,
    tables2: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    num_sms: Optional[int] = None,
) -> torch.Tensor:
    # we reconstruct the tensor using the fact that
    # `W.T = I @ W.T` and thus using the `qgemm` routine
    inputs = torch.eye(
        weight.shape[1],
        dtype=scales.dtype,
        device=scales.device)

    if num_sms is None:
        _qgemm = qgemm_simple
    else:
        _qgemm = QGEMM_SIMPLE_DICT[num_sms]

    weight_reconstructed = _qgemm(
        inputs,
        weight,
        scales,
        tables,
        tables2,
        workspace,
        num_bits,
        group_size)
    return weight_reconstructed.T


def unpack(
    weight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    group_size: int,
    num_sms_packed: Optional[int] = None,
) -> torch.Tensor:

    # the scales needs to be just ones
    scales = torch.ones_like(scales)
    # the tables need to return the original values
    tables = torch.arange(
        2 ** num_bits,
        dtype=scales.dtype,
        device=scales.device)
    tables2 = make_qmap2_from_qmap(tables)

    return reconstruct(
        weight=weight,
        scales=scales,
        tables=tables,
        tables2=tables2,
        workspace=workspace,
        num_bits=num_bits,
        group_size=group_size,
        num_sms=num_sms_packed)
