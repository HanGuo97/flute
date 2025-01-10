LLAMA3_8B_SHAPES = [
    (1024 , 4096 ),
    (4096 , 4096 ),
    (4096 , 14336),
    (6144 , 4096 ),
    (14336, 4096 ),
]

LLAMA3_70B_SHAPES = [
    (1024 , 8192 ),
    (8192 , 8192 ),
    (8192 , 28672),
    (10240, 8192 ),
    (28672, 8192 ),
]

LLAMA3_70B_SHAPES_TP2 = [
    (5120 , 8192 ),
    (8192 , 4096 ),
    (8192 , 14336),
    (14336, 8192 ),
]

LLAMA3_70B_SHAPES_TP4 = [
    (2560 , 8192 ),
    (7168 , 8192 ),
    (8192 , 2048 ),
    (8192 , 7168 ),
]

LLAMA3_405B_SHAPES = [
    (2048 , 16384),
    (2560 , 16384),
    (5120 , 16384),
    (16384, 2048 ),
    (16384, 4096 ),
    (16384, 6656 ),
    (16384, 16384),
    (16384, 53248),
    (16384, 13312),
    (53248, 16384),
    (20480, 16384),
    (26624, 16384),
    (13312, 16384),
    (106496, 16384),
]

LLAMA3_EXTRA_SHAPES_VLLM = [
    (28672, 4096 ),
    (57344, 8192 ),
]

GEMMA2_9B_SHAPES = [
    (2048 , 3584 ),
    (3584 , 4096 ),
    (3584 , 14336),
    (4096 , 3584 ),
    (14336, 3584 ),
    (8192 , 3584 ),
    (28672, 3584 ),
]

GEMMA2_27B_SHAPES = [
    (2048 , 4608 ),
    (4096 , 4608 ),
    (4608 , 4096 ),
    (4608 , 36864),
    (36864, 4608 ),
    (8192 , 4608 ),
    (73728, 4608 ),
    (4608 , 2048 ),
    (4608 , 18432),
    (4608 , 1024 ),
    (4608 , 9216 ),
    (18432, 4608 ),
]


LLAMA3_SHAPES = (
    LLAMA3_8B_SHAPES +
    LLAMA3_70B_SHAPES +
    LLAMA3_70B_SHAPES_TP2 +
    LLAMA3_70B_SHAPES_TP4 +
    LLAMA3_405B_SHAPES +
    LLAMA3_EXTRA_SHAPES_VLLM)

GEMMA2_SHAPES = (
    GEMMA2_9B_SHAPES +
    GEMMA2_27B_SHAPES)

SUPPORTED_SHAPES = (
    LLAMA3_SHAPES +
    GEMMA2_SHAPES)

if len(SUPPORTED_SHAPES) != len(set(SUPPORTED_SHAPES)):
    raise ValueError("Duplicate shapes in SUPPORTED_SHAPES")
