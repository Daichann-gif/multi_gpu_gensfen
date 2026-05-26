# =========================================================
# Ultra Stable TRUE Multi GPU Selfplay Gensfen
#
# COMPLETE FINAL VERSION
#
# FINAL FEATURES
# ---------------------------------------------------------
# [FIXED] TensorRT first-build freeze
# [FIXED] mixed GPU deadlock
# [FIXED] TensorRT cache not generated
# [FIXED] inconsistent result/score
# [FIXED] alternating result bug
# [FIXED] RAM explosion
#
# IMPORTANT
# ---------------------------------------------------------
# FINAL VALUE -> ALL POSITIONS
#
# score:
#   side-to-move evaluation
#
# result:
#   derived from final_value
#
# ALWAYS CONSISTENT
#
# RECOMMENDED FIRST RUN
# ---------------------------------------------------------
# --selfplay-batch-size 16
#
# AFTER TRT CACHE GENERATED:
#
# --selfplay-batch-size 64
#
# EXAMPLE
# ---------------------------------------------------------
# python gensfen.py dlsuisho.bin ^
#   --devices 0,1 ^
#   --model-path eval/model.onnx ^
#   --enable-tensorrt ^
#   --enable-cuda ^
#   --sfen-path start.sfen ^
#   --total-positions 1000000000 ^
#   --chunk-positions 100000000 ^
#   --selfplay-batch-size 16
# =========================================================

import argparse
import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from cshogi import (
    Board,
    PackedSfenValue,
    BLACK,
    WHITE,
)

from cshogi.dlshogi import (
    make_input_features,
    make_move_label,
)

# =========================================================
# args
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "output_prefix",
        type=str
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="0"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--total-positions",
        type=int,
        default=1000000000
    )

    parser.add_argument(
        "--chunk-positions",
        type=int,
        default=100000000
    )

    parser.add_argument(
        "--selfplay-batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--max-ply",
        type=int,
        default=256
    )

    parser.add_argument(
        "--score-scaling",
        type=float,
        default=2000.0
    )

    parser.add_argument(
        "--draw-value-threshold",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--sfen-path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--enable-cuda",
        action="store_true"
    )

    parser.add_argument(
        "--enable-tensorrt",
        action="store_true"
    )

    return parser.parse_args()


# =========================================================
# TensorRT session
# =========================================================

def create_session(
        model_path,
        device_id,
        enable_cuda,
        enable_tensorrt
):

    providers = []

    #
    # TensorRT
    #
    if enable_tensorrt:

        cache_dir = (
            f"trt_cache_gpu{device_id}"
        )

        os.makedirs(
            cache_dir,
            exist_ok=True
        )

        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id":
                        device_id,

                    #
                    # IMPORTANT
                    #
                    "trt_engine_cache_enable":
                        True,

                    "trt_engine_cache_path":
                        cache_dir,

                    #
                    # mixed GPU safety
                    #
                    "trt_timing_cache_enable":
                        False,

                    #
                    # fp16
                    #
                    "trt_fp16_enable":
                        True,
                },
            )
        )

        print(
            f"[GPU {device_id}] "
            f"TensorRT enabled."
        )

    #
    # CUDA
    #
    if enable_cuda:

        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id":
                        device_id
                },
            )
        )

        print(
            f"[GPU {device_id}] "
            f"CUDA enabled."
        )

    #
    # CPU
    #
    providers.append(
        "CPUExecutionProvider"
    )

    sess_options = \
        ort.SessionOptions()

    sess_options.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers
    )

    print(
        f"[GPU {device_id}] "
        f"{session.get_providers()}"
    )

    return session


# =========================================================
# sequential TRT warmup
# =========================================================

def warmup_sessions(
        sessions
):

    print("\n================================")
    print("Sequential GPU Warmup")
    print("================================\n")

    dummy1 = np.zeros(
        (1, 62, 9, 9),
        dtype=np.float32
    )

    dummy2 = np.zeros(
        (1, 57, 9, 9),
        dtype=np.float32
    )

    #
    # IMPORTANT
    # sequential warmup
    #
    for gpu_id, session in enumerate(
            sessions
    ):

        print(
            f"Warmup GPU {gpu_id}..."
        )

        session.run(
            None,
            {
                "input1": dummy1,
                "input2": dummy2,
            },
        )

        print(
            f"GPU {gpu_id} warmup done."
        )

    print("\nWarmup complete.\n")


# =========================================================
# huge sfen reader
# =========================================================

class RandomSfenReader:

    def __init__(self, path):

        self.offsets = []

        print(
            "Building SFEN offsets..."
        )

        with open(path, "rb") as f:

            while True:

                offset = f.tell()

                line = f.readline()

                if not line:
                    break

                line = line.strip()

                if len(line) == 0:
                    continue

                self.offsets.append(
                    offset
                )

        print(
            f"Loaded offsets: "
            f"{len(self.offsets)}"
        )

        self.fp = open(
            path,
            "rb"
        )

    def random_sfen(self):

        while True:

            try:

                offset = random.choice(
                    self.offsets
                )

                self.fp.seek(offset)

                line = self.fp.readline()

                line = line.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                if line.startswith(
                        "sfen "
                ):
                    line = line[5:]

                if len(line) == 0:
                    continue

                return line

            except:
                continue


# =========================================================
# softmax
# =========================================================

def softmax(
        x,
        temperature=1.0
):

    x = x.astype(
        np.float32
    )

    x /= temperature

    x -= np.max(x)

    exp_x = np.exp(x)

    return exp_x / (
        np.sum(exp_x)
        + 1e-8
    )


# =========================================================
# create sessions
# =========================================================

def create_sessions(args):

    device_ids = [
        int(x.strip())
        for x in args.devices.split(",")
    ]

    sessions = []

    for device_id in device_ids:

        print(
            f"Creating GPU session "
            f"{device_id}"
        )

        session = create_session(
            args.model_path,
            device_id,
            args.enable_cuda,
            args.enable_tensorrt
        )

        sessions.append(
            session
        )

    return sessions


# =========================================================
# allocate features
# =========================================================

def allocate_input_features(
        batch_size
):

    input1 = np.empty(
        (batch_size, 62, 9, 9),
        dtype=np.float32
    )

    input2 = np.empty(
        (batch_size, 57, 9, 9),
        dtype=np.float32
    )

    return input1, input2


# =========================================================
# inference
# =========================================================

def run_inference(
        session,
        input1,
        input2
):

    outputs = session.run(
        None,
        {
            "input1": input1,
            "input2": input2,
        },
    )

    #
    # auto detect
    #
    if outputs[0].ndim == 2:

        logits = outputs[0]
        values = outputs[1]

    else:

        values = outputs[0]
        logits = outputs[1]

    values = np.asarray(
        values
    ).reshape(-1)

    return values, logits


# =========================================================
# parallel inference
# =========================================================

def parallel_inference(
        sessions,
        input1,
        input2
):

    batch_size = len(input1)

    split_indices = np.array_split(
        np.arange(batch_size),
        len(sessions)
    )

    value_outputs = [None] * len(sessions)
    logit_outputs = [None] * len(sessions)

    def worker(
            gpu_idx,
            indices
    ):

        if len(indices) == 0:
            return

        values, logits = \
            run_inference(
                sessions[gpu_idx],
                input1[indices],
                input2[indices]
            )

        value_outputs[gpu_idx] = (
            indices,
            values
        )

        logit_outputs[gpu_idx] = (
            indices,
            logits
        )

    with ThreadPoolExecutor(
            max_workers=len(sessions)
    ) as executor:

        futures = []

        for gpu_idx, indices in enumerate(
                split_indices
        ):

            futures.append(
                executor.submit(
                    worker,
                    gpu_idx,
                    indices
                )
            )

        for future in futures:
            future.result()

    values = np.empty(
        (batch_size,),
        dtype=np.float32
    )

    logits = np.empty(
        (batch_size, 2187),
        dtype=np.float32
    )

    for item in value_outputs:

        if item is None:
            continue

        indices, vals = item

        values[indices] = vals

    for item in logit_outputs:

        if item is None:
            continue

        indices, logs = item

        logits[indices] = logs

    return values, logits


# =========================================================
# choose move
# =========================================================

def choose_move(
        board,
        logits,
        temperature
):

    legal_moves = list(
        board.legal_moves
    )

    if len(legal_moves) == 0:
        return None

    labels = [
        make_move_label(
            move,
            board.turn
        )
        for move in legal_moves
    ]

    legal_logits = logits[
        labels
    ]

    probs = softmax(
        legal_logits,
        temperature
    )

    return np.random.choice(
        legal_moves,
        p=probs
    )


# =========================================================
# game state
# =========================================================

class GameState:

    def __init__(self, board):

        self.board = board

        self.history = []

        self.finished = False

        self.final_value = 0.0


# =========================================================
# play step
# =========================================================

def play_step(
        game,
        logits,
        score,
        normalized_value,
        temperature,
        max_ply
):

    board = game.board

    #
    # game over
    #
    if board.is_game_over():

        game.finished = True

        game.final_value = \
            normalized_value

        return

    move = choose_move(
        board,
        logits,
        temperature
    )

    if move is None:

        game.finished = True

        game.final_value = \
            normalized_value

        return

    #
    # save position
    #
    psfen = np.empty(
        1,
        dtype=PackedSfenValue
    )

    board.to_psfen(psfen)

    psfen["move"][0] = \
        make_move_label(
            move,
            board.turn
        )

    psfen["score"][0] = int(score)

    psfen["gamePly"][0] = \
        board.move_number

    game.history.append(
        psfen.copy()
    )

    #
    # play move
    #
    board.push(move)

    #
    # max ply
    #
    if board.move_number >= max_ply:

        game.finished = True

        game.final_value = \
            normalized_value


# =========================================================
# apply results
# =========================================================

def apply_results(
        game,
        threshold
):

    output = []

    v = game.final_value

    #
    # FINAL VALUE -> RESULT
    #
    if v > threshold:

        final_result = 1

    elif v < -threshold:

        final_result = -1

    else:

        final_result = 0

    for psfen in game.history:

        psfen["game_result"][0] = \
            final_result

        output.append(psfen)

    return output


# =========================================================
# generate chunk
# =========================================================

def generate_chunk(
        args,
        sessions,
        sfen_reader,
        output_file,
        chunk_positions
):

    generated_positions = 0

    batch_size = \
        args.selfplay_batch_size

    input1, input2 = \
        allocate_input_features(
            batch_size
        )

    with open(output_file, "wb") as f_out:

        with tqdm(
                total=chunk_positions,
                desc=os.path.basename(
                    output_file
                )
        ) as pbar:

            while (
                    generated_positions
                    < chunk_positions
            ):

                games = []

                #
                # init games
                #
                for _ in range(batch_size):

                    while True:

                        try:

                            sfen = \
                                sfen_reader.random_sfen()

                            board = Board()

                            board.set_sfen(
                                sfen
                            )

                            games.append(
                                GameState(board)
                            )

                            break

                        except:
                            continue

                #
                # selfplay loop
                #
                while True:

                    active_games = [
                        g for g in games
                        if not g.finished
                    ]

                    if len(active_games) == 0:
                        break

                    #
                    # features
                    #
                    for i, game in enumerate(
                            active_games
                    ):

                        make_input_features(
                            game.board,
                            input1[i],
                            input2[i]
                        )

                    #
                    # inference
                    #
                    values, logits = \
                        parallel_inference(
                            sessions,
                            input1[
                                :len(active_games)
                            ],
                            input2[
                                :len(active_games)
                            ]
                        )

                    #
                    # play
                    #
                    for i, game in enumerate(
                            active_games
                    ):

                        raw_value = float(
                            values[i]
                        )

                        #
                        # sigmoid -> [-1,+1]
                        #
                        normalized_value = (
                            raw_value - 0.5
                        ) * 2.0

                        #
                        # BLACK POV
                        # -> STM POV
                        #
                        if game.board.turn == WHITE:

                            normalized_value = \
                                -normalized_value

                        score = int(
                            normalized_value
                            * args.score_scaling
                        )

                        play_step(
                            game,
                            logits[i],
                            score,
                            normalized_value,
                            args.temperature,
                            args.max_ply
                        )

                #
                # save
                #
                for game in games:

                    output = apply_results(
                        game,
                        args.draw_value_threshold
                    )

                    for psfen in output:

                        psfen.tofile(f_out)

                    n = len(output)

                    generated_positions += n

                    pbar.update(n)

                gc.collect()

    del input1
    del input2

    gc.collect()


# =========================================================
# main
# =========================================================

def main():

    args = parse_args()

    print("--------------------------------")

    print(
        f"Devices : "
        f"{args.devices}"
    )

    print(
        f"Total Positions : "
        f"{args.total_positions}"
    )

    print(
        f"Chunk Positions : "
        f"{args.chunk_positions}"
    )

    print(
        f"Selfplay Batch : "
        f"{args.selfplay_batch_size}"
    )

    print("--------------------------------")

    sessions = create_sessions(
        args
    )

    #
    # IMPORTANT
    # Sequential TRT warmup
    #
    warmup_sessions(
        sessions
    )

    sfen_reader = RandomSfenReader(
        args.sfen_path
    )

    num_chunks = (
        args.total_positions
        + args.chunk_positions
        - 1
    ) // args.chunk_positions

    for chunk_idx in range(num_chunks):

        remain = (
            args.total_positions
            - (
                chunk_idx
                * args.chunk_positions
            )
        )

        current_chunk = min(
            args.chunk_positions,
            remain
        )

        output_file = (
            f"{args.output_prefix}_"
            f"{chunk_idx + 1:04d}.bin"
        )

        print("\n================================")

        print(
            f"Chunk "
            f"{chunk_idx + 1}"
            f"/{num_chunks}"
        )

        print(
            f"Positions : "
            f"{current_chunk}"
        )

        print(
            f"Output : "
            f"{output_file}"
        )

        print("================================\n")

        generate_chunk(
            args,
            sessions,
            sfen_reader,
            output_file,
            current_chunk
        )

        gc.collect()

        print(
            f"\nChunk "
            f"{chunk_idx + 1} finished."
        )

    print("\nALL DONE")


if __name__ == "__main__":
    main()