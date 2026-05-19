# =========================================================
# Ultra Stable Multi GPU Gensfen
#
# FIXED VERSION
#
# Fixes:
# ---------------------------------------------------------
# - NO infinite buffer growth
# - NO RAM explosion
# - Fixed-size search tree
# - Seed recycling
# - Chunk memory reset
# - Multi GPU
# - TensorRT / CUDA
# - Billion-scale stable generation
# =========================================================

import argparse
import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from cshogi import (
    Board,
    PackedSfenValue,
)

from cshogi.dlshogi import (
    make_input_features,
    make_move_label,
)

import utils


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
        "--seed-size",
        type=int,
        default=1000000
    )

    parser.add_argument(
        "--sfen-path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--policy-moves",
        type=int,
        default=2
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768
    )

    #
    # IMPORTANT
    # fixed maximum tree size
    #
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2000000
    )

    parser.add_argument(
        "--score-scaling",
        type=float,
        default=600.0
    )

    parser = utils.configure_session_args(
        parser
    )

    return parser.parse_args()


# =========================================================
# duplicate checker
# =========================================================

class DuplicateChecker:

    def __init__(self):

        self.hashes = set()

    def mark(self, h):

        self.hashes.add(h)

    def check(self, h):

        return h in self.hashes


# =========================================================
# FIXED SIZE BUFFER
# =========================================================

class BatchBuffer:

    def __init__(
            self,
            capacity,
            batch_size,
            dtype
    ):

        self.capacity = max(
            capacity,
            batch_size
        )

        self.batch_size = batch_size

        self.dtype = dtype

        #
        # FIXED MEMORY
        #
        self.data = np.empty(
            self.capacity,
            dtype=dtype
        )

        self.size = 0

    def push(self, arr):

        if len(arr) == 0:
            return

        #
        # IMPORTANT
        # NO MEMORY EXPANSION
        #
        free_space = (
            len(self.data)
            - self.size
        )

        if free_space <= 0:
            return

        #
        # random truncation
        #
        if len(arr) > free_space:

            indices = np.random.choice(
                len(arr),
                size=free_space,
                replace=False
            )

            arr = arr[indices]

        self.data[
            self.size:
            self.size + len(arr)
        ] = arr

        self.size += len(arr)

    def pop(self):

        if self.size == 0:

            return np.empty(
                0,
                dtype=self.dtype
            )

        n = min(
            self.batch_size,
            self.size
        )

        result = self.data[
            self.size - n:
            self.size
        ].copy()

        self.size -= n

        return result

    def empty(self):

        return self.size == 0


# =========================================================
# softmax
# =========================================================

def softmax(
        x,
        temperature=1.0
):

    x = x.astype(np.float32)

    x /= temperature

    x -= np.max(x)

    exp_x = np.exp(x)

    return exp_x / (
        np.sum(exp_x) + 1e-8
    )


# =========================================================
# features
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
# sessions
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

        args.device_id = device_id

        session = utils.create_session(
            args
        )

        sessions.append(session)

    return sessions


# =========================================================
# inference
# =========================================================

def parallel_inference(
        sessions,
        input1,
        input2
):

    split_indices = np.array_split(
        np.arange(len(input1)),
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

        values, logits = utils.inference(
            input1[indices],
            input2[indices],
            sessions[gpu_idx]
        )

        value_outputs[gpu_idx] = values
        logit_outputs[gpu_idx] = logits

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

    batch_values = np.concatenate(
        [
            x for x in value_outputs
            if x is not None
        ],
        axis=0
    )

    batch_logits = np.concatenate(
        [
            x for x in logit_outputs
            if x is not None
        ],
        axis=0
    )

    return batch_values, batch_logits


# =========================================================
# load sfens
# =========================================================

def load_initial_sfens(path):

    if path is None:

        return [
            "lnsgkgsnl/1r5b1/"
            "ppppppppp/9/9/9/"
            "PPPPPPPPP/1B5R1/"
            "LNSGKGSNL b - 1"
        ]

    sfens = []

    with open(
            path,
            "r",
            encoding="utf-8-sig"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("sfen "):
                line = line[5:]

            sfens.append(line)

    return sfens


# =========================================================
# convert
# =========================================================

def convert_to_psfens(sfens):

    board = Board()

    psfens = np.empty(
        len(sfens),
        dtype=PackedSfenValue
    )

    for i, sfen in enumerate(sfens):

        board.set_sfen(sfen)

        board.to_psfen(
            psfens[i:i + 1]
        )

        psfens["gamePly"][i] = 1

    return psfens


# =========================================================
# generate chunk
# =========================================================

def generate_chunk(
        args,
        sessions,
        output_file,
        chunk_positions,
        seed_sfens
):

    board = Board()

    input1, input2 = \
        allocate_input_features(
            args.batch_size
        )

    generated = np.empty(
        args.batch_size
        * args.policy_moves,
        dtype=PackedSfenValue
    )

    duplicate_checker = \
        DuplicateChecker()

    sfens_buffer = BatchBuffer(
        args.buffer_size,
        args.batch_size,
        dtype=PackedSfenValue
    )

    #
    # initial tree
    #
    sfens_buffer.push(
        seed_sfens
    )

    generated_positions = 0

    #
    # fixed seed pool
    #
    next_seed_pool = []

    next_seed_count = 0

    with open(output_file, "wb") as f_out:

        with tqdm(
                total=chunk_positions,
                desc=os.path.basename(
                    output_file
                )
        ) as bar:

            while (
                    generated_positions
                    < chunk_positions
            ):

                if sfens_buffer.empty():

                    print(
                        "Buffer empty"
                    )

                    break

                sfens = sfens_buffer.pop()

                #
                # features
                #
                for i, sfen in enumerate(
                        sfens["sfen"]
                ):

                    board.set_psfen(sfen)

                    make_input_features(
                        board,
                        input1[i],
                        input2[i]
                    )

                #
                # inference
                #
                batch_values, batch_logits = \
                    parallel_inference(
                        sessions,
                        input1[:len(sfens)],
                        input2[:len(sfens)]
                    )

                scores = (
                    batch_values.flatten()
                    * args.score_scaling
                )

                pos_count = 0

                #
                # expand
                #
                for i, logits in enumerate(
                        batch_logits
                ):

                    board.set_psfen(
                        sfens["sfen"][i]
                    )

                    duplicate_checker.mark(
                        board.zobrist_hash()
                    )

                    legal_moves = list(
                        board.legal_moves
                    )

                    if len(legal_moves) == 0:
                        continue

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
                        args.temperature
                    )

                    sampled_moves = \
                        np.random.choice(
                            legal_moves,
                            size=min(
                                args.policy_moves,
                                len(legal_moves)
                            ),
                            replace=False,
                            p=probs
                        )

                    for move in sampled_moves:

                        board.push(move)

                        h = board.zobrist_hash()

                        if not duplicate_checker.check(h):

                            board.to_psfen(
                                generated[
                                    pos_count:
                                    pos_count + 1
                                ]
                            )

                            generated["score"][
                                pos_count
                            ] = int(scores[i])

                            generated["gamePly"][
                                pos_count
                            ] = (
                                sfens["gamePly"][i]
                                + 1
                            )

                            generated["padding"][
                                pos_count
                            ] = 1

                            pos_count += 1

                        board.pop()

                #
                # no generated
                #
                if pos_count == 0:
                    continue

                #
                # write
                #
                sfens.tofile(f_out)

                #
                # next tree
                #
                new_sfens = generated[
                    :pos_count
                ].copy()

                #
                # randomize tree
                #
                np.random.shuffle(
                    new_sfens
                )

                #
                # IMPORTANT
                # fixed search tree size
                #
                sfens_buffer.push(
                    new_sfens
                )

                #
                # LIMITED seed recycling
                #
                if (
                        next_seed_count
                        < args.seed_size
                ):

                    take = min(
                        10000,
                        len(new_sfens),
                        args.seed_size
                        - next_seed_count
                    )

                    next_seed_pool.append(
                        new_sfens[:take]
                    )

                    next_seed_count += take

                generated_positions += \
                    len(sfens)

                bar.update(
                    len(sfens)
                )

    #
    # next chunk seeds
    #
    if len(next_seed_pool) == 0:

        next_seed_sfens = seed_sfens

    else:

        next_seed_sfens = np.concatenate(
            next_seed_pool,
            axis=0
        )

    #
    # cleanup
    #
    del sfens_buffer
    del duplicate_checker
    del generated
    del input1
    del input2
    del next_seed_pool

    gc.collect()

    return next_seed_sfens


# =========================================================
# main
# =========================================================

def main():

    args = parse_args()

    print("--------------------------------")

    print(
        f"Devices            : "
        f"{args.devices}"
    )

    print(
        f"Total Positions    : "
        f"{args.total_positions}"
    )

    print(
        f"Chunk Positions    : "
        f"{args.chunk_positions}"
    )

    print(
        f"Buffer Size        : "
        f"{args.buffer_size}"
    )

    print(
        f"Seed Size          : "
        f"{args.seed_size}"
    )

    print("--------------------------------")

    #
    # load once
    #
    initial_sfens = load_initial_sfens(
        args.sfen_path
    )

    seed_sfens = convert_to_psfens(
        initial_sfens
    )

    #
    # chunk count
    #
    num_chunks = (
        args.total_positions
        + args.chunk_positions
        - 1
    ) // args.chunk_positions

    for chunk_idx in range(num_chunks):

        sessions = create_sessions(
            args
        )

        start_pos = (
            chunk_idx
            * args.chunk_positions
        )

        remain = (
            args.total_positions
            - start_pos
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
            f"Output : "
            f"{output_file}"
        )

        print(
            f"Positions : "
            f"{current_chunk}"
        )

        print("================================\n")

        #
        # generate
        #
        seed_sfens = generate_chunk(
            args,
            sessions,
            output_file,
            current_chunk,
            seed_sfens
        )

        #
        # cleanup TensorRT
        #
        del sessions

        gc.collect()

        print(
            f"\nChunk "
            f"{chunk_idx + 1} "
            f"finished."
        )

        print(
            f"Next seed size : "
            f"{len(seed_sfens)}"
        )

        print(
            "Memory cleaned.\n"
        )

    print("\nALL DONE")


if __name__ == "__main__":
    main()