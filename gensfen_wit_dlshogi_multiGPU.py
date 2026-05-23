# =========================================================
# Ultra Stable Multi GPU Selfplay Gensfen
#
# FINAL STM VERSION
#
# DESIGN
# ---------------------------------------------------------
# score/result are BOTH STM-relative
#
# EXAMPLE
# ---------------------------------------------------------
# final winner = BLACK
#
# BLACK turn:
#   score  +500
#   result  1
#
# WHITE turn:
#   score  -500
#   result -1
#
# IMPORTANT
# ---------------------------------------------------------
# This is the NORMAL AlphaZero/dlshogi style.
#
# score:
#   current position evaluation
#
# result:
#   final game outcome
#
# Both flip with side-to-move.
#
# WEAK CLAMP
# ---------------------------------------------------------
# contradictory scores are weakened
# but NOT forcibly flipped
#
# +800 -> +200
# -600 -> -150
#
# FIXES
# ---------------------------------------------------------
# [FIXED] sign confusion
# [FIXED] NaN softmax
# [FIXED] TRT cache collision
# [FIXED] multi GPU freeze
# [FIXED] None unpack crash
# [FIXED] score=0 spam
#
# =========================================================

import argparse
import os
import gc
import random
import numpy as np

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from cshogi import (
    Board,
    PackedSfenValue,
    BLACK,
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
        "output_prefix"
    )

    parser.add_argument(
        "--model-path",
        required=True
    )

    parser.add_argument(
        "--sfen-path",
        required=True
    )

    parser.add_argument(
        "--devices",
        default="0"
    )

    parser.add_argument(
        "--total-positions",
        type=int,
        default=1000000
    )

    parser.add_argument(
        "--chunk-positions",
        type=int,
        default=100000
    )

    parser.add_argument(
        "--batch-size",
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
        "--value-scale",
        type=float,
        default=2000.0
    )

    parser.add_argument(
        "--draw-threshold",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--enable-tensorrt",
        action="store_true"
    )

    parser.add_argument(
        "--enable-cuda",
        action="store_true"
    )

    return parser.parse_args()


# =========================================================
# random sfen reader
# =========================================================

class RandomSfenReader:

    def __init__(self, path):

        self.offsets = []

        print("building sfen offsets...")

        with open(path, "rb") as f:

            while True:

                off = f.tell()

                line = f.readline()

                if not line:
                    break

                if line.strip():
                    self.offsets.append(off)

        print(
            f"loaded {len(self.offsets)} sfens"
        )

        self.fp = open(path, "rb")

    def random(self):

        while True:

            try:

                off = random.choice(
                    self.offsets
                )

                self.fp.seek(off)

                line = self.fp.readline()

                s = line.decode(
                    errors="ignore"
                ).strip()

                if s.startswith("sfen "):
                    s = s[5:]

                if s:
                    return s

            except:
                continue


# =========================================================
# sigmoid -> [-1,+1]
# =========================================================

def value_to_black(v):

    v = float(v)

    v = np.nan_to_num(v)

    v = np.clip(v, 0.0, 1.0)

    return (
        v * 2.0 - 1.0
    )


# =========================================================
# black result
# =========================================================

def black_result(vb, threshold):

    if vb > threshold:
        return 1

    if vb < -threshold:
        return -1

    return 0


# =========================================================
# STM score
# =========================================================

def stm_score(
        value_black,
        turn,
        scale
):

    if turn == BLACK:
        stm = value_black
    else:
        stm = -value_black

    return int(stm * scale)


# =========================================================
# STM result
# =========================================================

def stm_result(
        final_black_result,
        turn
):

    if turn == BLACK:
        return final_black_result
    else:
        return -final_black_result


# =========================================================
# safe softmax
# =========================================================

def softmax(x, temperature):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    x = np.nan_to_num(x)

    temperature = max(
        temperature,
        1e-6
    )

    x /= temperature

    x -= np.max(x)

    exp_x = np.exp(x)

    s = np.sum(exp_x)

    if (
        not np.isfinite(s)
        or s <= 0
    ):
        return np.ones_like(
            exp_x
        ) / len(exp_x)

    return exp_x / s


# =========================================================
# choose move
# =========================================================

def choose_move(
        board,
        logits,
        temperature
):

    moves = list(
        board.legal_moves
    )

    if len(moves) == 0:
        return None

    labels = [
        make_move_label(
            m,
            board.turn
        )
        for m in moves
    ]

    legal_logits = logits[
        labels
    ]

    probs = softmax(
        legal_logits,
        temperature
    )

    return np.random.choice(
        moves,
        p=probs
    )


# =========================================================
# create sessions
# =========================================================

def create_sessions(args):

    ids = [
        int(x.strip())
        for x in args.devices.split(",")
    ]

    sessions = []

    for device_id in ids:

        print(
            f"creating gpu {device_id}"
        )

        args.device_id = device_id

        #
        # IMPORTANT
        #
        args.trt_engine_cache_path = (
            f"trt_cache_gpu_{device_id}"
        )

        os.makedirs(
            args.trt_engine_cache_path,
            exist_ok=True
        )

        sessions.append(
            utils.create_session(args)
        )

    return sessions


# =========================================================
# parallel inference
# =========================================================

def inference_parallel(
        sessions,
        x1,
        x2
):

    batch = len(x1)

    splits = np.array_split(
        np.arange(batch),
        len(sessions)
    )

    def worker(
            gpu_id,
            idx
    ):

        if len(idx) == 0:
            return (
                idx,
                None,
                None
            )

        try:

            v, l = utils.inference(
                x1[idx],
                x2[idx],
                sessions[gpu_id]
            )

            v = np.nan_to_num(
                np.asarray(v).reshape(-1)
            )

            l = np.nan_to_num(
                np.asarray(l)
            )

            return (
                idx,
                v,
                l
            )

        except Exception as e:

            print(
                f"[gpu {gpu_id}] inference failed:",
                e
            )

            return (
                idx,
                None,
                None
            )

    results = []

    with ThreadPoolExecutor(
            max_workers=len(sessions)
    ) as ex:

        futures = []

        for gpu_id, idx in enumerate(splits):

            futures.append(
                ex.submit(
                    worker,
                    gpu_id,
                    idx
                )
            )

        for f in futures:
            results.append(
                f.result()
            )

    values = np.zeros(
        batch,
        dtype=np.float32
    )

    logits = np.zeros(
        (batch, 2187),
        dtype=np.float32
    )

    for idx, v, l in results:

        if (
            v is None
            or l is None
        ):
            continue

        values[idx] = v
        logits[idx] = l

    return values, logits


# =========================================================
# game
# =========================================================

class Game:

    def __init__(self, board):

        self.board = board

        self.history = []

        self.finished = False

        self.final_value_black = 0.0


# =========================================================
# play step
# =========================================================

def play_step(
        game,
        logits,
        value_black,
        args
):

    board = game.board

    game.final_value_black = (
        value_black
    )

    #
    # repetition
    #
    if board.is_draw():

        game.finished = True

        game.final_value_black = 0.0

        return

    #
    # mate
    #
    if board.is_game_over():

        game.finished = True

        return

    move = choose_move(
        board,
        logits,
        args.temperature
    )

    if move is None:

        game.finished = True

        return

    #
    # save PSV
    #
    ps = np.empty(
        1,
        dtype=PackedSfenValue
    )

    board.to_psfen(ps)

    ps["move"][0] = make_move_label(
        move,
        board.turn
    )

    #
    # STM score
    #
    ps["score"][0] = stm_score(
        value_black,
        board.turn,
        args.value_scale
    )

    ps["gamePly"][0] = (
        board.move_number
    )

    #
    # save turn
    #
    ps["padding"][0] = (
        board.turn
    )

    game.history.append(
        ps.copy()
    )

    #
    # play move
    #
    board.push(move)

    #
    # max ply
    #
    if (
        board.move_number
        >= args.max_ply
    ):

        game.finished = True


# =========================================================
# finalize
# =========================================================

def finalize(
        game,
        args
):

    #
    # BLACK fixed result
    #
    final_black_result = black_result(
        game.final_value_black,
        args.draw_threshold
    )

    out = []

    for ps in game.history:

        turn = ps["padding"][0]

        #
        # STM result
        #
        result = stm_result(
            final_black_result,
            turn
        )

        score = int(
            ps["score"][0]
        )

        # =================================================
        # WEAK CLAMP
        # =================================================

        if result > 0 and score < 0:

            score = int(score * 0.25)

        elif result < 0 and score > 0:

            score = int(score * 0.25)

        #
        # draw soften
        #
        if result == 0:

            score = int(score * 0.25)

        #
        # final clamp
        #
        score = max(
            -32000,
            min(32000, score)
        )

        ps["score"][0] = score

        ps["game_result"][0] = result

        out.append(ps)

    return out


# =========================================================
# generate chunk
# =========================================================

def generate_chunk(
        args,
        sessions,
        reader,
        out_file,
        chunk_size
):

    batch = args.batch_size

    x1 = np.empty(
        (batch, 62, 9, 9),
        dtype=np.float32
    )

    x2 = np.empty(
        (batch, 57, 9, 9),
        dtype=np.float32
    )

    generated = 0

    with open(out_file, "wb") as f:

        with tqdm(
                total=chunk_size
        ) as pbar:

            while generated < chunk_size:

                games = []

                #
                # init games
                #
                for _ in range(batch):

                    while True:

                        try:

                            b = Board()

                            b.set_sfen(
                                reader.random()
                            )

                            games.append(
                                Game(b)
                            )

                            break

                        except:
                            continue

                #
                # selfplay
                #
                while True:

                    active = [
                        g for g in games
                        if not g.finished
                    ]

                    if len(active) == 0:
                        break

                    #
                    # features
                    #
                    for i, g in enumerate(active):

                        make_input_features(
                            g.board,
                            x1[i],
                            x2[i]
                        )

                    #
                    # inference
                    #
                    vals, logs = (
                        inference_parallel(
                            sessions,
                            x1[:len(active)],
                            x2[:len(active)]
                        )
                    )

                    #
                    # play
                    #
                    for i, g in enumerate(active):

                        vb = value_to_black(
                            vals[i]
                        )

                        play_step(
                            g,
                            logs[i],
                            vb,
                            args
                        )

                #
                # save
                #
                for g in games:

                    out = finalize(
                        g,
                        args
                    )

                    for ps in out:
                        ps.tofile(f)

                    n = len(out)

                    generated += n

                    pbar.update(n)

                gc.collect()

    gc.collect()


# =========================================================
# main
# =========================================================

def main():

    args = parse_args()

    print("--------------------------------")
    print("FINAL STM SELFPLAY")
    print("--------------------------------")

    print(
        f"devices: {args.devices}"
    )

    print(
        f"total positions: {args.total_positions}"
    )

    print(
        f"chunk positions: {args.chunk_positions}"
    )

    sessions = create_sessions(
        args
    )

    reader = RandomSfenReader(
        args.sfen_path
    )

    chunks = (
        args.total_positions
        + args.chunk_positions
        - 1
    ) // args.chunk_positions

    for i in range(chunks):

        remain = (
            args.total_positions
            - i * args.chunk_positions
        )

        chunk_size = min(
            remain,
            args.chunk_positions
        )

        out_file = (
            f"{args.output_prefix}_"
            f"{i:04d}.bin"
        )

        print(
            f"\nchunk {i+1}/{chunks}"
        )

        generate_chunk(
            args,
            sessions,
            reader,
            out_file,
            chunk_size
        )

        gc.collect()

    print("\nDONE")


if __name__ == "__main__":
    main()