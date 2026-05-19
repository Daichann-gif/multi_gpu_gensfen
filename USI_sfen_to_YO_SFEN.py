import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from cshogi import Board, move_to_usi
from cshogi.dlshogi import make_input_features

MODEL_PATH = "eval/model.onnx"

BATCH_SIZE = 128
ALPHA = 0.3


# =========================
# TensorRT session
# =========================
def create_session():
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_fp16_enable": True,
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print("Providers:", session.get_providers())
    return session


# =========================
# inference
# =========================
def infer(session, f1, f2):
    inputs = {
        session.get_inputs()[0].name: f1,
        session.get_inputs()[1].name: f2
    }
    policy, score = session.run(None, inputs)
    return policy, score


# =========================
# feature buffer
# =========================
def alloc(batch):
    f1 = np.zeros((batch, 62, 9, 9), dtype=np.float32)
    f2 = np.zeros((batch, 57, 9, 9), dtype=np.float32)
    return f1, f2


# =========================
# mate-aware score conversion
# =========================
def convert_score_with_mate(v):
    # mate扱い（簡易検出）
    if v > 3000:
        return 5000 + min(5000, int(v))
    if v < -3000:
        return -5000 + max(-5000, int(v))

    # 通常評価
    return int(max(-5000, min(5000, v * 5000)))


# =========================
# move selection
# =========================
def select_move(board, session, f1_buf, f2_buf):
    moves = list(board.legal_moves)

    if not moves:
        return None

    f1_list = []
    f2_list = []

    for m in moves:
        board.push(m)
        make_input_features(board, f1_buf[0], f2_buf[0])

        f1_list.append(f1_buf[0].copy())
        f2_list.append(f2_buf[0].copy())

        board.pop()

    f1_batch = np.array(f1_list, dtype=np.float32)
    f2_batch = np.array(f2_list, dtype=np.float32)

    policy, score = infer(session, f1_batch, f2_batch)

    best_score = -1e9
    best_move = None

    for i, m in enumerate(moves):
        s = policy[i].max() + ALPHA * score[i][0]

        if s > best_score:
            best_score = s
            best_move = m

    return best_move


# =========================
# convert (teacher generator)
# =========================
def convert(input_file, output_file):
    session = create_session()
    f1, f2 = alloc(BATCH_SIZE)
    board = Board()

    with open(input_file) as f:
        sfens = [x.strip() for x in f if x.strip()]

    with open(output_file, "w") as out, tqdm(total=len(sfens)) as bar:

        for sfen in sfens:
            board.set_sfen(sfen)

            move = select_move(board, session, f1, f2)

            if move is None:
                bar.update(1)
                continue

            # =========================
            # ROOT evaluation（重要）
            # =========================
            make_input_features(board, f1[0], f2[0])
            _, score = infer(session, f1[None, 0], f2[None, 0])
            value = float(score[0][0])

            int_score = convert_score_with_mate(value)

            # =========================
            # result
            # =========================
            if value > 0.3:
                result = 1
            elif value < -0.3:
                result = -1
            else:
                result = 0

            # =========================
            # ply固定
            # =========================
            ply = 512

            usi = move_to_usi(move)

            # =========================
            # output
            # =========================
            out.write(
                f"sfen {sfen}\n"
                f"move {usi}\n"
                f"score {int_score}\n"
                f"ply {ply}\n"
                f"result {result}\n"
                f"e\n"
            )

            bar.update(1)

    print("完了")


if __name__ == "__main__":
    convert("input.txt", "output.txt")