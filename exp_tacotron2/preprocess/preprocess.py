import argparse
import sys
from concurrent.futures import ProcessPoolExecutor #並列処理用標準module
from pathlib import Path

import librosa
import numpy as np
from nnmnkwii.io import hts
from nnmnkwii.preprocessing import mulaw_quantize
from scipy.io import wavfile
from tqdm import tqdm
# from ttslearn.dsp import logmelspectrogram
# from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence
# from ttslearn.util import pad_1d
from dsp import logmelspectrogram
from openjtalk import pp_symbols, text_to_sequence
from util import pad_1d


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--mu", type=int, default=256, help="mu")
    return parser



def preprocess(
    wav_file,
    lab_file,
    sr,
    mu,
    in_dir,
    out_dir,
    wave_dir,
):
    assert wav_file.stem == lab_file.stem
    # Make tacotron2's input : load the fullcontextlabel & transform to 音律付き音素列 & to numerical data np_array
    labels = hts.load(lab_file) 
    PP = pp_symbols(labels.contexts) 
    in_feats = np.array(text_to_sequence(PP), dtype=np.int64)

    # Make tacotron2's output : culculate the melspectrogram
    _sr, x = wavfile.read(wav_file) #scipy・wavfile ＝ 音声読み込み
    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64) #サンプル値を正規化 ＝ hfcaptain:int32 => to float64
    x = librosa.resample(x, orig_sr=_sr, target_sr=sr)
    out_feats = logmelspectrogram(x, sr) #メルスペクトログラムを計算

    assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
    # hfcデータセットからは音素アライメントが取得できないため、固定したフレーム数を切り取る （前後20）# 冒頭： 50 ミリ秒、末尾： 100 ミリ秒
    # start_frame = int(labels.start_times[1] / 125000)
    # end_frame = int(labels.end_times[-2] / 125000)
    # start_frame = max(0, start_frame - int(0.050 / 0.0125))
    # end_frame = min(len(out_feats), end_frame + int(0.100 / 0.0125))
    start_frame = 20
    end_frame = -20
    out_feats = out_feats[start_frame:end_frame] #余分な無音部分を切り詰める

#wavenet 特徴量 : μ-law音声サンプル
    # 時間領域で音声の長さを調整
    x = x[int(start_frame * 0.0125 * sr) :]
    length = int(sr * 0.0125) * out_feats.shape[0]
    x = pad_1d(x, length) if len(x) < length else x[:length] # 無音期間が足りない場合にはpaddingして、多い場合には切り出す

    # 特徴量のアップサンプリングを行う都合上、音声波形の長さはフレームシフトで割り切れる必要があります
    assert len(x) % int(sr * 0.0125) == 0

    # mu-law量子化
    x = mulaw_quantize(x, mu)

    # save to files
    utt_id = lab_file.stem
    np.save(in_dir / f"{utt_id}-feats.npy", in_feats, allow_pickle=False)
    np.save(
        out_dir / f"{utt_id}-feats.npy",
        out_feats.astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        wave_dir / f"{utt_id}-feats.npy",
        x.astype(np.int64),
        allow_pickle=False,
    )




if __name__ == "__main__":
    # コマンドライン引数解析
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    # データ実態のpathを配列で格納
    wav_files = [Path(args.wav_root) / "all" / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_files = [Path(args.lab_root) / "all" / f"{utt_id}.lab" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_tacotron"
    out_dir = Path(args.out_dir) / "out_tacotron"
    wave_dir = Path(args.out_dir) / "out_wavenet" 
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wave_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor: #並列処理用標準 method
        futures = [
            executor.submit(
                preprocess, 
                wav_file,
                lab_file,
                args.sample_rate,
                args.mu,
                in_dir, #dump/~/org/train/in_tacotronに保存していく
                out_dir, #dump/~/org/train/out_tacotron
                wave_dir, ##dump/~/org/train/out_wavenet
            )
            for wav_file, lab_file in zip(wav_files, lab_files)
        ]
        for future in tqdm(futures):
            future.result()
