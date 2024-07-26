import sys, os
import argparse
import re
from pathlib import Path
from tqdm import tqdm

import pyopenjtalk as pjt


# コマンドライン引数の解析
def get_parser():
    parser = argparse.ArgumentParser(
        description="label preparation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_text_path", type=str, help="input text's file ().txt) path")
    parser.add_argument("out_lab_dir", type=str, help="output directory path for (.label) file")
    return parser

# labファイルをout_dirに保存する
def make_label_file(label, name, out_dir):
    new_label_file = os.path.join(out_dir, f'{name}.lab')
    with open(new_label_file, "w") as lf:
        lf.write(label)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    out_path = Path(args.out_lab_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(args.in_text_path, "r") as text_file:
        content = text_file.readlines()

        for i, text in tqdm(enumerate(content), desc="Processing", leave=False):
            # テキストファイルにはそのテキストの分類名がついてるためそれを取り出してファイル名にして保存
            texts = text.split()
            joined_text = "".join(texts[1:])
            name = texts[0]

            label_array = pjt.extract_fullcontext(joined_text)
            label = "\n".join(label_array)

            make_label_file(label, name, out_path)