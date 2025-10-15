import os
import sys
import csv
import argparse

def main(args):
    input_dir = args.input_dir
    output_path = args.output

    # 规范输出：若 output 是目录或不以 .csv 结尾，则写入 merged.csv
    if os.path.isdir(output_path) or not output_path.lower().endswith(".csv"):
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "merged.csv")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_file = output_path

    # 收集 input_dir 下的所有 csv 文件（仅一层，不递归）
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    csv_files = [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.lower().endswith(".csv") and os.path.isfile(os.path.join(input_dir, name))
    ]

    if not csv_files:
        # 若没有可用 CSV，创建空文件并返回
        with open(output_file, "w", newline="", encoding="utf-8"):
            pass
        return

    wrote_header = False
    expected_header = None

    with open(output_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)

        for path in csv_files:
            with open(path, "r", newline="", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                try:
                    header = next(reader)
                except StopIteration:
                    # 空文件
                    continue

                if not wrote_header:
                    writer.writerow(header)
                    expected_header = header
                    wrote_header = True
                else:
                    # 头部不一致则跳过该文件
                    if header != expected_header:
                        continue

                # 追加数据行
                for row in reader:
                    if not row:
                        continue
                    writer.writerow(row)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #set simulation start and end cycle
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results_process")

    args = parser.parse_args()

    main(args)
