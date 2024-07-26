#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () { #処理を出力でトレースする関数
    set -x
    $@
    set +x
}

# get this directory's path
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

COMMON_ROOT=./common
. $COMMON_ROOT/yaml_parser.sh || exit 1;

eval $(parse_yaml "./exe.yaml" "")

train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

stage=0
stop_stage=0

. $COMMON_ROOT/parse_options.sh || exit 1;

# set the directory of saving data during the examination
dumpdir=dump
dump_org_dir=$dumpdir/${spk}_sr${sample_rate}/org
dump_norm_dir=$dumpdir/${spk}_sr${sample_rate}/norm

# experiment name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname





# skip the stage of downdoading the data
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"
    mkdir -p downloads
    #downloadsディレクトリに保存していく
    if [ ! -d downloads/hi-fi-captain ]; then 
        cd downloads
        curl -LO https://ast-astrec.nict.go.jp/release/hi-fi-captain/hfc_ja-JP_M.zip
        unzip -o hfc_ja-JP_M.zip
        cd -
    fi
fi





# stage 0 ： separete data for appropriate exp & make full-context-label from text-label 
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Delete_&_Divide Data & Preparation of the full-context-label"

    wav_path="./downloads/hi-fi-captain/ja-JP/male/wav"
    text_path="./downloads/hi-fi-captain/ja-JP/male/text"
    if [ -d $wav_path/dev ]; then
        rm -r $wav_path/dev ;
        echo "removed wav-dev"
    fi
    if [ -d $wav_path/eval ]; then
        rm -r $wav_path/eval ;
        echo "removed wav-eval"
    fi
    if [ -d $wav_path/train_non_parallel ]; then
        rm -r $wav_path/train_non_parallel ;
        echo "removed wav-train_non_parallel"
    fi

    if [ -f $text_path/dev.txt ]; then
        rm $text_path/dev.txt ;
        echo "removed text-dev"
    fi
    if [ -f $text_path/eval.txt ]; then
        rm $text_path/eval.txt ;
        echo "removed text-eval"
    fi
    if [ -f $text_path/train_non_parallel.txt ]; then
        rm $text_path/train_non_parallel.txt ;
        echo "removed text-train_non_parallel"
    fi

    mkdir -p "$wav_root"
    if [ -d $wav_path/train_parallel ]; then
        mv "$wav_path/train_parallel" "$wav_root/all"
        echo "moved text-train_parallel as all"
    fi
    mkdir -p "$text_root"
    if [ -f $text_path/train_parallel.txt ]; then
        mv -i "$text_path/train_parallel.txt" "$text_root/all.txt"
        echo "moved text-train_parallel as all"
    fi

    if [ ! -d "$wav_path/train_parallel" ] && [ ! -f "$text_path/train_parallel.txt" ] && [ -d "downloads/hi-fi-captain" ]; then
        rm -r "downloads/hi-fi-captain"
        echo "delete previous data-folder"
    fi


    echo "make full-context-label & shuffle & train/dev/eval split"
    xrun python preprocess/label_preparation.py $text_root/all.txt $lab_root/all

    mkdir -p data
    find $wav_root/all -name "*.wav" -exec basename {} .wav \; | sort > data/utt_list.txt
    sort -R data/utt_list.txt > data/utt_list_shuf.txt

    total_lines=$(wc -l < data/utt_list_shuf.txt)
    train_lines=$(echo "$total_lines * 80 / 100" | bc)
    deveval_lines=$(echo "$total_lines * 20 / 100" | bc)
    head -n $train_lines data/utt_list_shuf.txt > data/train.list
    tail -n $deveval_lines data/utt_list_shuf.txt > data/deveval.list
    dev_lines=$(echo "$deveval_lines * 50 / 100" | bc)
    head -n $dev_lines data/deveval.list > data/dev.list
    tail -n $dev_lines data/deveval.list > data/eval.list
    rm -f data/deveval.list
fi



# stage 1 ： Preprocessing
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for Tacotron"
    for s in ${datasets[@]}; do
        # args : utt_ist / wav_root / lab_root / out_dir + named_args
        xrun python preprocess/preprocess.py data/$s.list $wav_root $lab_root $dump_org_dir/$s \
            --n_jobs $n_jobs --sample_rate $sample_rate --mu $mu
    done
fi



# Stage 2 : Normalizing the data
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature normalization"

    # 学習済みデータから統計量を求めてjoblibファイルに保存
    xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
        $dump_org_dir/$train_set/out_tacotron/ \
        $dump_org_dir/out_tacotron_scaler.joblib

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/
    # 保存した統計量から標準化を行う
    for s in ${datasets[@]}; do
        xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
            $dump_org_dir/out_tacotron_scaler.joblib \
            $dump_org_dir/$s/out_tacotron/ \
            $dump_norm_dir/$s/out_tacotron/ --n_jobs $n_jobs
        # 波形データは手動でコピー
        # find $dump_org_dir/$s/out_tacotron/ -name "*-wave.npy" -exec cp "{}" $dump_norm_dir/$s/out_tacotron \;
        # mkdir -p $dump_norm_dir/$s/out_wavenet
        # find $dump_org_dir/$s/out_wavenet/ -name "*-feats.npy" -exec cp "{}" $dump_norm_dir/$s/out_wavenet/ \;
        # 韻律記号付き音素列は手動でコピー
        # rm -rf $dump_norm_dir/$s/in_tacotron
        cp -r $dump_org_dir/$s/out_wavenet $dump_norm_dir/$s/
        cp -r $dump_org_dir/$s/in_tacotron $dump_norm_dir/$s/
    done
fi



# Train Stepはretrain_tacotron.ipynbにて行う



# Test Stage 
if [ ${stage} -le 100 ] && [ ${stop_stage} -ge 100 ]; then
    echo "Test Stage"
    echo "wav count : $(find "$wav_root/all" -type f -name "*.wav" | grep -c "")"
    echo "label count : $(find "$lab_root/all" -type f -name "*.lab" | grep -c "")"
    # wav_path="./downloads/hi-fi-captain/ja-JP/male/wav"
    # echo "WAV dev count: $(find "$wav_path/dev" -type f -name "*.wav" | grep -c "")"
    # text_path="./downloads/hi-fi-captain/ja-JP/male/text"
    # echo "text count: $(grep -c '' "$text_path/dev.txt")"
fi