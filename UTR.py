from transformers import (
    MBartForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import torch
from torch.utils.data import random_split
from tokenizers import ByteLevelBPETokenizer
from transformers import MBartConfig
from transformers import RobertaTokenizerFast
from transformers import pipeline
import utils.codonBox as cb
import numpy as np

import random


def get_utr(seq, tokenizer_name, model_name, use_box, use_spe):
    # get tokenzier and pipeline
    if use_spe:
        tokenizer = RobertaTokenizerFast.from_pretrained(f"./{tokenizer_name}", max_len=512,
                                                         additional_special_tokens=["<CDS>", "<5UTR>", "<3UTR>"])
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(f"./{tokenizer_name}")
    translator = pipeline("translation", model=model_name, tokenizer=tokenizer)
    if seq != '':
        if use_box:
            seq = cb.codon_to_box(seq)
        if use_spe:
            seq = "<CDS>" + seq
        seq_u = translator(seq, return_text=True, max_length=512)[0]['translation_text']
        return seq_u


def get_full_sequence(input_file, output_file, tokenizer_name, model_name_5u, model_name_3u, use_box, use_spe):
    file_5u = output_file.replace('full', '5u')
    file_3u = output_file.replace('full', '3u')
    with open(input_file, 'r') as f:
        seqs = f.read().split('\n')
        full_seqs, seqs_5u, seqs_3u = list(), list(), list()
        for seq in seqs:
            if seq != '':
                seq_5u = get_utr(seq, tokenizer_name, model_name_5u, use_box, use_spe)
                seq_3u = get_utr(seq, tokenizer_name, model_name_3u, use_box, use_spe)
                print(f'3:{len(seq_3u)}')
                print(f'5:{len(seq_5u)}')
                full_seq = seq_5u + seq + seq_3u+'\n'
                full_seqs.append(full_seq)
                seqs_3u.append(seq_3u+'\n')
                seqs_5u.append(seq_5u+'\n')
        # write results
        with open(output_file, 'w') as f_w:
            for i in full_seqs:
                f_w.write(i)
        with open(file_3u, 'w') as f_3:
            for i in seqs_3u:
                f_3.write(i)
        with open(file_5u, 'w') as f_5:
            for i in seqs_5u:
                f_5.write(i)


def get_single_sequence(input_seq, tokenizer_name, model_name_5u, model_name_3u, use_box=True, use_spe=True):
    seq_5u = get_utr(input_seq, tokenizer_name, model_name_5u, use_box, use_spe)
    seq_3u = get_utr(input_seq, tokenizer_name, model_name_3u, use_box, use_spe)
    return seq_5u, input_seq, seq_3u
