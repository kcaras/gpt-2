from nltk.corpus import brown
import random
import os

def produce_brown_files():
    cats = ['romance', 'humor', 'government']
    for cat in cats:
        sents = brown.sents(categories=cat)
        out_f = open('brown_{}.txt'.format(cat), 'w', encoding='utf-8')
        for sent in sents:
            out_line = ' '.join(sent) + '\n'
            out_line = out_line.replace(' ; ;', '').replace(' ? ?', '?').replace(' ! !', '!').replace('``', '')
            out_f.write(out_line)
        out_f.close()


def read_supreme():
    supreme = 'gpt-2/supreme.conversations.txt'
    out_s = open('gpt-2/supreme.txt', 'w', encoding='utf-8')
    f = open(supreme, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    print("Num Lines: {}".format(len(lines)))
    for line in lines:
        split = line.split('+++$+++')
        out_line = split[-1]
        out_s.write(out_line)
    out_s.close()


def combine_files(fname1, fname2):
    f1 = open(fname1, 'r', encoding='utf-8')
    total_lines = f1.readlines()
    f1.close()
    f2 = open(fname2, 'r', encoding='utf-8')
    lines2 = f2.readlines()
    f2.close()
    total_lines.extend(lines2)
    random.shuffle(total_lines)

    f3name = "{}_{}.txt".format(os.path.basename(fname1).replace('.txt', ''), os.path.basename(fname2).replace('.txt', ''))
    f3 = open(f3name, 'w', encoding='utf-8')
    for line in total_lines:
        f3.write(line)
    f3.close()



if __name__ == '__main__':
    #combine_files('gpt-2/supreme.txt', 'gpt-2/brown_romance.txt')
    import gpt_2_simple as gpt2

    model_name = "117M"
    gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/117M_Romance_Supreme/

