from nltk.corpus import brown
import random
import os, csv, json


def parse_kdrama():
    kdir = '/home/twister/Dropbox (GaTech)/Chicken/Datasets/Korean_drama_recaps'
    fout = open('kdrama.txt', 'w', encoding='utf-8')
    for i in range(1, 4):
        fin = open('{}/cleaned_kdrama_{}.txt'.format(kdir, i), 'r', encoding='utf-8')
        fout.write(fin.read().replace('<EOS>', ' '))
        fin.close()
    fout.close()


def combine(data1, data2):
    data_path1 = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_wang/Data/{}.txt'.format(data1)
    data_path2 = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_wang/Data/{}.txt'.format(data2)
    out_file_path = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_wang/Data/{}_{}.txt'.format(data1,data2)
    f1 = open(data_path1, 'r', encoding='utf-8')
    lines1 = f1.readlines()
    f1.close()
    f2 = open(data_path2, 'r', encoding='utf-8')
    lines2 = f2.readlines()
    f2.close()
    lines1.extend(lines2)
    random.shuffle(lines1)
    out_file = open(out_file_path, 'w', encoding='utf-8')
    out_file.writelines(lines1)
    #out_file.write(lines2)
    out_file.close()


def parse_urban_dictionary():
    path = 'words.json'
    jf = open(path, 'r', encoding='utf-8')
    #words = json.load(jf)
    #jf.close()
    outf_path = 'urban_dictionary.txt'
    outf = open(outf_path, 'w', encoding='utf-8')
    for line in jf:
        words = json.loads(line)
        ex = words['example']
        outf.write(ex.replace('\n', '') + '\n')
    outf.close()
    jf.close()


def produce_brown_files():
    cats = ['romance', 'humor', 'government']
    for cat in cats:
        sents = brown.sents(categories=cat)
        out_f = open('brown_{}.txt'.format(cat), 'w', encoding='utf-8')
        for sent in sents:
            out_line = ' '.join(sent) + '\n'
            out_line = out_line.replace(' ; ;', '').replace(' ? ?', '?').replace(' ! !', '!').replace('``', '').replace(' \'\'', '').replace('\'\'', '')
            out_f.write(out_line)
        out_f.close()


def read_dnd():
    out_lines = []
    with open('dd_bios.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            if i > 0:
                out_lines.append(row[4])
        out_file = 'dd_bios.txt'
        of = open(out_file, 'w', encoding='utf-8')
        of.write('\n'.join(out_lines))
        of.close()


def produce_shakespere():
    out_lines = []
    with open('Shakespeare_data.csv', 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            if i > 0:
                out_lines.append(row[-1])
    out_file = 'shakespere.txt'
    of = open(out_file, 'w', encoding='utf-8')
    of.write('\n'.join(out_lines))
    of.close()


def read_supreme():
    supreme = 'supreme.conversations.txt'
    out_s = open('supreme.txt', 'w', encoding='utf-8')
    f = open(supreme, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    print("Num Lines: {}".format(len(lines)))
    for line in lines:
        split = line.split('+++$+++')
        out_line = split[-1]
        out_s.write(out_line)
    out_s.close()


def read_movies():
    movies = 'movie_lines.txt'
    out_s = open('movies.txt', 'w', encoding='utf-8')
    f = open(movies, 'r', encoding='utf-8',errors='ignore')
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

def clean_reddit_jokes():
    f = open('reddit_jokes.txt', 'r', encoding='utf-8')
    lines = f.read()
    f.close()
    new_lines = lines.replace('===', '')
    f_new = open('reddit_jokes.txt', 'w', encoding='utf-8')
    f_new.write(new_lines)
    f_new.close()

