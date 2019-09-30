import matplotlib.pyplot as plt
import numpy as np
import model, sample, encoder
import json

def create_word_chart(model_name, run_name1, run_name2, log_dir, ex_num):
    enc = encoder.get_encoder(model_name)
    text_path = '{}/{}/{}_{}/text.txt'.format(log_dir, ex_num, run_name1, run_name2)
    tfile = open(text_path, 'r', encoding='utf-8')
    tlines = tfile.readlines()
    tfile.close()
    text = tlines[0].split()
    encoded = [int(num) for num in tlines[1].split(' ')]
    probs1 = []
    probs2 = []
    syms = []
    for t, num in enumerate(encoded):
        sym = enc.decoder[num]
        sym = bytearray([enc.byte_decoder[s] for s in sym]).decode('utf-8')
        if sym != '\n':
            #text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
            try:
                with open('{}/{}/{}_{}/{}/{}_logits1.json'.format(log_dir,ex_num, run_name1, run_name2, run_name1, t)) as l1f:
                    logit1_dict = json.load(l1f)

                print('Max1: {}'.format(max(logit1_dict.values())))
                print('Sum1: {}'.format(sum(logit1_dict.values())))

                with open('{}/{}/{}_{}/{}/{}_logits2.json'.format(log_dir, ex_num, run_name1, run_name2, run_name2, t)) as l2f:
                    logit2_dict = json.load(l2f)
                    print('Max2: {}'.format(max(logit2_dict.values())))
                    print('Sum2: {}'.format(sum(logit2_dict.values())))

            except:
                with open('{}/{}/{}_{}/{}/{}_logits1.json'.format(log_dir,ex_num, run_name1, run_name2, run_name1, len(text)-1)) as l1f:
                    logit1_dict = json.load(l1f)

                with open('{}/{}/{}_{}/{}/{}_logits2.json'.format(log_dir,ex_num, run_name1, run_name2, run_name2, len(text)-1)) as l2f:
                    logit2_dict = json.load(l2f)
            if sym.strip() not in logit1_dict:
                prob1 = 0
            else:
                prob1 = logit1_dict[sym.strip()]
            if sym.strip() not in logit2_dict:
                prob2 = 0
            else:
                prob2 = logit2_dict[sym.strip()]

            sym = sym.strip()
            if sym in syms:
                while sym in syms:
                    sym += ' '
            syms.append(sym)
            probs1.append(prob1)
            probs2.append(prob2)

    plt.title('Word Probabilities')
    plt.plot(syms, probs1, label=run_name1)
    plt.plot(syms, probs2, label=run_name2)
    plt.legend()
    plt.show()
    plt.savefig('{}/{}/{}_{}/fig.png'.format(log_dir,ex_num, run_name1, run_name2))
    plt.clf()

if __name__ == '__main__':
    model_name = '117M'
    run_name1 = 'brown_romance'
    run_name2 = 'cornell_supreme'
    log_dir = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/logs'
    ex_num = 'ex1'
    create_word_chart(model_name, run_name1 , run_name2, log_dir, ex_num)
