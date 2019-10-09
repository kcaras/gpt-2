import matplotlib.pyplot as plt
import numpy as np
import model, sample, encoder
import json
import math


def create_sentence_chart(loss_dict, ex_num, run_name1, run_name2, logits_used, repeat, weight1, weight2, display_combined=True):
    combined = 'combined'
    x = ['s{}'.format(i) for i in range(len(loss_dict[combined]))]
    plt.figure(figsize=(20, 20))
    plt.title('Probs : {} {}'.format(run_name1, run_name2))
    for loss_name in loss_dict.keys():
        if loss_name != combined:
            plt.plot(x, loss_dict[loss_name], label=loss_name)

    if display_combined:
        plt.plot(x, loss_dict[combined], label=combined)

    plt.legend()
    plt.savefig(
        '/home/twister/Dropbox (GaTech)/caras_graphs/{}_{}_{}_{}_{}_{}_{}.png'.format(ex_num, repeat, logits_used, run_name1, run_name2, weight1, weight2))
    plt.show()
    #plt.clf()


def create_sentence_chart_not_gen(loss_dict, ex_num, run_names, repeat):

    x = ['s{}'.format(i) for i in range(len(loss_dict[run_names[0]]))]
    plt.figure(figsize=(20, 20))
    plt.title('Probs :')

    for loss_name in loss_dict.keys():
        plt.plot(x, loss_dict[loss_name], label=loss_name)

    plt.legend()
    plt.savefig(
        '/home/twister/Dropbox (GaTech)/caras_graphs/{}_{}_{}.png'.format(ex_num, repeat, '{}'.format('_'.join(run_names))))
    plt.show()
    #plt.clf()



def create_word_chart(model_name, run_name1, run_name2, log_dir, ex_num, logits_used, display_combined=True):
    enc = encoder.get_encoder(model_name)
    text_path = '{}/{}/{}/{}_{}/text.txt'.format(log_dir, ex_num, logits_used, run_name1, run_name2)
    tfile = open(text_path, 'r', encoding='utf-8')
    tlines = tfile.readlines()
    tfile.close()
    text = tlines[0].split()
    encoded = [int(num) for num in tlines[1].split(' ')]
    probs1 = []
    probs2 = []
    probs3 = []
    syms = []
    combined = 'combined'
    for t, num in enumerate(encoded):
        sym = enc.decoder[num]
        sym = bytearray([enc.byte_decoder[s] for s in sym]).decode('utf-8')
        if sym != '\n':
            #text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
            try:
                with open('{}/{}/{}/{}_{}/{}/{}_logits1.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, run_name1, t)) as l1f:
                    logit1_dict = json.load(l1f)
                    print('Max1: {}'.format(max(logit1_dict.values())))
                    #d = sum([math.exp(val) for val in logit1_dict.values()])
                    #print('Sum1: {}'.format(d))
                    print('Sum1: {}'.format(sum(logit1_dict.values())))

                with open('{}/{}/{}/{}_{}/{}/{}_logits2.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, run_name2, t)) as l2f:
                    logit2_dict = json.load(l2f)
                    print('Max2: {}'.format(max(logit2_dict.values())))
                    #d = sum([math.exp(val) for val in logit2_dict.values()])
                    #print('Sum2: {}'.format(d))
                    print('Sum2: {}'.format(sum(logit2_dict.values())))

                if display_combined:
                    with open('{}/{}/{}/{}_{}/{}/{}_logits.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, combined, t)) as l3f:
                        logit_dict = json.load(l3f)
                        print('Max: {}'.format(max(logit_dict.values())))
                        #d = sum([math.exp(val) for val in logit_dict.values()])
                        #print('Sum: {}'.format(d))
                        print('Sum1: {}'.format(sum(logit_dict.values())))

            except:
                with open('{}/{}/{}/{}_{}/{}/{}_logits1.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, run_name1, len(text)-1)) as l1f:
                    logit1_dict = json.load(l1f)

                with open('{}/{}/{}/{}_{}/{}/{}_logits2.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, run_name2, len(text)-1)) as l2f:
                    logit2_dict = json.load(l2f)

                if display_combined:
                    with open('{}/{}/{}/{}_{}/{}/{}_logits.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, combined, len(text)-1)) as l3f:
                        logit_dict = json.load(l3f)

            if sym.strip() not in logit1_dict:
                prob1 = 0
            else:
                prob1 = logit1_dict[sym.strip()]

            if sym.strip() not in logit2_dict:
                prob2 = 0
            else:
                prob2 = logit2_dict[sym.strip()]

            if display_combined:
                if sym.strip() not in logit_dict:
                    prob3 = 0
                else:
                    prob3 = logit_dict[sym.strip()]

            sym = sym.strip()
            if sym in syms:
                while sym in syms:
                    sym += ' '
            syms.append(sym)
            probs1.append(prob1)
            probs2.append(prob2)

            if display_combined:
                probs3.append(prob3)
    plt.figure(figsize=(20, 20))
    plt.title('Word Probabilities')
    plt.plot(syms, probs1, label=run_name1)
    plt.plot(syms, probs2, label=run_name2)

    if display_combined:
        plt.plot(syms, probs3, label='combined')

    plt.legend()
    plt.savefig('/home/twister/Dropbox (GaTech)/caras_graphs/{}_{}_{}_{}.png'.format(ex_num, logits_used, run_name1, run_name2))
    plt.show()
    plt.clf()

if __name__ == '__main__':
    model_name = '117M'
    run_name1 = 'scifi'
    run_name2 = 'cornell_supreme'
    log_dir = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/logs'
    ex_num = 'ex_combined'
    logits_used = 1
    create_word_chart(model_name, run_name1 , run_name2, log_dir, ex_num, logits_used, display_combined=True)
