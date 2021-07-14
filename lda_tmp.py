import numpy as np
import nltk
import pandas as pd

'''
rawdoc = ["bull market investor generally positive about economy",
             "bear market situation share price fall rapidly",
             "stock market open higher",
             "strong buy pharma stocks",
             "Japan stock rebound month low",
             "Japan stock market drop",
             "beat player win tennis tournament",
             "fans cheer team player win",
             "team compete win tournament",
             "play tennis grass court",
             "huge fans team"
             ]
rawdocs = [nltk.tokenize.word_tokenize(i) for i in rawdoc]
'''


def lda_model(rawdocs, num_topic, iteration):
    newlist = []
    for i in range(len(rawdocs)):
        newlist.extend(rawdocs[i])
    word = list(set(newlist))
    num = list(range(0, len(word)))
    dictionary = dict(zip(word, num))

    raw_num = []
    for i in range(len(rawdocs)):
        numlist = []
        for word in rawdocs[i]:
            numlist.append(dictionary[word])
        raw_num.append(numlist)

    num_doc = len(raw_num)
    num_word = len(dictionary)
    alpha = 1
    beta = 1

    # doc-topic matrix
    dt = np.zeros([num_doc, num_topic])
    # topic-word matrix
    wt = np.zeros([num_topic, num_word])
    initial_doc = []
    for i, doc in enumerate(raw_num):
        initial_doc_tmp = []
        print(i)
        theta = np.random.dirichlet(alpha * np.ones(num_topic))  # 文件-主題骰子
        print(theta)
        for j, word in enumerate(doc):
            print(j, word)
            k = np.random.multinomial(1, theta).argmax()
            initial_doc_tmp.append(k)
            dt[i, k] += 1
            wt[k, word] += 1
        initial_doc.append(initial_doc_tmp)

    # gibbs sampling

    for iter in range(iteration):
        print("iter:", iter)

        for d, doc in enumerate(raw_num):
            print("\n", d, doc)
            for index, v in enumerate(doc):
                z = initial_doc[d][index]   # group of word
                dt[d, z] -= 1   # 去掉單字
                wt[z, v] -= 1

                left = (wt[:, v] + beta) / (np.sum(wt, axis=1) + num_word*beta)
                right = (dt[d, :] + alpha) / (np.sum(dt[d, :]) + num_topic*alpha)
                print(left, right)

                z = np.random.multinomial(1, left*right/np.sum(left*right)).argmax()
                dt[d, z] += 1   # 補回單字
                wt[z, v] += 1

                initial_doc[d][index] = z

    # topic-word probability
    phi = np.divide((wt + beta),(np.sum(wt, axis=1) + num_word*beta).reshape(num_topic,1))

    # topic-document probability
    theta = (dt + alpha)/(np.sum(dt, axis=1).reshape(num_doc, 1) + num_topic*alpha)

    key_list = []
    for key, value in dictionary.items():
        key_list.append(key)

    df = pd.DataFrame({"word": key_list, "topic0": phi[0], "topic1": phi[1]})
    return df
