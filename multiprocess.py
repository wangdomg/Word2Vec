
# -*- coding:utf-8 -*-
import multiprocessing
import numpy as np
import pickle
from data_util import formal_prepare_w2v_data
import gensim
import json


class myProcess(multiprocessing.Process):
    def __init__(self, con, docs_dict, model, vocab, idf):
        multiprocessing.Process.__init__(self)
        self.con = con
        self.docs_dict = docs_dict
        self.model = model
        self.vocab = vocab
        self.idf = idf

    # title长度差，结果越大越相似
    def extract_sentece_length_diff(self, sen1, sen2):
        sen1 = sen1.split('_'); sen2 = sen2.split('_')
        return 1 - abs(len(sen1) - len(sen2)) / float(max(len(sen1), len(sen2)))

    # n-gram相似度，结果越大越相似
    def extract_ngram_sim(self, sen1, sen2, max_ngram=3):
        sen1 = sen1.split('_'); sen2 = sen2.split('_')
        # 定义n_gram的方法
        def get_ngram(sen, ngram_value):
            result = []
            for i in range(len(sen)):
                if i + ngram_value < len(sen) + 1:
                    result.append('_'.join(sen[i:i + ngram_value]))
            return result

        # 计算ngram之间的相似度
        def get_ngram_sim(q1_ngram, q2_ngram):
            q1_dict = {}
            q2_dict = {}
            for token in q1_ngram:
                if token not in q1_dict:
                    q1_dict[token] = 1
                else:
                    q1_dict[token] = q1_dict[token] + 1
            q1_count = np.sum([value for key, value in q1_dict.items()])

            for token in q2_ngram:
                if token not in q2_dict:
                    q2_dict[token] = 1
                else:
                    q2_dict[token] = q2_dict[token] + 1
            q2_count = np.sum([value for key, value in q2_dict.items()])

            # ngram1有但是ngram2没有
            q1_count_only = np.sum([value for key, value in q1_dict.items() if key not in q2_dict])
            # ngram2有但是ngram1没有
            q2_count_only = np.sum([value for key, value in q2_dict.items() if key not in q1_dict])
            # ngram1和ngram2都有的话，计算value的差值
            q1_q2_count = np.sum([abs(value - q2_dict[key]) for key, value in q1_dict.items() if key in q2_dict])
            # ngram1和ngram2的总值
            all_count = q1_count + q2_count
            return (1 - float(q1_count_only + q2_count_only + q1_q2_count) / (float(all_count) + 0.00000001))

        ngram_feature = {}
        for ngram_value in range(max_ngram):
            ngram1 = get_ngram(sen1, ngram_value + 1)
            ngram2 = get_ngram(sen2, ngram_value + 1)
            ngram_sim = get_ngram_sim(ngram1, ngram2)
            ngram_feature[ngram_value+1] = ngram_sim

        return ngram_feature

    # 1gram的不相似度，结果越大越不相似
    def extract_1gram_dissim(self, sen1, sen2):
        sen1 = sen1.split('_'); sen2 = sen2.split('_')
        # 计算ngram之间的相似度
        def get_1gram_dissim(q1_ngram, q2_ngram):
            q1_dict = {}
            q2_dict = {}
            for token in q1_ngram:
                if token not in q1_dict:
                    q1_dict[token] = 1
                else:
                    q1_dict[token] = q1_dict[token] + 1
            q1_count = np.sum([value for key, value in q1_dict.items()])

            for token in q2_ngram:
                if token not in q2_dict:
                    q2_dict[token] = 1
                else:
                    q2_dict[token] = q2_dict[token] + 1
            q2_count = np.sum([value for key, value in q2_dict.items()])

            # ngram1有但是ngram2没有
            q1_count_only = np.sum([value for key, value in q1_dict.items() if key not in q2_dict])
            # ngram2有但是ngram1没有
            q2_count_only = np.sum([value for key, value in q2_dict.items() if key not in q1_dict])
            # ngram1和ngram2的总值
            all_count = q1_count + q2_count
            return float(q1_count_only + q2_count_only) / (float(all_count) + 0.00000001)
        
        dissim = get_1gram_dissim(sen1, sen2)
        return dissim

    # jaccard距离相似度，结果越大越相似
    def extract_jaccard_sim(self, sen1, sen2):
        sen1 = sen1.split('_'); sen2 = sen2.split('_')
        set1 = set(sen1); set2 = set(sen2)
        same_word_len = len(set1 & set2)
        jaccard_sim = same_word_len / float(len(set1 | set2))
        return jaccard_sim

    # IDF*embedding的相似度
    def extract_idf_sim(self, sen1, sen2, model, vocab, idf):
        sen1 = sen1.split('_'); sen2 = sen2.split('_')
        # 计算sentence的embedding
        def get_sen_embedding(sen, model, vocab, idf):
            sen_vec = np.array([0.0]*50)
            for word in sen:
                if word in vocab:
                    sen_vec += idf[word] * model[word]
            return sen_vec

        sen1_vec = get_sen_embedding(sen1, model, vocab, idf)
        sen2_vec = get_sen_embedding(sen2, model, vocab, idf)
        distance = {}
        # 余弦相似度
        distance['cos_sim'] = float(np.dot(sen1_vec, sen2_vec)) / (np.linalg.norm(sen1_vec) * np.linalg.norm(sen2_vec))
        # 欧氏距离
        distance['euc_dist'] = np.linalg.norm(sen1_vec - sen2_vec)  
        # 曼哈顿距离
        distance['manh_dist'] = np.linalg.norm(sen1_vec - sen2_vec, ord=1)
        # 切比雪夫距离
        distance['che_dist'] = np.linalg.norm(sen1_vec - sen2_vec, ord=np.inf)

        return distance

    # 编辑距离
    def normal_leven(self, str1, str2):
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        # 创建矩阵
        matrix = [0 for n in range(len_str1 * len_str2)]
        #矩阵的第一行
        for i in range(len_str1):
            matrix[i] = i
        # 矩阵的第一列
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1
        # 根据状态转移方程逐步得到编辑距离
        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,matrix[j*len_str1+(i-1)]+1,matrix[(j-1)*len_str1+(i-1)] + cost)

        return matrix[-1]  # 返回矩阵的最后一个值，也就是编辑距离

    # 文档与簇的特征
    def extract_rule_feature_from_doc_and_cluster(self, doc, id_clusters, all_docs, model, vocab, idf):
        def get_cluster_feature(single_doc, tmp_doc):
            cluster_feature = {}
            same_author = len(set(single_doc['authors']) & set(tmp_doc['authors']))
            same_org = len(set(single_doc['orgs']) & set(tmp_doc['orgs']))
            same_venue = len(set(single_doc['venues']) & set(tmp_doc['venues']))
            cluster_feature['author_ratio'] = float(same_author) / len(set(single_doc['authors']))
            cluster_feature['org_ratio'] = float(same_org) / len(set(single_doc['orgs']))
            cluster_feature['venue_ratio'] = float(same_venue) / len(set(single_doc['venues']))
            if tmp_doc['author_org'] != '' and single_doc['author_org'] != '':
                cluster_feature['org_cover_ratio'] = len(set(single_doc['author_org'].split('_')) & set(tmp_doc['author_org'].split('_'))) / float(len(set(single_doc['author_org'].split('_'))))
            else:
                cluster_feature['org_cover_ratio'] = 0.0  # 簇文章中没有跟当前文档重名的作者
            cluster_feature['org_edit_dist'] = self.normal_leven(single_doc['author_org'], tmp_doc['author_org'])
            return cluster_feature

        pos_data = []; neg_data = []

        # 当前文档
        single_doc = {'authors':[], 'orgs':[], 'venues':[], 'author_org':''}
        for author in doc['authors']:
            single_doc['authors'].append(author['name'])
            single_doc['orgs'].append(author['org'])
            if author['name'] == doc['author']:
                single_doc['author_org'] = author['org']
        single_doc['venues'].append(doc['venue'][-1])

        # 遍历所有簇构造特征
        for id_cluster in id_clusters:
            if doc['id'] in id_cluster:  # 正例
                for i in range(len(id_cluster)):
                    # 簇里的每一个文档
                    cluster_doc = all_docs[id_cluster[i]]  
                    tmp_doc = {'authors':[], 'orgs':[], 'venues':[], 'author_org':''}
                    for author in cluster_doc['authors']:
                        tmp_doc['authors'].append(author['name'])
                        tmp_doc['orgs'].append(author['org'])
                        if author['name'] == doc['author']:
                            tmp_doc['author_org'] = author['org']
                    tmp_doc['venues'].append(cluster_doc['venue'][-1])

                    # 簇级特征
                    cluster_feature = get_cluster_feature(single_doc, tmp_doc)

                    cluster_feature['length_diff'] = self.extract_sentece_length_diff(doc['title'], cluster_doc['title'])

                    tmp_feature = self.extract_ngram_sim(doc['title'], cluster_doc['title'])
                    cluster_feature['1gram_sim'] = tmp_feature[1]; cluster_feature['2gram_sim'] = tmp_feature[2]; cluster_feature['3gram_sim'] = tmp_feature[3]
                    
                    cluster_feature['1gram_dis'] = self.extract_1gram_dissim(doc['title'], cluster_doc['title'])
                    
                    cluster_feature['jaccard_sim'] = self.extract_jaccard_sim(doc['title'], cluster_doc['title'])
                    
                    tmp_feature = self.extract_idf_sim(doc['title'], cluster_doc['title'], model, vocab, idf)
                    cluster_feature['cos_sim'] = tmp_feature['cos_sim']; cluster_feature['euc_dist'] = tmp_feature['euc_dist']
                    cluster_feature['manh_dist'] = tmp_feature['manh_dist']; cluster_feature['che_dist'] = tmp_feature['che_dist']
                    
                    tmp_feature = self.extract_keywords_feature_from_doc_and_doc(doc['keywords'], cluster_doc['keywords'])
                    cluster_feature['same_word_ratio'] = tmp_feature['same_word_ratio']
                    cluster_feature['uniq_word_ratio'] = tmp_feature['uniq_word_ratio']
                    cluster_feature['same_words_ratio'] = tmp_feature['same_words_ratio']
                    pos_data.append(cluster_feature)
            else:
                cluster_doc = all_docs[id_cluster[0]]  
                tmp_doc = {'authors':[], 'orgs':[], 'venues':[], 'author_org':''}
                for author in cluster_doc['authors']:
                    tmp_doc['authors'].append(author['name'])
                    tmp_doc['orgs'].append(author['org'])
                    if author['name'] == doc['author']:
                        tmp_doc['author_org'] = author['org']
                tmp_doc['venues'].append(cluster_doc['venue'][-1])

                # 簇级特征
                cluster_feature = get_cluster_feature(single_doc, tmp_doc)

                cluster_feature['length_diff'] = self.extract_sentece_length_diff(doc['title'], cluster_doc['title'])
                tmp_feature = self.extract_ngram_sim(doc['title'], cluster_doc['title'])
                cluster_feature['1gram_sim'] = tmp_feature[1]; cluster_feature['2gram_sim'] = tmp_feature[2]; cluster_feature['3gram_sim'] = tmp_feature[3]
                cluster_feature['1gram_dis'] = self.extract_1gram_dissim(doc['title'], cluster_doc['title'])
                cluster_feature['jaccard_sim'] = self.extract_jaccard_sim(doc['title'], cluster_doc['title'])
                tmp_feature = self.extract_idf_sim(doc['title'], cluster_doc['title'], model, vocab, idf)
                cluster_feature['cos_sim'] = tmp_feature['cos_sim']; cluster_feature['euc_dist'] = tmp_feature['euc_dist']
                cluster_feature['manh_dist'] = tmp_feature['manh_dist']; cluster_feature['che_dist'] = tmp_feature['che_dist']
                tmp_feature = self.extract_keywords_feature_from_doc_and_doc(doc['keywords'], cluster_doc['keywords'])
                cluster_feature['same_word_ratio'] = tmp_feature['same_word_ratio']
                cluster_feature['uniq_word_ratio'] = tmp_feature['uniq_word_ratio']
                cluster_feature['same_words_ratio'] = tmp_feature['same_words_ratio']
                neg_data.append(cluster_feature)

        return pos_data, neg_data

    # 将特征写入文件
    def write_feature(self, cluster_features, cato, name):
        atts = ['author_ratio', 'org_ratio', 'venue_ratio', 'org_cover_ratio', 'org_edit_dist', 'length_diff',
                    '1gram_sim', '2gram_sim', '3gram_sim', '1gram_dis', 'jaccard_sim', 'cos_sim', 'euc_dist',
                    'manh_dist', 'che_dist', 'same_word_ratio', 'uniq_word_ratio', 'same_words_ratio']
        if cato == 'pos':
            f = open(name+'_pos_data.txt', 'a')
        else:
            f = open(name+'_neg_data.txt', 'a')
        for cluster_feature in cluster_features:
            line = ''
            for att in atts:
                line = line + str(cluster_feature[att]) + ' '
            line = line[:-1] + '\n'
            f.write(line)
        f.close()

    # 文档与锚文档的keywords的特征
    def extract_keywords_feature_from_doc_and_doc(self, keywords1, keywords2):
        def get_key_word_feature(q1_1gram, q2_1gram):
            q1_dict = {}
            q2_dict = {}
            for token in q1_1gram:
                if token not in q1_dict:
                    q1_dict[token] = 1
                else:
                    q1_dict[token] = q1_dict[token] + 1
            q1_count = np.sum([value for key, value in q1_dict.items()])

            for token in q2_1gram:
                if token not in q2_dict:
                    q2_dict[token] = 1
                else:
                    q2_dict[token] = q2_dict[token] + 1
            q2_count = np.sum([value for key, value in q2_dict.items()])

            # ngram1有但是ngram2没有
            q1_count_only = np.sum([value for key, value in q1_dict.items() if key not in q2_dict])
            # ngram2有但是ngram1没有
            q2_count_only = np.sum([value for key, value in q2_dict.items() if key not in q1_dict])
            # ngram2和ngram1都有的词
            common_count = np.sum([abs(value + q2_dict[key]) for key, value in q1_dict.items() if key in q2_dict])
            # ngram1和ngram2的总值
            all_count = q1_count + q2_count

            # 相同词所占的比例
            same_word_ratio = float(common_count) / all_count
            # 不同词所占的比例
            uniq_word_ratio = float(q1_count_only + q2_count_only) / all_count

            return same_word_ratio, uniq_word_ratio

        keyword_feature = {}

        if not keywords1 or not keywords2:
            keyword_feature['same_word_ratio'] = 0.0; keyword_feature['same_words_ratio'] = 0.0
            keyword_feature['uniq_word_ratio'] = 1.0
            return keyword_feature

        q1_1gram = []; q2_1gram = []
        for item in keywords1:
            q1_1gram.extend(item.split('_'))
        for item in keywords2:
            q2_1gram.extend(item.split('_'))
        
        keyword_feature['same_word_ratio'], keyword_feature['uniq_word_ratio'] = get_key_word_feature(q1_1gram, q2_1gram)

        keyword_feature['same_words_ratio'], _ = get_key_word_feature(keywords1, keywords2)

        return keyword_feature

    def run(self):
        for name, id_clusters in self.con:
            for id_cluster in id_clusters:
                doc = self.docs_dict[id_cluster[0]]
                doc['id'] = id_cluster[0]; doc['author'] = name
                pos_data, neg_data = self.extract_rule_feature_from_doc_and_cluster(doc, id_clusters, self.docs_dict, model, vocab, idf)
                self.write_feature(pos_data, 'pos', name)
                self.write_feature(neg_data, 'neg', name)


def get_idf():
    print ('get idf.....')
    with open('idf.pkl', 'rb') as f:
        idf = pickle.load(f)
    return idf

def get_model():
    print ('load w2v model.....')
    model = gensim.models.Word2Vec.load('opendac/data/emb/w2v.model')
    vocab = list(model.wv.vocab.keys())
    return model, vocab

def read_data():
    print ('read all data.....')
    with open('opendac/data/v2_train_names.json', 'r') as f:
        name_alldocsid_dict = json.load(f)
    with open('opendac/data/v2_train_pubs_raw.json', 'r') as f:
        docs_dict = json.load(f)
    with open('opendac/data/v2_train_results.json', 'r') as f:
        name_docidclu_dict = json.load(f)

    return name_alldocsid_dict, docs_dict, name_docidclu_dict


if __name__ == '__main__':

    idf = get_idf()
    model, vocab = get_model()
    name_alldocsid_dict, docs_dict, name_docidclu_dict = read_data()
    print ('total name count: '+str(len(name_alldocsid_dict)))
    i = 1
    con = []
    for name, id_clusters in name_docidclu_dict.items():
        con.append((name, id_clusters))
    epoch = int(len(con) / 10)
    for i in range(0, len(con), epoch):
        tmpcon = con[i:i+epoch]
        myProcess(tmpcon, docs_dict, model, vocab, idf).start()