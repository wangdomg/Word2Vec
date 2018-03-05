# include <iostream>
# include <fstream>
# include <cstring>
# include <cmath>
# include <map>
# include <vector>
# include <ctime>
# include <cstdlib>
# include <thread>  // c++11添加的线程标准库，之前都是用系统接口pthread
# include <iomanip>

using namespace std;

time_t start_time = time(0);  // 程序开始的时间，用来记录程序的运行时间

// 单词的结构体
struct single_word 
{
    int freq;  // 这个词在语料中出现的频次
    string word;  // 这个词是啥
    vector<int> code;  // 这个词的Huffman编码，即0-1串
    vector<int> path;  // 从根节点到这个词所在的叶子节点的路径(不包括这个叶子节点)，存的是路径上每一个节点在HuffmanTree数组中的下标(这里我们用一个数组来表示一棵Huffman树)
    int codelen;  // 单词的Huffman编码的长度
};

vector<single_word> vocab;  // 词汇表，将所有的词按照词频降序排列
int vocab_size;  // 词汇表中词的个数
map<string, int> vocab_hash;  // 哈希表，根据一个词找到这个词在词汇表中的下标
vector<vector<double> > word_embeddings;  // embedding词典，根据一个词可以找到这个词的embedding
vector< vector<double> > setas;  // Huffman树中的𝜽数组
map< int, vector<double> > betas;  // 负采样中的𝜷数组，key是单词在vocab中的下标

long long file_size;

int table_size = 1e8;  // 负采样映射表的大小
vector<int> sample_table(table_size, 0);  // 负采样映射表

int max_sentence_length = 1000;  // 训练时，构造的句子的最大长度

// 用来近似计算sigmoid值的
int max_exp = 6;  // -𝜽x的允许区间[-6,6]
int exp_table_size = 1000;  // 将[-6,6]均分成1000份，然后看sigmoid的-𝜽x落在哪个区间，这样可以加快计算
vector<double> expTable(exp_table_size);

// 用来动态改变学习率alpha的
int train_words = 0;  // 训练文件中的单词个数
int word_count_actual = 0;  // 多个线程目前总的处理的单词个数
double start_alpha;  // 初始的学习率alpha

// 从命令行接收到的参数
string train_file="data.txt", output_file="wdvec.txt";
int embedding_size = 200;
int cbow = 0, skip_gram = 0, hs = 0, negative = 0;
double sample = 0;
double alpha = 0.05;
int window = 5, iters = 2, min_count = 5, threads_num = 1;


// 自定义的排序函数，按照词频降序排列
bool sortbyfreq(single_word sw1, single_word sw2)
{
    return sw1.freq > sw2.freq;  // 注意这里必须是严格的大于或者小于，不能有等于
}

// 读取训练数据，训练数据需要是已经分好词的数据
void read_data()
{
    ifstream fin(train_file);

    string word;
    train_words = 0;  // 训练文件中的单词个数
    int wc = 0;  // 词汇表中的单词个数

    // 一直读取文件，直到文件末尾
    while(fin>>word)  
    {
        train_words++;
        map<string, int>::iterator it = vocab_hash.find(word);
        if(it==vocab_hash.end())
        {
            single_word sw;
            sw.freq = 1;
            sw.word = word;
            vocab.push_back(sw);
            vocab_hash.insert(pair<string, int>(word, wc));
            wc++;
        }
        else
            vocab[vocab_hash[word]].freq += 1;
    }

    fin.clear();
    fin.seekg(0, ios::end);
    file_size = fin.tellg();  // 获取文件大小，单位是字节
    cout<<"file_size: "<<file_size<<endl;

    // 将vocab中的单词按照词频降序排列
    sort(vocab.begin(), vocab.end(), sortbyfreq);

    // 剔除词频太小的词
    if(min_count)
    {
        string tmp;
        while(vocab.back().freq<min_count) 
        {
            vocab.pop_back();
        }
        // 重新写vocab_hash
        vocab_hash.clear();
        vocab_size = vocab.size();
        for(int i=0; i<vocab_size; i++)
        {
            vocab_hash.insert(pair<string, int>(vocab[i].word, i));
        }
    }

    vocab_size = vocab.size();  // 词汇表中词的个数

    // train_words = 0;
    // for(auto sw : vocab) {train_words += sw.freq;}

    cout<<"Vocab size: "<<vocab_size<<endl;
    cout<<"Words in train file: "<<train_words<<endl;
}


// 创建Huffman树
void create_huffmantree()
{
    vector<int> count;  // 保存的是节点的词频
    vector<int> parent(2*vocab_size-1, 0);  // 保存的是节点之间的父子关系
    vector<int> childflag(2*vocab_size-1, 0);  // 保存的是节点是左孩子还是右孩子，左孩子为1，右孩子为0

    for(int i=0; i<vocab_size; i++) count.push_back(vocab[i].freq);  // count的前半部分保存的是叶子节点的词频
    for(int i=vocab_size; i<2*vocab_size-1; i++) count.push_back((int)1e10);  // count的后半部分保存的是中间节点的词频，中间节点的数目比叶子节点的数目少1

    int pos1 = vocab_size - 1; int pos2 = vocab_size;  // pos1从中间向前半部分找词频最小的，pos2从中间向后半部分找词频最小的
    int min1 = 0; int min2 = 0;  // min1和min2分别记录的是词频最小的两个节点在count中的下标

    for(int i=0; i<vocab_size-1; i++)
    {
        // 找第一个词频最小的节点
        if(pos1 >= 0)
        {
            if(count[pos1] < count[pos2]) {min1 = pos1; pos1--;}
            else {min1 = pos2; pos2++;}
        }
        else{min1 = pos2; pos2++;}
        // 找第二个词频最小的节点
        if(pos1>=0)
        {
            if(count[pos1] < count[pos2]) {min2 = pos1; pos1--;}
            else {min2 = pos2; pos2++;}
        }
        else {min2 = pos2; pos2++;}
        // 添加新的节点
        count[vocab_size+i] = count[min1] + count[min2];
        parent[min1] = vocab_size+i; parent[min2] = vocab_size+i;
        childflag[min1] = 1;  // 把min1设置为左孩子
    }

    // 填写每一个节点的Huffman编码和路径
    for(int i=0; i<vocab_size; i++)
    {
        int j = i;
        while(j != 2*vocab_size-2)
        {
            vocab[i].code.insert(vocab[i].code.begin(), childflag[j]);
            vocab[i].path.insert(vocab[i].path.begin(), parent[j]-vocab_size);  // 这里之所以要减去vocab_size，是因为父节点的下标是从vocab_size开始的，而𝜽的下标是从0开始的
            j = parent[j];
        }
        vocab[i].codelen = vocab[i].code.size();  // 这样之后就不用多次调用size()函数了
    }
}


// 实现负采样(通过均等划分和不均等划分之间的映射实现)，参考: http://blog.csdn.net/itplus/article/details/37998797
void create_sampletable()
{
    int i = 0;
    double power = 0.75;  // 其实可以按单词的词频来设置单词的权重，不过word2vec中加了这样一个幂运算
    double train_words_pow = 0.0;

    for(int j=0; j<vocab_size; j++) {train_words_pow += pow(vocab[j].freq, power);}  // 用于权重归一化

    double dlen = pow(vocab[i].freq, power)/train_words_pow;  // 遍历过的词的权重的累加

    for(int j=0; j<table_size; j++)
    {
        sample_table[j] = i;
        if(j/(double)table_size > dlen)  // 应该是到下一个词的权重范围了
        {
            i++;
            dlen += pow(vocab[i].freq, power)/train_words_pow;
        }
        if(i>=vocab_size) {i--;}  // 按理说i是不会>=vocab_size的
    }
}


// 初始化神经网络
void init_net()
{
    // 初始化所有的词向量(注意，不是初始化为全零)
    // srand((unsigned)time(NULL));  // 真随机化
    for(int i=0; i<vocab_size; i++)
    {
        vector<double> embedding(embedding_size, 0.0);
        for(int j=0; j<embedding_size; j++)
        {
            embedding[j] = ((double)rand()/RAND_MAX - 0.5) / embedding_size;
        } 
        word_embeddings.push_back(embedding);
    }

    // 如果采用hs，初始化相应的数据结构
    if(hs)
    {
        // 初始化所有的𝜽全零
        for(int i=0; i<vocab_size-1; i++) {vector<double> seta(embedding_size, 0.0); setas.push_back(seta);}
        // 建树
        create_huffmantree();
    }
    
    // 如果采用负采样，初始化相应的数据结构
    if(negative)
    {
        // 初始化所有的𝜷全零
        for(int i=0; i<vocab_size; i++)
        {
            vector<double> beta(embedding_size, 0.0);
            betas.insert(pair< int, vector<double> > (i, beta));
        }
        // 建立负采样表
        create_sampletable();
    }

    // 近似计算sigmoid值
    for(int i=0; i<exp_table_size; i++) 
    {
        expTable[i] = exp((i/(double)exp_table_size*2-1)*max_exp); // Precompute the exp() table
        expTable[i] = expTable[i]/(expTable[i]+1);                   // Precompute f(x) = x / (x + 1)
    }
}

// 单个线程从文件中读取自己负责的那部分数据
void create_trainsentence(ifstream &fin, vector<int> &sentence, int &sentence_length)
{
    int wc = 0;
    string word;
    map<string, int>::iterator it;
    while(wc<max_sentence_length && !fin.eof())
    {
        fin>>word;
        it = vocab_hash.find(word);
        if(it!=vocab_hash.end()) {sentence[wc]=vocab_hash[word]; wc++;}
    }
    sentence_length = wc;
}

// 将训练好的词向量写入文件
void write_vecfile()
{
    int a;
    vector<double> b;
    ofstream fout(output_file);
    for(auto word : vocab)
    {
        fout<<word.word<<"  ";
        a = vocab_hash[word.word];
        b = word_embeddings[a];
        for(auto embedding : b) {fout<<embedding<<"  ";}
        fout<<"\n";
    }
}

// 单个训练进程
void train_thread(int thread_id)
{
    int local_iter = 0;  // 当前的迭代次数
    
    vector<int> sentence(max_sentence_length);  // 从训练数据中构造的句子，存的是单词在vocab中的下标
    int sentence_length = 0;  // 从训练数据中读取的句子的长度
    int sentence_position = 0;
    
    int window_actual;  // 全局变量window表示一个词的上下文范围，而在word2vec源码中，每一次参考的词的范围实际上是在[0,window]中随机取的

    int word;  // 当前词
    int context_word;  // 当前词的上下文词
    int context_word_count;  // 当前词的上下文个数

    vector<double> x(embedding_size);  // 隐层向量
    vector<double> e(embedding_size);  // 一个累加值，用来更新参数

    double f, q, g;  // 进行梯度更新的时候需要用到的量，f是sigmoid函数的中的𝜽*x，q是f代入sigmoid之后的值

    int label;  // 负采样的时候标识一个样本是正样本(1)还是负样本(0)
    int target;  // 负采样时采样到的单词

    ifstream fin(train_file);
    fin.seekg(file_size/threads_num*(thread_id), ios::beg);  // 当前线程在文件中的起始读取位置

    int word_count = 0, last_word_count = 0;

    time_t now_time;

    // 进行多次迭代
    while(local_iter < iters)  
    {
        local_iter++;
        
        while(1)  // 每一次迭代的时候，并不是把数据一次全部读进来，而是分多次读进来，每一次读一小部分，避免内存不够
        {
            create_trainsentence(fin, sentence, sentence_length);  // 上一个句子遍历完了，要新读进来一个句子

            // 在单次迭代中训练，依次遍历sentence中的每一个词，采用SGD
            while(sentence_position < sentence_length)
            {
                // 动态改变alpha，随着训练的进行，alpha会变得越来越小
                if(word_count-last_word_count > 10000)
                {
                    now_time = time(0);
                    word_count_actual += word_count - last_word_count;
                    last_word_count = word_count;
                    alpha = start_alpha * (1 - word_count_actual / (double)(iters * train_words + 1));
                    if (alpha < start_alpha * 0.0001) alpha = start_alpha * 0.0001;
                    // cout不是线程安全的，要靠自己去线程同步，比较麻烦; printf是线程安全的，也就是自己做了线程同步的处理。
                    printf("Alpha: %f; Process: %.2f%%; Words/sec: %ld \r", alpha, word_count_actual*100/(double)(iters*train_words+1), word_count_actual/(now_time-start_time+1));
                    fflush(stdout);
                }

                word = sentence[sentence_position];
                word_count++;

                for(int c=0; c<embedding_size; c++) {x[c] = 0.0;}  // 初始化x和e
                for(int c=0; c<embedding_size; c++) {e[c] = 0.0;}
                window_actual = (int)rand() % window + 1;

                // 如果是cbow结构，则进行相应的处理
                if(cbow)
                {
                    // 读取当前词的上下文，构造隐层向量
                    context_word_count = 0;
                    for(int i=sentence_position-window_actual; i<=sentence_position+window_actual; i++)
                    {
                        if(i != sentence_position)
                        {
                            if(i < 0) {continue;}
                            if(i >= sentence_length) {break;}
                            context_word = sentence[i];
                            for(int c=0; c<embedding_size; c++) {x[c] += word_embeddings[context_word][c];}
                            context_word_count++;
                        }
                    }

                    // 如果当前词的上下文存在，那么就进行一次训练
                    if(context_word_count)
                    {
                        for(int c=0; c<embedding_size; c++) {x[c] = x[c] / (double)context_word_count;}  // 将隐层向量做一个平均
                        
                        // 如果是hs，进行相应的处理
                        if(hs)
                        {
                            for(int d=0; d<vocab[word].codelen; d++)
                            {
                                f = 0.0;
                                for(int c=0; c<embedding_size; c++) {f += x[c] * setas[vocab[word].path[d]][c];}
                                if(f <= -max_exp) {continue;}
                                else if(f >= max_exp) {continue;}
                                else q = expTable[(int)((f+max_exp)*(exp_table_size/max_exp/2))];
                                g = (1 - vocab[word].code[d] - q) * alpha;
                                for(int c=0; c<embedding_size; c++) {e[c] += g * setas[vocab[word].path[d]][c];}
                                for(int c=0; c<embedding_size; c++) {setas[vocab[word].path[d]][c] += g * x[c];}
                            }
                        }

                        // 如果是negative，进行相应的处理
                        if(negative)
                        {
                            for(int d=0; d<negative+1; d++)
                            {
                                if(d==0) {target = word; label = 1;}  // 当前词是正样本，其它词都是负样本
                                else
                                {
                                    target = sample_table[(int)rand()%table_size];
                                    if(target==word) {continue;}
                                    label = 0;
                                }
                                f = 0.0;
                                for(int c=0; c<embedding_size; c++) {f += x[c] * betas[target][c];}
                                if(f < -max_exp) {q=0;}
                                else if(f > max_exp) {q=1;}
                                else {q = expTable[(int)((f+max_exp)*(exp_table_size/max_exp/2))];}
                                g = (label - q) * alpha;
                                for(int c=0; c<embedding_size; c++) {e[c] += g * betas[target][c];}
                                for(int c=0; c<embedding_size; c++) {betas[target][c] += g * x[c];}
                            }
                        }

                        // hs和negative对embedding的更新都是一样的，更新当前词的上下文词的embedding，不更新当前词
                        for(int i=sentence_position-window_actual; i<=sentence_position+window_actual; i++)
                        {
                            if(i != sentence_position)
                            {
                                if(i < 0) {continue;}
                                if(i >= sentence_length) {break;}
                                context_word = sentence[i];
                                for(int c=0; c<embedding_size; c++) {word_embeddings[context_word][c] += e[c];}
                            }
                        }
                    }
                }

                // skip_gram结构
                if(skip_gram)
                {
                    for(int c=0; c<embedding_size; c++) {x[c] = word_embeddings[word][c];}  // 这些上下文词都对于同一个中心词
                    // 对每一个上下文词都要进行处理
                    for(int i=sentence_position-window_actual; i<=sentence_position+window_actual; i++)
                    {
                        if(i != sentence_position)
                        {
                            if(i < 0) {continue;}
                            if(i >= sentence_length) {break;}
                            for(int c=0; c<embedding_size; c++) {e[c] = 0.0;}
                            context_word = sentence[i];

                            // 如果是hs，则进行相应处理
                            if(hs)
                            {
                                for(int d=0; d<vocab[context_word].codelen; d++)
                                {
                                    f = 0.0;
                                    for(int c=0; c<embedding_size; c++) {f += x[c] * setas[vocab[context_word].path[d]][c];}
                                    if(f <= -max_exp) {continue;}
                                    else if(f >= max_exp) {continue;}
                                    else q = expTable[(int)((f+max_exp)*(exp_table_size/max_exp/2))];
                                    g = (1 - vocab[context_word].code[d] - q) * alpha;
                                    for(int c=0; c<embedding_size; c++) {e[c] += g * setas[vocab[context_word].path[d]][c];}
                                    for(int c=0; c<embedding_size; c++) {setas[vocab[context_word].path[d]][c] += g * x[c];}
                                }
                            }

                            // 如果是negative，则进行相应处理
                            if(negative)
                            {
                                for(int d=0; d<negative+1; d++)
                                {
                                    if(d==0) {target = context_word; label = 1;}  // 需要注意的是这里是对word进行负采样，而不是对上下文词进行负采样!!!
                                    else
                                    {
                                        target = sample_table[(int)rand()%table_size];
                                        if(target==context_word) {continue;}
                                        label = 0;
                                    }
                                    f = 0.0;
                                    for(int c=0; c<embedding_size; c++) {f += x[c] * betas[target][c];}
                                    if(f < -max_exp) {q=0;}
                                    else if(f > max_exp) {q=1;}
                                    else {q = expTable[(int)((f+max_exp)*(exp_table_size/max_exp/2))];}
                                    g = (label - q) * alpha;
                                    for(int c=0; c<embedding_size; c++) {e[c] += g * betas[target][c];}
                                    for(int c=0; c<embedding_size; c++) {betas[target][c] += g * x[c];}
                                }
                            }

                            // hs和negative更新embedding的方式是一样的
                            for(int c=0; c<embedding_size; c++) {word_embeddings[word][c] += e[c];}
                        }
                    }
                }
                
                sentence_position++;  // 处理句子中的下一个词
            }  // 遍历句子
            sentence_position = 0;
            // 如果已经读到文件结尾，或者当前线程需要处理的数据都读完了，那么将文件指针放回本线程起始的位置
            if(fin.eof() || fin.tellg()>=file_size/threads_num*(thread_id+1)) {fin.clear(); fin.seekg(file_size/threads_num*(thread_id), ios::beg); break;}
        }  // 多次构造句子
    }  // 外层迭代
}


// 训练模型
void train_model()
{
    start_alpha = alpha;
    // 开多线程训练模型，并且线程之间没有加锁，所以有可能造成精度上的损失
    thread threads[threads_num];
    cout<<"threads_num: "<<threads_num<<endl;

    for(int i=0; i<threads_num; i++) {threads[i] = thread(train_thread, i);}
    for(int i=0; i<threads_num; i++) {threads[i].join();}

    // 将训练好的词向量写入文件
    write_vecfile();
}


// 从命令行中解析参数，str是要获取的参数名
int ArgPos(string str, int argc, char* argv[]) 
{
    for (int i = 1; i < argc; i++) 
    {
        if (str==string(argv[i]))
        {
            if (i == argc - 1) {cout<<"Argument missing for "<<str<<"\n"; exit(1);}
            return i;
        }
    }
    return -1;
}

int main(int argc, char* argv[])
{
    int i;
    if ((i = ArgPos("-embedding_size", argc, argv)) > 0) embedding_size = atoi(argv[i+1]);
    if ((i = ArgPos("-train_file", argc, argv)) > 0) train_file = string(argv[i+1]);
    if ((i = ArgPos("-output_file", argc, argv)) > 0) output_file = string(argv[i+1]);
    if ((i = ArgPos("-cbow", argc, argv)) > 0) cbow = atoi(argv[i+1]);
    if ((i = ArgPos("-skip_gram", argc, argv)) > 0) skip_gram = atoi(argv[i+1]);
    if ((i = ArgPos("-hs", argc, argv)) > 0) hs = atoi(argv[i+1]);
    if ((i = ArgPos("-negative", argc, argv)) > 0) negative = atoi(argv[i+1]);
    if ((i = ArgPos("-sample", argc, argv)) > 0) sample = atof(argv[i+1]);  // 对高频词进行下采样(注意，这里不是负采样)
    if ((i = ArgPos("-alpha", argc, argv)) > 0) alpha = atof(argv[i+1]);
    if ((i = ArgPos("-window", argc, argv)) > 0) window = atoi(argv[i+1]);
    if ((i = ArgPos("-iters", argc, argv)) > 0) iters = atoi(argv[i+1]);
    if ((i = ArgPos("-min_count", argc, argv)) > 0) min_count = atoi(argv[i+1]);  // 单词出现频率的下限
    if ((i = ArgPos("-threads_num", argc, argv)) > 0) threads_num = atoi(argv[i+1]);  // 线程个数

    cout<<"训练数据为: "<<train_file<<endl;

    // 时钟，测试运行时间(注意不能用clock，因为clock不支持多线程)
    time_t cstart, cend;

    // 读取数据
    cstart = time(0);
    read_data();
    cend = time(0);
    cout<<"Read data runs "<<(cend-cstart)<<endl;

    // 初始化网络结构
    cstart = time(0);
    init_net();
    cend = time(0);
    cout<<"Inititalize net runs "<<(cend-cstart)<<endl;

    // 模型训练
    cstart = time(0);
    train_model();
    cend = time(0);
    cout<<endl;
    cout<<"Train model runs "<<(cend-cstart)<<endl;
}