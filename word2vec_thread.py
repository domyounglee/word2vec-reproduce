import os
import numpy as np
import _pickle as pickle
import random as rn
import time
import math as m
import threading

class vocab_word:
    #trace from the root
    point=[]
    #huffman code
    code=[]
    def __init__(self, word, count):
        self.count = count
        self.word = word
     
    #object print 
    def __repr__(self):
        return repr(( self.word,self.count))


hs=0
negative=10
table_size=int(1e7)
EXP_TABLE_SIZE=1000
MAX_EXP=6
MAX_SENTENCE_LENGTH=1000
layer_size=200

vocab_size=np.int64(0)
vocab=[]
vocab_dic={}
train_words=np.int64(0)
min_count=5
table=[]
iteration=2


power=0.75
window=7
cbow=1

alpha = 0.025

sample=1e-4
worker=4

_filename_ = 'output.txt'
file_size= os.path.getsize(_filename_)



#making a exponential table 
expTable=[]
for i in range(EXP_TABLE_SIZE):
    expTable.append(m.exp((float(i)/EXP_TABLE_SIZE*2-1)*MAX_EXP))
    expTable[i]=expTable[i]/(expTable[i]+1)
    #print expTable[i]


#in case of load
vocab_dic= pickle.load( open( "vocab_dic.p", "rb" ) )
vocab=pickle.load( open( "vocab.p", "rb" ) )
table=pickle.load( open( "table.p", "rb" ) )
vocab_size=len(vocab)
#text8:16718844
train_words=765887525
print(vocab[0].code)
print(vocab[-1].word)
print("vocabulary loaded")

    
def ReadWord(text_file):
    # Reads a single word from a file
    # assuming SPACE + TAB + EOL to be word boundaries
    a=0
    one_word=''
    MAX_STRING=100
    
    for one_char in iter(lambda: text_file.read(1), ""):
        if(one_char==13):continue
        if(one_char==' ' or one_char=='\t' or one_char=='\n'):
            if(a>0):
                break
                
            if(one_char=='\n'): 
                return "</s>"
            else: continue
        
        a+=1
        one_word+=one_char
        if(a==MAX_STRING): return one_word
    return one_word

def ReadWordIndex(pfile):
    word=ReadWord(pfile)
    #End of File
    if( word==''):return -1
    #whether it is the key in the vocab
    if(word in vocab_dic):return vocab_dic[word]
    #not then word=-2
    else: return -2
    
"""

   

def AddWordToVocab(word):
    global vocab_size
    vocab.append(vocab_word(word, 0))
    vocab_dic[word]=vocab_size
    vocab_size+=1
    return  vocab_size-1
    

def SortVocab():
    global vocab_size,train_words,vocab
   
    train_words=0
    vocab_dic.clear()
    vocab=vocab[0:1]+sorted(vocab[1:], key=lambda vocabulary: vocabulary.count,reverse=True)
    size=vocab_size
    print(len(vocab))
    del_idx=0
    print(vocab[0].word)
    vocab_dic[vocab[0]]=0
    #remove unneccessary list element

    for i in range(1,size):
        
        if(vocab[i].count<min_count):
            del_idx=i
            break
            
        else:
            word=vocab[i].word
            #re build vocab_dict 
            vocab_dic[word]=i
            train_words+=vocab[i].count
            
    vocab[:]=vocab[0:del_idx]
    vocab_size=len(vocab)



    


def LearnVocabFromTrainFile():
    global train_words
    word=''
    global vocab_size
    AddWordToVocab("</s>")
    print(vocab[0].word)
    print("flag1")
    with open('text8', 'r') as train_file:
        while(1):
            word=ReadWord(train_file)
            if(word==''):break
            train_words+=1
            
            if(vocab_dic.has_key(word)):
                vocab[vocab_dic[word]].count+=1
            else: 
                a=AddWordToVocab(word)
                vocab[a].count=1
    print("flag2")
    SortVocab()
    print("flag3")
    #if(debug_mode>0):
    #    print("Vocab_size",vocabsize)
    #    print("Train_words",train_words)
    pickle.dump( vocab_dic, open( "vocab_dic.p", "wb" ) )
    pickle.dump( vocab, open( "vocab.p", "wb" ) )
 








def InitUnigramTable():
    
    train_words_pow=0
    
    for a in range(vocab_size):
        train_words_pow+=vocab[a].count**power
    
    i=0
    d1=(vocab[i].count**power)/train_words_pow
    for a in range(int(table_size)):
        table.append(i)
        if(a/float(table_size)>d1):
            i+=1
            d1+=(vocab[i].count**power)/train_words_pow
        if(i>vocab_size):
            i=vocab_size-1
    pickle.dump( table, open( "table.p", "wb" ) )

def CreateBinaryTree():
    
    pos1=vocab_size-1
    pos2=vocab_size
    min1i=0
    min2i=0
    count=[vocab[i].count for i in range(vocab_size)]+[1e15 for i in range(vocab_size-1)]
    binary=[0 for i in range(2*vocab_size)]
    parent_node=[0 for i in range(2*vocab_size)]

    #think of a as each non-leaf node index that are going to connet with other nodes
    for a in range(vocab_size-1):
        #next, find two smallest nodes 'min1i, min2i'. First, find the min1i
        #if pos1 has not passed the left border of "count[]"
        if(pos1>=0):
            if(count[pos1]<count[pos2]):
                min1i=pos1
                pos1-=1
            else:
                min1i=pos2
                pos2+=1
        else:
            min1i=pos2
            pos2+=1
        #second,  find the min2i
        if(pos1>=0):
            if(count[pos1]<count[pos2]):
                min2i=pos1
                pos1-=1
            else:
                min2i=pos2
                pos2+=1
        else:
            min2i=pos2
            pos2+=1
        #already found the two sons, add their counts as their father's count
        count[vocab_size + a] = count[min1i] + count[min2i]
        #record their father's position in "count[]"
        parent_node[min1i] = vocab_size + a
        parent_node[min2i] = vocab_size + a
        # let the code choosing the second son is "1", the first son is naturally "0" for the initialization of "binary[]"
        binary[min2i] = 1;
        
    #think of each a as leaf node index
    for a in range(vocab_size):
        code=[] #huffman code
        point=[] #trace 
        b = a; # "b" is used to find its father, starting from itself
        i = 0; # "i" is for counting the number of its ancestors, namely the length of its code
        #find all ancestors by down-up style
        while(1):
            code.append(binary[b])
            point.append(b)
            i+=1
            b = parent_node[b]
            if(b==2*vocab_size-2):break
        vocab[a].code_len=i
        vocab[a].point=[x-vocab_size for x in point]+[vocab_size-2]
        vocab[a].point.reverse()
        vocab[a].code=code
        vocab[a].code.reverse()
        
"""
def InitNet():
    global syn0
    global syn1
    global syn1neg
  
    syn0=(np.random.rand( vocab_size,layer_size)-0.5)/float(layer_size)
    if(hs):
        syn1=np.zeros( (vocab_size,layer_size), dtype=np.float64 )
    if(negative>0):
        syn1neg=np.zeros( (vocab_size,layer_size), dtype=np.float64 )
   





def TrainModelThread(id):
    global alpha
    global syn0
    global syn1
    global syn1neg
    global iteration 
    word=''
    last_word='' 
    sent_pos = 0
    sent_len=0
    word_count = 0
    word_count_actual=0
    last_word_count = 0
    sen=[]
    l1=-1
    l2=-1
    target=-1
    label=-1
    starting_alpha=alpha
    jj=0
    
    neu1=np.zeros( layer_size, dtype=np.float64 )
    neu1e=np.zeros( layer_size, dtype=np.float64 )
    fi = open(_filename_, 'r',encoding='UTF8')
    fi.seek(file_size*float(id)/worker,0)
    print("id : " , id,"current pos: " , fi.tell())



    while(1):
        
        if(word_count - last_word_count > 10000):
            word_count_actual += word_count - last_word_count
            last_word_count = word_count
            alpha = starting_alpha * (1 - word_count_actual / float(train_words + 1))
            if (alpha < starting_alpha * 0.0001): alpha = starting_alpha * 0.0001
           
        #make a new sen
        if(sent_pos==0):
            jj+=1
            if(jj%1000==0):
                print(id," : make sent 1000")
            

            while(1):
                word=ReadWordIndex(fi)
                if(word==-1):break #EOF
                if(word==-2): #not in vocab
                    continue
                word_count+=1
                if(word==0):break #<\s>
                
                #sample is threshold
                if(sample>0):
                    f=vocab[word].count/float(train_words)
                    
                    p=1-m.sqrt(sample/(f))
                    if(rn.random()<p):continue 
                
                sen.append(word)
                if(len(sen)>=MAX_SENTENCE_LENGTH):break 
            sent_len=len(sen)
            
           ##random 
            
            
        if (sent_len==0):continue


        if(word==-1 or word_count > train_words / float(worker)):#EOF
            word_count_actual+=word_count - last_word_count
            iteration-=1
            if(iteration==0):break
            word_count=0
            last_word_count=0
            sent_pos=0
            fi.seek(file_size*float(id)/worker,0)
            continue 


        word=sen[sent_pos]
        if(word==-2):continue 
        
        #initialize        
        neu1=np.zeros( layer_size, dtype=np.float64 )
        neu1e=np.zeros( layer_size, dtype=np.float64 )
        
        b = rn.randint(0,10**8)%window
        
      
        #train the CBOW architecture
        if(cbow):
            
            #in->hidden 
            for a in range(b,window * 2 + 1 - b):
                if(a!=window):
                    c=sent_pos-window+a
                    if (c < 0) : continue
                    if (c >= sent_len) :continue
                    #target word is sen[sentence_position]
                    last_word = sen[c]
                    #here, last_word should be a context word
                    if (last_word == -1) : continue
                    #sum the context words' embeddings
                    #for i in range(layer_size): neu1[i] += syn0[last_word][i]
                    neu1 += syn0[last_word]
            
            #if it is hierachial softmax
            if(hs):
                for d in range(len(vocab[word].code)):
                    #f is similarity : v'(j)*h
                    f = 0
                    l2 = vocab[word].point[d] 
                    # Propagate hidden -> output : similarity of v'(j)*h
                    f = np.dot(neu1,syn1[l2])
                
                    if (f <= -MAX_EXP): continue
                    elif (f >= MAX_EXP): continue
                    else: f = expTable[int(((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))]
                    
                    g = (1 - vocab[word].code[d] - f) * alpha
                    
                    # Propagate errors output -> hidden
                    neu1e += g * syn1[l2]
                    #Learn weights hidden -> output
                    syn1[l2] += g * neu1


            #if it is negative sampling    
            if (negative>0): 
                for i in range(negative+1):
                    if i==0:
                        target=word
                        label=1
                    else:
                        rand_num=rn.randint(0,int(table_size-1))
                        target=table[rand_num]
                        if target == 0 : target = rand_num % (vocab_size - 1) + 1
                        if target == word : continue
                        label = 0
                    
                    l2=target
                    f = 0
                    f = np.dot(neu1, syn1neg[l2])
                    if (f >= MAX_EXP) : 
                        g = (label - 1) * alpha
                    elif (f <= -MAX_EXP) : 
                        g = (label - 0) * alpha
                    else : 
                        g = (label - expTable[int(((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))]) * alpha
                    neu1e += g * syn1neg[l2]
                    syn1neg[l2] += g * neu1
                
                
            #hidden -> in    update all the c words
            for i in range(b,window * 2 + 1 - b):
                if(i!=window):
                    c=sent_pos-window+i
                    if (c < 0) : continue
                    if (c >= len(sen)) :continue
                    #target word is sen[sentence_position]
                    last_word = sen[c]
                    #here, last_word should be a context word
                    if (last_word == -1) : continue
                    syn0[last_word]+=neu1e

                    

        
        else: #Train skip-gram
            #in->hidden 
            for a in range(b,window * 2 + 1 - b):
                if(a!=window):
                    c=sent_pos-window+a
                    if (c < 0) : continue
                    if (c >= len(sen)) :continue
                    #target word is sen[sentence_position]
                    last_word = sen[c]
                    #here, last_word should be a context word
                    if (last_word == -1) : continue
                    l1=last_word
                    #for i in range(layer_size): neu1[i] += syn0[last_word][i]
                    neu1e=np.zeros( layer_size, dtype=np.float64 )
                    #if it is hierachial softmax
                    if(hs):
                        for d in range(len(vocab[word].code)):
                            #f is similarity : v'(j)*h
                            f = 0
                            l2 = vocab[word].point[d] 
                            # Propagate hidden -> output : similarity of v'(j)*h
                            f += np.dot(syn0[l1],syn1[l2])
                            if (f <= -MAX_EXP): continue
                            elif (f >= MAX_EXP): continue
                            else: f = expTable[int(((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))]
                            
                            g = (1 - vocab[word].code[d] - f) * alpha
                            # Propagate errors output -> hidden
                            neu1e += g * syn1[l2]
                            #Learn weights hidden -> output
                            syn1[l2] += g * syn0[l1]

                    #if it is negative sampling    
                    if (negative>0): 
                        for i in range(negative+1):
                            if i==0:
                                target=word
                                label=1
                            else:
                                rand_num=rn.randint(0,int(table_size-1))
                                target=table[rand_num]
                                if target == 0 : target = rand_num % (vocab_size - 1) + 1
                                if target == word : continue
                                label = 0

                            f = 0
                            f = np.dot(syn0[l1], syn1neg[target])
                            if f > MAX_EXP : g = (label - 1) * alpha
                            elif f < -MAX_EXP : g = (label - 0) * alpha
                            else : g = (label - expTable[int(((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))]) * alpha;
                            neu1e += g * syn1neg[target]
                            syn1neg[target] += g * syn0[l1]

                    #hidden -> in    update words
                    syn0[l1]+=neu1e
                
                
        sent_pos+=1
        if(sent_pos>=sent_len):
            del sen[:]
            sent_pos=0
                
    
                













def TrainModel():
    global vocab_index 
    vocab_index={}
    global vocab_invindex 
    vocab_invindex={}
    threadlist=[]
    #LearnVocabFromTrainFile()
    

    InitNet()
    print("initialize finished")
    if(negative>0):
        table=pickle.load( open( "table.p", "rb" ) )
        #InitUnigramTable()
        print("Table finished")

    print("train starts")   
    try:
        """
        thread1=threading.Thread(target=TrainModelThread, args=(0,))
        thread2=threading.Thread(target=TrainModelThread, args=(1,))
        thread3=threading.Thread(target=TrainModelThread, args=(2,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()

        """
        
        for i in range(worker):
            threadlist.append(threading.Thread(target=TrainModelThread, args=(i,)))
        
        for i in range(worker):
            threadlist[i].start()

        for i in range(worker):
            threadlist[i].join()
        
    except:
        print("Error: unable to start thread")

    while 1:
        pass

    print("train finished")

    

    #save word index and weight matrix
    for i in range(vocab_size):
        vocab_index[vocab[i].word]=i
    
    #save word index and weight matrix
    for i in range(vocab_size):
        vocab_invindex[i]=vocab[i].word
    
    #save with pickle

    pickle.dump( syn0, open( "WeightMatrix.p", "wb" ) )
    pickle.dump( vocab_index, open( "vocab_index.p", "wb" ) )
    pickle.dump( vocab_invindex, open( "vocab_invindex.p", "wb" ) )
    
    return vocab_index,vocab_invindex





#print  top k  
def print_topk(query,idx,inv_idx,k):
    rank={}
    topk=[]
    for i in range(syn0.shape[0]):
        if(i==idx[query]):continue 
        sim=np.dot(syn0[idx[query]],syn0[i])
        rank[i]=sim
    topk=sorted(rank.items(), key=lambda x: x[1],reverse=True)
    
    for i in range(k):
        print(inv_idx[topk[i][0]])
    





def main():
    start_time = time.time()
    #train model 
    idx,invidx=TrainModel()
    print("---trainning time %s seconds ---" % (time.time() - start_time))
    #find top-k related word vectors 
    while(1):
        query=raw_input()
        if(vocab_index.has_key(query)):
            print_topk(query,idx,invidx,10)
            
        else: 
            print("It's not in the vocab")
            



    

if __name__ == "__main__":
    main()

    



