import os
import numpy as np
import cPickle as pickle


vocab_size=np.int64(0)
vocab=[]
vocab_dic={}
train_words=np.int64(0)
min_count=5


class vocab_word:
    point=[]
    code=[]
    def __init__(self, word, count):
        self.count = count
        self.word = word
     
    #object print 
    def __repr__(self):
        return repr(( self.word,self.count))
    
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
                if(one_char=='\n'):
                    text_file.seek(-1,1)
                break
                
            if(one_char=='\n'): 
                return "</s>"
            else: continue
        
        a+=1
        one_word+=one_char
        if(a==MAX_STRING): return one_word
    return one_word

def AddWordToVocab(word):
    global vocab_size
    vocab.append(vocab_word(word, 0))
    vocab_dic[word]=vocab_size
    vocab_size+=1
    return  vocab_size-1
    

def SortVocab():
    global vocab_size
    global train_words
    global vocab
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
    with open('test.txt', 'r') as train_file:
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
 
#save with pickle
pickle.dump( vocab, open( "vocab.p", "wb" ) )
pickle.dump( vocab_dic, open( "vocab_dic.p", "wb" ) )

LearnVocabFromTrainFile()
print(vocab)
print(vocab_size)
print(len(vocab_dic))