//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6 //sigmoid(6) ~ 1, sigmoid(-6)~0
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;//count
  int *point;
  char *word, *code, codelen; // huffman code & word 
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter=1,file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;// i is the index of vocab[i]
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;//carriage return
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin); //passing 배y '\n'
        break;
      }
      if (ch == '\n') {// if it starts with \n from the beginning
        strcpy(word, (char *)"</s>"); // 엔터면 그거저장한다. vocab[0] 에 있음 
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0; // 마무리로 null 값 넣어준다. 
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word(index) in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash]; // vocabulary에 있으면 
    hash = (hash + 1) % vocab_hash_size; // 없으면 move to next slot
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++; 
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size; //빈자리 찾아서 +1만큼씩 이동 
  vocab_hash[hash] = vocab_size - 1;//vocab_hash에는  원래 vocab_size 저장
  return vocab_size - 1; // 원래 vocab_size 반환
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  //infrequent word  제거하고 vocab 앞으로 당김 . 
  
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}


// Create binary Huffman tree using the word counts
// Frequent words will have short unique binary codes
void CreateBinaryTree() {
  // min1i is the first son, and min2i is the second son; point[] is used to store the index of all father nodes
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];  //code is 0-1 sequence
  //전체 크기를 W 의 vocabulary(leaf node) 크기와 W'의 vocab이라고 생각할 수 있는 tree의 non-leafnode들을 다 한꺼번에 할당하였다. 
  long long *count = (long long *)calloc(vocab_size * 2 +1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 +1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 +1, sizeof(long long));
  //initialize count[], left part stores the occur times of words, and the right part is initialized by infinite
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  // pos1 scans from the end of the left part of "count[]" to its beginning, while pos2 scans from the beginning of the second part of "count[]"
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  //following a step within the second part of the count vector
  //in following for loop, "a" is a pointer scanning the second part of "count[]", because the non-leaf nodes is "one" less than the leaf nodes, hence "a" only needs to 
  //scan "vocab_size-1" entries.
  for (a = 0; a < vocab_size - 1; a++) {
    // next, find two smallest nodes 'min1i, min2i'. First, find the min1i
    if (pos1 >= 0) //if pos1 has not passed the left border of "count[]"
    {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {  //count 가 엄청 큰 숫자만 있을땐 이경우 있을 수 있음 전체를 두개씩 쌍으로 짝짓겠다는 뜻임
      min1i = pos2;
      pos2++;
    }
    //second,  find the min2i
    if (pos1 >= 0)
    {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    //already found the two sons, add their counts as their father's count
    count[vocab_size + a] = count[min1i] + count[min2i];
    //record their father's position in "count[]"
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    // let the code choosing the second son is "1", the first son is naturally "0" for the initialization of "binary[]"
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word "a"
  for (a = 0; a < vocab_size; a++)
  {
    b = a; // "b" is used to find its father, starting from itself
    i = 0; // "i" is for counting the number of its ancestors, namely the length of its code
    //find all ancestors by down-up style
    while (1)
    {
      code[i] = binary[b];
      point[i] = b;//point[] stores the index of its ancestors from itself to the highest (not consider the root) it's like a trace from bottom to top 
      i++;
      b = parent_node[b];
      //if "b" meets the root, over. because there are only vocab_size-1 non-leaf nodes, hence the index of root is vocab_size * 2 - 2 in "count[]"
      if (b == vocab_size * 2 - 2) break;
    }//while outputs the temporary code[] and point[], both are down-up style
    vocab[a].codelen = i; //i is the length of code
    //next, convert above temporary code[] and point[] inversely. First, let the first ancestor be the root (index is vocab_size-2)
    vocab[a].point[0] = vocab_size - 2; //W'의 인덱스를 의미하므로 2*vocab_size-2가 아니라 vocab_size-2
    for (b = 0; b < i; b++)
    {
      vocab[a].code[i - b - 1] = code[b];
      //note that point[0] is the index of the word itself, so "point[0] - vocab_size" is surely a negative number. Putting this negative number into the       
      // "vocab[a].point[i]" (where "i" is the code length) as a flag.
      vocab[a].point[i - b] = point[b] - vocab_size; // syn1의 index로 바꿔주는 것이다. W' 
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

//vocabulary 만든다. 
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1; //initialize to 1
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>"); // 공백문자 먼저 집어넣는다. 
  while (1) {
    ReadWord(word, fin); //단어 하나 읽어들여서 
    if (feof(fin)) break; // 파일 마지막이면 탈출
    train_words++;//train한 총단어수 증가 

    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }

    i = SearchVocab(word); //word가 vocab에 있는지 없는지 확인 & word position반환 
    //단어 카운트 
    if (i == -1) {//vocab에 없으면 
      a = AddWordToVocab(word); // vocab에 word를 셋팅
      vocab[a].cn = 1;
    } else vocab[i].cn++;

    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();  //  로드팩터 0.7 넘으면 vocab 줄인다 
  }
   // vocab 다 만들어졌으면 sorting한다. 
  SortVocab(); //binary tree 만들때 huffman code 사용하기 때문에 필요함 
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}




//SaveVocab ReadVocab 는 파일 io에서 나온것임 
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}




void InitNet() {
  long long a, b;
  //시작주소를 128byte의 배수로 할당한다는것이다. 
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real)); // Allocate memory to make Matrix W 
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));// Allocate memory to make Matrix W' from syn1
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1[a * layer1_size + b] = 0; // Initialize W' all to zero 
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real)); // Allocate memory to make Matrix W' from syn1neg
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1neg[a * layer1_size + b] = 0; // Initialize W' all to zero 
  }
  // layer size 나눠주는 이유는 sigmoid function에 x'*x를 집어 넣으면 -6보다 작거나 6보다 클수 있다. x'*x/layersize로 해주므로 dot product 값이 -6 에서 6 사이로 놀 수 있게 해준다.  
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)  // Initialize the Matrix W to random number 
   syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size; // rand_max is the largest value that rand() function can return  0 ~ rand_max
  CreateBinaryTree();
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label,local_iter = iter;
  unsigned long long next_random = (long long)id; // thread id 
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real)); //projection layer 
  real *neu1e = (real *)calloc(layer1_size, sizeof(real)); // error layer 
  FILE *fi = fopen(train_file, "rb"); // read train file as binary 
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi, (file_size / (long long)num_threads) * (long long)id, SEEK_SET);//thread id 마다 시작점을 파일 의  (thread id/ 총 thread크기) 로 출발시킨다. 
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter*train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter*train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }


    if (sentence_length == 0) { //초기과정 
      //문장하나 만들자
      while (1) {
        word = ReadWordIndex(fi); // word하나 읽어들임 index
        if (feof(fi)) break;
        if (word == -1) continue; // 없으면 패스 
        word_count++; 
        if (word == 0) break; // <\s> 즉 엔터이면 break 왜냐하면 다른 글이기 때문이다.  
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn; //ran produces some number which is a function of the frequency of a word in the processed text, and train_words the total number of tokens (previously smoothed by an alpha term, but let's ignore this).
          next_random = next_random * (unsigned long long)25214903917 + 11; //next_random is assigned a value by a function which looks like a Linear Congruential Generator. I am getting this information from here: 
          if (ran < (next_random & 0xFFFF) / (real)65536) continue; //next_random is masked to keep only the lowest 16 bits unchanged. So I now have a binary number with max (decimal) value 65535. Divided by 65536, this gives me a number between 0 and 1, The hexadecimal notation 0xffff makes it clear that all bits in the number are 1. The 65535 is the same number
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--; 
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }



    word = sen[sentence_position];
    if (word == -1) continue;
    //initialize neul, neule
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11; 
    b = next_random % window;  // b에 의해 target word의 window 앞에둘지 뒤에둘지 중간에둘지 정한다 램덤하게 
    //평균내지 않고 그냥 합친거 
    //train the cbow architecture
    //cbow 와 skip gram 의 차이는 hs or negative를 indent 하느냐 안하느냐 로 구분할 수 있다. 
    if (cbow) {  
      // in -> hidden
      //namely sum all the embedding of context words by element-wise style, produce the information of hidden layer "neul[]"
      // windowsize 5 일때 b=1이면 456 , b=4이면  123456789   '5'는 target word 자리이다. 

      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {//a 가 window이면 target word이므로 trainning할때 포함시키면안됨 
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        //target word is sen[sentence_position]
        last_word = sen[c]; //here, last_word should be a context word
        if (last_word == -1) continue;
        //sum the context words' embeddings
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      }
      //in hierarchical softmax, each target word is replaced by its ancestors for computation. Namely, use each father's feature vector to 
      //compute with the sum of context words. Father's feature vector is also the weight vector from hidden layer to output layer.
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        //f is simil : v'(j)*h
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output : similarity of v'(j)*h
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate ..'g' is the error 
        //code 0 or code 1 이면 sigmoid 값이 1 이되도록 훈련시킬것인데 코드에따라 값이 다르므로 0일땐 simil 이 + 쪽으로가게끔 1일땐 - 쪽으로가게끔 트레이닝 시킨다. 
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
      }

      // NEGATIVE SAMPLING 
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        //positive target
        if (d == 0) {
          target = word;
          label = 1;
        } 
        //negative target 
        else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1; // <\s>이면 다시 설정 
          if (target == word) continue; // 자기 자신이면 패스 
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }


      // hidden -> in    C개 워드 다 없데이트 
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } 


    else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c]; //context 
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0; // 한 단어당 한번 trainning시키므로 
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue; // gradient가 0이므로 
          else if (f >= MAX_EXP) continue;// gradient가 0이므로 
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; // 실제 f 값을 EXP_TABLE_SIZE 의 몇등분 인지로 바꿔준다. 
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden         :EHi
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;

  //if it has a vocab input file read it else train vocab
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //if there is a save  vocab file save it 
  if (save_vocab_file[0] != 0) SaveVocab();
  //
  if (output_file[0] == 0) return;
  InitNet(); //  initialize W , W' 
  if (negative > 0) InitUnigramTable(); // if you choosed negative make InitUnigramTalbe()
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      if (vocab[a].word != NULL) {
        fprintf(fo, "%s ", vocab[a].word);
      }
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {// Run K-means on the word vectors
    // save the word classes
    // run kmeans on word vectors to get word classes
    int clcn = classes, iter = 10, closeid;
    // sizes of each cluster
    int *centcn = (int *)malloc(classes * sizeof(int));

    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    // classes of each word in vocab
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    // center vector
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    // initialize class labels of words in a wheel way
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
      // iterative training
    for (a = 0; a < iter; a++) {
      // reset centers to all zeros
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
        // reset cluster size to 1 element
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      // for each word (for each feature of it)
      // center_vec += word_vec
      // center_size += 1
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          // cl[c] is the cluster index of word at c
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      // for each cluster (for each feature of cluster center)
      // cent_vec /= cluster_size
      // cent_vec `~ normalized by l2 norm
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          // taking average
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        // closev = l2 norm of the center vec
        // normalize the center vec by its l2 norm
        // NORMALIZATION OF CENTER VECTORS FOR LATER DISTANCE COMPARISON
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }

     // ASSIGN each word to the corresponding center
      // for each word, for each cluster, 
      // calculate the dist between the word vec and the cluster center 
      // (cluster vecs have all been normalized, so just use the inner product)
      // find the closest cluster center to the current word vector
      // closev (the closest dist so far), closeid (the closest cluster id so far)
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  free(pt);
  DestroyVocab();
}
//It is fundamental to c that char**x and char*x[] are two ways of expressing the same thing. Both declare that the parameter receives a pointer to an array of pointers. Recall that you can always write:
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    //str 은 있지만 parameter가 존재 하지않을때 단, 마지막이어야된다. 
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
//옵션을 하나도 입력하지 않으면 argc 는 1이 됩니다. 즉 argc 는 항상 1 이상입니다. 0이 되지 않습니다.
int main(int argc, char **argv) {
  int i;
  clock_t before;
  double result;
  //argc는 명령행 옵션의 갯수 (1은 자기 실행파일 만 있을때)
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;

  //모든 parameter를 변수에 저장 
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);

  //allocate memory space to maintain vocabulary of handled corpus. calloc will save you a call of memset.
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int)); // key is hash value is position 
  // c언어는 파이썬처럼 dictionary를 만들수 없다 vocab[word]= vocab_word 
  //따라서 word를 hash 값으로 바꾸고 그것에 index를 설정해주면 그  index갖고  vocab_word를 indexing할 수 있다. 
  // 그냥 hash 없이 word:index 로하면 vocab 자체를 찾으려고 전체를 탐색해야된다.그래서 해쉬를 사용하면  O(n) -> O(1)로 바꿔준다. 
  // vocab_hash[word_hash]=word_idx -> 이거 해주는 이유가 vocab[word_hash]=vocab_word 로하면 배열을 효율적으로 사용하지 못하기 때문이다. 
  // vocab[word_idx]=vocab_word
  // 위와같은 결과로 인해 우리는 vocab[word]= vocab_word  와같은 기능을 구현할 수 있다. 



  //prepare a table to look up values of sigmoid function quickly.
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  //EXP_TABLE_SIZE is 1000 as default
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }

  before = clock();

  //train the model 
  TrainModel();
  result=(double)(clock()-before)/CLOCKS_PER_SEC;

  printf("it took %5.2f sec\n",result);

  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;
}
