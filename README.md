# word2vec-reproduce (python으로 날로 짠 word2vec)
## code
commented myself the original word2vec code in korean & english 
-	word2vec.c 

reproduced myself 
-	word2vec.py 
-	word2vec_thread.py : used thread (debugging required)
-	evalutation.ipynb : evaluation 
-	make_voca : generates only vocabulary
materials & plot

## reference 

reference code: https://code.google.com/archive/p/word2vec/

word2vec 설명 슬라이드 : http://www.slideshare.net/lewuathe/drawing-word2vec  
Hierarchial softmax 코드설명 : https://yinwenpeng.wordpress.com/2013/09/26/hierarchical-softmax-in-neural-network-language-model/
sigmoid 코드 설명 :  https://yinwenpeng.wordpress.com/2013/12/18/word2vec-gradient-calculation/comment-page-1/
word2vec 코드 코멘트 : https://yinwenpeng.wordpress.com//?s=word2vec&search=Go
word2vec 설명 : https://joneswongml.wordpress.com/2014/01/11/word2vec/


### thread programming
쓰레드 프로그래밍 
http://www.joinc.co.kr/w/Site/Thread/Beginning/WhatThread
http://www.joinc.co.kr/w/man/3/pthread_create
http://repilria.tistory.com/228

### etc 
메모리 alignment : http://egloos.zum.com/wonchuri/v/2127834
			http://extern.tistory.com/2
			http://ikpil.com/359
			http://egloos.zum.com/studyfoss/v/5409933

gnu library : http://database.sarang.net/study/glibc/3.htm

c reference : http://itguru.tistory.com/104
		  http://www.joinc.co.kr/w/man/3/memset
		http://forum.falinux.com/zbxe/index.php?document_srl=413244&mid=C_LIB
		http://echosf.net/lecture/assembly/ (어셈블리 )

align : http://egloos.zum.com/studyfoss/v/5409933


자료구조 
해쉬 : http://juggernaut.tistory.com/entry/%ED%95%B4%EC%8B%9C-%ED%85%8C%EC%9D%B4%EB%B8%94-%EC%97%B4%EB%A6%B0-%EC%A3%BC%EC%86%8C-%EB%B0%A9%EC%8B%9D

