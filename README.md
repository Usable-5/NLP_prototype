# NLP_prototype
Chatbot prototype based on tensorflow transformer, KoGPT2

### Data
- Q(질문), A(답변) 구조로 구성
- 본 모델에서는 label (답변 구분)은 사용하지 않음
  

### Output (KoGPT2)
    Input:  나랑 영화 보자
    Output 1:  응 시간 될 것 같아
    Output 2:  응 가야지

    Input:  커피 한 잔 할까?
    Output 1:  곧 마실 거야
    Output 2:  응응 내려갈게

    Input:  좀 이따가 밥먹을래?
    Output 1:  그래~
    Output 2:  글쎄
    
### Output (Transformer)  
    Input: 나랑 영화 보자
    Output: 그래 !

    Input: 내일 시간 괜찮을까요?
    Output: 네 괜찮습니다 !

    Input: 커피 한 잔 할까?
    Output: 그래 !


---
### Reference
Kochat: https://github.com/hyunwoongko/kochat  
Transformer Chatbot: https://github.com/ukairia777/tensorflow-transformer  
KoGPT2 Chatbot: https://github.com/ukairia777/tensorflow-kogpt2-chatbot