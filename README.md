# Multi-Patching

- 제 3회 ETRI 휴먼이해 인공지능 논문경진대회

# TEAM USINMKO
- 유선우 (DGIST)
- 이재현 (DGIST)
- 김대원 (DGIST)
- 최기원 (DGIST)

# Preprocessing
```
>> python3 ./preprocess/Preprocess.py
```

참고 사항
```
ㄴ data
    ㄴ val_dataset
    ㄴ test_dataset
    ㄴ val_label.csv
    ㄴ user01-06
    ㄴ user07-10
    ㄴ user11-12
    ㄴ user21-25
    ㄴ user26-30
    ㄴ train_label.csv
```

# How to Inference

## stage 1
```
>> python3 ./model/stage1_test.py
```
새로운 dataset을 inference하기 위해서는 './data_npy/'의 val과 test 폴더에 .npy 파일이 저장되어 있어야 한다.
또한 새롭게 진행한 train model을 적용하려면 해당 모델 주소로 수정해야한다.

## stage 2
```
>> python3 ./model/stage2_test.py
```
stage 1 test 주의 사항과 동일


# How to Train

## stage 1
```
>> python3 ./model/stage1_train.py
```
새로운 dataset을 학습하기 위해서는 코드 내의 'hr', 'pedo' 주소를 수정해야한다.

## stage 2
```
>> python3 ./model/stage2_train.py
```


