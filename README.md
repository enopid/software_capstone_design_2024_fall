# GNN based 3D Object classification model using Eigen pooling
# Project Summary
 그래프의 형태로 3d 객체를 다루는 기존 연구에서 전체 노드수를 줄이는 풀링에 있어서 몇가지 한계가 존재한다. 풀링을 사용하지 않는 MDC-GCN은 풀링을 사용하지 않은 만큼 층이 깊어짐에 따라 연산량이 크게 증가해 성능을 확보하기 위한 깊은 층을 구성하기 어려워 그래프를 사용하는 네트워크 중 비교적 낮은 성능을 보여준다. RimeshGNN과 meshformer의 경우 각각 edge-pooling과 bfs-pooling을 사용하는데 각 풀링 방식은 그래프의 구조적 의존성으로 인해 풀링될 요소의 선택과 갱신과정이 순차적으로 이루어져야되기 때문에 병렬화가 어려워 풀링시 시간이 오래걸린다는 단점이 있다. 

 본 과제에서는 그래프를 다루는 모델들의 풀링 방식을 개선하기 위해 매쉬를 그래프로 간주하여 GAT(Graph Attention Transformer)을 활용하여 네트워크를 구성하고 Eigen 풀링을 통한 병렬연산이 가능한 풀링 방식을 통해 연산량을 줄이면서 기존 모델의 성능을 유지하는 것을 목표로 한다.

# Code instruction
- 모델은 GAT와 Eigen pooling의 블록으로 이루어져있으며 전체적인 네트워크 구조는 하단의 그림과 같다. 
 
- 네트워크의 입력은 페이스 별로 13차원의 입력이 들어가며 페이스별 정보를 담은 13차원의 정보로 구성된다. 13차원의 정보는 페이스별 영역의 넓이, 법선의 방향, 중심점 7차원의 페이스에대한 직접적인 정보와 인접한 페이스와의 이면각 정보, 버텍스 법선과 페이브버선의 내적에 해당하는 곡률 정보에 해당하는 6차원의 주변 정보로 이루어져있다.

- GAT Block은 GAT 구조에 Relu와 batch norm을 결합한 구조이다. GAT Block을 거치면서 엣지로 연결된 이웃 버텍스로 버텍스의 정보가 전파가 되고 병합이 된다.

- Base Block은 GAT Block을 3번 거친 후 Eigen 풀링이 결합된 형태이다. Eigen Pooling는 입력 버텍스중 주변 노드 간의 관계를 고려하여 중요 노드만 남기는 풀링 계층의 역할을 수행한다. 모델은 총 3번의 Base Block을 거친 후 최종적으로 얻어낸 각 버텍스별 피쳐를 Classification block으로 넘긴다. 

- Classification Block은 Global Mean Pooling을 수행 하여 각 버텍스의 순서에 상관없이 독립적으로 결과가 나오도록 각 피쳐별로 모든 버텍스에 대해 풀링을 진행한다. 일정한 2번의 Linear layer를 거쳐 최종적인 입력의 클래스를 식별한다.
  
![image](https://github.com/user-attachments/assets/cfda2d9b-4a06-4e74-9ff5-75a81fe30a0f)

# Demo
기본환경 설정
- python 3.7
- CUDA 11.3
- Pytorch 1.10.0
```python
pip install -r requirements.txt
```
데이터셋 다운로드
```bash
bash ./scripts/manifold40/get_data.sh
```
데이터셋 전처리
```bash
bash ./scripts/manifold40/prepare_face_data.sh
```
모델 학습
```bash
bash ./scripts/manifold40/train.sh
```
# Conclusion and Future Work
 본 과제에서는 Eigen 풀링을 사용한 그래프 기반 3d 객체 인식 모델을 설계하였다. 성능 평가 결과 기존 모델의 성능을 유지하면서 풀링으로 연산량을 줄여 eigen pooling은 기존의 그래프기반 네트워크에서 사용되는 풀링방식과 다르게 병렬연산을 통해서 풀링 과정에서의 성능 향상이 가능하다. 모델을 더 확장하면 3d 객체의 버텍스별로 라벨링을 수행하는 segmentation과 같은 수행가능할 것으로 예상된다.


<img src="https://github.com/user-attachments/assets/a76b2f34-44db-483e-818f-27eca7cdfd2b" width="300" height="200"/>
<img src="https://github.com/user-attachments/assets/06868adf-dfc3-4e36-95f4-2cf6f3a98b86" width="400" height="100"/>
