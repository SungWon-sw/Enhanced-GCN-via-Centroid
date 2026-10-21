# 독립 baseline 모델

이 폴더 전체를 다른 프로젝트에 복사하면 `baseline` 모델과 VN 전처리를 사용할 수
있습니다. 현재 프로젝트의 **GCN + separator-tree VN** 구현을 분리한 것으로,
일반 GCN만 사용하는 모델은 아닙니다.

```text
baseline_model/
├── __init__.py       # 외부에서 사용하는 import
├── model.py          # GCN, GCNConv, GCNConvVN, PPANodeEncoder, build_model
├── preprocessing.py  # separator 계산, VNData 배치 처리, prepare_graph
├── config.yaml       # 현재 모델·전처리 설정
├── example.py        # 합성 그래프로 CPU 학습 1 step
├── requirements.txt
└── README.md
```

원본 `src/`와 연결된 import나 심볼릭 링크가 없는 독립 복사본입니다.
GCN+, 학습 루프, Optuna, 데이터셋 다운로드·캐시, 데이터, 가중치는 포함하지 않습니다.
이후 이 폴더에서 모델을 수정해도 원본 코드에는 반영되지 않습니다.

## 옮기고 실행하기

현재 프로젝트 루트에서, 목적지 경로를 실제 프로젝트 경로로 바꿔 실행하세요.

```bash
cp -r baseline_model /path/to/your/project/
cd /path/to/your/project

# 기존 gcnet 가상환경을 사용한다면 추가 설치 없이 실행할 수 있습니다.
# 새 환경에서는 사용할 CPU/CUDA에 맞게 PyTorch를 설치한 뒤:
python -m pip install -r baseline_model/requirements.txt
python -m baseline_model.example
```

예제는 데이터 다운로드 없이 `logits: (2, 37)`과 `CPU forward/backward OK`를
출력합니다. 원본 환경의 PyTorch는 `2.7.1+cu118`이며, requirements는 CUDA를
강제하지 않습니다. 새 환경에서의 라이브러리 설치는 별도로 필요합니다.

## 다른 학습 코드에서 사용하기

아래 `raw_graphs`는 PyG `Data` 객체 목록입니다. OGB 데이터셋을 사용하는 경우
기존 split 인덱스로 train/valid/test를 구분한 뒤 각 그래프를 전처리하세요.

```python
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

import baseline_model
from baseline_model import build_model, prepare_graph

config_path = Path(baseline_model.__file__).with_name('config.yaml')
cfg = yaml.safe_load(config_path.read_text())
model = build_model(cfg, num_classes=37)

# 전처리는 그래프별로 한 번 수행하고 결과를 재사용합니다.
graphs = [prepare_graph(graph, cfg) for graph in raw_graphs]
loader = DataLoader(graphs, batch_size=32, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for batch in loader:
    batch = batch.to(device)
    logits = model(batch)  # [배치 그래프 수, num_classes]
    loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1).long())
    # optimizer.zero_grad(), loss.backward(), optimizer.step() 등은 실험 코드에 작성
```

입력은 현재 ogbg-ppa 형식을 유지합니다.

- `num_nodes`: 1 이상인 노드 수.
- `edge_index`: `[2, E]` 크기의 long 텐서. PPA처럼 무방향 간선은 양방향으로 넣습니다.
- `edge_attr`: `[E, 7]` 크기의 간선 특징. 간선 순서는 `edge_index`와 같아야 합니다.
- `y`: 학습 시 사용할 클래스 인덱스.
- `x`: `prepare_graph`가 길이 `num_nodes`의 0 벡터로 생성합니다. 원래 노드 특징은
  사용하지 않습니다. 다른 노드 특징이나 다른 차원의 간선 특징을 쓰려면 인코더를 수정해야 합니다.

`prepare_graph`는 원본을 복사하고 VN 필드를 붙인 `VNData`를 반환합니다.
**배치로 묶기 전에** 적용하고, 반환된 `VNData`를 그대로 PyG DataLoader에 넣으세요.
일반 `Data`로 다시 만들면 VN 번호의 배치 오프셋이 잘못될 수 있습니다.
학습 모드의 BatchNorm에는 배치 전체의 실제 노드 수와 VN 수가 각각 2 이상이어야 합니다.
단일 노드 그래프 하나를 추론할 때는 `model.eval()`을 사용하세요.

전처리는 CPU에서 separator를 계산합니다. 대규모 데이터는 결과를 미리 저장하거나
자체 캐시를 사용하세요. 위 목록 예제는 전처리한 그래프를 모두 메모리에 보관합니다.

`config.yaml`에는 분리 당시 설정을 넣었습니다. `build_model({})`과
`prepare_graph(graph)`처럼 설정을 생략하면 원본 코드의 기본값을 사용하므로,
현재 실험을 재현할 때는 **두 함수에 동일한 cfg를 전달**하세요.

## 기존 baseline 가중치 사용

모델 구조와 파라미터 이름을 유지했으므로 **같은 모델 설정**으로 만든 baseline의
`best_model.pt`를 불러올 수 있습니다. 가중치 파일과 해당 실행의 `config.yaml`은
필요하면 별도로 복사하세요. 전처리 설정도 해당 실험과 같아야 합니다.

```python
cfg = yaml.safe_load(Path('saved_run/config.yaml').read_text())
model = build_model(cfg, num_classes=37)
weights = torch.load('saved_run/best_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(weights, strict=True)
model.eval()
```

`last_checkpoint.pt`는 전체 학습 상태를 담고 있으므로 `checkpoint['config']`로
모델을 만들고 `checkpoint['model']`을 불러옵니다. 이 패키지는 optimizer와 scheduler를
포함한 학습 재개 기능을 제공하지 않습니다. GCN+ 가중치는 구조가 달라 사용할 수 없습니다.
