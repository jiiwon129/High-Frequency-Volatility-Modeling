# High-Frequency Volatility Modeling

200개 종목의 1분 로그가격으로 공분산 행렬을 예측하고, 그 예측으로 구성한 포트폴리오의 위험을 평가한 프로젝트입니다. POSTECH IME 연구실 인턴에서 다룬 FIVAR 연구를 바탕으로, 2026년 10월에 기존 코드와 평가 절차를 다시 점검했습니다.

## 분석 내용

- 사전평균화와 잡음 보정으로 일별 연속 공분산을 추정하고 점프 성분을 분리했습니다.
- 날짜별 공통 성분을 제거한 뒤, GICS 산업분류를 적용해 개별 성분을 구성했습니다.
- Huber 회귀와 L1 규제를 적용해 공통·개별 성분의 다음 날 값을 예측했습니다.
- 2018~2019년 497거래일에 대해 전날까지의 정보로 투자 비중을 정하고, 당일 10분 수익률로 실현 위험을 계산했습니다.
- 점프 행렬을 보정하는 순서만 바꿔 포트폴리오 위험과 투자 비중이 어떻게 달라지는지 비교했습니다.

## 결과

2018~2019년, 총 익스포저 한도 3.0, 연 250거래일 기준입니다.

| 모형과 점프 처리 | 연율화 위험 |
|---|---:|
| H-LASSO: 합산 후 PSD 보정 | 10.63% |
| H-LASSO: 점프 제외 | 8.11% |
| **H-LASSO: 점프를 먼저 PSD 보정** | **8.05%** |
| POET: 점프를 먼저 PSD 보정 | 8.72% |

같은 점프 보정 방식과 투자 제약에서 H-LASSO의 위험은 POET보다 약 **0.67%포인트, 상대적으로 7.6% 낮았습니다**. 이는 이 데이터와 평가 조건에서 관찰한 결과이며 수익률 개선을 의미하지 않습니다.

![Jump covariance comparison](results/jump_comparison.svg)

점프 추정 행렬은 두 추정량의 차이여서 음수 고유값을 가질 수 있습니다. 이를 그대로 더한 뒤 전체 행렬을 보정하면 일부 방향의 추정 위험이 거의 0이 됩니다. 점프 행렬부터 보정했을 때 기존 연속 공분산의 위험을 깎아 내리는 문제가 사라졌고, 같은 H-LASSO 예측에서도 포트폴리오 위험이 10.63%에서 8.05%로 달라졌습니다. 이 비교는 새 예측모형의 성능 향상이 아니라 **행렬 처리 방식에 대한 민감도 분석**입니다.

## 실행

Python 3.11 이상 환경에서 다음과 같이 실행합니다.

```bash
pip install -r requirements.txt
python experiments/fivar_rolling_evaluation.py \
  --prices-csv data/log_price.csv \
  --gics-csv data/gicslist.csv \
  --workers 2
```

가격 CSV는 첫 열에 시각, 나머지 200개 열에 로그가격이 있어야 합니다. GICS CSV는 헤더가 있는 한 열이며 가격 열과 같은 종목 순서를 사용합니다. 입력 파일은 공개하지 않습니다. 계산한 일별 행렬과 예측은 `cache/`에 저장됩니다.

예측을 저장한 뒤 포트폴리오 평가만 다시 실행할 수 있습니다.

```bash
python experiments/fivar_rolling_evaluation.py --stage evaluate
python -m unittest discover -s tests
```

자료 없이도 [결과 노트북](notebooks/research_workflow.ipynb), [위험 비교표](results/portfolio_risk.csv), [행렬 예측오차](results/matrix_prediction_errors.csv)를 확인할 수 있습니다.

## 해석 범위

논문에서 선택한 학습 기간 251일, 고유벡터 추정 기간 22일, 요인 수 3, 시차 1과 튜닝 상수를 사용했습니다. 전체 튜닝 탐색을 다시 수행하지는 않았습니다. BIC의 세부 구현, 점프 임계값, 장 마감 가격 처리와 PSD 보정에는 명시적인 가정이 남아 있습니다.

논문 Figure 5의 H-LASSO 위험은 약 7.9%로 읽히며, 이번 결과 8.05%와 약 0.15%포인트 차이가 있습니다. 원 저자의 PSD 처리와 같은 구현인지 확인되지 않았으므로 완전 재현으로 표현하지 않습니다. 점프 처리 비교는 기존 평가기간의 차이를 확인한 뒤 수행한 후속 분석입니다. 자세한 설정과 남은 차이는 [분석 방법](docs/methodology_notes.md)에 정리했습니다.

원본 가격·산업분류 자료, 제공된 reference/solution 노트북, 종목별 중간 행렬은 포함하지 않았습니다. 공개된 결과표는 이번 재실행에서 계산한 집계 결과입니다.

## Reference

Shin, M., Kim, D., Wang, Y., & Fan, J. (2025).
*Factor and Idiosyncratic VAR Volatility Matrix Models for Heavy-Tailed High-Frequency Financial Observations.*

[arXiv:2109.05227](https://arxiv.org/abs/2109.05227)
