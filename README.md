# High-Frequency Volatility Modeling

> S&P 500 주요 200개 종목의 1분 로그가격과 GICS 산업분류를 활용해 고차원 변동성 행렬의 추정·분해·예측 절차를 구현한 POSTECH IME 연구실 인턴 프로젝트

## Overview

- **Research topic**: 고빈도 금융 데이터 기반 변동성 행렬 추정 및 예측
- **Data**: 200개 종목의 1분 로그가격과 GICS 산업분류
- **Main reference**: FIVAR (Factor and Idiosyncratic VAR Volatility Matrix Models)
- **My role**: 데이터 구조 확인, 논문 수식·알고리즘 해석, Python 구현, 중간 결과 비교
- **Tech**: Python, NumPy, pandas, scikit-learn, joblib

## Research Flow

### 1. PRVM estimation
1분 로그가격으로부터 pre-averaging과 noise correction을 적용해 일별 변동성 행렬을 추정하고, jump truncation 이후 PSD projection을 수행했습니다.

### 2. Factor / idiosyncratic decomposition
최근 변동성 행렬의 고유벡터를 이용해 공통 factor 성분을 분리하고, 잔차 행렬에는 GICS 산업분류 기반 hard thresholding을 적용했습니다.

### 3. Eigenvalue dynamics
factor 및 idiosyncratic 성분의 고유값 시계열을 구성하고, robust regression 기반 예측 실험을 구현했습니다.

### 4. Matrix reconstruction & evaluation
예측한 factor / idiosyncratic 고유값을 다시 변동성 행렬로 복원하고, 다음 시점의 행렬과 비교하는 평가 실험을 연결했습니다.

## Repository Structure

```text
.
├── README.md
├── src/
│   ├── prvm_estimation.py
│   └── fivar_components.py
├── experiments/
│   └── fivar_rolling_evaluation.py
├── notebooks/
│   └── research_workflow.ipynb
├── docs/
│   └── methodology_notes.md
├── requirements.txt
└── .gitignore
```

## Notes

이 저장소는 **포트폴리오용 정리본**입니다.

- 연구실에서 제공받은 원본 가격 데이터와 산업분류 파일은 포함하지 않았습니다.
- 제공받은 reference/solution notebook은 포함하지 않았습니다.
- 최종 연구 노트북에는 여러 실험 버전이 함께 남아 있어, 이 저장소에는 대표성이 높은 구현만 분리해 정리했습니다.
- 논문의 전체 실증 결과나 성능표를 정확히 재현했다고 주장하지 않습니다. 논문의 핵심 추정·분해·예측 절차를 Python으로 구현하고 결과를 비교한 경험을 보여주는 목적입니다.

## Reference

Shin, M., Kim, D., Wang, Y., & Fan, J. (2025).
*Factor and Idiosyncratic VAR Volatility Matrix Models for Heavy-Tailed High-Frequency Financial Observations.*