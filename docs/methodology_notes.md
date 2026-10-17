# Methodology Notes

## 프로젝트 범위

최종 연구 노트북에서 다음 흐름을 구현·검토했습니다.

1. 1분 로그가격 데이터 구조 및 분석 기간 확인
2. pre-averaging 기반 PRVM 추정
3. jump truncation 및 noise correction
4. PSD projection
5. factor / idiosyncratic 성분 분리
6. GICS 산업분류 기반 idiosyncratic hard thresholding
7. factor / idiosyncratic eigenvalue 예측 실험
8. 변동성 행렬 재구성 및 MSPE 평가

## 공개하지 않은 자료

- 연구실 제공 원본 가격 데이터
- GICS 원본 파일
- 제공 reference/solution notebook
- 중간 `.npy` 산출물

## 해석상의 주의

최종 연구 노트북에는 여러 구현·수정 버전이 함께 남아 있습니다. 이 저장소는 논문의 전체 empirical result를 완전히 재현하는 패키지가 아니라, 연구 인턴 기간에 구현한 핵심 분석 절차를 포트폴리오용으로 정리한 것입니다.

특히 예측·MSPE 실험은 구현 과정의 한 버전이며, 논문 성능표 전체의 정확한 reproduction으로 표현하지 않습니다.
