---
title: "선형 회귀와 로지스틱 회귀"
excerpt: "회귀의 기본적인 개념"

categories:
  - 통계학
tags:
  - Statistics
  - Machine Learning
  - Data Science

toc: true
toc_sticky: true

use_math: true
---

## 회귀 분석
회귀 분석(Regression Analysis)은 학창시절 방정식 문제와 유사합니다.<br/>
함수 'f(x) = y'라고 할 때, x와 y의 관계를 밝히는 것, 다시 말하면 둘 이상의 변수 사이의 함수 관계를 찾는 것입니다.<br/>
이때 x는 독립변수(Independent variable) 혹은 설명변수(Explanatory variable)라고 하고, y는 종속변수(Dependent variable) 혹은 반응변수(Response variable)라고 합니다.<br/>
정리하면 "독립변수와 종속변수 사이의 상호 관련성을 규명하는 것"이라고 할 수 있습니다.<br/>

## 선형 회귀
### 정의
선형 회귀(Linear Regression)이란 변수의 관계를 직선 형태로 가정하고 분석하는 것입니다.<br/>
### 용어
$y=\beta x + \epsilon$

### 기본 가정
적절한 선형 회귀 분석을 위해서는 다음 네가지 조건을 만족해야합니다.<br/>
1. 선형성 : 독립변수와 종속변수의 관계가 선형적임
2. 독립성 : 서로 다른 독립변수 간 상관관계가 없음
3. 등분산성 : 독립변수의 변화에 따른 오차의 분산이 일정함
4. 정규성 : 오차의 학률 분포가 정규 분포를 따름

## 로지스틱 회귀

### 정의
### Odds
### sigmoid

+ 다중 로지스틱 회귀
+ 요약
