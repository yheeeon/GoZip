"""
모델 학습 및 평가 모듈 - 3중 분류
"""
import warnings
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from lightgbm import early_stopping, log_evaluation
from xgboost.callback import EarlyStopping

warnings.filterwarnings("ignore")


class ModelTrainer:
    """모델 학습 및 평가 클래스"""

    def __init__(self):
        self.results = []
        self.best_model_name = None
        self.best_model = None
        self.models = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.feature_names = None
        
        # Cost-Sensitive Learning: 비용 행렬 정의
        self.cost_matrix = np.array([
            [0,  1,  4],  # 저렴을 비쌈으로 예측 시 비용 4 (치명적)
            [1,  0,  1],  # 적정을 저렴/비쌈으로 예측 시 비용 1
            [4,  1,  0]   # 비쌈을 저렴으로 예측 시 비용 4 (치명적)
        ])

    @staticmethod
    def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        분류 평가 지표 계산

        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            y_pred_proba: 예측 확률 (ROC-AUC 계산용, optional)

        Returns:
            평가 지표 딕셔너리
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ROC-AUC (확률이 제공된 경우)
        if y_pred_proba is not None:
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class="ovr",
                    average="macro"
                )
            except:
                metrics["roc_auc_ovr"] = 0.0

        return metrics
    
    @staticmethod
    def calculate_extreme_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Extreme Error Rate 계산 (치명적 오류율)
        
        Args:
            y_true: 실제 레이블 (0=저렴, 1=적정, 2=비쌈)
            y_pred: 예측 레이블
        
        Returns:
            extreme error 지표 딕셔너리
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # 치명적 오류: 저렴(0) ↔ 비쌈(2)
        extreme_low_to_high = cm[0, 2] if cm.shape[0] > 2 else 0  # 저렴 → 비쌈
        extreme_high_to_low = cm[2, 0] if cm.shape[0] > 2 else 0  # 비쌈 → 저렴
        
        total_samples = len(y_true)
        total_extreme_errors = extreme_low_to_high + extreme_high_to_low
        
        # 클래스별 샘플 수
        n_low = np.sum(y_true == 0)
        n_high = np.sum(y_true == 2)
        
        return {
            "extreme_error_count": total_extreme_errors,
            "extreme_error_rate": total_extreme_errors / total_samples if total_samples > 0 else 0,
            "low_to_high_count": extreme_low_to_high,
            "low_to_high_rate": extreme_low_to_high / n_low if n_low > 0 else 0,
            "high_to_low_count": extreme_high_to_low,
            "high_to_low_rate": extreme_high_to_low / n_high if n_high > 0 else 0,
        }
    
    @staticmethod
    def adjust_threshold(y_pred_proba: np.ndarray, conservative_factor: float = 1.2) -> np.ndarray:
        """
        Threshold 조정: 저렴(0) 예측을 보수적으로
        
        Args:
            y_pred_proba: 예측 확률 (n_samples, 3)
            conservative_factor: 저렴 클래스 threshold 조정 계수 (>1이면 보수적)
        
        Returns:
            조정된 예측 레이블
        """
        adjusted_proba = y_pred_proba.copy()
        
        # 저렴(0) 클래스의 확률을 낮춰서 예측을 보수적으로 만듦
        # 즉, 저렴으로 예측하려면 더 확실해야 함
        adjusted_proba[:, 0] = adjusted_proba[:, 0] / conservative_factor
        
        # 비쌈(2) 클래스도 약간 보수적으로 (선택적)
        # adjusted_proba[:, 2] = adjusted_proba[:, 2] / 1.1
        
        # 재정규화
        adjusted_proba = adjusted_proba / adjusted_proba.sum(axis=1, keepdims=True)
        
        # 최대 확률 클래스 선택
        return np.argmax(adjusted_proba, axis=1)
    
    def calculate_cost_based_weights(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        base_weight: float = 1.0
    ) -> np.ndarray:
        """
        비용 행렬 기반 샘플 가중치 계산
        
        각 샘플의 예상 오분류 비용을 계산하여 가중치로 사용.
        비용이 높을 것으로 예상되는 샘플에 더 높은 가중치 부여.
        
        Args:
            y_true: 실제 레이블 (n_samples,)
            y_pred_proba: 예측 확률 (n_samples, 3)
            base_weight: 기본 가중치 (최소값)
        
        Returns:
            샘플별 가중치 (n_samples,)
        """
        n_samples = len(y_true)
        weights = np.ones(n_samples) * base_weight
        
        for i in range(n_samples):
            true_class = int(y_true[i])
            
            # 예상 비용 계산: E[Cost] = Σ P(pred=j) × Cost(true=i, pred=j)
            expected_cost = 0.0
            for pred_class in range(3):
                prob = y_pred_proba[i, pred_class]
                cost = self.cost_matrix[true_class, pred_class]
                expected_cost += prob * cost
            
            # 예상 비용을 가중치로 사용 (비용이 높을수록 가중치 증가)
            # +1을 더해서 최소 가중치가 base_weight가 되도록
            weights[i] = base_weight + expected_cost
        
        # 정규화 (평균이 1이 되도록)
        weights = weights / weights.mean()
        
        return weights

    def train_models(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None,
    ) -> pd.DataFrame:
        """
        분류 모델 학습 및 평가

        Args:
            models: 모델 딕셔너리
            X_train: 학습 데이터 (전처리 완료)
            y_train: 학습 타깃 (클래스 레이블 0, 1, 2)
            X_val: 검증 데이터
            y_val: 검증 타깃 (클래스 레이블)
            X_test: 테스트 데이터
            y_test: 테스트 타깃 (클래스 레이블)
            feature_names: 피처 이름 리스트 (SHAP 분석용)

        Returns:
            결과 데이터프레임
        """
        self.results = []
        self.models = models
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.feature_names = feature_names

        for name, clf in models.items():
            print(f"\n{'=' * 60}")
            print(f"🚀 학습 중: {name}")
            print(f"{'=' * 60}")

            # 1) 학습 (early_stopping 사용)
            if name == "XGBoost":
                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=200,
                )
            else:  # LightGBM - Cost-Sensitive Learning with Sample Weights
                print(f"\n🎯 [{name}] Cost-Sensitive Learning 적용 (2단계 학습)")
                
                # Stage 1: 초기 학습 (class_weight만 사용)
                print(f"   [Stage 1] 초기 학습 (class_weight 기반)...")
                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        early_stopping(stopping_rounds=50, verbose=False),
                        log_evaluation(period=200)
                    ],
                )
                
                # Stage 2: 비용 기반 Sample Weight 계산 및 재학습
                print(f"   [Stage 2] 비용 기반 Sample Weight 계산...")
                initial_proba = clf.predict_proba(X_train)
                sample_weights = self.calculate_cost_based_weights(
                    y_train, 
                    initial_proba,
                    base_weight=1.0
                )
                
                print(f"   [Stage 2] Sample Weight 통계:")
                print(f"      - 평균: {sample_weights.mean():.4f}")
                print(f"      - 최소: {sample_weights.min():.4f}")
                print(f"      - 최대: {sample_weights.max():.4f}")
                print(f"      - 표준편차: {sample_weights.std():.4f}")
                
                # 재학습 (sample_weight 적용)
                print(f"   [Stage 2] Sample Weight 기반 재학습...")
                clf.fit(
                    X_train, y_train,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        early_stopping(stopping_rounds=50, verbose=False),
                        log_evaluation(period=200)
                    ],
                )
                print(f"   ✅ Cost-Sensitive Learning 완료!")


            # 2) 예측
            pred_tr = clf.predict(X_train)
            pred_val = clf.predict(X_val)
            pred_test = clf.predict(X_test)

            # 예측 확률
            pred_tr_proba = clf.predict_proba(X_train)
            pred_val_proba = clf.predict_proba(X_val)
            pred_test_proba = clf.predict_proba(X_test)

            # 3) 평가 지표 계산
            metrics_tr = self.eval_metrics(y_train, pred_tr, pred_tr_proba)
            metrics_val = self.eval_metrics(y_val, pred_val, pred_val_proba)
            metrics_test = self.eval_metrics(y_test, pred_test, pred_test_proba)
            
            # ★★★ Extreme Error Rate 계산 ★★★
            extreme_tr = self.calculate_extreme_error_rate(y_train, pred_tr)
            extreme_val = self.calculate_extreme_error_rate(y_val, pred_val)
            extreme_test = self.calculate_extreme_error_rate(y_test, pred_test)
            
            # ★★★ Threshold 조정 예측 (Test만) ★★★
            pred_test_adjusted = self.adjust_threshold(pred_test_proba, conservative_factor=1.2)
            metrics_test_adj = self.eval_metrics(y_test, pred_test_adjusted, pred_test_proba)
            extreme_test_adj = self.calculate_extreme_error_rate(y_test, pred_test_adjusted)

            # 4) 결과 저장 (Extreme Error Rate 포함)
            self.results.append({
                "model": name,
                "accuracy_train": metrics_tr["accuracy"],
                "precision_macro_train": metrics_tr["precision_macro"],
                "recall_macro_train": metrics_tr["recall_macro"],
                "f1_macro_train": metrics_tr["f1_macro"],
                "f1_weighted_train": metrics_tr["f1_weighted"],
                "roc_auc_train": metrics_tr.get("roc_auc_ovr", 0.0),
                "extreme_error_rate_train": extreme_tr["extreme_error_rate"],
                "accuracy_val": metrics_val["accuracy"],
                "precision_macro_val": metrics_val["precision_macro"],
                "recall_macro_val": metrics_val["recall_macro"],
                "f1_macro_val": metrics_val["f1_macro"],
                "f1_weighted_val": metrics_val["f1_weighted"],
                "roc_auc_val": metrics_val.get("roc_auc_ovr", 0.0),
                "extreme_error_rate_val": extreme_val["extreme_error_rate"],
                "accuracy_test": metrics_test["accuracy"],
                "precision_macro_test": metrics_test["precision_macro"],
                "recall_macro_test": metrics_test["recall_macro"],
                "f1_macro_test": metrics_test["f1_macro"],
                "f1_weighted_test": metrics_test["f1_weighted"],
                "roc_auc_test": metrics_test.get("roc_auc_ovr", 0.0),
                "extreme_error_rate_test": extreme_test["extreme_error_rate"],
                # Threshold 조정 결과
                "f1_macro_test_adjusted": metrics_test_adj["f1_macro"],
                "extreme_error_rate_test_adjusted": extreme_test_adj["extreme_error_rate"],
            })

            # 5) 결과 출력 (Extreme Error Rate 포함)
            print(f"\n📈 [{name}] 성능 지표:")
            print(f"   [Train] Acc: {metrics_tr['accuracy']:.4f} | F1(macro): {metrics_tr['f1_macro']:.4f} | Extreme Err: {extreme_tr['extreme_error_rate']:.4f} ({extreme_tr['extreme_error_count']}건)")
            print(f"   [Val]   Acc: {metrics_val['accuracy']:.4f} | F1(macro): {metrics_val['f1_macro']:.4f} | Extreme Err: {extreme_val['extreme_error_rate']:.4f} ({extreme_val['extreme_error_count']}건)")
            print(f"   [Test]  Acc: {metrics_test['accuracy']:.4f} | F1(macro): {metrics_test['f1_macro']:.4f} | Extreme Err: {extreme_test['extreme_error_rate']:.4f} ({extreme_test['extreme_error_count']}건)")
            
            # ★★★ Extreme Error 상세 분석 ★★★
            print(f"\n🚨 [{name}] Extreme Error 상세 (Test):")
            print(f"   - 저렴→비쌈: {extreme_test['low_to_high_count']}건 (저렴 중 {extreme_test['low_to_high_rate']:.2%})")
            print(f"   - 비쌈→저렴: {extreme_test['high_to_low_count']}건 (비쌈 중 {extreme_test['high_to_low_rate']:.2%})")
            
            # ★★★ Threshold 조정 결과 ★★★
            print(f"\n🎯 [{name}] Threshold 조정 결과 (Test):")
            print(f"   - 기본:     F1(macro): {metrics_test['f1_macro']:.4f} | Extreme Err: {extreme_test['extreme_error_rate']:.4f}")
            print(f"   - 조정 후:  F1(macro): {metrics_test_adj['f1_macro']:.4f} | Extreme Err: {extreme_test_adj['extreme_error_rate']:.4f}")
            extreme_reduction = (extreme_test['extreme_error_rate'] - extreme_test_adj['extreme_error_rate']) / extreme_test['extreme_error_rate'] * 100 if extreme_test['extreme_error_rate'] > 0 else 0
            print(f"   - Extreme Error 감소율: {extreme_reduction:.1f}%")

            # 6) 혼동 행렬 출력 (Test)
            print(f"\n📊 [{name}] Test 혼동 행렬:")
            cm = confusion_matrix(y_test, pred_test)
            print(f"   실제\\예측  저렴(0)  적정(1)  비쌈(2)")
            for i, row in enumerate(cm):
                class_name = ["저렴(0)", "적정(1)", "비쌈(2)"][i]
                print(f"   {class_name:8s}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

            # 7) 클래스별 성능 출력 (Test)
            print(f"\n📋 [{name}] Test 클래스별 성능:")
            report = classification_report(
                y_test, pred_test,
                target_names=["저렴(0)", "적정(1)", "비쌈(2)"],
                digits=4
            )
            print(report)
            
            # ===== Cost-Sensitive 평가 추가 =====
            print(f"\n{'=' * 60}")
            print(f"💰 [{name}] Cost-Sensitive 평가 (비용 기반 메트릭)")
            print(f"{'=' * 60}")
            
            # 비용 행렬 정의 (저렴=0, 적정=1, 비쌈=2)
            cost_matrix = np.array([
                [0,  1,  4],  # 저렴을 비쌈으로 예측 시 비용 4 (치명적)
                [1,  0,  1],  # 적정을 저렴/비쌈으로 예측 시 비용 1
                [4,  1,  0]   # 비쌈을 저렴으로 예측 시 비용 4 (치명적)
            ])
            
            print(f"\n📋 비용 행렬:")
            print("              예측: 저렴(0)  적정(1)  비쌈(2)")
            print("  실제: 저렴(0)     0         1        4  ← 치명적 오류")
            print("        적정(1)     1         0        1")
            print("        비쌈(2)     4         1        0  ← 치명적 오류")
            
            # 총 비용 계산
            total_cost = 0
            for true_idx, pred_idx in zip(y_test, pred_test):
                total_cost += cost_matrix[true_idx, pred_idx]
            
            avg_cost = total_cost / len(y_test)
            
            print(f"\n▶ 총 예측 비용:        {total_cost:,}")
            print(f"▶ 샘플당 평균 비용:    {avg_cost:.4f}")
            
            # 비용 가중 혼동 행렬
            cm_cost = cm * cost_matrix
            print(f"\n▶ 비용 가중 혼동 행렬:")
            print(f"   실제\\예측  저렴(0)  적정(1)  비쌈(2)")
            for i, row in enumerate(cm_cost):
                class_name = ["저렴(0)", "적정(1)", "비쌈(2)"][i]
                print(f"   {class_name:8s}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
            
            # 치명적 오류 분석
            critical_errors_low_to_high = cm[0, 2]  # 저렴(0) → 비쌈(2)
            critical_errors_high_to_low = cm[2, 0]  # 비쌈(2) → 저렴(0)
            total_critical = critical_errors_low_to_high + critical_errors_high_to_low
            
            print(f"\n🚨 치명적 오류 분석:")
            print(f"  - 저렴→비쌈:           {critical_errors_low_to_high}건 (비용: {critical_errors_low_to_high * 4:,})")
            print(f"  - 비쌈→저렴:           {critical_errors_high_to_low}건 (비용: {critical_errors_high_to_low * 4:,})")
            print(f"  - 총 치명적 오류:      {total_critical}건 (비용: {total_critical * 4:,})")
            
            # 클래스별 비용 기여도
            class_costs = cm_cost.sum(axis=1)
            print(f"\n▶ 클래스별 비용 기여도:")
            for i, class_name in enumerate(["저렴(0)", "적정(1)", "비쌈(2)"]):
                contribution = class_costs[i]/total_cost*100 if total_cost > 0 else 0
                print(f"  {class_name:8s}: {class_costs[i]:>6,.0f} ({contribution:.1f}%)")
            
            print(f"\n💡 해석:")
            print(f"  - 현재 모델은 샘플당 평균 {avg_cost:.2f}의 비용을 발생시킵니다.")
            if total_critical > 0:
                print(f"  - 치명적 오류(저렴↔비쌈)가 {total_critical}건 발생했습니다.")
                print(f"  - Cost-Sensitive Learning 적용 시 이를 개선할 수 있습니다.")
            else:
                print(f"  - 치명적 오류가 없습니다! 매우 우수한 성능입니다.")
            
            print(f"{'=' * 60}")
            # ===== Cost-Sensitive 평가 끝 =====

        # 결과 정리
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("f1_macro_test", ascending=False)

        # 최고 성능 모델 저장
        self.best_model_name = results_df.iloc[0]["model"]
        self.best_model = models[self.best_model_name]
        best_f1 = results_df.iloc[0]["f1_macro_test"]

        print(f"\n{'=' * 60}")
        print(f"📊 전체 모델 비교 (Test F1-Macro 기준)")
        print(f"{'=' * 60}")
        print(results_df.to_string(index=False))
        print(f"\n🏆 최고 성능 모델: {self.best_model_name} (Test F1-Macro = {best_f1:.4f})")

        return results_df

    def save_model(
        self,
        preprocessor: Any,
        output_dir: str = "./models",
        model_filename: str = None
    ) -> str:
        """
        학습된 모델과 전처리기를 pkl 파일로 저장

        Args:
            preprocessor: 전처리 파이프라인
            output_dir: 저장 디렉토리
            model_filename: 저장할 파일명 (None이면 자동 생성)

        Returns:
            저장된 파일 경로
        """
        if self.best_model is None:
            raise ValueError("학습된 모델이 없습니다. train_models()를 먼저 실행하세요.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if model_filename is None:
            model_filename = f"price_model_{self.best_model_name.lower()}.pkl"

        full_path = output_path / model_filename

        # 모델과 전처리기를 함께 저장
        model_bundle = {
            "model": self.best_model,
            "preprocessor": preprocessor,
            "model_name": self.best_model_name,
        }

        with open(full_path, "wb") as f:
            pickle.dump(model_bundle, f)

        print(f"\n✅ 모델 저장 완료: {full_path}")
        print(f"   - 모델: {self.best_model_name}")
        print(f"   - 파일 크기: {full_path.stat().st_size / 1024 / 1024:.2f} MB")

        return str(full_path)

    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        저장된 모델 로드

        Args:
            model_path: 모델 파일 경로

        Returns:
            모델 번들 딕셔너리
        """
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)

        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"   - 모델: {model_bundle['model_name']}")

        return model_bundle

    def create_shap_explainer(
        self,
        model_name: str = None,
        background_size: int = 100
    ):
        """
        SHAP Explainer 생성

        Args:
            model_name: 분석할 모델 이름 (None이면 best_model 사용)
            background_size: 배경 데이터 크기

        Returns:
            ModelExplainer 객체
        """
        from analysis.explainer import ModelExplainer

        if model_name is None:
            if self.best_model is None:
                raise ValueError("학습된 모델이 없습니다. train_models()를 먼저 실행하세요.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if self.models is None or model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'을 찾을 수 없습니다.")
            model = self.models[model_name]

        if self.X_train is None:
            raise ValueError("학습 데이터가 저장되지 않았습니다. train_models()를 먼저 실행하세요.")

        # Explainer 생성
        explainer = ModelExplainer(
            model=model,
            model_name=model_name,
            feature_names=self.feature_names
        )

        # TreeExplainer 생성
        explainer.create_explainer(
            X_background=self.X_train,
            background_size=background_size
        )

        return explainer

    def analyze_shap(
        self,
        model_name: str = None,
        data_type: str = "test",
        max_samples: int = 1000,
        output_dir: str = "./shap_plots",
        save_plots: bool = True
    ):
        """
        SHAP 분석 실행

        Args:
            model_name: 분석할 모델 이름 (None이면 best_model 사용)
            data_type: 분석할 데이터 ("train", "val", "test")
            max_samples: 분석할 최대 샘플 수
            output_dir: 플롯 저장 디렉토리
            save_plots: 플롯 저장 여부

        Returns:
            ModelExplainer 객체
        """
        print(f"\n{'=' * 70}")
        print(f"🔍 SHAP 분석 시작")
        print(f"{'=' * 70}")

        # Explainer 생성
        explainer = self.create_shap_explainer(model_name=model_name)

        # 분석할 데이터 선택
        if data_type == "train":
            X = self.X_train
        elif data_type == "val":
            X = self.X_val
        elif data_type == "test":
            X = self.X_test
        else:
            raise ValueError(f"data_type은 'train', 'val', 'test' 중 하나여야 합니다. 입력값: {data_type}")

        print(f"   - 분석 데이터: {data_type}")
        print(f"   - 데이터 크기: {X.shape}")
        print(f"   - 최대 샘플 수: {max_samples}")

        # SHAP values 계산
        shap_values, X_sample = explainer.compute_shap_values(
            X=X,
            max_samples=max_samples
        )

        # 플롯 저장
        if save_plots:
            explainer.save_all_plots(
                X=X_sample,
                output_dir=output_dir,
                max_display=20,
                sample_indices=[0, 1, 2]
            )

        print(f"\n{'=' * 70}")
        print(f"✅ SHAP 분석 완료!")
        print(f"{'=' * 70}")

        return explainer
