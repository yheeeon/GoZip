"""
학습된 모델로 Trust Score 예측 (psycopg2 버전)

Django ORM 대신 psycopg2 사용 → scripts 컨테이너에서 실행 가능
"""
import os
import sys
import pickle
import psycopg2
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix


def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'realestate'),
        user=os.getenv('POSTGRES_USER', 'postgres'),
        password=os.getenv('POSTGRES_PASSWORD', 'postgres')
    )


def load_model():
    """모델 로드"""
    # 여러 경로 시도
    possible_paths = [
        '/app/scripts/03_import/trust/final_trust_model.pkl',  # Docker 절대 경로 (scripts가 /app에 마운트됨)
        '/scripts/03_import/trust/final_trust_model.pkl',
        'scripts/03_import/trust/final_trust_model.pkl',
        '03_import/trust/final_trust_model.pkl',
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다. 시도한 경로: {possible_paths}")
    
    print(f"  ✓ 모델 경로: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['scaler'], model_data['feature_names']


def create_features(broker_row):
    """
    Feature 생성 (학습 시와 동일한 14개)
    broker_row: dict 형태의 DB row
    """
    # 기본 값 추출
    거래완료 = broker_row.get('completed_deals') or 0
    등록매물 = broker_row.get('registered_properties') or 0
    공인중개사수 = broker_row.get('brokers_count') or 0
    중개보조원수 = broker_row.get('assistants_count') or 0
    일반직원수 = broker_row.get('staff_count') or 0
    
    # 대표자 구분
    대표자구분명 = broker_row.get('representative_type') or "공인중개사"
    
    # 1-3. 거래 지표 (로그 변환)
    등록매물_log = np.log1p(등록매물)
    총거래활동량 = 거래완료 + 등록매물
    총거래활동량_log = np.log1p(총거래활동량)
    
    # 1인당 거래량
    총_직원수 = 공인중개사수 + 중개보조원수 + 일반직원수
    총_직원수_safe = max(총_직원수, 1)
    일인당_거래량 = 총거래활동량 / 총_직원수_safe
    일인당_거래량_log = np.log1p(일인당_거래량)
    
    # 4-6. 인력 지표
    중개보조원_비율 = 중개보조원수 / 총_직원수_safe
    자격증_보유_인원 = 공인중개사수 + 중개보조원수
    자격증_보유비율 = 자격증_보유_인원 / 총_직원수_safe
    
    # 7-9. 운영 경험
    운영기간_년 = 0
    등록일 = broker_row.get('registration_date')
    if 등록일:
        try:
            if isinstance(등록일, str):
                from dateutil import parser
                등록일 = parser.parse(등록일).date()
            today = datetime.now().date()
            영업일수 = (today - 등록일).days
            운영기간_년 = 영업일수 / 365.25
        except:
            pass
    
    공인중개사_비율 = 공인중개사수 / 총_직원수_safe
    숙련도_지수 = 운영기간_년 * 공인중개사_비율
    운영_안정성 = 1 if 운영기간_년 >= 3 else 0
    
    # 10. 조직 구조
    대형사무소 = 1 if 총_직원수 >= 2 else 0
    
    # 11-12. 대표자 자격
    대표_공인중개사 = 1 if 대표자구분명 == "공인중개사" else 0
    대표_법인 = 1 if 대표자구분명 == "법인" else 0
    
    # 13. 지역 경쟁 강도 (평균값)
    지역_경쟁강도 = 50
    
    # 14. 1층 여부
    주소 = broker_row.get('address') or ""
    일층_여부 = 1 if ("1층" in 주소 or "101호" in 주소 or "102호" in 주소) else 0
    
    # 14개 Feature
    features = [
        등록매물_log,
        총거래활동량_log,
        일인당_거래량_log,
        총_직원수,
        중개보조원_비율,
        자격증_보유비율,
        운영기간_년,
        숙련도_지수,
        운영_안정성,
        대형사무소,
        대표_공인중개사,
        대표_법인,
        지역_경쟁강도,
        일층_여부
    ]
    
    return features


def predict():
    """예측 실행"""
    print("\n" + "=" * 70)
    print(" " * 15 + "Trust Score 예측 시작")
    print("=" * 70 + "\n")
    
    # 1. 모델 로드
    print("1. 모델 로드 중...")
    try:
        model, scaler, feature_names = load_model()
        print(f"  ✓ Feature: {feature_names}")
    except Exception as e:
        print(f"  ✗ 모델 로드 실패: {e}")
        return
    
    # 2. DB 연결 및 Broker 조회
    print("\n2. Broker 데이터 조회 중...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 테이블 존재 확인
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'landbroker'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("  ⚠ landbroker 테이블이 존재하지 않습니다.")
            print("  → 중개사 데이터가 아직 Import되지 않았습니다.")
            print("  → Trust Score 예측을 건너뜁니다.")
            cur.close()
            conn.close()
            return
        
        cur.execute("""
            SELECT landbroker_id, office_name, completed_deals, registered_properties,
                   brokers_count, assistants_count, staff_count, 
                   registration_date, address
            FROM landbroker
        """)
        
        columns = [desc[0] for desc in cur.description]
        brokers = [dict(zip(columns, row)) for row in cur.fetchall()]
        total = len(brokers)
        
        if total == 0:
            print("  ⚠ landbroker 테이블에 데이터가 없습니다.")
            cur.close()
            conn.close()
            return
            
        print(f"  ✓ {total}개 broker")
    except Exception as e:
        print(f"  ✗ DB 조회 실패: {e}")
        return
    
    # 3. 예측
    print("\n3. 예측 실행 중...")
    updated = 0
    grade_dist = {'A': 0, 'B': 0, 'C': 0}
    
    # Cost-Sensitive 평가를 위한 예측 결과 저장
    all_predictions = []
    all_true_labels = []  # 실제 레이블 (DB에 저장된 기존 trust_score)
    
    for i, broker in enumerate(brokers, 1):
        try:
            # Feature 생성
            features = create_features(broker)
            features_df = pd.DataFrame([features], columns=feature_names)
            
            # 스케일링 및 예측
            features_scaled = scaler.transform(features_df)
            pred = model.predict(features_scaled)[0]
            
            # 등급 매핑
            if isinstance(pred, str):
                trust_score = pred
                pred_numeric = {'A': 2, 'B': 1, 'C': 0}.get(pred, 0)
            else:
                grade_map = {0: 'C', 1: 'B', 2: 'A'}  # 수정: 0=C, 1=B, 2=A
                trust_score = grade_map.get(pred, 'C')
                pred_numeric = pred
            
            # 기존 trust_score 조회 (평가용)
            cur.execute("""
                SELECT trust_score FROM landbroker WHERE landbroker_id = %s
            """, (broker['landbroker_id'],))
            existing_score = cur.fetchone()
            if existing_score and existing_score[0]:
                true_label = {'A': 2, 'B': 1, 'C': 0}.get(existing_score[0], 0)
                all_true_labels.append(true_label)
                all_predictions.append(pred_numeric)
            
            # DB 업데이트
            cur.execute("""
                UPDATE landbroker 
                SET trust_score = %s, trust_score_updated_at = CURRENT_TIMESTAMP
                WHERE landbroker_id = %s
            """, (trust_score, broker['landbroker_id']))
            
            updated += 1
            grade_dist[trust_score] += 1
            
            if i % max(total // 10, 1) == 0:
                print(f"  진행: {i}/{total} ({i/total*100:.1f}%)")
        
        except Exception as e:
            print(f"  ✗ 예측 실패 ({broker.get('office_name', 'Unknown')}): {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    
    # 4. 결과
    print("\n" + "=" * 70)
    print(" " * 20 + "예측 완료")
    print("=" * 70)
    print(f"\n  성공: {updated}개")
    print(f"\n  등급 분포:")
    print(f"    A등급(골드):   {grade_dist['A']}개 ({grade_dist['A']/max(total,1)*100:.1f}%)")
    print(f"    B등급(실버):   {grade_dist['B']}개 ({grade_dist['B']/max(total,1)*100:.1f}%)")
    print(f"    C등급(브론즈): {grade_dist['C']}개 ({grade_dist['C']/max(total,1)*100:.1f}%)")
    
    # ===== Cost-Sensitive 평가 추가 =====
    if len(all_true_labels) > 0 and len(all_predictions) > 0:
        print("\n" + "=" * 70)
        print(" " * 15 + "💰 Cost-Sensitive 평가")
        print("=" * 70)
        
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        
        # F1 점수 계산
        try:
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            print(f"\n📊 F1 Score:")
            print(f"  - F1-Macro:    {f1_macro:.4f}")
            print(f"  - F1-Weighted: {f1_weighted:.4f}")
        except Exception as e:
            print(f"\n⚠️ F1 Score 계산 실패: {e}")
        
        # 혼동 행렬
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            grade_names = ['C(브론즈)', 'B(실버)', 'A(골드)']
            
            print(f"\n📋 혼동 행렬:")
            print("              예측: C(브론즈)  B(실버)  A(골드)")
            for i, grade_name in enumerate(grade_names):
                row_values = "  ".join([f"{val:>9}" for val in cm[i]])
                print(f"  실제: {grade_name:9s}  {row_values}")
        except Exception as e:
            print(f"\n⚠️ 혼동 행렬 계산 실패: {e}")
        
        # 비용 행렬 기반 평가
        try:
            # 비용 행렬 정의 (C=0, B=1, A=2)
            cost_matrix = np.array([
                [0,  2,  10],  # C를 A로 예측 시 비용 10 (치명적)
                [1,  0,   5],  # B를 A로 예측 시 비용 5
                [1,  1,   0]   # A를 C/B로 예측 시 비용 1 (보수적, 허용)
            ])
            
            print(f"\n💰 비용 행렬:")
            print("              예측: C(브론즈)  B(실버)  A(골드)")
            print("  실제: C(브론즈)     0          2         10  ← 치명적 오류")
            print("        B(실버)       1          0          5")
            print("        A(골드)       1          1          0  ← 보수적 오류")
            
            # 총 비용 계산
            total_cost = 0
            for true_idx, pred_idx in zip(y_true, y_pred):
                total_cost += cost_matrix[true_idx, pred_idx]
            
            avg_cost = total_cost / len(y_true)
            
            print(f"\n▶ 총 예측 비용:        {total_cost:,}")
            print(f"▶ 샘플당 평균 비용:    {avg_cost:.4f}")
            
            # 비용 가중 혼동 행렬
            cm_cost = cm * cost_matrix
            print(f"\n▶ 비용 가중 혼동 행렬:")
            print("              예측: C(브론즈)  B(실버)  A(골드)")
            for i, grade_name in enumerate(grade_names):
                row_values = "  ".join([f"{val:>9}" for val in cm_cost[i]])
                print(f"  실제: {grade_name:9s}  {row_values}")
            
            # 치명적 오류 분석 (C→A)
            critical_errors = cm[0, 2]  # C(0) → A(2)
            total_c = cm[0].sum()
            
            print(f"\n🚨 치명적 오류 분석 (C→A):")
            print(f"  - 발생 건수:           {critical_errors}건")
            if total_c > 0:
                print(f"  - 전체 C등급 중 비율:  {critical_errors/total_c*100:.1f}%")
            print(f"  - 이 오류의 총 비용:   {critical_errors * 10:,}")
            
            # 클래스별 비용 기여도
            class_costs = cm_cost.sum(axis=1)
            print(f"\n▶ 클래스별 비용 기여도:")
            for i, grade_name in enumerate(grade_names):
                contribution = class_costs[i]/total_cost*100 if total_cost > 0 else 0
                print(f"  {grade_name:11s}: {class_costs[i]:>6,.0f} ({contribution:.1f}%)")
            
            print("\n💡 해석:")
            print(f"  - 현재 모델은 샘플당 평균 {avg_cost:.2f}의 비용을 발생시킵니다.")
            if critical_errors > 0:
                print(f"  - 가장 위험한 오류(C→A)가 {critical_errors}건 발생했습니다.")
                print(f"  - Cost-Sensitive Learning 적용 시 이를 개선할 수 있습니다.")
            else:
                print(f"  - 치명적 오류가 없습니다! 매우 우수한 성능입니다.")
        
        except Exception as e:
            print(f"\n⚠️ 비용 기반 평가 실패: {e}")
    
    else:
        print("\n⚠️ 기존 trust_score가 없어 Cost-Sensitive 평가를 수행할 수 없습니다.")
        print("   (첫 예측 시에는 평가 불가, 두 번째 예측부터 평가 가능)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        predict()
    except Exception as e:
        print(f"\n✗ 오류: {e}")
        import traceback
        traceback.print_exc()
