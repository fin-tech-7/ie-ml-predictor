import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import pickle

import base64, os

# 🔥 추가: 성능 최적화 설정 (지원되는 옵션만 사용)
# st.set_option('deprecation.showPyplotGlobalUse', False)  # 지원되지 않는 옵션 제거
# st.set_option('deprecation.showfileUploaderEncoding', False)  # 지원되지 않는 옵션 제거
def load_gif_b64(filename):
    path = os.path.join('data', filename)
    try:
        with open(path, 'rb') as f:
            return 'data:image/gif;base64,' + base64.b64encode(f.read()).decode()
    except Exception:
        return None
hat_b64 = load_gif_b64('hat-unscreen.gif')
dance_b64 = load_gif_b64('dance-unscreen.gif')
left_b64 = load_gif_b64('left-unscreen.gif') or hat_b64
right_b64 = load_gif_b64('right-unscreen.gif') or dance_b64
unbias_b64 = load_gif_b64('unbias-unscreen.gif') or hat_b64

# 방향별 내향형 GIF 추가
hat_right_b64 = load_gif_b64('hat-unscreen-right.gif') or hat_b64
hat_left_b64 = load_gif_b64('hat-unscreen.gif') or hat_b64

# 메인화면용 변수명 통일
hat_gif_base64 = hat_b64
dance_gif_base64 = dance_b64

# 🔥 수정: GIF 파일을 base64로 인코딩하여 사용하는 전역 함수 (캐시 추가)
@st.cache_data(ttl=7200)  # 2시간 캐시로 로딩 속도 향상
def get_gif_base64(gif_filename):
    try:
        with open(f"data/{gif_filename}", "rb") as gif_file:
            gif_data = gif_file.read()
            gif_base64 = base64.b64encode(gif_data).decode()
            return f"data:image/gif;base64,{gif_base64}"
    except FileNotFoundError:
        st.error(f"❌ data/{gif_filename} 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"❌ GIF 파일 로드 중 오류: {e}")
        return None

@st.cache_data(ttl=3600)  # 🔥 추가: 1시간 캐시로 로딩 속도 향상
def load_and_prepare_data():
    # 훈련 데이터 로드
    df = pd.read_csv('./data/train.csv')
    
    # 범주형 컬럼을 숫자로 변환
    map01 = {"Yes": 1, "No": 0}
    bin_cols = ["Stage_fear", "Drained_after_socializing"]
    for c in bin_cols:
        if c in df.columns:
            df[c] = df[c].map(map01).astype("float")
    
    # 피처/타깃 지정
    target = "Personality"
    id_col = "id"
    feature_cols = [c for c in df.columns if c not in [id_col, target]]
    
    X = df[feature_cols]
    y = df[target]
    
    return X, y, feature_cols

# CatBoost 모델 훈련 및 저장
@st.cache_resource
def train_and_save_model(X, y, model_path="./model/catboost_model.cbm", validation_path="./model/validation_data.pkl"):
    """
    모델을 학습시키고 저장하는 함수
    """
    # 이미 모델이 저장되어 있다면 로드
    if os.path.exists(model_path) and os.path.exists(validation_path):
        try:
            # 저장된 모델 로드
            model = CatBoostClassifier()
            model.load_model(model_path)
            
            # 검증 데이터 로드
            with open(validation_path, 'rb') as f:
                validation_data = pickle.load(f)
            
            X_val, y_val = validation_data['X_val'], validation_data['y_val']
            
            return model, X_val, y_val
            
        except Exception as e:
            st.warning(f"⚠️ 저장된 모델 로드 실패: {e}.")
    
    return model, X_val, y_val

# 모델 성능 평가 및 저장
@st.cache_data
def evaluate_and_save_model_performance(_model, X_val, y_val, performance_path="./model/model_performance.pkl"):
    """
    모델 성능을 평가하고 저장하는 함수
    """
    import pickle
    import os
    
    # 이미 성능 결과가 저장되어 있다면 로드
    if os.path.exists(performance_path):
        try:
            with open(performance_path, 'rb') as f:
                performance_data = pickle.load(f)
            return performance_data
            
        except Exception as e:
            st.warning(f"⚠️ 저장된 성능 결과 로드 실패: {e}.")
    
# 모델 예측 결과를 일관된 숫자로 변환하는 헬퍼 함수
def get_extrovert_probability(model, X_sample):
    """
    모델 예측 결과에서 외향성 확률을 일관되게 반환
    Extrovert -> 1, Introvert -> 0으로 변환
    """
    try:
        # 확률 예측
        proba = model.predict_proba(X_sample)
        
        # 클래스 순서 확인
        if len(model.classes_) == 2:
            if 'Extrovert' in model.classes_ and 'Introvert' in model.classes_:
                extrovert_idx = list(model.classes_).index('Extrovert')
                return proba[:, extrovert_idx]  # Extrovert 확률 반환
        else:
            # 단일 클래스인 경우
            return proba[:, 0]
    except Exception as e:
        st.error(f"예측 확률 계산 중 오류: {e}")
        return np.zeros(len(X_sample))

# 실시간 예측 함수
def show_realtime_prediction(model):
    # 실제 답변이 있는지 확인
    required_fields = ["time_alone", "stage_fear", "social_events", "going_outside", 
                      "drained_social", "friends_circle", "post_frequency"]
    
    answered_fields = [field for field in required_fields if field in st.session_state.answers]
    
    if len(answered_fields) < 3:  # 최소 3개 이상 답변해야 예측
        st.info("💡 더 많은 질문에 답변해주세요! 최소 3개 이상 답변하면 성향을 분석해드릴게요.")
        return
    
    # 기본값은 사용하지 않고, 답변된 것만 사용
    current_answers = {}
    for field in required_fields:
        if field in st.session_state.answers:
            current_answers[field] = st.session_state.answers[field]
        else:
            # 답변되지 않은 필드는 건너뛰기
            st.warning(f"⚠️ {field} 질문에 답변해주세요.")
            return
    
    # 예측 수행
    stage_fear_num = 1 if current_answers["stage_fear"] == "Yes" else 0
    drained_social_num = 1 if current_answers["drained_social"] == "Yes" else 0
    user_input = np.array([[
        current_answers["time_alone"], stage_fear_num, current_answers["social_events"], 
        current_answers["going_outside"], drained_social_num, current_answers["friends_circle"], 
        current_answers["post_frequency"]
    ]])
    extro_prob = get_extrovert_probability(model, user_input)[0]
    intro_prob = 1 - extro_prob
    
    # GIF 파일들을 base64로 변환
    hat_gif = get_gif_base64("hat-unscreen.gif")
    dance_gif = get_gif_base64("dance-unscreen.gif")
    
    # 실시간 펭귄 시각화 표시
    st.markdown("---")
    st.markdown("### 🐧 실시간 펭귄 시각화")
    
    # 성향에 따른 설명
    if extro_prob >= 0.5:
        st.success(f"🔍 현재 추정: 내향적 성향 ({intro_prob:.1%})")
    else:
        st.success(f"🔍 현재 추정: 외향적 성향 ({extro_prob:.1%})")
    
    # 가로 슬라이더 형태의 펭귄 시각화
    st.markdown("### 🐧 실시간 펭귄 시각화")
    
    # left-unscreen.gif도 추가로 로드
    left_gif = get_gif_base64("left-unscreen.gif")
    
    # 가로 슬라이더 시각화
    st.markdown(f"""
    <div style="position: relative; margin: 20px 0;">
        <!-- 제목과 수치 -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-weight: bold; color: #2c3e50; font-size: 16px;">현재 추정 위치</span>
            <span style="font-size: 18px; color: #e74c3c; font-weight: bold;">{extro_prob:.2f}</span>
        </div>
        
        <!-- 가로 슬라이더 바 -->
        <div style="position: relative; height: 80px; border: 1px solid #eee; border-radius: 10px; overflow: hidden; background: #fafafa;">
            
            <!-- 내향형 펭귄 (왼쪽 끝) - 이글루 배경 -->
            <div style="position: absolute; left: 5px; top: 50%; transform: translateY(-50%); width: 80px; height: 80px; border-radius: 10px; overflow: hidden;">
                <img src="data:image/jpeg;base64,{get_gif_base64('igloo.jpg')}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                <div style="position: absolute; right: 5px; top: 60%; transform: translateY(-50%); width: 90px; height: 90px; border-radius: 50%; overflow: hidden;">
                    <img src="data:image/gif;base64,{hat_gif}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
            </div>
            
            <!-- 사용자 펭귄 (성향에 따라 위치) -->
            <div style="position: absolute; left: {extro_prob * 80 + 10}%; top: 70%; transform: translateY(-50%); width: 60px; height: 60px; border-radius: 50%; overflow: hidden; z-index: 10;">
                <img src="data:image/gif;base64,{left_gif}" style="width: 100%; height: 100%; object-fit: cover;">
            </div>
            
            <!-- 외향형 펭귄 (오른쪽 끝) - 파티 배경 -->
            <div style="position: absolute; right: 5px; top: 50%; transform: translateY(-50%); width: 80px; height: 80px; border-radius: 10px; overflow: hidden;">
                <img src="data:image/jpeg;base64,{get_gif_base64('party.jpeg')}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
                <div style="position: absolute; left: 5px; top: 60%; transform: translateY(-50%); width: 90px; height: 90px; overflow: hidden;">
                    <img src="data:image/gif;base64,{dance_gif}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

# 메인 앱
def main():
    # 🔥 추가: 페이지 설정 최적화 (로딩 속도 향상)
    st.set_page_config(
        page_title="내향/외향 예측", 
        page_icon="🧑‍🤝‍🧑", 
        layout="wide",
        initial_sidebar_state="collapsed"  # 사이드바 접기로 초기 로딩 속도 향상
    )
    
    # 요약 및 예측 단계에서만 상단 여백 추가 (Deploy bar와 겹치지 않도록)
    if "wizard_step" in st.session_state and st.session_state.wizard_step == 8:
        st.markdown("""
        <div style="height: 80px; width: 100%; background: transparent; margin: 0; padding: 0;"></div>
        """, unsafe_allow_html=True)
    
    st.title("🧑‍🤝‍🧑 내향/외향 성향 스펙트럼 설문")
    
    # 데이터 로드 및 모델 로딩/훈련
    X, y, feature_cols = load_and_prepare_data()
    
    # 모델 로딩 또는 훈련 (캐시된 결과 사용)
    model, X_val, y_val = train_and_save_model(X, y)
    
    # 모델 성능 평가 결과 로딩 또는 계산 (캐시된 결과 사용)
    performance_data = evaluate_and_save_model_performance(model, X_val, y_val)
    
    # 탭 구성
    tab2, tab1 = st.tabs(["📝 설문 입력 & 분석", "📊 성향 분포 시각화"])
    
    with tab1:
        st.markdown("### 📊 성향 분포 시각화")
        
        # 🔥 수정: 샘플 데이터 크기 최적화 (로딩 속도 향상)
        sample_size = min(50, len(X_val))  # 100 → 50으로 감소
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        sample_X = X_val.iloc[sample_indices]
        sample_y = y_val.iloc[sample_indices]
        
        # 예측 확률 계산
        sample_probs = model.predict_proba(sample_X)
        
        # 새로운 헬퍼 함수를 사용하여 외향성 확률 계산
        extro_probs = get_extrovert_probability(model, sample_X)
        
        # 데이터프레임 생성
        sample_df = sample_X.copy()
        sample_df['Personality'] = sample_y
        sample_df['extro_prob'] = extro_probs
        sample_df['name'] = [f"Person_{i+1}" for i in range(len(sample_df))]
        
        # 툴팁 텍스트 생성
        def make_hover_text(row):
            text = f"<b>{row['name']}</b><br>"
            text += f"실제 성향: {row['Personality']}<br>"
            text += f"외향성 확률: {row['extro_prob']:.3f}<br>"
            
            # 컬럼명을 한국어로 매핑
            column_mapping = {
                'Time_spent_Alone': '혼자 있는 시간',
                'Stage_fear': '무대 공포증',
                'Social_event_attendance': '사회적 이벤트 참여',
                'Going_outside': '외출 빈도',
                'Drained_after_socializing': '사교 후 지침함',
                'Friends_circle_size': '친구 그룹 크기',
                'Post_frequency': '게시물 작성 빈도'
            }
            
            for col in feature_cols:
                if col in row:
                    korean_name = column_mapping.get(col, col)
                    value = row[col]
                    if pd.isna(value):
                        text += f"{korean_name}: 없음<br>"
                    else:
                        text += f"{korean_name}: {value}<br>"
            return text
        
        # 원형 구조로 시각화
        fig = go.Figure()
        
        # 중심점 (0.5, 0.5)
        center_x, center_y = 0.5, 0.5
        circle_radius = 0.35  # 원의 반지름을 더 크게 (화면의 50% 정도)
        
        # 각 사람을 원형으로 배치
        for i, row in sample_df.iterrows():
            # 360도 전체로 분산 (0 ~ 2π)
            angle = 2 * np.pi * i / len(sample_df)  # 균등하게 360도 분산
            
            # 외향성 확률에 따라 중심에서의 거리 결정
            extro_prob = row['extro_prob']
            
            # 성향에 따라 위치 결정
            if row['Personality'] == 'Introvert':
                # 내향형: 원 밖에 배치 - 외향성 확률에 따라 거리 조정
                # 외향성 확률이 높을수록 원 테두리에 가깝게
                min_distance = 0.45  # 최소 거리
                max_distance = 0.55  # 최대 거리
                # 외향성 확률이 높을수록 원 테두리에 가까워짐 (거리 감소)
                distance = max_distance - (extro_prob * (max_distance - min_distance))
                
                # 좌우 분산을 더 강화
                horizontal_offset = np.random.uniform(-0.1, 0.1)  # 좌우 분산
                vertical_offset = np.random.uniform(-0.05, 0.05)   # 상하 분산
                x = center_x + distance * np.cos(angle) + horizontal_offset
                y = center_y + distance * np.sin(angle) + vertical_offset
                color = 'blue'
                symbol = 'circle'
                # 내향형을 나타내는 펭귄 이모지 (임시)
                emoji = "🐧"  # 펭귄 이모지
            else:
                # 외향형: 원 안에 배치 - 외향성 확률에 따라 거리 조정
                # 외향성 확률이 높을수록 원 중심에 가깝게
                min_distance = 0.05  # 최소 거리 (중심에 가까움)
                max_distance = circle_radius * 0.6  # 최대 거리 (원 안쪽)
                # 외향성 확률이 높을수록 중심에 가까워짐 (거리 감소)
                distance = max_distance - (extro_prob * (max_distance - min_distance))
                
                # 좌우 분산을 더 강화
                horizontal_offset = np.random.uniform(-0.08, 0.08)  # 좌우 분산
                vertical_offset = np.random.uniform(-0.03, 0.03)   # 상하 분산
                x = center_x + distance * np.cos(angle) + horizontal_offset
                y = center_y + distance * np.sin(angle) + vertical_offset
                color = 'red'
                symbol = 'diamond'
                # 외향형을 나타내는 펭귄 이모지 (임시)
                emoji = "🐧"  # 펭귄 이모지
            
            # 개별 점 추가
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=30,  # 아이콘 크기를 더 크게
                    symbol=symbol,
                    color=color,
                    line=dict(width=3, color='white')  # 테두리를 더 굵게
                ),
                text=[emoji],  # 펭귄 이모지로 변경
                textposition="middle center",
                textfont=dict(size=24),  # 텍스트 크기도 더 키움
                hovertext=make_hover_text(row),
                hoverinfo="text",
                showlegend=False
            ))
        
        # GIF 파일을 base64로 인코딩하여 사용
        import base64, os
        
        def get_gif_base64(gif_filename):
            try:
                with open(f"data/{gif_filename}", "rb") as gif_file:
                    gif_data = gif_file.read()
                    gif_base64 = base64.b64encode(gif_data).decode()
                    return f"data:image/gif;base64,{gif_base64}"
            except FileNotFoundError:
                st.error(f"❌ data/{gif_filename} 파일을 찾을 수 없습니다.")
                return None
            except Exception as e:
                st.error(f"❌ GIF 파일 로드 중 오류: {e}")
                return None
        
        # 내향형과 외향형 GIF 로드
        hat_gif_base64 = get_gif_base64("hat-unscreen.gif")
        dance_gif_base64 = get_gif_base64("dance-unscreen.gif")
        
        # 방향별 내향형 GIF 추가
        hat_right_gif_base64 = get_gif_base64("hat-unscreen-right.gif") or hat_gif_base64
        hat_left_gif_base64 = get_gif_base64("hat-unscreen.gif") or hat_gif_base64
        

        
        if hat_gif_base64 and dance_gif_base64:
            # HTML을 사용하여 GIF로 아이콘 표시
            html_content = """
            <style>
            .person-icon {
                position: absolute;
                width: 120px;  /* 🔥 수정: 100px → 120px로 더 크게 증가 */
                height: 120px; /* 🔥 수정: 100px → 120px로 더 크게 증가 */
                border-radius: 50%;
                cursor: pointer;
                overflow: hidden;
                box-shadow: none;
                border: none;
                outline: none;
                /* 초기 위치는 중앙에 고정 */
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                transition: none;
                /* 수평 뒤집기 우선순위 높임 */
                transform-style: preserve-3d;
                backface-visibility: hidden;
            }
            .person-icon.animate {
                transition: left 10s cubic-bezier(0.25, 0.46, 0.45, 0.94), 
                            top 10s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                z-index: 10;
            }
            .person-icon:hover {
                transform: scale(1.2) translate(-50%, -50%); /* 🔥 수정: 1.3 → 1.2로 조정 */
                z-index: 1000;
                box-shadow: none;
            }
            
            /* 호버 시 GIF 이미지 확대 */
            .person-icon:hover .gif-image {
                transform: scale(1.2); /* 🔥 수정: 1.3 → 1.2로 조정 */
            }
            .gif-container {
                position: relative;
                width: 100%;
                height: 800px; /* 🔥 수정: 700px → 800px로 더 높게 증가 */
                background: white;
                border-radius: 10px;
                overflow: hidden;
            }
            .center-circle {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 350px;  /* 🔥 수정: 300px → 350px로 증가 */
                height: 350px; /* 🔥 수정: 300px → 350px로 증가 */
                border-radius: 50%;
                background-color: rgba(211, 211, 211, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px; /* 🔥 수정: 20px → 24px로 증가 */
                font-weight: bold;
                color: transparent;
            }
            .gif-image {
                width: 100%;
                height: 100%;
                object-fit: cover; /* 🔥 유지: 비율 유지하면서 컨테이너에 맞춤 */
                border-radius: 50%;
                background: transparent;
                /* transition 제거 - 수평 뒤집기가 즉시 적용되도록 */
                /* 🔥 추가: GIF 품질 개선 */
                image-rendering: -webkit-optimize-contrast;
                image-rendering: crisp-edges;
            }
            
            /* CSS 애니메이션 제거 - JavaScript로 대체 */
            
            @keyframes penguin-walk {
                0% { 
                    transform: translateY(0px) translateX(0px); 
                }
                25% { 
                    transform: translateY(-2px) translateX(1px); 
                }
                50% { 
                    transform: translateY(0px) translateX(0px); 
                }
                75% { 
                    transform: translateY(-1px) translateX(-1px); 
                }
                100% { 
                    transform: translateY(0px) translateX(0px); 
                }
            }
            
            @keyframes penguin-walk-right {
                0% { 
                    transform: translateY(0px) translateX(0px); 
                }
                25% { 
                    transform: translateY(-3px) translateX(2px); 
                }
                50% { 
                    transform: translateY(0px) translateX(0px); 
                }
                75% { 
                    transform: translateY(-2px) translateX(-2px); 
                }
                100% { 
                    transform: translateY(0px) translateX(0px); 
                }
            }
            
            @keyframes penguin-dance {
                0%, 100% { transform: rotate(0deg); }
                25% { transform: rotate(5deg); }
                75% { transform: rotate(-5deg); }
            }
            .introvert .gif-image {
                filter: brightness(2.2) contrast(1.3);
            }
            .extrovert .gif-image {
                filter: brightness(1.0) contrast(1.0);
            }
            /* 방향별 GIF 사용으로 수평 뒤집기 불필요 */
            /* .hat-right와 .hat-left 클래스는 이제 사용하지 않음 */
            
            /* 방향별 GIF 사용으로 애니메이션 중 수평 뒤집기 불필요 */
            /* .hat-right와 .hat-left 클래스는 이제 사용하지 않음 */
            
            /* 방향별 GIF 사용으로 호버 상태 수평 뒤집기 불필요 */
            /* .hat-right와 .hat-left 클래스는 이제 사용하지 않음 */
            </style>
            
            <div class="gif-container">
                <div class="center-circle">외향 zone</div>
            """
            
            # 각 사람을 HTML로 배치
            for i, row in sample_df.iterrows():
                # 360도 전체로 균등하게 분산 (0 ~ 2π)
                # 첫 번째 사람을 12시 방향(π/2)에서 시작하여 시계방향으로 배치
                angle = (2 * np.pi * i / len(sample_df)) + np.pi/2
                
                # 각도를 -π ~ π 범위로 정규화
                while angle > np.pi:
                    angle -= 2 * np.pi
                while angle < -np.pi:
                    angle += 2 * np.pi
                
                # 외향성 확률에 따라 중심에서의 거리 결정
                extro_prob = row['extro_prob']
                
                # 성향에 따라 위치 결정
                if row['Personality'] == 'Introvert':
                    # 내향형: 원 밖에 배치하되 화면 안에서 보이도록 조정
                    min_distance = 0.35  # 0.45에서 0.35로 줄임
                    max_distance = 0.45  # 0.55에서 0.45로 줄임
                    distance = max_distance - (extro_prob * (max_distance - min_distance))
                    person_class = "introvert"
                    
                    # 방향에 따라 다른 GIF 사용
                    if -np.pi/2 <= angle <= np.pi/2:  # 오른쪽 반원 (오른쪽으로 이동)
                        gif_src = hat_right_gif_base64  # hat-unscreen-right.gif 사용
                        direction_class = "right-direction"
                    else:  # 왼쪽 반원 (왼쪽으로 이동)
                        gif_src = hat_left_gif_base64  # hat-unscreen.gif 사용
                        direction_class = "left-direction"
                else:
                    # 외향형: 원 안에 배치
                    min_distance = 0.05
                    max_distance = 0.3
                    distance = max_distance - (extro_prob * (max_distance - min_distance))
                    person_class = "extrovert"
                    gif_src = dance_gif_base64  # 외향형은 dance-unscreen.gif 사용
                    direction_class = "extrovert-direction"
                
                # 좌표 계산 (HTML 좌표계에 맞춤)
                center_x_html = 50  # 50%
                center_y_html = 50  # 50%
                
                # 거리를 퍼센트로 변환
                distance_percent = distance * 100
                
                # 각도에 따른 위치 계산
                x_percent = center_x_html + distance_percent * np.cos(angle)
                y_percent = center_y_html + distance_percent * np.sin(angle)
                
                # 랜덤 오프셋 추가
                x_offset = np.random.uniform(-2, 2)
                y_offset = np.random.uniform(-2, 2)
                x_percent += x_offset
                y_percent += y_offset
                
                # 툴팁 정보 생성
                tooltip_info = f"""
                <b>{row['name']}</b><br>
                실제 성향: {row['Personality']}<br>
                외향성 확률: {row['extro_prob']:.3f}<br>
                """
                
                # 컬럼명을 한국어로 매핑
                column_mapping = {
                    'Time_spent_Alone': '혼자 있는 시간',
                    'Stage_fear': '무대 공포증',
                    'Social_event_attendance': '사회적 이벤트 참여',
                    'Going_outside': '외출 빈도',
                    'Drained_after_socializing': '사교 후 지침함',
                    'Friends_circle_size': '친구 그룹 크기',
                    'Post_frequency': '게시물 작성 빈도'
                }
                
                for col in feature_cols:
                    if col in row:
                        korean_name = column_mapping.get(col, col)
                        value = row[col]
                        if pd.isna(value):
                            tooltip_info += f"{korean_name}: 없음<br>"
                        else:
                            tooltip_info += f"{korean_name}: {value}<br>"
                
                # 방향별 GIF 사용으로 클래스 분류 불필요
                # 각 GIF가 이미 올바른 방향으로 설정되어 있음
                person_class_with_hat = person_class
                
                # HTML 요소 추가 - 성향에 따라 다른 GIF 사용, 초기 위치는 중앙
                html_content += f"""
                <div class="person-icon {person_class_with_hat} {direction_class}" 
                     data-target-x="{x_percent}" data-target-y="{y_percent}"
                     title="{tooltip_info}">
                    <img src="{gif_src}" alt="펭귄" class="gif-image">
                </div>
                """
                

            
            html_content += """
            </div>
            
            <script>
            (function() {
                // CSS 애니메이션 대신 JavaScript로 애니메이션 구현
                function animatePenguins() {
                    
                    // 더 확실한 선택자 사용
                    var allPenguins = document.querySelectorAll('.person-icon');
                    var rightPenguins = [];
                    var leftPenguins = [];
                    var extrovertPenguins = [];
                    
                    // 각 펭귄을 분류
                    allPenguins.forEach(function(penguin) {
                        if (penguin.classList.contains('right-direction')) {
                            rightPenguins.push(penguin.querySelector('.gif-image'));
                        } else if (penguin.classList.contains('left-direction')) {
                            leftPenguins.push(penguin.querySelector('.gif-image'));
                        } else if (penguin.classList.contains('extrovert-direction')) {
                            extrovertPenguins.push(penguin.querySelector('.gif-image'));
                        }
                    });
                
                    
                    // 🔥 수정: 오른쪽 방향 펭귄들 애니메이션 - 겹칠 때 랜덤으로 앞뒤 순서 변경
                    rightPenguins.forEach(function(img, index) {
                        if (img) {
                            setInterval(function() {
                                // 🔥 추가: 겹칠 때 랜덤으로 z-index 변경
                                var randomZIndex = 10 + Math.floor(Math.random() * 20); // 10~29 랜덤
                                img.parentElement.style.zIndex = randomZIndex;
                                
                                // 오른쪽으로 걸어가는 느낌
                                img.style.transform = 'translateX(2px) translateY(-1px)';
                                setTimeout(function() {
                                    img.style.transform = 'translateX(-1px) translateY(0px)';
                                }, 300);
                                setTimeout(function() {
                                    img.style.transform = 'translateX(0px) translateY(0px)';
                                }, 600);
                            }, 2000 + (index * 100));
                        }
                    });
                    
                    // 🔥 수정: 왼쪽 방향 펭귄들 애니메이션 - 겹칠 때 랜덤으로 앞뒤 순서 변경
                    leftPenguins.forEach(function(img, index) {
                        if (img) {
                            setInterval(function() {
                                // 🔥 추가: 겹칠 때 랜덤으로 z-index 변경
                                var randomZIndex = 10 + Math.floor(Math.random() * 20); // 10~29 랜덤
                                img.parentElement.style.zIndex = randomZIndex;
                                
                                // 왼쪽으로 걸어가는 느낌
                                img.style.transform = 'translateX(-2px) translateY(-1px)';
                                setTimeout(function() {
                                    img.style.transform = 'translateX(1px) translateY(0px)';
                                }, 300);
                                setTimeout(function() {
                                    img.style.transform = 'translateX(0px) translateY(0px)';
                                }, 600);
                            }, 3000 + (index * 150));
                        }
                    });
                    
                    // 🔥 수정: 외향형 펭귄들 애니메이션 - 겹칠 때 랜덤으로 앞뒤 순서 변경
                    extrovertPenguins.forEach(function(img, index) {
                        if (img) {
                            setInterval(function() {
                                // 🔥 추가: 겹칠 때 랜덤으로 z-index 변경
                                var randomZIndex = 10 + Math.floor(Math.random() * 20); // 10~29 랜덤
                                img.parentElement.style.zIndex = randomZIndex;
                                
                                // 자연스러운 흔들림
                                img.style.transform = 'rotate(2deg)';
                                setTimeout(function() {
                                    img.style.transform = 'rotate(-2deg)';
                                }, 200);
                                setTimeout(function() {
                                    img.style.transform = 'rotate(0deg)';
                                }, 400);
                            }, 1500 + (index * 50));
                        }
                    });
                }
                
                // 🔥 추가: GIF들이 겹칠 때 주기적으로 z-index를 랜덤하게 변경하는 함수
                function updateRandomZIndex() {
                    // 모든 펭귄 아이콘의 z-index를 주기적으로 랜덤하게 변경
                    var allPenguinIcons = document.querySelectorAll('.person-icon');
                    allPenguinIcons.forEach(function(icon) {
                        // 🔥 수정: 더 넓은 범위의 z-index로 랜덤하게 설정
                        var randomZIndex = 5 + Math.floor(Math.random() * 50); // 5~54 랜덤
                        icon.style.zIndex = randomZIndex;
                    });
                    
                    console.log('펭귄들의 z-index 랜덤 업데이트 완료!');
                }
                
                // 🔥 추가: 3초마다 z-index 랜덤 업데이트
                setInterval(updateRandomZIndex, 3000);
                
                // 페이지 로드 후 애니메이션 시작
                setTimeout(animatePenguins, 1000);
                
                function startAnimation() {
                    var icons = document.querySelectorAll('.person-icon');
                    var animatedCount = 0;
                    
                    icons.forEach(function(icon, index) {
                        var targetX = icon.getAttribute('data-target-x');
                        var targetY = icon.getAttribute('data-target-y');
                        
                        if (targetX && targetY) {
                            // 각 펭귄마다 약간의 지연을 두어 순차적으로 애니메이션
                            setTimeout(function() {
                                                            // 애니메이션 클래스 추가
                            icon.classList.add('animate');
                            
                            // 강제 리플로우
                            icon.offsetHeight;
                            
                            // 목표 위치로 이동
                            icon.style.left = targetX + '%';
                            icon.style.top = targetY + '%';
                            
                            // 방향별 GIF 사용으로 수평 뒤집기 불필요
                            // 각 GIF가 이미 올바른 방향으로 설정되어 있음
                                
                                animatedCount++;
                                
                                // 모든 애니메이션이 완료되면 콘솔에 로그
                                if (animatedCount === icons.length) {
                                    console.log('모든 펭귄 애니메이션 완료!');
                                }
                            }, index * 100); // 각 펭귄마다 100ms씩 지연
                        }
                    });
                }
                
                // 페이지 로드 후 1초 뒤에 애니메이션 시작
                setTimeout(startAnimation, 1000);
                
                // 추가로 2초 후에도 한 번 더 시도 (혹시 늦게 로드된 경우)
                setTimeout(function() {
                    var icons = document.querySelectorAll('.person-icon');
                    var unanimatedIcons = Array.from(icons).filter(function(icon) {
                        return !icon.classList.contains('animate');
                    });
                    
                    if (unanimatedIcons.length > 0) {
                        console.log('지연된 펭귄들 애니메이션 시작:', unanimatedIcons.length);
                        startAnimation();
                    }
                }, 3000);
            })();
            </script>
            """
            
            # HTML 표시
            st.components.v1.html(html_content, height=800) # 🔥 수정: 700 → 800으로 높이 증가
            
            st.caption("↕ 마우스를 펭귄 아이콘에 올리면 해당 사람의 상세 정보를 볼 수 있습니다.")
        
        else:
            # GIF 로드 실패 시 대안으로 이모지 사용
            st.warning("⚠️ GIF 파일을 로드할 수 없어 이모지로 대체합니다.")
            
            # Plotly로 이모지 시각화
            fig = go.Figure()
            
            # 중심점 (0.5, 0.5)
            center_x, center_y = 0.5, 0.5
            circle_radius = 0.35
            
            # 각 사람을 원형으로 배치
            for i, row in sample_df.iterrows():
                # 360도 전체로 분산 (0 ~ 2π)
                angle = 2 * np.pi * i / len(sample_df)
                
                # 외향성 확률에 따라 중심에서의 거리 결정
                extro_prob = row['extro_prob']
                
                # 성향에 따라 위치 결정
                if row['Personality'] == 'Introvert':
                    # 내향형: 원 밖에 배치
                    min_distance = 0.45
                    max_distance = 0.55
                    distance = max_distance - (extro_prob * (max_distance - min_distance))
                    color = 'blue'
                    symbol = 'circle'
                    emoji = "🐧"
                else:
                    # 외향형: 원 안에 배치
                    min_distance = 0.05
                    max_distance = circle_radius * 0.6
                    distance = max_distance - (extro_prob * (max_distance - min_distance))
                    color = 'red'
                    symbol = 'diamond'
                    emoji = "🐧"
                
                # 개별 점 추가
                fig.add_trace(go.Scatter(
                    x=[center_x + distance * np.cos(angle)],
                    y=[center_y + distance * np.sin(angle)],
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        symbol=symbol,
                        color=color,
                        line=dict(width=3, color='white')
                    ),
                    text=[emoji],
                    textposition="middle center",
                    textfont=dict(size=24),
                    hovertext=make_hover_text(row),
                    hoverinfo="text",
                    showlegend=False
                ))
            
            # 중심 원 추가
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = center_x + circle_radius * np.cos(theta)
            circle_y = center_y + circle_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=5),
                showlegend=False
            ))
            
            # 원 안에 "외향 zone" 텍스트 추가
            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='text',
                text=["외향<br>zone"],
                textposition="middle center",
                textfont=dict(size=20, color='black'),
                showlegend=False
            ))
            
            # 레이아웃 설정
            fig.update_layout(
                xaxis=dict(title="", range=[0, 1], showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=False),
                yaxis=dict(title="", range=[0, 1], showgrid=True, gridcolor='lightgray', zeroline=False, showticklabels=False),
                height=600, width=700, showlegend=False,
                title="펭귄 이모지로 성향 분포 시각화 (마우스를 올리면 상세 정보 표시)",
                plot_bgcolor='white'
            )
            
            # 정확한 원형을 위해 aspectmode 설정
            fig.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("↕ 마우스를 펭귄 이모지에 올리면 해당 사람의 상세 정보를 볼 수 있습니다.")
    
    with tab2:
        current_step = 0
        # 9번 탭(요약 및 예측)이 아닐 때만 헤더와 스타일 표시
        if current_step != 8:
            # st.markdown("### 📝 나의 성향 입력하기")

            # 스타일 - 버튼/칩/진행바/카드
            st.markdown(
                """
                <style>
                .wizard-wrap{padding:20px 22px;border-radius:16px;background:rgba(255,255,255,.7);border:1px solid rgba(0,0,0,.06);box-shadow:0 8px 28px rgba(15,23,42,.08)}
                .progress-outer{height:10px;background:#eef2ff;border-radius:999px;overflow:hidden;border:1px solid #e5e7eb;margin:4px 0 10px}
                .progress-inner{height:100%;background:linear-gradient(90deg,#5b8cff,#9b7bff)}
                .step-chips{margin:6px 0 2px}
                .step-chip{display:inline-block;padding:6px 10px;border-radius:999px;font-size:12px;margin-right:6px;background:#f3f4f6;color:#111827;border:1px solid #e5e7eb}
                .step-chip.active{background:linear-gradient(135deg,#5b8cff,#9b7bff);color:#fff;border-color:transparent}
                
                /* Enhanced button styles with better visual feedback */
                .stButton>button{
                    border-radius:12px;
                    padding:12px 14px;
                    font-weight:700;
                    border:0;
                    background:linear-gradient(135deg,#7aa2ff,#9b7bff);
                    color:#fff;
                    box-shadow:0 6px 18px rgba(91,140,255,.35);
                    transition:all .2s ease;
                    position:relative;
                    overflow:hidden;
                }
                
                .stButton>button:hover{
                    transform:translateY(-2px);
                    box-shadow:0 10px 24px rgba(91,140,255,.45);
                    background:linear-gradient(135deg,#8bb3ff,#ac8cff);
                }
                
                .stButton>button:active{
                    transform:translateY(0);
                    box-shadow:0 4px 12px rgba(91,140,255,.3);
                }
                
                /* Enhanced selected state with better visual feedback */
                .stButton>button:disabled{
                    opacity:1;
                    background:linear-gradient(135deg,#10b981,#059669);
                    color:#fff;
                    box-shadow:0 8px 25px rgba(16,185,129,.4);
                    border:3px solid #34d399;
                    transform:scale(1.05);
                    position:relative;
                    font-weight:800;
                    text-shadow:0 1px 2px rgba(0,0,0,0.3);
                }
                
                /* Add a strong glow effect for selected buttons */
                .stButton>button:disabled::before{
                    content:'';
                    position:absolute;
                    top:-4px;
                    left:-4px;
                    right:-4px;
                    bottom:-4px;
                    background:linear-gradient(135deg,#10b981,#34d399,#6ee7b7);
                    border-radius:16px;
                    z-index:-1;
                    opacity:0.8;
                    animation:strong-pulse 1.5s infinite;
                }
                
                /* Stronger pulse animation for selected buttons */
                @keyframes strong-pulse{
                    0%, 100% { opacity: 0.8; transform: scale(1); }
                    50% { opacity: 1; transform: scale(1.1); }
                }
                
                /* Add a prominent checkmark icon for selected buttons */
                .stButton>button:disabled::after{
                    content:'✓';
                    position:absolute;
                    top:50%;
                    right:18px;
                    transform:translateY(-50%);
                    font-size:20px;
                    font-weight:bold;
                    color:#fff;
                    text-shadow:0 2px 4px rgba(0,0,0,0.5);
                    background:rgba(255,255,255,0.2);
                    border-radius:50%;
                    width:24px;
                    height:24px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                }
                
                /* live score bar */
                .score-card{margin:8px 0 14px;padding:12px;border-radius:14px;background:rgba(243,244,246,.7);border:1px solid #e5e7eb}
                .score-bar{height:14px;background:#e5e7eb;border-radius:999px;position:relative;overflow:hidden}
                .score-fill{position:absolute;left:0;top:0;height:100%;background:linear-gradient(90deg,#93c5fd,#60a5fa,#34d399)}
                .score-marker{position:absolute;top:-6px;width:2px;height:26px;background:#111827;border-radius:1px}
                </style>
                """,
                unsafe_allow_html=True,
            )

        # 멀티스텝 상태 초기화
        if "wizard_step" not in st.session_state:
            st.session_state.wizard_step = 0
        if "answers" not in st.session_state:
            st.session_state.answers = {
                "name": "",
                "time_alone": None,
                "stage_fear": None,
                "social_events": None,
                "going_outside": None,
                "drained_social": None,
                "friends_circle": None,
                "post_frequency": None,
            }
        if "temp_answers" not in st.session_state:
            st.session_state.temp_answers = {}  # 빈 상태로 초기화
        if "prob_history" not in st.session_state:
            st.session_state.prob_history = []
        # 🔥 추가: 이전 설문에서의 실제 펭귄 위치 저장
        if "penguin_positions" not in st.session_state:
            st.session_state.penguin_positions = []

        steps = [
            "펭귄 닉네임",
            "혼자 있는 시간",
            "무대 공포증",
            "한달내 행사 참여",
            "일주일 외출 빈도",
            "사회생활 기빨림",
            "친구 몇명?",
            "일주일내 인스타 포스트 빈도",
            "요약 및 예측",
        ]
        total_steps = len(steps)
        current_step = st.session_state.wizard_step

        # 진행률/칩 표시 (9번 탭이 아닐 때만)
        if current_step != 8:
            percent = int((current_step) / (total_steps - 1) * 100)
            st.markdown(f'<div class="progress-outer"><div class="progress-inner" style="width:{percent}%"></div></div>', unsafe_allow_html=True)
            chips_html = ''.join([f'<span class="step-chip {"active" if i==current_step else ""}">{i+1}. {label}</span>' for i,label in enumerate(steps)])
            st.markdown(f'<div class="step-chips">{chips_html}</div>', unsafe_allow_html=True)

        # 현재까지 응답으로 추정 확률 계산 함수
        feature_means = X.mean()
        
        def estimate_extro_prob(ans_dict):
            value_by_feature = {
                'Time_spent_Alone': ans_dict.get('time_alone', None),
                'Stage_fear': 1.0 if ans_dict.get('stage_fear', 'No') == 'Yes' else 0.0,
                'Social_event_attendance': ans_dict.get('social_events', None),
                'Going_outside': ans_dict.get('going_outside', None),
                'Drained_after_socializing': 1.0 if ans_dict.get('drained_social', 'No') == 'Yes' else 0.0,
                'Friends_circle_size': ans_dict.get('friends_circle', None),
                'Post_frequency': ans_dict.get('post_frequency', None),
            }
            
            row = []
            for col in feature_cols:
                v = value_by_feature.get(col, None)
                if v is None:
                    mv = feature_means.get(col, 0.0)
                    try:
                        if np.isnan(mv):
                            mv = 0.0
                    except Exception:
                        mv = 0.0
                    row.append(mv)
                else:
                    row.append(float(v))
            
            arr = np.array([row])
            prob = model.predict_proba(arr)[0]
            # classes_ indexing
            if 'Extrovert' in list(model.classes_):
                ext_idx = list(model.classes_).index('Extrovert')
            else:
                ext_idx = 1 if len(prob) > 1 else 0
            return float(prob[ext_idx])

        # 라이브 스코어: 펭귄 트랙 시각화 (9번 탭이 아닐 때만 표시)
        if current_step != 8:  # 9번 탭(요약 및 예측)이 아닐 때만
            # 0번째 단계(이름 입력)에서는 예측하지 않음
            if current_step == 0:
                live_prob = 0.5  # 중립값
                prev_prob = None
                delta = None
            else:
                # 1번째 단계부터 예측 확률 계산 (확정된 답변만 사용)
                # temp_answers는 제외하고 확정된 answers만 사용
                current_answers = st.session_state.answers.copy()
                
                live_prob = estimate_extro_prob(current_answers)
                
                # 이전 단계와 비교
                if len(st.session_state.prob_history) > 0:
                    prev_prob = st.session_state.prob_history[-1]  # 마지막 값 사용
                    delta = live_prob - prev_prob
                    print(current_step, "delta", delta)
            
            # 🔥 수정: prob_history 누적 방식 변경 - 단계별로 딱 하나씩만 저장
            if len(st.session_state.prob_history) <= current_step:
                st.session_state.prob_history.append(live_prob)
                print(f"새로운 확률 저장: Step {current_step} = {live_prob:.3f}")
            else:
                st.session_state.prob_history[current_step] = live_prob
                print(f"기존 확률 업데이트: Step {current_step} = {live_prob:.3f}")
            
            # 🔥 추가: prob_history 상태 디버깅
            print(f"현재 prob_history: {[f'{i}:{p:.3f}' for i, p in enumerate(st.session_state.prob_history)]}")
            
                                        # 🔥 수정: 펭귄 위치를 먼저 계산하고, 이동 방향에 따라 GIF 선택
            # 🔥 수정: start_pos와 end_pos를 모든 경우에 확실하게 정의
            if current_step in (0, 1):
                # 초기 단계: 항상 unbias
                walker_b64 = unbias_b64
                penguin_position = 50
                start_pos = 50
                end_pos = 50
            else:
                # 🔥 수정: 펭귄 위치 계산 (확정된 답변 기반)
                if live_prob <= 0.5:
                    # 내향형 ~ 중립: 10% ~ 50% 범위
                    penguin_position = 10 + (live_prob * 80)
                else:
                    # 중립 ~ 외향형: 50% ~ 90% 범위
                    penguin_position = 50 + ((live_prob - 0.5) * 80)
                
                # 🔥 추가: 현재 펭귄 위치를 저장 (다음 설문의 start_pos로 사용)
                if len(st.session_state.penguin_positions) <= current_step:
                    st.session_state.penguin_positions.append(penguin_position)
                    print(f"새로운 펭귄 위치 저장: Step {current_step} = {penguin_position:.1f}%")
                else:
                    st.session_state.penguin_positions[current_step] = penguin_position
                    print(f"기존 펭귄 위치 업데이트: Step {current_step} = {penguin_position:.1f}%")
                
                # 🔥 추가: penguin_positions 상태 디버깅
                print(f"현재 penguin_positions: {[f'{i}:{p:.1f}%' for i, p in enumerate(st.session_state.penguin_positions)]}")
                
                # 🔥 수정: start_pos와 end_pos 정확하게 계산
                # 이전 펭귄 위치가 있으면 그것을 start_pos로 사용
                if current_step == 0:
                    start_pos = 50  # 첫 번째 단계는 중앙에서 시작
                    print(f"Step {current_step}: 첫 번째 단계, start_pos = 50% (중앙)")
                elif len(st.session_state.penguin_positions) <= current_step - 1:
                    # 🔥 수정: 이전 단계의 펭귄 위치가 아직 저장되지 않은 경우
                    start_pos = 50
                    print(f"Step {current_step}: 이전 단계 펭귄 위치 없음, start_pos = 50% (중앙)")
                else:
                    # 🔥 수정: 이전 단계의 실제 펭귄 위치 사용
                    start_pos = int(st.session_state.penguin_positions[current_step - 1])
                    print(f"Step {current_step}: 이전 단계 펭귄 위치 사용, start_pos = {start_pos}%")
                
                # 🔥 수정: end_pos는 현재 펭귄 위치와 동일
                end_pos = int(penguin_position)
                
                # 0~100 범위 제한
                start_pos = max(4, min(96, start_pos))
                end_pos = max(4, min(96, end_pos))
                
                print(f"위치 계산: start_pos = {start_pos}%, end_pos = {end_pos}%")
                
                # 🔥 수정: walker_b64 애니메이션 방향 결정 개선
                if end_pos > start_pos:
                    walker_b64 = right_b64
                    print(f"오른쪽 이동: {start_pos}% → {end_pos}% (right 펭귄)")
                else:
                    walker_b64 = left_b64
                    print(f"왼쪽 이동: {start_pos}% → {end_pos}% (left 펭귄)")
                
                # 🔥 추가: walker_b64 디버깅
                print(f"Debug - walker_b64 설정: {walker_b64[:50]}..." if walker_b64 else "Debug - walker_b64: None")
            
            # 🔥 추가: 디버깅을 위한 변수 상태 출력
            print(f"Debug - Step {current_step}: start_pos={start_pos}, end_pos={end_pos}, penguin_position={penguin_position}")

            # JS로 start -> end로 부드럽게 이동하도록 구성
            st.components.v1.html(
                f"""
                <style>
                .penguin-card{{margin:8px 0 14px;padding:12px 14px;border-radius:14px;background:rgba(243,244,246,.7);border:1px solid #e5e7eb}}
                .penguin-track{{position:relative;height:120px}}
                .rail{{position:absolute;left:6%;right:6%;top:48px;height:6px;background:#e5e7eb;border-radius:999px}}
                .endpoint{{position:absolute;top:8px;width:80px;height:80px;border-radius:50%;overflow:hidden}}
                .endpoint img{{width:100%;height:100%;object-fit:cover;border-radius:50%}}
                .endpoint.left{{left:0}}
                .endpoint.right{{right:0}}
                .walker{{position:absolute;top:20px;left:{start_pos}%;transform:translateX(-50%);width:80px;height:80px;border-radius:50%;overflow:hidden;transition:left 4s ease-in-out}}
                .walker img{{width:100%;height:100%;object-fit:cover;border-radius:50%}}
                .labels{{display:flex;justify-content:space-between;font-size:12px;margin-top:8px;color:#4b5563}}
                .rail::before {{
                    content: '';
                    position: absolute;
                    top: -2px;
                    left: 0;
                    right: 0;
                    height: 10px;
                    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #10b981);
                    border-radius: 999px;
                    opacity: 0.3;
                }}
                
                @keyframes userBounce {{
                    0%, 100% {{ transform: translateX(-50%) translateY(-50%); }}
                    50% {{ transform: translateX(-50%) translateY(-50%); }}
                }}
                
                @keyframes penguin-walk {{
                    0%, 100% {{ transform: translateX(-50%) translateY(-50%); }}
                    25% {{ transform: translateX(-52%) translateY(-50%); }}
                    50% {{ transform: translateX(-50%) translateY(-50%); }}
                    75% {{ transform: translateX(-48%) translateY(-50%); }}
                }}
                
                .user-penguin {{
                    /* 초기 transition 제거 - 갑작스러운 이동 방지 */
                    transition: none;
                }}
                
                .introvert-penguin, .extrovert-penguin {{
                    transition: all 0.3s ease;
                    cursor: pointer;
                }}
                
                /* 내향 펭귄 밝기 증가 */
                .introvert-penguin img {{
                    filter: brightness(1.5) contrast(1.2);
                }}
                
                /* 펭귄 호버 시 크기 증가 */
                .introvert-penguin:hover, .extrovert-penguin:hover {{
                    transform: scale(1.1);
                }}
                </style>
                <div class="penguin-visualization">
                  <div style="display:flex;justify-content:space-between;margin-bottom:15px;align-items:center">
                    <span style="font-weight:bold;color:#2c3e50;">현재 추정 위치</span>
                  </div>
                  
                  <!-- 가로 슬라이더 형태의 펭귄 시각화 -->
                  <div class="penguin-space" style="position:relative;height:120px;border:1px solid #eee;border-radius:10px;margin:10px 0;overflow:hidden;background:#fafafa;">
                    
                    <!-- 🔥 수정: 내향형 펭귄 (왼쪽 끝) - 이글루 배경 (겹침 방지) -->
                    <div style="position:absolute;left:5px;top:50%;transform:translateY(-50%);width:100px;height:100px;border-radius:10px;overflow:hidden;z-index:5;">
                      <img src="{get_gif_base64('igloo.jpg') or ''}" style="width:100%;height:100%;object-fit:cover;border-radius:10px;">
                      <div style="position:absolute;right:2px;top:60%;transform:translateY(-50%);width:90px;height:90px;border-radius:50%;overflow:hidden;filter:brightness(1.5) contrast(1.2);">
                        <img src="{hat_b64 or ''}" style="width:100%;height:100%;object-fit:cover;">
                      </div>
                    </div>
                    
                    <!-- 🔥 수정: 사용자 펭귄 (성향에 따라 위치) - 더 높은 z-index -->
                    <div class="user-penguin" id="user-penguin" style="
                      position:absolute;top:57%;left:{penguin_position}%;transform:translateY(-50%);
                      width:100px;height:100px;border-radius:50%;overflow:hidden;z-index:15;
                    ">
                      <img src="{walker_b64 or ''}" style="width:100%;height:100%;object-fit:cover;">
                    </div>
                    
                    <!-- 🔥 수정: 외향형 펭귄 (오른쪽 끝) - 파티 배경 (겹침 방지) -->
                    <div style="position:absolute;right:5px;top:50%;transform:translateY(-50%);width:100px;height:100px;border-radius:10px;overflow:hidden;z-index:5;">
                      <img src="{get_gif_base64('party.jpeg') or ''}" style="width:100%;height:100%;object-fit:cover;border-radius:10px;">
                      <div style="position:absolute;left:2px;top:60%;transform:translateY(-50%);width:90px;height:90px;border-radius:50%;overflow:hidden;">
                        <img src="{dance_b64 or ''}" style="width:100%;height:100%;object-fit:cover;">
                      </div>
                    </div>
                  </div>
            
                </div>
                <script>
                (function(){{
                  var userPenguin = document.getElementById('user-penguin');
                  if(!userPenguin) return;
                  
                  
                  // GIF 파일들을 변수로 정의
                  var leftB64 = '{left_b64 or ""}';
                  var rightB64 = '{right_b64 or ""}';
                  var unbiasB64 = '{unbias_b64 or ""}';
                  
                  // 🔥 수정: 성향에 따라 펭귄 GIF 변경 (Python에서 설정된 walker_b64 사용)
                  function updatePenguinGIF() {{
                    var penguinImg = userPenguin.querySelector('img');
                    if (!penguinImg) return;
                    
                    // 🔥 수정: Python에서 설정된 walker_b64 직접 사용
                    var walkerB64 = '{walker_b64 or ""}';
                    console.log('Python에서 설정된 walker_b64:', walkerB64 ? walkerB64.substring(0, 50) + '...' : 'None');
                    
                    if (walkerB64 && walkerB64.length > 10) {{
                        // Python에서 설정된 GIF 사용
                        penguinImg.src = walkerB64;
                        console.log('GIF 변경됨:', walkerB64 === leftB64 ? 'LEFT' : (walkerB64 === rightB64 ? 'RIGHT' : 'UNBIAS'));
                    }} else {{
                        // 🔥 수정: fallback 로직 (기존 방식)
                        var currentStep = {current_step};
                        var liveProb = {live_prob};
                        
                        if (currentStep <= 1) {{
                           // 초기 단계 또는 중립 (0.4~0.6): unbias-unscreen.gif 사용
                           penguinImg.src = unbiasB64;
                        }} else if (liveProb > 0.5) {{
                           // 외향적: right-unscreen.gif 사용
                           penguinImg.src = rightB64;
                        }} else {{
                           // 내향적: left-unscreen.gif 사용
                           penguinImg.src = leftB64;
                        }}
                    }}
                  }}
                  
                  // 사용자 펭귄 위치 계산 및 애니메이션
                  function animateUserPosition() {{
                    // Python에서 계산한 정확한 위치 사용
                    var targetLeft = {penguin_position};
                    
                    // 위치가 변경된 경우에만 애니메이션 적용
                    var currentLeft = parseFloat(userPenguin.style.left);
                    if (Math.abs(currentLeft - targetLeft) > 1) {{
                      // start_pos에서 end_pos까지 천천히 걸어가기
                      userPenguin.style.left = targetLeft + '%';
                      userPenguin.style.transition = 'left 4s ease-in-out';
                      
                      // 걸음걸이 애니메이션 추가
                      userPenguin.style.animation = 'penguin-walk 2s infinite';
                    }}
                    
                    // 성향에 따른 색상 변화 - 원래 색상 유지
                    userPenguin.style.filter = 'drop-shadow(3px 3px 6px rgba(0,0,0,0.4))';
                    
                    // GIF 업데이트
                    updatePenguinGIF();
                  }}
                  
                  // 펭귄들 호버 효과
                  var introvertPenguins = document.querySelectorAll('.introvert-penguin');
                  var extrovertPenguins = document.querySelectorAll('.extrovert-penguin');
                  
                  introvertPenguins.forEach(function(penguin) {{
                    penguin.addEventListener('mouseenter', function() {{
                      this.style.transform = 'scale(1.2) rotate(-5deg)';
                      this.style.transition = 'all 0.3s ease';
                    }});
                    
                    penguin.addEventListener('mouseleave', function() {{
                      this.style.transform = 'scale(1) rotate(0deg)';
                    }});
                  }});
                  
                  extrovertPenguins.forEach(function(penguin) {{
                    penguin.addEventListener('mouseenter', function() {{
                      this.style.transform = 'scale(1.2) rotate(5deg)';
                      this.style.transition = 'all 0.3s ease';
                    }});
                    
                    penguin.addEventListener('mouseleave', function() {{
                      this.style.transform = 'scale(1) rotate(0deg)';
                    }});
                  }});
                  
                  // 초기 GIF 설정
                  updatePenguinGIF();
                  
                  // 움직임 제어: start_pos에서 end_pos로 자연스럽게 이동
                  var startPos = {start_pos};
                  var endPos = {end_pos};
                  
                  // 초기 위치를 start_pos로 설정
                  userPenguin.style.left = startPos + '%';
                  userPenguin.style.transition = 'none';  // 초기 위치는 애니메이션 없이
                  
                  if (startPos !== endPos) {{
                    // 움직임이 있는 경우: 애니메이션 활성화
                    console.log('펭귄 움직임:', startPos + '% → ' + endPos + '%');
                    
                    // 잠시 후 end_pos까지 천천히 걸어가기
                    setTimeout(function() {{
                      userPenguin.style.transition = 'left 4s ease-in-out';
                      userPenguin.style.left = endPos + '%';
                      userPenguin.style.animation = 'penguin-walk 2s infinite';
                    }}, 500);
                  }} else {{
                    // 움직임이 없는 경우: 현재 위치에 고정
                    console.log('펭귄 위치 고정:', startPos + '%');
                    userPenguin.style.left = startPos + '%';
                    userPenguin.style.transition = 'none';
                    userPenguin.style.animation = 'none';
                  }}
                  
                  // 주기적 애니메이션 제거 - 펭귄이 밑으로 내려가지 않도록
                  // setInterval(function() {{
                  //   userPenguin.style.animation = 'none';
                  //   setTimeout(function() {{
                  //     userPenguin.style.animation = 'userBounce 2s infinite';
                  //   }}, 10);
                  // }}, 4000);
                }})();
                </script>
                """,
                height=180,  # 🔥 수정: 160 → 180으로 높이 증가
            )
        # 옵션 버튼 그리드 렌더러 (임시 저장, 다음 단계 버튼으로 적용)
        def render_option_buttons(options, field, columns=6, formatter=lambda x: str(x)):
            # temp_answers에서만 확인 (기본값 무시)
            temp_selected = st.session_state.temp_answers.get(field)
            
            # 기본값은 설정하지 않음 (사용자가 직접 선택해야 함)

            step_cols = columns
            for i in range(0, len(options), step_cols):
                row = options[i:i + step_cols]
                cols = st.columns(len(row))
                for j, opt in enumerate(row):
                    is_sel = (temp_selected == opt) if temp_selected is not None else False
                    label = f"✅ {formatter(opt)}" if is_sel else formatter(opt)
                    if cols[j].button(label, key=f"{field}_opt_{i+j}", use_container_width=True, disabled=is_sel):
                        st.session_state.temp_answers[field] = opt
                        # 🔥 수정: 즉시 UI 업데이트를 위해 rerun 추가
                        st.rerun()

        # 한 페이지 한 입력 UI (애니메이션 형식)
        if current_step == 0:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #1f77b4; font-size: 2.5em; margin: 0;">🐧</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">펭귄에게 닉네임을 부여해주세요!</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                name_input = st.text_input(
                    "닉네임 입력",
                    value=st.session_state.answers.get("name", ""),
                    key="name_input",
                    placeholder="예: 펭펭이, 귀요미, 뚱뚱이...",
                    label_visibility="collapsed"
                )
                if name_input:
                    st.session_state.temp_answers["name"] = name_input

            # st.markdown(
            #     """
            #     <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 20px 0;">
            #         <h4>🎉 펭귄 닉네임 등록 완료!</h4>
            #         <p>이제 재미있는 질문들에 답변하면서 당신만의 성향을 발견해보세요! 🐧</p>
            #     </div>
            #     """,
            #     unsafe_allow_html=True,
            # )

        elif current_step == 1:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #e74c3c; font-size: 2.5em; margin: 0;">🏠</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">하루 중 혼자 있는 시간은 얼마나 되나요?</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 🔥 슬라이더를 버튼 그리드로 변경
            render_option_buttons(list(range(0, 11)), "time_alone", columns=6, 
                                formatter=lambda x: f"{x}시간")

        elif current_step == 2:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #9b59b6; font-size: 2.5em; margin: 0;">🎭</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">무대에 서면 떨리나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">많은 사람들 앞에서 발표할 때의 느낌을 말해주세요!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                no_selected = st.session_state.temp_answers.get("stage_fear") == "No"
                button_text = "✅ 😎 아니요, 괜찮아요!" if no_selected else "😎 아니요, 괜찮아요!"
                if st.button(button_text, key="stage_fear_no", use_container_width=True):
                    st.session_state.temp_answers["stage_fear"] = "No"
                    # 🔥 수정: 즉시 UI 업데이트를 위해 rerun 추가
                    st.rerun()

            with col2:
                yes_selected = st.session_state.temp_answers.get("stage_fear") == "Yes"
                button_text = "✅ 😰 네, 떨려요..." if yes_selected else "😰 네, 떨려요..."
                if st.button(button_text, key="stage_fear_yes", use_container_width=True):
                    st.session_state.temp_answers["stage_fear"] = "Yes"
                    # 🔥 수정: 즉시 UI 업데이트를 위해 rerun 추가
                    st.rerun()

        elif current_step == 3:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #f39c12; font-size: 2.5em; margin: 0;">🎉</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">한 달에 몇 번이나 행사에 참여하나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">생일파티, 동창회, 회식 등 재미있는 모임들! 🎊</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_option_buttons(list(range(0, 11)), "social_events", columns=6)

        elif current_step == 4:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #2ecc71; font-size: 2.5em; margin: 0;">🚶‍♀️</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">일주일에 몇 번이나 외출하나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">집콕 vs 외출러버, 당신은 어느 쪽? 🏃‍♂️</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_option_buttons(list(range(0, 8)), "going_outside", columns=8)

        elif current_step == 5:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #e67e22; font-size: 2.5em; margin: 0;">😴</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">사교 후에 기빨리나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">친구들과 놀고 난 후 에너지 상태는? ⚡</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_option_buttons(["No", "Yes"], "drained_social", columns=2)

        elif current_step == 6:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #3498db; font-size: 2.5em; margin: 0;">👥</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">친구가 몇 명이나 되나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">소수의 베프 vs 많은 친구들, 당신의 선택은? 🤝</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_option_buttons(list(range(0, 16)), "friends_circle", columns=8)

        elif current_step == 7:
            st.markdown(
                """
                <div style="text-align: center; margin: 20px 0;">
                    <h2 style="color: #9b59b6; font-size: 2.5em; margin: 0;">📱</h2>
                    <h3 style="color: #2c3e50; margin: 10px 0;">일주일에 인스타에 몇 번 포스팅하나요?</h3>
                    <p style="color: #7f8c8d; font-size: 0.9em;">인스타러버 vs 숨은 인스타, 당신은? 📸</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_option_buttons(list(range(0, 11)), "post_frequency", columns=6)

        elif current_step == 8:
            # 설문 요약은 제거하고 바로 예측 결과로 이동
            ans = st.session_state.answers

            # 예측 수행
            if ans["name"]:
                stage_fear_num = 1 if ans["stage_fear"] == "Yes" else 0
                drained_social_num = 1 if ans["drained_social"] == "Yes" else 0

                user_input = np.array([[
                    ans["time_alone"],
                    stage_fear_num,
                    ans["social_events"],
                    ans["going_outside"],
                    drained_social_num,
                    ans["friends_circle"],
                    ans["post_frequency"],
                ]])

                # 새로운 헬퍼 함수를 사용하여 예측 결과 계산
                prediction = model.predict(user_input)[0]
                extro_prob = get_extrovert_probability(model, user_input)[0]
                intro_prob = 1 - extro_prob

                # 스타일이 적용된 테이블로 표시
                st.markdown("""
                <style>
                    .survey-table {
                        background: white;
                        border-radius: 15px;
                        padding: 20px;
                        margin: 20px 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }
                    .survey-table h4 {
                        color: #40bbd1;
                        text-align: center;
                        margin-bottom: 20px;
                    }
                    .step-number {
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        padding: 8px 15px;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 0.9em;
                    }
                    .question-text {
                        color: #2c3e50;
                        font-weight: 600;
                        font-size: 1.1em;
                    }
                    .answer-text {
                        color: #e74c3c;
                        font-weight: bold;
                        font-size: 1.1em;
                    }
                </style>
                """, unsafe_allow_html=True)
                
   
                st.markdown("""
                <div style="text-align: center; margin: 20px 0 20px 0; padding-top: 1rem;">
                    <h4 style="color: #40bbd1; font-size: 1.5em; margin: 10px 0;">나는 외향펭귄일까 내향펭귄일까?</h4>
                    <p style="color: #7f8c8d; font-size: 0.9em;">스크롤하여 각 특성별로 펭귄들이 어떻게 분포되어 있는지 확인해보세요! 📊</p>
                </div>
                """, unsafe_allow_html=True)
                
                
                # 3.py 스타일의 배경색과 스크롤 섹션 CSS
                st.markdown("""
                <style>
                    /* 전체 앱 배경색 설정 */
                    .stApp { 
                        background-color: #ADD8E6; 
                    }
                </style>
                """, unsafe_allow_html=True)
                
                try:
                    SCALE = 1.2  # 🔥 수정: 1.5 → 1.2로 감소 (로딩 속도 향상)
                    BASE_W, BASE_H = 720, 420
                    PLOT_W, PLOT_H = int(BASE_W * SCALE), int(BASE_H * SCALE)
                    MAX_PENGUINS = 150  # 🔥 수정: 300 → 150으로 감소 (로딩 속도 향상)
                    RANDOM_SEED = 11
                    

                    column_titles = {
                        'Time_spent_Alone': '하루에 혼자 있는 시간',
                        'Stage_fear': '무대 공포증',
                        'Social_event_attendance': '한달내 행사 참여',
                        'Going_outside': '일주일 외출 빈도',
                        'Drained_after_socializing': '사회생활 기빨림',
                        'Friends_circle_size': '친구 몇명?',
                        'Post_frequency': '일주일내 인스타 포스트 빈도'
                    }
                    bin_cols = ['Drained_after_socializing', 'Stage_fear']
                    selected_cols = list(column_titles.keys())
                    display_titles = [column_titles[col] for col in selected_cols]
                    
                                        
                    # 데이터 준비
                    X, y, feature_cols = load_and_prepare_data()

                    # ⬇️ 학습/검증셋과 모델 받기 (변경된 반환값 사용)
                    model, X_val, y_val = train_and_save_model(X, y)

                    # ⬇️ 검증셋에서 예측 및 정답 맞힌 샘플만 풀로 사용
                    y_pred_val = model.predict(X_val)
                    mask_correct = (np.asarray(y_pred_val).ravel() == np.asarray(y_val).ravel())
                    X_pool = X_val.iloc[mask_correct]
                    y_pool = y_val.iloc[mask_correct]

                    # ⬇️ 풀에서의 외향성 확률
                    extro_prob_pool = get_extrovert_probability(model, X_pool)
                

                    #  (Extrovert/Introvert 50:50 균형) — 풀(y_pool) 기준
                    rng = np.random.RandomState(RANDOM_SEED)
                    classes = list(pd.Series(y_pool).unique())

                    # 🔥 문제 해결: 클래스 순서를 명시적으로 지정
                    if 'Extrovert' in classes and 'Introvert' in classes:
                        cls_extrovert = 'Extrovert'
                        cls_introvert = 'Introvert'
                    else:
                        # 클래스명이 다른 경우 첫 번째와 두 번째 사용
                        cls_extrovert = classes[0]
                        cls_introvert = classes[1]
                    
                    y_pool_np = np.asarray(y_pool)
                    idx_extrovert = np.where(y_pool_np == cls_extrovert)[0]
                    idx_introvert = np.where(y_pool_np == cls_introvert)[0]
                    
                    # 🔥 문제 해결: 셔플 제거하여 인덱스 일관성 유지
                    per_cls = min(len(idx_extrovert), len(idx_introvert), MAX_PENGUINS // 2)
                    sample_idx = np.concatenate([idx_extrovert[:per_cls], idx_introvert[:per_cls]])
                    
                    # 🔥 문제 해결: 최종 셔플 제거하여 순서 유지
                    # rng.shuffle(sample_idx)  # 이 줄 제거

                    # 각 컬럼의 전체 데이터 범위 미리 계산 (y축 범위 설정용)
                    column_ranges = {}
                    for col in selected_cols:
                        is_bin = col in bin_cols
                        if is_bin:
                            # 이진 컬럼: 0~1 범위
                            column_ranges[col] = [0, 1]
                        else:
                            # 연속 컬럼: 전체 데이터의 실제 범위
                            full_col_values = X_pool[col].astype(float).values
                            min_val = np.nanmin(full_col_values)
                            max_val = np.nanmax(full_col_values)
                            column_ranges[col] = [min_val, max_val * 1.1]

                    # 🔥 완전히 다른 접근법: extro_prob_pool을 제거하고 고정된 X값 생성
                    # extro_prob_pool은 모델 예측 결과로 컬럼마다 다를 수 있음
                    # 대신 균등 분포로 고정된 X값 생성
                    num_samples = len(sample_idx)
                    extro_prob_fixed = np.linspace(0.05, 0.95, num_samples)  # 0.05 ~ 0.95 범위로 균등 분포

                    
                    # fig_jsons + dots_data 생성 — 풀(X_pool) 기준
                    fig_jsons = []
                    dots_data = {}
                    

                    for col in selected_cols:
                        is_bin = col in bin_cols
                        
                        # sample_idx에 맞춰서 y_vals 생성 (인덱싱 문제 해결)
                        y_vals = X_pool.iloc[sample_idx][col].astype(float).values
                        
                        # 🔥 수정: 고정된 X값 사용
                        x_vals = extro_prob_fixed  # 모든 컬럼에서 동일한 X값 사용

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,  # 샘플링된 데이터 사용
                            mode='markers',
                            marker=dict(size=6, opacity=0),  # 오버레이 펭귄만 보이게 숨김
                            showlegend=False, hoverinfo='skip'
                        ))

                        fig.update_layout(
                            width=PLOT_W, height=PLOT_H,
                            margin=dict(l=60, r=30, t=60, b=60),
                            template='plotly_white',
                            font=dict(family='Pretendard, Apple SD Gothic Neo, sans-serif', size=18),
                            plot_bgcolor='rgba(255,255,255,0.3)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            showlegend=False
                        )

                        fig.update_xaxes(
                            title=dict(text='외향 <—> 내향', font=dict(color='white', size=18)),
                            range=[0, 1],
                            zeroline=False,
                            showline=True, linewidth=0.5, linecolor='#edfcff', mirror=True,
                            gridcolor='rgba(148,163,184,0.3)',
                            ticks='outside', ticklen=6, tickcolor='#edfcff',
                            tickfont=dict(color='black', size=20)
                        )

                        if is_bin:
                            fig.update_yaxes(
                                title=dict(text=column_titles[col], font=dict(color='white', size=18)),
                                range=[-0.5, 1.5],
                                dtick=1,
                                tickmode='array',
                                tickvals=[0, 1],
                                ticktext=['No', 'Yes'],
                                zeroline=False,
                                showline=True, linewidth=0.5, linecolor='#edfcff', mirror=True,
                                showgrid=True, gridcolor='#edfcff', gridwidth=1,
                                tickfont=dict(color='white', size=15)
                            )
                        else:
                            fig.update_yaxes(
                                title=dict(text='', font=dict(color='white', size=18)),
                                range=[0, float(np.nanmax(y_vals))*1.1],
                                zeroline=False,
                                showline=True, linewidth=0.5, linecolor='#edfcff', mirror=True,
                                gridcolor='#edfcff',
                                tickfont=dict(color='white', size=15)
                            )

                        fig_jsons.append(fig.to_json())

                        # dots_data — 풀의 균형 샘플에서 좌표 생성
                        coords = []
                        for i in range(len(sample_idx)):
                            xv = float(x_vals[i])  # 이미 sample_idx로 필터링된 x_vals 사용
                            yv = float(y_vals[i])  # y_vals[i]는 이미 sample_idx에 맞춰져 있음
                            if np.isfinite(xv) and np.isfinite(yv):
                                coords.append({'x': xv, 'y': yv, 'is_user': False})



                        dots_data[col] = coords
                        
                    # 설문자 데이터 추가
                    if 'prediction' in locals() and (extro_prob is not None) and 'answers' in st.session_state:
                        ans = st.session_state.answers
                        user_coords = []
                        
                        # 각 컬럼별로 설문자 데이터 생성
                        for col in selected_cols:
                            user_y_val = None
                            if col == 'Time_spent_Alone':
                                user_y_val = ans['time_alone']
                            elif col == 'Stage_fear':
                                user_y_val = 1 if ans['stage_fear'] == 'Yes' else 0
                            elif col == 'Social_event_attendance':
                                user_y_val = ans['social_events']
                            elif col == 'Going_outside':
                                user_y_val = ans['going_outside']
                            elif col == 'Drained_after_socializing':
                                user_y_val = 1 if ans['drained_social'] == 'Yes' else 0
                            elif col == 'Friends_circle_size':
                                user_y_val = ans['friends_circle']
                            elif col == 'Post_frequency':
                                user_y_val = ans['post_frequency']
                            
                            if user_y_val is not None:
                                user_coord = {
                                    'x': extro_prob,
                                    'y': float(user_y_val),
                                    'is_user': True,
                                    'prediction': prediction,
                                    'extro_prob': extro_prob
                                }
                                user_coords.append(user_coord)
                        
                        # 각 컬럼의 dots_data에 설문자 데이터 추가
                        for col in selected_cols:
                            if col in dots_data:
                                dots_data[col].extend(user_coords)
                        
                    # 3.py와 동일한 펭귄 GIF 로드 (아래 부분은 기존 코드 그대로 이어짐)
                    def _read_as_data_url(path: str) -> str:
                        import base64
                        from pathlib import Path
                        p = Path(path)
                        if not p.exists():
                            return ''
                        mime = 'image/gif' if p.suffix.lower() == '.gif' else 'image/png'
                        return f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode('ascii')

                    peng_dance = _read_as_data_url('./data/dance-unscreen.gif')
                    peng_sleep = _read_as_data_url('./data/sleeping-unscreen.gif')
                                        
                    # 3.py와 동일한 스크롤 섹션 생성
                    st.markdown("""
                    <style>
                        .stApp { background-color: #ADD8E6; }
                        .scroll-section {
                        min-height: 100vh;
                        display: flex;
                        align-items: flex-start;
                        # padding-top: 12vh;
                        padding-left: 60px;
                        background-size: cover !important;
                        background-position: center !important;
                        background-repeat: no-repeat !important;
                        }
                        .scroll-title { font-size: 48px; font-weight: 800; color: #40bbd1; }
                        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
                    </style>
                    """, unsafe_allow_html=True)

                    # ⬇️ 추가: 섹션 배경용 이미지 로더 (/data/img/<컬럼명>.png → data URL)
                    def _bg_data_url_for(col: str) -> str:
                        try:
                            import base64
                            from pathlib import Path
                            p = Path(f'./data/img/{col}.png')
                            if not p.exists():
                                return ''
                            return 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode('ascii')
                        except Exception:
                            return ''

                    for i, title in enumerate(display_titles):
                        col_key = selected_cols[i]
                        bg_url = _bg_data_url_for(col_key)
                        style_attr = f"background-image:url('{bg_url}');" if bg_url else ""
                        st.markdown(
                            f"<section class='scroll-section' id='sec-{i}' data-idx='{i}' style=\"{style_attr}\">"
                            #f"<div class='scroll-title'>{title}</div>"
                            f"</section>",
                            unsafe_allow_html=True,
                        )

                    # 3.py와 동일한 JavaScript 코드 (펭귄 오버레이 + 스크롤 애니메이션)
                    import json
                    st.components.v1.html(f"""
                                            
                    <script>
                    (function(){{
                        const figs = {json.dumps(fig_jsons)};
                        const colNames = {json.dumps(selected_cols)};
                        const dotsData = {json.dumps(dots_data)};
                        const parentWin = window.parent;
                        const doc = parentWin.document;

                        // 펭귄 GIF (항상 dance / sleeping만 사용)
                        const PENG_DANCE = "{peng_dance}";
                        const PENG_SLEEP = "{peng_sleep}";

                        const N = (dotsData[colNames[0]] || []).length;

                        // 고정 홀더
                        let holder = doc.getElementById('fixed-graph-holder');
                        if (!holder) {{
                        holder = doc.createElement('div');
                        holder.id = 'fixed-graph-holder';
                        holder.style.position = 'fixed';
                        holder.style.bottom = '50px';  // 페이지 하단에 위치
                        holder.style.left = '50%';
                        holder.style.transform = 'translateX(-50%)';
                        holder.style.width = '{PLOT_W}px';
                        holder.style.height = '{PLOT_H}px';
                        holder.style.zIndex = '9999';
                        holder.style.pointerEvents = 'auto';
                        holder.style.background = 'transparent';

                        // Plotly 그래프 영역
                        const plotDiv = doc.createElement('div');
                        plotDiv.id = 'fixed-graph-plot';
                        plotDiv.style.position = 'absolute';
                        plotDiv.style.inset = '0';
                        plotDiv.style.pointerEvents = 'none';  // ✅ 그래프만 이벤트 막기 (툴팁 방해 방지)
                        plotDiv.style.background = 'rgba(255,255,255,0.8)'; // 반투명 흰 배경
                        plotDiv.style.borderRadius = '12px';                // 모서리 둥글게
                        plotDiv.style.boxShadow = '0 8px 24px rgba(0,0,0,0.15)'; // 그림자 효과


                        holder.appendChild(plotDiv);

                        // 펭귄 오버레이 레이어
                        const overlay = doc.createElement('div');
                        overlay.id = 'penguin-overlay';
                        overlay.style.position = 'absolute';
                        overlay.style.inset = '0';
                        overlay.style.pointerEvents = 'auto';
                        overlay.style.zIndex = '10000';        // ✅ 그래프보다 위

                        holder.appendChild(overlay);

                        doc.body.appendChild(holder);

                        // 펭귄 생성 (이미지) — 표본 수에 맞춰 생성 (설문자 제외)
                        window.penguins = [];
                        for (let i = 0; i < N; i++) {{
                            const img = doc.createElement('img');
                            img.style.position = 'absolute';
                            img.style.top = '0px';
                            img.style.left = '0px';
                            img.style.transform = 'translate(-50%, -50%)';
                            img.style.width = '60px';
                            img.style.height = '60px';
                            img.style.borderRadius = '50%';
                            img.style.opacity = '0';
                            img.style.pointerEvents = 'auto';
                            img.style.zIndex = '10001';        // ✅ overlay보다 위
                            img.style.cursor = 'pointer';
                            
                            // 일반 펭귄용 스타일
                            img.classList.add('penguin-img');
                            overlay.appendChild(img);

                            const pengObj = {{
                                el: img,
                                x: {PLOT_W/2}, 
                                y: {PLOT_H/2}, 
                                tx: {PLOT_W/2}, 
                                ty: {PLOT_H/2},
                                lastState: 'dance',
                                isIntro: false,
                                meta: null,
                                isUser: false,  // 일반 펭귄
                                penguinId: i
                            }};
                            
                            // 초기 위치를 그래프 영역 내부로 제한
                            pengObj.x = Math.max(60 + 30, Math.min({PLOT_W} - 30 - 30, pengObj.x));
                            pengObj.y = Math.max(60 + 30, Math.min({PLOT_H} - 60 - 30, pengObj.y));
                            pengObj.tx = pengObj.x;
                            pengObj.ty = pengObj.y;
                            img.__pengRef = pengObj;
                            window.penguins.push(pengObj);
                        }}

                        // 🔥 설문자 펭귄을 완전히 별도로 생성
                        window.userPenguin = null;
                        if (dotsData[colNames[0]] && dotsData[colNames[0]].some(c => c.is_user)) {{
                            const userData = dotsData[colNames[0]].find(c => c.is_user);
                            if (userData) {{
                                const userImg = doc.createElement('img');
                                userImg.style.position = 'absolute';
                                userImg.style.top = '0px';
                                userImg.style.left = '0px';
                                userImg.style.transform = 'translate(-50%, -50%)';
                                userImg.style.width = '60px';
                                userImg.style.height = '60px';
                                userImg.style.borderRadius = '50%';
                                userImg.style.opacity = '0';
                                userImg.style.pointerEvents = 'auto';
                                userImg.style.zIndex = '10002';        // 일반 펭귄보다 위
                                userImg.style.cursor = 'pointer';
                                userImg.classList.add('penguin-img', 'user-penguin');
                                
                                // 🔥 설문자 펭귄 GIF를 즉시 설정
                                const userExtroProb = userData.extro_prob || 0.5;
                                const userGif = (userExtroProb < 0.5) ? PENG_SLEEP : PENG_DANCE;
                                userImg.src = userGif;
                                
                                overlay.appendChild(userImg);
                                
                                window.userPenguin = {{
                                    el: userImg,
                                    x: {PLOT_W/2}, 
                                    y: {PLOT_H/2}, 
                                    tx: {PLOT_W/2}, 
                                    ty: {PLOT_H/2},
                                    isUser: true,
                                    userGif: userGif,  // 🔥 GIF 고정
                                    meta: userData
                                }};
                                
                                console.log('설문자 펭귄 별도 생성 완료! GIF:', userGif === PENG_SLEEP ? 'SLEEP' : 'DANCE');
                            }}
                        }}
                        

                        
                        // 설문자 펭귄 스타일 CSS 추가
                        const style = doc.createElement('style');
                        style.textContent = `
                            .penguin-img.user-penguin {{
                                border: 4px solid #ff6b6b !important;
                                background: rgba(255, 107, 107, 0.2) !important;
                                box-shadow: 0 0 20px rgba(255, 107, 107, 0.6) !important;
                                z-index: 1000 !important;
                            }}
                            .penguin-img.user-penguin:hover {{
                                transform: scale(1.2) translate(-50%, -50%) !important;
                                border-color: #ff4757 !important;
                                background: rgba(255, 71, 87, 0.3) !important;
                            }}
                        `;
                        doc.head.appendChild(style);
                        }}

                        const plotDiv = doc.getElementById('fixed-graph-plot');

                        // Plotly 라이브러리 로드 함수
                        function loadPlotly(cb) {{
                            if (parentWin.Plotly) return cb();
                            const s = doc.createElement('script');
                            // 최신 Plotly.js 버전 사용 (구버전 plotly-latest.min.js 대신)
                            s.src = 'https://cdn.plot.ly/plotly-2.27.1.min.js';
                            s.onload = cb;
                            doc.head.appendChild(s);
                        }}

                        // 그래프 렌더링 함수
                        function renderGraph(index) {{
                            const fig = JSON.parse(figs[index]);
                            const currentCol = colNames[index];
                            
                            // 현재 컬럼에 따른 y축 범위 설정
                            let yAxisRange = [0, 10]; // 기본값
                            let yAxisTitle = '';
                            
                            // 컬럼별 y축 범위와 제목 설정
                            if (currentCol === 'Stage_fear' || currentCol === 'Drained_after_socializing') {{
                                // 이진 컬럼: -0.5 ~ 1.5
                                yAxisRange = [0, 1];
                                yAxisTitle = currentCol === 'Stage_fear' ? '무대 공포증' : '사교 후 지침함';
                            }} else if (currentCol === 'Time_spent_Alone') {{
                                // 혼자 있는 시간: 0 ~ 10
                                yAxisRange = [0, 10];
                                yAxisTitle = '하루에 혼자 있는 시간';
                            }} else if (currentCol === 'Social_event_attendance') {{
                                // 한달내 행사 참여: 0 ~ 10
                                yAxisRange = [0, 10];
                                yAxisTitle = '한달내 행사 참여';
                            }} else if (currentCol === 'Going_outside') {{
                                // 일주일 외출 빈도: 0 ~ 7
                                yAxisRange = [0, 7];
                                yAxisTitle = '일주일 외출 빈도';
                            }} else if (currentCol === 'Friends_circle_size') {{
                                // 친구 몇명: 0 ~ 15
                                yAxisRange = [0, 15];
                                yAxisTitle = '친구 몇명?';
                            }} else if (currentCol === 'Post_frequency') {{
                                // 일주일내 인스타 포스트 빈도: 0 ~ 10
                                yAxisRange = [0, 10];
                                yAxisTitle = '일주일내 인스타 포스트 빈도';
                            }}
                            
                            // layout을 직접 수정하여 x축과 y축 설정 강제 적용
                            const updatedLayout = {{
                                ...fig.layout,
                                xaxis: {{
                                    title: '내향 <—> 외향',
                                    titlefont: {{color: 'black', size: 20}},
                                    range: [0, 1],
                                    zeroline: false,
                                    showline: true,
                                    linewidth: 0.5,
                                    linecolor: '#edfcff',
                                    mirror: true,
                                    gridcolor: 'rgba(148,163,184,0.3)',
                                    ticks: 'outside',
                                    ticklen: 6,
                                    tickcolor: '#edfcff',
                                    tickfont: {{color: 'black', size: 20}},
                                    showticklabels: true
                                }},
                                yaxis: {{
                                    title: '',
                                    titlefont: {{color: 'white', size: 18}},
                                    range: yAxisRange,
                                    zeroline: false,
                                    showline: true,
                                    linewidth: 0.5,
                                    linecolor: '#edfcff',
                                    mirror: true,
                                    gridcolor: '#edfcff',
                                    tickfont: {{color: 'black', size: 20}},
                                    showticklabels: true
                                }}
                            }};
                            
                            // newPlot으로 완전히 새로 그리기 (react 대신)
                            parentWin.Plotly.newPlot(plotDiv, fig.data, updatedLayout, {{displayModeBar: false}});
                            
                            // 추가로 relayout으로 한 번 더 확실하게 적용
                            setTimeout(() => {{
                                parentWin.Plotly.relayout(plotDiv, {{
                                    'xaxis.title': '내향 <—> 외향',
                                    'xaxis.titlefont.color': 'black',
                                    'xaxis.titlefont.size': 20,
                                    'xaxis.range': [0, 1],
                                    'xaxis.zeroline': false,
                                    'xaxis.showline': true,
                                    'xaxis.linewidth': 0.5,
                                    'xaxis.linecolor': '#edfcff',
                                    'xaxis.mirror': true,
                                    'xaxis.gridcolor': 'rgba(148,163,184,0.3)',
                                    'xaxis.ticks': 'outside',
                                    'xaxis.ticklen': 6,
                                    'xaxis.tickcolor': '#edfcff',
                                    'xaxis.tickfont.color': 'black',
                                    'xaxis.tickfont.size': 20,
                                    'xaxis.showticklabels': true,
                                    'yaxis.title': yAxisTitle,
                                    'yaxis.titlefont.color': 'white',
                                    'yaxis.titlefont.size': 18,
                                    'yaxis.range': yAxisRange,
                                    'yaxis.zeroline': false,
                                    'yaxis.showline': true,
                                    'yaxis.linewidth': 0.5,
                                    'yaxis.linecolor': '#edfcff',
                                    'yaxis.mirror': true,
                                    'yaxis.gridcolor': '#edfcff',
                                    'yaxis.tickfont.color': 'black',
                                    'yaxis.tickfont.size': 20,
                                    'yaxis.showticklabels': true
                                }});
                            }}, 100);
                            
                            setTimeout(() => movePenguins(colNames[index]), 200);
                        }}

                        // Y값 포맷팅 함수
                        function formatY(colName, y) {{
                            const binCols = new Set(['Stage_fear', 'Drained_after_socializing']);
                            if (binCols.has(colName)) return (Number(y) >= 0.5 ? 'Yes' : 'No');
                            const num = Number(y);
                            return Number.isFinite(num) ? num.toFixed(2) : String(y);
                        }}

                        // 펭귄 이동 함수
                        function movePenguins(colName) {{
                            try {{
                                const coords = dotsData[colName] || [];
                                const total = window.penguins.length;
                                
                                // 그래프 영역의 실제 플롯 영역 계산 (마진 제외)
                                const plotArea = {{
                                    left: 60,    // 왼쪽 마진
                                    right: {PLOT_W} - 30,  // 오른쪽 마진
                                    top: 60,     // 상단 마진
                                    bottom: {PLOT_H} - 60   // 하단 마진
                                }};

                                // 🔥 설문자 펭귄 별도 처리
                                if (window.userPenguin && coords.some(c => c.is_user)) {{
                                    const userData = coords.find(c => c.is_user);
                                    if (userData) {{
                                        const userExtroProb = userData.extro_prob || 0.5;
                                        const userY = userData.y;
                                        
                                        // x 좌표: 외향형 확률
                                        const x01 = Math.max(0, Math.min(1, Number(userExtroProb) || 0));
                                        const userX = plotArea.left + x01 * (plotArea.right - plotArea.left);
                                        
                                        // y 좌표: 각 컬럼의 실제 범위를 고려하여 계산
                                        let userYPos;
                                        if (colName === 'Stage_fear' || colName === 'Drained_after_socializing') {{
                                            userYPos = plotArea.bottom + (userY * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Time_spent_Alone') {{
                                            userYPos = plotArea.bottom + ((userY / 11) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Social_event_attendance') {{
                                            userYPos = plotArea.bottom + ((userY / 10) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Going_outside') {{
                                            userYPos = plotArea.bottom + ((userY / 7) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Friends_circle_size') {{
                                            userYPos = plotArea.bottom + ((userY / 15) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Post_frequency') {{
                                            userYPos = plotArea.bottom + ((userY / 10) * (plotArea.top - plotArea.bottom));
                                        }} else {{
                                            userYPos = plotArea.bottom + (userY * (plotArea.top - plotArea.bottom));
                                        }}
                                        
                                        // 위치 제한
                                        userYPos = Math.max(plotArea.top + 30, Math.min(plotArea.bottom - 30, userYPos));
                                        
                                        window.userPenguin.tx = userX;
                                        window.userPenguin.ty = userYPos;
                                        window.userPenguin.el.style.opacity = '1';
                                        
                                        // 툴팁 설정
                                        const ext = userExtroProb * 100;
                                        const intv = 100 - ext;
                                        const yText = formatY(colName, userY);
                                        window.userPenguin.el.title = '설문자\\n값: ' + yText + '\\n외향: ' + ext.toFixed(1) + '% / 내향: ' + intv.toFixed(1) + '%';
                                    }}
                                }}
                                
                                for (let i = 0; i < total; i++) {{
                                    try {{
                                        const p = window.penguins[i];
                                        if (!p || !coords[i]) {{
                                            if (p) {{
                                                p.tx = -80; 
                                                p.ty = -80;
                                                p.el.style.opacity = '0';
                                                p.isIntro = false;
                                                p.meta = null;
                                                p.isUser = false;
                                                // 설문자 스타일 제거
                                                p.el.classList.remove('user-penguin');
                                            }}
                                            continue;
                                        }}
                                        
                                        const c = coords[i];
                                        
                                        // x 좌표: 0~1 범위를 플롯 영역의 left~right로 변환
                                        const x01 = Math.max(0, Math.min(1, Number(c.x) || 0)); // 안전클램프
                                        let px = plotArea.left + x01 * (plotArea.right - plotArea.left);
                                        
        
                                        
                                        // y 좌표: 각 컬럼의 실제 범위를 고려하여 계산 (수정)
                                        let py;
                                        if (colName === 'Stage_fear' || colName === 'Drained_after_socializing') {{
                                            // 이진 컬럼: 0~1 범위
                                            py = plotArea.bottom + (c.y * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Time_spent_Alone') {{
                                            // 혼자 있는 시간: 0~11 범위
                                            py = plotArea.bottom + ((c.y / 11) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Social_event_attendance') {{
                                            // 한달내 행사 참여: 0~10 범위
                                            py = plotArea.bottom + ((c.y / 10) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Going_outside') {{
                                            // 일주일 외출 빈도: 0~7 범위
                                            py = plotArea.bottom + ((c.y / 7) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Friends_circle_size') {{
                                            // 친구 몇명: 0~15 범위
                                            py = plotArea.bottom + ((c.y / 15) * (plotArea.top - plotArea.bottom));
                                        }} else if (colName === 'Post_frequency') {{
                                            // 일주일내 인스타 포스트 빈도: 0~10 범위
                                            py = plotArea.bottom + ((c.y / 10) * (plotArea.top - plotArea.bottom));
                                        }} else {{
                                            // 기본값: 0~1 범위로 가정
                                            py = plotArea.bottom + (c.y * (plotArea.top - plotArea.bottom));
                                        }}
                                        // 펭귄이 그래프 영역을 벗어나지 않도록 제한
                                        
                                        py = Math.max(plotArea.top + 30, Math.min(plotArea.bottom - 30, py));
                                        
                                        // 🔥 설문자 데이터는 별도 펭귄으로 처리 (일반 펭귄에서는 제외)
                                        if (c.is_user) {{
                                            // 일반 펭귄에서는 설문자 데이터 무시
                                            p.tx = -80; 
                                            p.ty = -80;
                                            p.el.style.opacity = '0';
                                            p.isUser = false;
                                            p.el.classList.remove('user-penguin');
                                            continue;
                                        }}

                                                                                    // 🔥 수정: 0.5 근처 데이터를 더 엄격하게 처리
                                            if (typeof c.x === 'number') {{
                                                if (c.x < 0.45) {{
                                                    p.isIntro = true;  // 명확한 내향형
                                                }} else if (c.x > 0.55) {{
                                                    p.isIntro = false; // 명확한 외향형
                                                }} else {{
                                                    // 0.45 ~ 0.55 범위: 중립 (기본값으로 처리)
                                                    p.isIntro = (c.x < 0.5);
                                                }}
                                            }} else {{
                                                p.isIntro = false;
                                            }}
                                            const desired = p.isIntro ? PENG_SLEEP : PENG_DANCE;



                                        if (!p.el.__gifStarted) {{
                                            p.el.src = desired;
                                            p.el.__gifStarted = true;
                                        }}

                                        // 일반 펭귄 툴팁: y값 + 외향/내향 %
                                        const ext = (typeof c.x === 'number' ? c.x : 0.5) * 100;
                                        const intv = 100 - ext;
                                        const yText = formatY(colName, c.y);
                                        p.el.title = '값: ' + yText + '\\n외향: ' + ext.toFixed(1) + '% / 내향: ' + intv.toFixed(1) + '%';

                                        p.tx = px;
                                        p.ty = py;
                                        p.el.style.opacity = '1';
                                        p.meta = {{ x: c.x, y: c.y, col: colName, isUser: c.is_user }};

                                    }} catch (penguinError) {{
                                        console.error('펭귄 ' + i + ' 처리 중 오류:', penguinError);
                                    }}
                                }}
                            }} catch (moveError) {{
                                console.error('movePenguins 함수 오류:', moveError);
                            }}
                        }}

                        // 애니메이션 함수
                        function animate() {{
                            try {{
                                const ease = 0.30;
                                
                                // 그래프 영역의 실제 플롯 영역 계산 (마진 제외)
                                const plotArea = {{
                                    left: 60,    // 왼쪽 마진
                                    right: {PLOT_W} - 30,  // 오른쪽 마진
                                    top: 60,     // 상단 마진
                                    bottom: {PLOT_H} - 60   // 하단 마진
                                }};
                                
                                // 🔥 설문자 펭귄 애니메이션 (별도 처리)
                                if (window.userPenguin && window.userPenguin.el) {{
                                    try {{
                                        const vx = window.userPenguin.tx - window.userPenguin.x;
                                        const vy = window.userPenguin.ty - window.userPenguin.y;
                                        window.userPenguin.x += vx * ease;
                                        window.userPenguin.y += vy * ease;
                                        
                                        // 펭귄이 그래프 영역을 벗어나지 않도록 제한
                                        window.userPenguin.x = Math.max(plotArea.left + 30, Math.min(plotArea.right - 30, window.userPenguin.x));
                                        window.userPenguin.y = Math.max(plotArea.top + 30, Math.min(plotArea.bottom - 30, window.userPenguin.y));
                                        
                                        window.userPenguin.el.style.transform = 'translate3d(' + window.userPenguin.x + 'px,' + window.userPenguin.y + 'px,0) translate(-50%, -50%)';
                                    }} catch (userPenguinError) {{
                                        console.error('설문자 펭귄 애니메이션 오류:', userPenguinError);
                                    }}
                                }}
                                
                                if (window.penguins && Array.isArray(window.penguins)) {{
                                    for (const p of window.penguins) {{
                                        try {{
                                            if (p && p.el) {{
                                                const vx = p.tx - p.x;
                                                const vy = p.ty - p.y;
                                                p.x += vx * ease;
                                                p.y += vy * ease;
                                                
                                                // 펭귄이 그래프 영역을 벗어나지 않도록 제한
                                                p.x = Math.max(plotArea.left + 30, Math.min(plotArea.right - 30, p.x));
                                                p.y = Math.max(plotArea.top + 30, Math.min(plotArea.bottom - 30, p.y));
                                                
                                                p.el.style.transform = 'translate3d(' + p.x + 'px,' + p.y + 'px,0) translate(-50%, -50%)';
                                            }}
                                        }} catch (penguinError) {{
                                            console.error('개별 펭귄 애니메이션 오류:', penguinError);
                                        }}
                                    }}
                                }}
                                requestAnimationFrame(animate);
                            }} catch (animateError) {{
                                console.error('animate 함수 오류:', animateError);
                                // 오류가 발생해도 애니메이션 계속
                                requestAnimationFrame(animate);
                            }}
                        }}

                        // 스크롤 옵저버 설정 함수
                        function setupObserver() {{
                            try {{
                                const sections = Array.from(doc.querySelectorAll('.scroll-section'));
                                let current = -1;
                                
                                if (sections.length === 0) {{
                                    console.warn('스크롤 섹션을 찾을 수 없습니다.');
                                    return;
                                }}
                                
                                const obs = new parentWin.IntersectionObserver((entries) => {{
                                    try {{
                                        let best = null;
                                        for (const e of entries) {{
                                            if (e.isIntersecting) {{
                                                if (!best || e.intersectionRatio > best.intersectionRatio) best = e;
                                            }}
                                        }}
                                        if (best) {{
                                            const idx = parseInt(best.target.getAttribute('data-idx'));
                                            if (idx !== current && !isNaN(idx)) {{
                                                current = idx;
                                                renderGraph(idx);
                                            }}
                                        }}
                                    }} catch (observerError) {{
                                        console.error('IntersectionObserver 콜백 오류:', observerError);
                                    }}
                                }}, {{ threshold: [0.25, 0.5, 0.75] }});
                                
                                sections.forEach(sec => obs.observe(sec));
                            }} catch (setupError) {{
                                console.error('setupObserver 함수 오류:', setupError);
                            }}
                        }}

                        // 초기화 함수
                        loadPlotly(() => {{
                            try {{
                                renderGraph(0);
                                animate();
                                setupObserver();
                            }} catch (initError) {{
                                console.error('초기화 중 오류:', initError);
                            }}
                        }});
                    }})();
                    </script>
                    """, height=1)

                except Exception as e:
                    st.error(f"❌ 3.py 스타일 시각화 로드 실패: {str(e)}")
                    st.info("💡 **문제 해결**: 데이터 로딩이나 모델 예측에 문제가 있을 수 있습니다.")

        st.markdown('</div>', unsafe_allow_html=True)


        # 애니메이션 네비게이션 버튼
        col_prev, col_next = st.columns(2)

        with col_prev:
            if st.button("◀ 이전 단계", disabled=current_step == 0, use_container_width=True, 
                    help="이전 질문으로 돌아가기"):
                st.session_state.wizard_step = max(0, current_step - 1)
                st.rerun()

        with col_next:
            # 🔥 각 단계별 입력 완료 여부 확인
            can_go_next = True
            
            if current_step == 0:  # 펭귄 닉네임
                if not st.session_state.temp_answers.get("name"):
                    can_go_next = False
            elif current_step == 1:  # 혼자 있는 시간
                if "time_alone" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 2:  # 무대 공포증
                if "stage_fear" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 3:  # 한달내 행사 참여
                if "social_events" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 4:  # 일주일 외출 빈도
                if "going_outside" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 5:  # 사회생활 기빨림
                if "drained_social" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 6:  # 친구 몇명?
                if "friends_circle" not in st.session_state.temp_answers:
                    can_go_next = False
            elif current_step == 7:  # 일주일내 인스타 포스트 빈도
                if "post_frequency" not in st.session_state.temp_answers:
                    can_go_next = False

            next_text = "다음 단계 ▶" if current_step < total_steps - 1 else "🎉 분석 완료!"
            if st.button(next_text, disabled=not can_go_next, use_container_width=True,
                    help="다음 질문으로 진행하기"):
                # 임시 저장소의 값들을 실제 답변에 적용
                if "temp_answers" in st.session_state:
                    for field, value in st.session_state.temp_answers.items():
                        if value is not None:  # None이 아닌 값만 적용
                            st.session_state.answers[field] = value

                # 다음 단계로 진행
                st.session_state.wizard_step = min(total_steps - 1, current_step + 1)
                st.rerun()

if __name__ == "__main__":
    main()
