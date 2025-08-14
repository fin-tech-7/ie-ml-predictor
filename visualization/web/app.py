import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# 데이터 로드 및 전처리
@st.cache_data
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

# CatBoost 모델 훈련
@st.cache_resource
def train_catboost_model(X, y):
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pool 구성
    tr_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val, y_val)
    
    # 모델 정의
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        auto_class_weights="Balanced",
        verbose=False
    )
    
    # 학습
    model.fit(tr_pool, eval_set=val_pool, use_best_model=True, early_stopping_rounds=100)
    
    return model, X_val, y_val

# SHAP 값 계산
def calculate_shap_values(model, X_sample, feature_names):
    # CatBoost 모델을 위한 SHAP 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 이진 분류의 경우 첫 번째 클래스(내향)에 대한 SHAP 값 사용
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    return shap_values

# 메인 앱
def main():
    st.set_page_config(page_title="내향/외향 예측", page_icon="🧑‍🤝‍🧑", layout="wide")
    
    st.title("🧑‍🤝‍🧑 내향/외향 성향 예측 & 분석")
    
    # 데이터 로드 및 모델 훈련
    with st.spinner("데이터 로딩 및 모델 훈련 중..."):
        X, y, feature_cols = load_and_prepare_data()
        model, X_val, y_val = train_catboost_model(X, y)
    
    # 탭 구성
    tab1, tab2 = st.tabs(["📊 성향 분포 시각화", "📝 설문 입력 & 분석"])
    
    with tab1:
        st.markdown("### 📊 성향 분포 시각화 (원형 구조)")
        
        # 샘플 데이터 생성 (실제 데이터에서 일부 추출)
        sample_size = min(100, len(X_val))
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        sample_X = X_val.iloc[sample_indices]
        sample_y = y_val.iloc[sample_indices]
        
        # 예측 확률 계산
        sample_probs = model.predict_proba(sample_X)
        
        # 클래스 순서에 따라 외향성 확률 결정
        if len(model.classes_) == 2:
            if 'Extrovert' in model.classes_ and 'Introvert' in model.classes_:
                extrovert_idx = list(model.classes_).index('Extrovert')
                extro_probs = sample_probs[:, extrovert_idx]
            else:
                # 기본값: 두 번째 클래스를 외향성으로 가정
                extro_probs = sample_probs[:, 1]
        else:
            # 단일 클래스인 경우
            extro_probs = sample_probs[:, 0]
        
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
        
        # GIF를 사용한 HTML 시각화 추가
        st.markdown("### 🐧 GIF 아이콘으로 성향 분포 시각화")
        
        # GIF 파일을 base64로 인코딩하여 사용
        import base64
        
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
        
        if hat_gif_base64 and dance_gif_base64:
            # HTML을 사용하여 GIF로 아이콘 표시
            html_content = """
            <style>
            .person-icon {
                position: absolute;
                width: 72px;
                height: 72px;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.3s ease;
                overflow: hidden;
                box-shadow: none;
                border: none;
                outline: none;
                /* 초기 위치는 중앙에 고정 */
                left: 50% !important;
                top: 50% !important;
                transform: translate(-50%, -50%);
            }
            .person-icon.animate {
                transition: left 2s cubic-bezier(0.25, 0.46, 0.45, 0.94), 
                            top 2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            }
            .person-icon:hover {
                transform: scale(1.3) translate(-50%, -50%);
                z-index: 1000;
                box-shadow: none;
            }
            .gif-container {
                position: relative;
                width: 100%;
                height: 600px;
                background: white;
                border-radius: 10px;
                overflow: hidden;
            }
            .center-circle {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 300px;
                height: 300px;
                border-radius: 50%;
                background-color: rgba(211, 211, 211, 0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                font-weight: bold;
                color: transparent;
            }
            .gif-image {
                width: 100%;
                height: 100%;
                object-fit: cover;
                border-radius: 50%;
                background: transparent;
            }
            .introvert .gif-image {
                filter: brightness(1.8) contrast(1.1);
            }
            .extrovert .gif-image {
                filter: brightness(1.0) contrast(1.0);
            }
            /* hat GIF 좌우 반전을 위한 스타일 */
            .hat-right .gif-image {
                transform: scaleX(-1);
            }
            .hat-left .gif-image {
                transform: scaleX(1);
            }
            </style>
            
            <div class="gif-container">
                <div class="center-circle">외향 zone</div>
            """
            
            # 각 사람을 HTML로 배치
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
                    person_class = "introvert"
                    gif_src = hat_gif_base64  # 내향형은 ht-unscreen.gif 사용
                else:
                    # 외향형: 원 안에 배치
                    min_distance = 0.05
                    max_distance = 0.3
                    distance = max_distance - (extro_prob * (max_distance - min_distance))
                    person_class = "extrovert"
                    gif_src = dance_gif_base64  # 외향형은 dance-unscreen.gif 사용
                
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
                
                # hat GIF 방향 결정 (각도에 따라 좌우 반전)
                if person_class == "introvert":
                    # 내향형: 각도에 따라 좌우 방향 결정
                    if -np.pi/2 <= angle <= np.pi/2:  # 오른쪽 반원
                        hat_class = "hat-right"  # 오른쪽으로 이동하는 hat
                    else:  # 왼쪽 반원
                        hat_class = "hat-left"   # 왼쪽으로 이동하는 hat
                    person_class_with_hat = f"{person_class} {hat_class}"
                else:
                    person_class_with_hat = person_class
                
                # HTML 요소 추가 - 성향에 따라 다른 GIF 사용, 초기 위치는 중앙
                html_content += f"""
                <div class="person-icon {person_class_with_hat}" 
                     data-target-x="{x_percent}" data-target-y="{y_percent}"
                     title="{tooltip_info}">
                    <img src="{gif_src}" alt="펭귄" class="gif-image">
                </div>
                """
            
            html_content += """
            </div>
            
            <script>
            (function() {
                // 페이지 로드 후 500ms 뒤에 애니메이션 시작
                setTimeout(function() {
                    var icons = document.querySelectorAll('.person-icon');
                    icons.forEach(function(icon) {
                        var targetX = icon.getAttribute('data-target-x');
                        var targetY = icon.getAttribute('data-target-y');
                        
                        if (targetX && targetY) {
                            // 애니메이션 클래스 추가
                            icon.classList.add('animate');
                            
                            // 강제 리플로우
                            icon.offsetHeight;
                            
                            // 목표 위치로 이동
                            icon.style.left = targetX + '%';
                            icon.style.top = targetY + '%';
                        }
                    });
                }, 500);
            })();
            </script>
            """
            
            # HTML 표시
            st.components.v1.html(html_content, height=600)
            
            st.caption("↕ 마우스를 펭귄 아이콘에 올리면 해당 사람의 상세 정보를 볼 수 있습니다.")
            st.info("💡 **GIF 아이콘 시각화**: 🐧 **내향형**은 hat-unscreen.gif, **외향형**은 dance-unscreen.gif를 사용합니다. 외향성 확률이 높을수록 원 중심에 가깝게, 낮을수록 원 테두리에 가깝게 배치됩니다.")
        
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
        st.markdown("### 📝 나의 성향 입력하기 (한 번에 하나씩)")

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
            /* fancy chip buttons */
            .stButton>button{border-radius:12px;padding:12px 14px;font-weight:700;border:0;background:linear-gradient(135deg,#7aa2ff,#9b7bff);color:#fff;box-shadow:0 6px 18px rgba(91,140,255,.35);transition:all .15s ease;}
            .stButton>button:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(91,140,255,.45)}
            .stButton>button:active{transform:translateY(0)}
            /* selected (disabled=true) appearance as ACTIVE chip */
            .stButton>button:disabled{opacity:1;background:linear-gradient(135deg,#5b8cff,#7f66ff);color:#fff;box-shadow:0 10px 28px rgba(127,102,255,.45);border:0}
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
                "time_alone": 5,
                "stage_fear": "No",
                "social_events": 3,
                "going_outside": 4,
                "drained_social": "No",
                "friends_circle": 8,
                "post_frequency": 3,
            }
        if "prob_history" not in st.session_state:
            st.session_state.prob_history = []

        steps = [
            "이름", "혼자 있는 시간", "무대 공포증", "사회적 이벤트 참여", "외출 빈도",
            "사교 후 지침함", "친구 그룹 크기", "게시물 작성 빈도", "요약 및 예측"
        ]
        total_steps = len(steps)
        current_step = st.session_state.wizard_step

        # 진행률/칩 표시
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

        # 라이브 스코어: 펭귄 트랙 시각화
        live_prob = estimate_extro_prob(st.session_state.answers)
        if len(st.session_state.prob_history) <= current_step:
            st.session_state.prob_history.append(live_prob)
        else:
            st.session_state.prob_history[current_step] = live_prob
        prev_prob = st.session_state.prob_history[current_step-1] if current_step > 0 and len(st.session_state.prob_history) > 1 else None
        delta = None if prev_prob is None else live_prob - prev_prob

        # GIF 로더
        import base64, os
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
        walker_b64 = right_b64 if (delta is None or delta >= 0) else left_b64

        # 시작/종료 위치 계산 (이전 확률에서 현재 확률로 이동)
        end_pos = int(live_prob * 100)
        end_pos = max(4, min(96, end_pos))
        if prev_prob is None:
            start_pos = end_pos
        else:
            start_pos = int(prev_prob * 100)
            start_pos = max(4, min(96, start_pos))

        # JS로 start -> end로 부드럽게 이동하도록 구성
        st.components.v1.html(
            f"""
            <style>
            .penguin-card{{margin:8px 0 14px;padding:12px 14px;border-radius:14px;background:rgba(243,244,246,.7);border:1px solid #e5e7eb}}
            .penguin-track{{position:relative;height:100px}}
            .rail{{position:absolute;left:6%;right:6%;top:48px;height:6px;background:#e5e7eb;border-radius:999px}}
            .endpoint{{position:absolute;top:8px;width:64px;height:64px;border-radius:50%;overflow:hidden}}
            .endpoint img{{width:100%;height:100%;object-fit:cover;border-radius:50%}}
            .endpoint.left{{left:0}}
            .endpoint.right{{right:0}}
            .walker{{position:absolute;top:20px;left:{start_pos}%;transform:translateX(-50%);width:56px;height:56px;border-radius:50%;overflow:hidden;transition:left 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)}}
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
            </style>
            <div class="penguin-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span>현재 추정 위치</span><span><b>{live_prob:.2f}</b>{("  (" + ("▶" if delta and delta>0 else "◀") + f" {abs(delta):.2f})" ) if delta is not None else ""}</span>
              </div>
              <div class="penguin-track">
                <div class="endpoint left"><img src="{hat_b64 or ''}"/></div>
                <div class="endpoint right"><img src="{dance_b64 or ''}"/></div>
                <div class="rail"></div>
                <div class="walker" id="walker"><img src="{walker_b64 or ''}"/></div>
              </div>
              <div class="labels"><span>0 (내향)</span><span>1 (외향)</span></div>
            </div>
            <script>
            (function(){{
              var walker = document.getElementById('walker');
              if(!walker) return;
              
              // 시작 위치 설정
              var startPos = {start_pos};
              var endPos = {end_pos};
              
              // 펭귄을 시작 위치에 배치
              walker.style.left = startPos + '%';
              walker.style.transition = 'none';
              
              // 강제 리플로우
              walker.offsetHeight;
              
              // 애니메이션 시작
              requestAnimationFrame(function(){{
                walker.style.transition = 'left 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
                walker.style.left = endPos + '%';
              }});
              
              // 애니메이션 완료 후 transition 제거 (다음 애니메이션을 위해)
              setTimeout(function(){{
                walker.style.transition = 'none';
              }}, 1200);
            }})();
            </script>
            """,
            height=160,
        )

        st.markdown('<div class="wizard-wrap">', unsafe_allow_html=True)

        # 옵션 버튼 그리드 렌더러 (한 번 클릭 시 자동 다음 단계)
        def render_option_buttons(options, field, columns=6, formatter=lambda x: str(x)):
            selected = st.session_state.answers.get(field)
            step_cols = columns
            for i in range(0, len(options), step_cols):
                row = options[i:i+step_cols]
                cols = st.columns(len(row))
                for j, opt in enumerate(row):
                    is_sel = (selected == opt)
                    label = ("✓ " if is_sel else "") + formatter(opt)
                    if cols[j].button(label, key=f"{field}_opt_{i+j}", use_container_width=True, disabled=is_sel):
                        st.session_state.answers[field] = opt
                        st.session_state.wizard_step = min(total_steps - 1, current_step + 1)
                        st.rerun()

        # 한 페이지 한 입력 UI (버튼 위주)
        if current_step == 0:
            st.subheader("🪪 이름")
            st.session_state.answers["name"] = st.text_input(
                "이름을 입력하세요", value=st.session_state.answers["name"], key="name_input"
            )
        elif current_step == 1:
            st.subheader("⏳ 혼자 있는 시간 (0=적음, 10=많음)")
            render_option_buttons(list(range(0, 11)), "time_alone", columns=6)
        elif current_step == 2:
            st.subheader("🎤 무대 공포증")
            render_option_buttons(["No", "Yes"], "stage_fear", columns=2)
        elif current_step == 3:
            st.subheader("🎟️ 사회적 이벤트 참여 (월)")
            render_option_buttons(list(range(0, 11)), "social_events", columns=6)
        elif current_step == 4:
            st.subheader("🚶 외출 빈도 (0=적음, 7=많음)")
            render_option_buttons(list(range(0, 8)), "going_outside", columns=8)
        elif current_step == 5:
            st.subheader("🧃 사교 후 지침함")
            render_option_buttons(["No", "Yes"], "drained_social", columns=2)
        elif current_step == 6:
            st.subheader("👥 친구 그룹 크기 (0=적음, 15=많음)")
            render_option_buttons(list(range(0, 16)), "friends_circle", columns=8)
        elif current_step == 7:
            st.subheader("✍️ 게시물 작성 빈도 (월)")
            render_option_buttons(list(range(0, 11)), "post_frequency", columns=6)
        else:
            # 요약 및 예측
            ans = st.session_state.answers
            st.subheader("요약")
            st.table(pd.DataFrame({
                "항목": ["이름", "혼자 있는 시간", "무대 공포증", "사회적 이벤트 참여", "외출 빈도", "사교 후 지침함", "친구 그룹 크기", "게시물 작성 빈도"],
                "값": [ans["name"], ans["time_alone"], ans["stage_fear"], ans["social_events"], ans["going_outside"], ans["drained_social"], ans["friends_circle"], ans["post_frequency"]]
            }))

            # 예측 수행
            if ans["name"]:
                stage_fear_num = 1 if ans["stage_fear"] == "Yes" else 0
                drained_social_num = 1 if ans["drained_social"] == "Yes" else 0
                user_input = np.array([[
                    ans["time_alone"], stage_fear_num, ans["social_events"], ans["going_outside"],
                    drained_social_num, ans["friends_circle"], ans["post_frequency"]
                ]])
                prediction = model.predict(user_input)[0]
                prediction_proba = model.predict_proba(user_input)[0]

                if prediction == "Introvert":
                    st.success(f"✅ {ans['name']}님의 예측 성향: 내향적 (Introvert)")
                    intro_prob = prediction_proba[0]; extro_prob = prediction_proba[1]
                else:
                    st.success(f"✅ {ans['name']}님의 예측 성향: 외향적 (Extrovert)")
                    intro_prob = prediction_proba[0]; extro_prob = prediction_proba[1]

                c1, c2 = st.columns(2)
                with c1: st.metric("내향성 확률", f"{intro_prob:.3f}")
                with c2: st.metric("외향성 확률", f"{extro_prob:.3f}")

                st.markdown("#### SHAP 분석")
                shap_values = calculate_shap_values(model, user_input, feature_cols)
                shap_df = pd.DataFrame({
                    '특성': feature_cols,
                    'SHAP_값': shap_values[0]
                }).sort_values('SHAP_값', key=abs, ascending=False)
                fig_shap = go.Figure()
                colors = ['red' if x < 0 else 'blue' for x in shap_df['SHAP_값']]
                fig_shap.add_trace(go.Bar(x=shap_df['SHAP_값'], y=shap_df['특성'], orientation='h', marker_color=colors))
                fig_shap.update_layout(height=360, margin=dict(l=10,r=10,b=10,t=10))
                st.plotly_chart(fig_shap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 네비게이션 버튼
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("◀ 이전", disabled=current_step == 0, use_container_width=True):
                st.session_state.wizard_step = max(0, current_step - 1)
                st.rerun()
        with col_next:
            can_go_next = True
            if current_step == 0 and not st.session_state.answers["name"]:
                can_go_next = False
            if st.button("다음 ▶" if current_step < total_steps - 1 else "완료", disabled=not can_go_next, use_container_width=True):
                st.session_state.wizard_step = min(total_steps - 1, current_step + 1)
                st.rerun()

if __name__ == "__main__":
    main()
