import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==================== I18N (EN/FA) ====================
# Basic bilingual support with a small translation dictionary.
# Fallback is English when a key is missing in Persian.
TRANSLATIONS = {
    'en': {
        'page.title': 'ML Education: Supervised Learning',
        'nav.title': '📚 Navigation',
        'nav.intro': 'Introduction',
        'nav.eda': 'Exploratory Data Analysis',
        'nav.preproc': 'Data Preprocessing',
        'nav.model': 'Modeling Sandbox',
        'nav.eval': 'Evaluation & Interpretation',
        'lang.label': 'Language',
        'lang.en': 'English',
        'lang.fa': 'Persian (فارسی)',
        'title.main': '🎓 Interactive Machine Learning Education Platform',
        'subtitle.main': 'Learn Supervised Learning Through Interactive Visualization',
        'intro.header': '📖 Introduction to Supervised Learning',
        'intro.supervised.h3': 'What is Supervised Learning?',
        'intro.supervised.p': 'Supervised Learning is a type of machine learning where we train a model using labeled data. We use known labels to teach a model to predict labels for new, unseen data.',
        'intro.classification.h3': 'What is Classification?',
        'intro.classification.p': 'Classification is a supervised learning task for predicting categorical outcomes (like Yes/No). Here, we predict whether a person will subscribe.',
        'intro.tf': '🎯 Target Variable vs. Features',
        'intro.target.title': 'Target Variable (Label)',
        'intro.target.desc': 'The Target Variable is what we want to predict (dependent variable). In our dataset: Subscribed (Yes/No).',
        'intro.features.title': 'Features (Independent Variables)',
        'intro.features.desc': 'Features are inputs used to make predictions (independent variables). In our dataset: Age, Income, Education, Gender.',
        'intro.preview': '📊 Sample Data Preview',
        'intro.dataset.overview': 'Dataset Overview:',
        'btn.apply_preproc': '🚀 Apply Preprocessing',
        'btn.train_model': '🎯 Train Model',
        'warn.need.preproc': '⚠️ Please complete Data Preprocessing first!',
        'info.how.preproc': "Navigate to 'Data Preprocessing' section and click 'Apply Preprocessing' button.",
        'eda.header': '🔍 Exploratory Data Analysis (EDA)',
        'eda.summary': '📈 Summary Statistics',
        'eda.numeric': 'Numeric Features',
        'eda.categorical': 'Categorical Features',
        'eda.missing.header': '🔴 Missing Values Analysis',
        'eda.missing.xlabel': 'Percentage Missing (%)',
        'eda.distributions': '📊 Data Distributions',
        'tab.numeric': 'Numeric Features',
        'tab.categorical': 'Categorical Features',
        'tab.target': 'Target Distribution',
        'chart.age': 'Age',
        'chart.income': 'Income',
        'chart.education': 'Education Level',
        'chart.gender': 'Gender',
        'chart.subscribed': 'Subscription Distribution',
        'metric.subscribed.yes': 'Subscribed (Yes)',
        'metric.subscribed.no': 'Not Subscribed (No)',
        'class.balance': 'Class Balance Ratio:',
        'pre.header': '🔧 Data Preprocessing Demo',
        'pre.missing': '1️⃣ Handling Missing Values',
        'pre.numeric': 'Numeric Features',
        'pre.numeric.radio': 'Imputation strategy for Age & Income:',
        'pre.numeric.mean': 'Mean Imputation',
        'pre.numeric.median': 'Median Imputation',
        'pre.categorical': 'Categorical Features',
        'pre.categorical.radio': 'Imputation strategy for Education:',
        'pre.categorical.mode': 'Mode (Most Frequent)',
        'pre.categorical.unknown': "Placeholder ('Unknown')",
        'pre.encoding': '2️⃣ Encoding Categorical Variables',
        'pre.encoding.radio': 'Select encoding method:',
        'pre.encoding.onehot': 'One-Hot Encoding',
        'pre.encoding.label': 'Label Encoding',
        'pre.scaling': '3️⃣ Feature Scaling',
        'pre.scaling.checkbox': 'Apply StandardScaler (Feature Scaling)',
        'pre.beforeafter': '📊 Before vs. After Preprocessing',
        'model.header': '🤖 Modeling Sandbox',
        'model.select': '🎯 Select Your Model',
        'model.radio': 'Choose a classification model:',
        'model.lr': 'Logistic Regression',
        'model.knn': 'K-Nearest Neighbors (KNN)',
        'model.knn.k': 'Select k (number of neighbors):',
        'model.coeff': 'Feature Coefficients',
        'model.decision.header': '🎨 Decision Boundary Visualization',
        'model.decision.info': 'Decision Boundary: The line/surface separating classes. For visualization, we use only the first two features.',
        'tab.train': 'Training Data',
        'tab.test': 'Test Data',
        'eval.header': '📊 Model Evaluation & Interpretation',
        'eval.split': '🔀 Train-Test Split',
        'metric.train.samples': 'Training Samples',
        'metric.train.pct': 'Training %',
        'metric.test.samples': 'Test Samples',
        'metric.test.pct': 'Test %',
        'eval.performance': '🎯 Model Performance',
        'metric.train.acc': 'Training Accuracy',
        'metric.test.acc': 'Test Accuracy',
        'metric.overfit.gap': 'Overfit Gap',
        'cm.header': '🔢 Confusion Matrix',
        'label.yes': 'Yes',
        'label.no': 'No',
        'label.age': 'Age',
        'label.income': 'Income',
        'legend.subscribed': 'Subscribed',
        'legend.no': 'No',
        'legend.yes': 'Yes',
    },
    'fa': {
        'page.title': 'آموزش یادگیری نظارت‌شده',
        'nav.title': '📚 ناوبری',
        'nav.intro': 'مقدمه',
        'nav.eda': 'تحلیل اکتشافی داده',
        'nav.preproc': 'پیش‌پردازش داده',
        'nav.model': 'محیط مدل‌سازی',
        'nav.eval': 'ارزیابی و تفسیر',
        'lang.label': 'زبان',
        'lang.en': 'انگلیسی',
        'lang.fa': 'فارسی',
        'title.main': '🎓 پلتفرم تعاملی آموزش یادگیری ماشین',
        'subtitle.main': 'یادگیری یادگیری نظارت‌شده با بصری‌سازی تعاملی',
        'intro.header': '📖 مقدمه‌ای بر یادگیری نظارت‌شده',
        'intro.supervised.h3': 'یادگیری نظارت‌شده چیست؟',
        'intro.supervised.p': 'در یادگیری نظارت‌شده، مدل با داده‌های برچسب‌دار آموزش می‌بیند تا برای داده‌های جدید برچسب پیش‌بینی کند.',
        'intro.classification.h3': 'رده‌بندی چیست؟',
        'intro.classification.p': 'رده‌بندی نوعی کار یادگیری نظارت‌شده برای پیش‌بینی خروجی‌های دسته‌ای (مثل بله/خیر) است. اینجا پیش‌بینی می‌کنیم آیا فرد مشترک می‌شود یا نه.',
        'intro.tf': '🎯 متغیر هدف در برابر ویژگی‌ها',
        'intro.target.title': 'متغیر هدف (برچسب)',
        'intro.target.desc': 'متغیر هدف چیزی است که می‌خواهیم پیش‌بینی کنیم. در این داده‌ها: اشتراک (بله/خیر).',
        'intro.features.title': 'ویژگی‌ها (متغیرهای مستقل)',
        'intro.features.desc': 'ویژگی‌ها ورودی‌هایی برای پیش‌بینی هستند: سن، درآمد، تحصیلات، جنسیت.',
        'intro.preview': '📊 پیش‌نمایش داده',
        'intro.dataset.overview': 'نمای کلی مجموعه‌داده:',
        'btn.apply_preproc': '🚀 اعمال پیش‌پردازش',
        'btn.train_model': '🎯 آموزش مدل',
        'warn.need.preproc': '⚠️ لطفاً ابتدا پیش‌پردازش داده را انجام دهید!',
        'info.how.preproc': "به بخش 'پیش‌پردازش داده' بروید و دکمه 'اعمال پیش‌پردازش' را بزنید.",
        'eda.header': '🔍 تحلیل اکتشافی داده (EDA)',
        'eda.summary': '📈 آمار خلاصه',
        'eda.numeric': 'ویژگی‌های عددی',
        'eda.categorical': 'ویژگی‌های رده‌ای',
        'eda.missing.header': '🔴 تحلیل مقادیر گمشده',
        'eda.missing.xlabel': 'درصد مقدار گمشده (%)',
        'eda.distributions': '📊 توزیع داده',
        'tab.numeric': 'ویژگی‌های عددی',
        'tab.categorical': 'ویژگی‌های رده‌ای',
        'tab.target': 'توزیع هدف',
        'chart.age': 'سن',
        'chart.income': 'درآمد',
        'chart.education': 'سطح تحصیلات',
        'chart.gender': 'جنسیت',
        'chart.subscribed': 'توزیع اشتراک',
        'metric.subscribed.yes': 'مشترک (بله)',
        'metric.subscribed.no': 'غیرمشترک (خیر)',
        'class.balance': 'نسبت توازن کلاس:',
        'pre.header': '🔧 نمایش پیش‌پردازش داده',
        'pre.missing': '1️⃣ مدیریت مقادیر گمشده',
        'pre.numeric': 'ویژگی‌های عددی',
        'pre.numeric.radio': 'روش جانشینی برای سن و درآمد:',
        'pre.numeric.mean': 'میانگین',
        'pre.numeric.median': 'میانه',
        'pre.categorical': 'ویژگی‌های رده‌ای',
        'pre.categorical.radio': 'روش جانشینی برای تحصیلات:',
        'pre.categorical.mode': 'مد (پراستفاده‌ترین)',
        'pre.categorical.unknown': "جایگزین ('نامشخص')",
        'pre.encoding': '2️⃣ کدگذاری متغیرهای رده‌ای',
        'pre.encoding.radio': 'روش کدگذاری را انتخاب کنید:',
        'pre.encoding.onehot': 'وان-هات (Dummy)',
        'pre.encoding.label': 'لیبل',
        'pre.scaling': '3️⃣ مقیاس‌بندی ویژگی‌ها',
        'pre.scaling.checkbox': 'اعمال استانداردسازی (StandardScaler)',
        'pre.beforeafter': '📊 قبل و بعد از پیش‌پردازش',
        'model.header': '🤖 محیط مدل‌سازی',
        'model.select': '🎯 انتخاب مدل',
        'model.radio': 'یک مدل رده‌بندی انتخاب کنید:',
        'model.lr': 'رگرسیون لجستیک',
        'model.knn': 'نزدیک‌ترین همسایه‌ها (KNN)',
        'model.knn.k': 'k (تعداد همسایه‌ها) را انتخاب کنید:',
        'model.coeff': 'ضرایب ویژگی',
        'model.decision.header': '🎨 بصری‌سازی مرز تصمیم',
        'model.decision.info': 'مرز تصمیم: حدّی که کلاس‌ها را جدا می‌کند. برای نمایش، فقط دو ویژگی اول را استفاده می‌کنیم.',
        'tab.train': 'دادهٔ آموزش',
        'tab.test': 'دادهٔ آزمون',
        'eval.header': '📊 ارزیابی و تفسیر مدل',
        'eval.split': '🔀 تفکیک آموزش/آزمون',
        'metric.train.samples': 'نمونه‌های آموزش',
        'metric.train.pct': '٪ آموزش',
        'metric.test.samples': 'نمونه‌های آزمون',
        'metric.test.pct': '٪ آزمون',
        'eval.performance': '🎯 کارایی مدل',
        'metric.train.acc': 'دقت آموزش',
        'metric.test.acc': 'دقت آزمون',
        'metric.overfit.gap': 'شکاف بیش‌برازش',
        'cm.header': '🔢 ماتریس درهم‌ریختگی',
        'label.yes': 'بله',
        'label.no': 'خیر',
        'label.age': 'سن',
        'label.income': 'درآمد',
        'legend.subscribed': 'اشتراک',
        'legend.no': 'خیر',
        'legend.yes': 'بله',
    }
}

def get_lang():
    return st.session_state.get('lang', 'en')

def tr(key: str) -> str:
    lang = get_lang()
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, TRANSLATIONS['en'].get(key, key))

# Set page configuration
st.set_page_config(
    page_title="ML Education: Supervised Learning",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    .term {
        font-weight: bold;
        color: #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA GENERATION ====================
@st.cache_data
def generate_dataset(n_samples=1000, missing_pct=0.07):
    """
    Generate synthetic demographic dataset with missing values
    """
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(40, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    
    income = np.random.normal(55000, 20000, n_samples)
    income = np.clip(income, 20000, 150000)
    
    education_levels = ['High School', 'Bachelor', 'Master']
    education = np.random.choice(education_levels, n_samples, p=[0.3, 0.5, 0.2])
    
    gender = np.random.choice(['Male', 'Female'], n_samples)
    
    # Create target variable with some logic
    # Higher income, age, and education increase subscription probability
    prob_subscribe = (
        0.3 * (age - age.min()) / (age.max() - age.min()) +
        0.4 * (income - income.min()) / (income.max() - income.min()) +
        0.3 * np.where(education == 'Master', 1, np.where(education == 'Bachelor', 0.5, 0))
    )
    prob_subscribe = np.clip(prob_subscribe, 0.1, 0.9)
    subscribed = np.random.binomial(1, prob_subscribe)
    subscribed = np.where(subscribed == 1, 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education': education,
        'Gender': gender,
        'Subscribed': subscribed
    })
    
    # Introduce missing values
    n_missing = int(n_samples * missing_pct)
    missing_indices_age = np.random.choice(n_samples, n_missing // 2, replace=False)
    missing_indices_income = np.random.choice(n_samples, n_missing // 3, replace=False)
    missing_indices_education = np.random.choice(n_samples, n_missing // 6, replace=False)
    
    df.loc[missing_indices_age, 'Age'] = np.nan
    df.loc[missing_indices_income, 'Income'] = np.nan
    df.loc[missing_indices_education, 'Education'] = np.nan
    
    return df

# ==================== HELPER FUNCTIONS ====================
def info_tooltip(term, definition):
    """Display educational tooltip"""
    with st.expander(f"ℹ️ What is {term}?"):
        st.info(definition)

def plot_decision_boundary(X, y, model, feature_names, resolution=100):
    """
    Plot decision boundary for 2D data
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.where(Z == 'Yes', 1, 0)
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot points
    y_numeric = np.where(y == 'Yes', 1, 0)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_numeric, cmap='RdYlBu', 
                        edgecolors='k', s=50, alpha=0.7)
    
    # Localize axis labels when common names are used
    name_map = {
        'Age': tr('label.age'),
        'Income': tr('label.income'),
    }
    x_label = name_map.get(feature_names[0], feature_names[0])
    y_label = name_map.get(feature_names[1], feature_names[1])
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title('Decision Boundary Visualization', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='#67a9cf', markersize=10, label=tr('legend.no')),
                      plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='#ef8a62', markersize=10, label=tr('legend.yes'))]
    ax.legend(handles=legend_elements, title=tr('legend.subscribed'))
    
    plt.tight_layout()
    return fig

# ==================== MAIN APP ====================
def main():
    st.title(tr('title.main'))
    st.markdown(f"### {tr('subtitle.main')}")
    
    # Sidebar language selector + navigation
    st.sidebar.title(tr('nav.title'))
    lang_display = st.sidebar.selectbox(tr('lang.label'), [TRANSLATIONS['en']['lang.en'], TRANSLATIONS['fa']['lang.fa']])
    st.session_state.lang = 'fa' if 'فارسی' in lang_display else 'en'
    if get_lang() == 'fa':
        st.markdown("""
        <style>
        html, body, [class*='css'] { direction: rtl; }
        .info-box { text-align: right; }
        </style>
        """, unsafe_allow_html=True)

    nav_items = [
        ('intro', tr('nav.intro')),
        ('eda', tr('nav.eda')),
        ('preproc', tr('nav.preproc')),
        ('model', tr('nav.model')),
        ('eval', tr('nav.eval')),
    ]
    nav_labels = [lbl for _, lbl in nav_items]
    selected_label = st.sidebar.radio('', nav_labels)
    section_key = next(k for k, v in nav_items if v == selected_label)
    
    # Generate dataset
    df = generate_dataset()
    
    # Initialize session state
    if 'preprocessed_df' not in st.session_state:
        st.session_state.preprocessed_df = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    
    # ==================== SECTION 1: INTRODUCTION ====================
    if section_key == "intro":
        st.header(tr('intro.header'))
        
        st.markdown("""
        <div class="info-box">
        <h3>{h3}</h3>
        <p>{p}</p>
        </div>
        """.format(h3=tr('intro.supervised.h3'), p=tr('intro.supervised.p')), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>{h3}</h3>
        <p>{p}</p>
        </div>
        """.format(h3=tr('intro.classification.h3'), p=tr('intro.classification.p')), unsafe_allow_html=True)
        
        st.subheader(tr('intro.tf'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **{title}**
            
            {desc}
            """.format(title=tr('intro.target.title'), desc=tr('intro.target.desc')), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            **{title}**
            
            {desc}
            """.format(title=tr('intro.features.title'), desc=tr('intro.features.desc')), unsafe_allow_html=True)
        
        st.subheader(tr('intro.preview'))
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"""
        **{tr('intro.dataset.overview')}**
        - Total samples: {len(df)}
        - Features: {', '.join(df.columns[:-1])}
        - Target: {df.columns[-1]}
        - Target classes: {', '.join(df['Subscribed'].unique())}
        """)
    
    # ==================== SECTION 2: EDA ====================
    elif section_key == "eda":
        st.header(tr('eda.header'))
        
        info_tooltip("Exploratory Data Analysis (EDA)", 
                    "EDA is the process of analyzing and visualizing data to understand its main " +
                    "characteristics, patterns, and potential issues before building models.")
        
        st.subheader(tr('eda.summary'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{tr('eda.numeric')}**")
            st.dataframe(df[['Age', 'Income']].describe())
        
        with col2:
            st.markdown(f"**{tr('eda.categorical')}**")
            cat_summary = pd.DataFrame({
                'Education': df['Education'].value_counts(),
                'Gender': df['Gender'].value_counts(),
                'Subscribed': df['Subscribed'].value_counts()
            })
            st.dataframe(cat_summary)
        
        st.subheader(tr('eda.missing.header'))
        
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage (%)': missing_pct
        })
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(missing_df)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            missing_df[missing_df['Missing Count'] > 0]['Percentage (%)'].plot(
                kind='barh', ax=ax, color='coral'
            )
            ax.set_xlabel(tr('eda.missing.xlabel'))
            ax.set_title(tr('eda.missing.header'))
            st.pyplot(fig)
        
        st.info("""
        **Why Missing Values Matter:** Missing data can bias our analysis and models. 
        We need to handle them appropriately before training our model.
        """)
        
        st.subheader(tr('eda.distributions'))
        
        tab1, tab2, tab3 = st.tabs([tr('tab.numeric'), tr('tab.categorical'), tr('tab.target')])
        
        with tab1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].hist(df['Age'].dropna(), bins=30, color='skyblue', edgecolor='black')
            axes[0].set_xlabel(tr('chart.age'))
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(tr('chart.age'))
            
            axes[1].hist(df['Income'].dropna(), bins=30, color='lightgreen', edgecolor='black')
            axes[1].set_xlabel(tr('chart.income'))
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(tr('chart.income'))
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            df['Education'].value_counts().plot(kind='bar', ax=axes[0], color='orange')
            axes[0].set_xlabel(tr('chart.education'))
            axes[0].set_ylabel('Count')
            axes[0].set_title(tr('chart.education'))
            axes[0].tick_params(axis='x', rotation=45)
            
            df['Gender'].value_counts().plot(kind='bar', ax=axes[1], color='purple')
            axes[1].set_xlabel(tr('chart.gender'))
            axes[1].set_ylabel('Count')
            axes[1].set_title(tr('chart.gender'))
            axes[1].tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 6))
                df['Subscribed'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%',
                                                     colors=['#ff9999', '#66b3ff'])
                ax.set_ylabel('')
                ax.set_title(tr('chart.subscribed'))
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Target Variable Balance")
                target_counts = df['Subscribed'].value_counts()
                st.metric(tr('metric.subscribed.yes'), target_counts.get('Yes', 0))
                st.metric(tr('metric.subscribed.no'), target_counts.get('No', 0))
                
                balance_ratio = min(target_counts.values) / max(target_counts.values)
                st.info(f"""
                **{tr('class.balance')}** {balance_ratio:.2f}
                
                A balanced dataset (ratio close to 1.0) helps models learn both classes equally well.
                """)
    
    # ==================== SECTION 3: PREPROCESSING ====================
    elif section_key == "preproc":
        st.header(tr('pre.header'))
        
        info_tooltip("Data Preprocessing",
                    "Data preprocessing involves transforming raw data into a clean, formatted dataset " +
                    "that machine learning algorithms can understand and process effectively.")
        
        st.subheader(tr('pre.missing'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{tr('pre.numeric')}**")
            numeric_options = {
                'Mean Imputation': tr('pre.numeric.mean'),
                'Median Imputation': tr('pre.numeric.median'),
            }
            numeric_choice_label = st.radio(
                tr('pre.numeric.radio'),
                list(numeric_options.values()),
                help="Mean: Replace missing values with the average. Median: Replace with the middle value."
            )
            numeric_strategy = next(k for k, v in numeric_options.items() if v == numeric_choice_label)
            
            info_tooltip("Imputation",
                        "Imputation is the process of replacing missing values with substituted values. " +
                        "Mean imputation uses the average, while median is more robust to outliers.")
        
        with col2:
            st.markdown(f"**{tr('pre.categorical')}**")
            categorical_options = {
                "Mode (Most Frequent)": tr('pre.categorical.mode'),
                "Placeholder ('Unknown')": tr('pre.categorical.unknown'),
            }
            categorical_choice_label = st.radio(
                tr('pre.categorical.radio'),
                list(categorical_options.values()),
                help="Mode: Replace with the most common value. Placeholder: Use a special 'Unknown' category."
            )
            categorical_strategy = next(k for k, v in categorical_options.items() if v == categorical_choice_label)
        
        st.subheader(tr('pre.encoding'))
        
        encoding_options = {
            'One-Hot Encoding': tr('pre.encoding.onehot'),
            'Label Encoding': tr('pre.encoding.label'),
        }
        encoding_choice_label = st.radio(
            tr('pre.encoding.radio'),
            list(encoding_options.values()),
            help="Choose how to convert categorical variables to numbers."
        )
        encoding_method = next(k for k, v in encoding_options.items() if v == encoding_choice_label)
        
        col1, col2 = st.columns(2)
        with col1:
            info_tooltip("One-Hot Encoding",
                        "Creates separate binary (0/1) columns for each category. Example: " +
                        "Education → Education_HighSchool, Education_Bachelor, Education_Master")
        
        with col2:
            info_tooltip("Label Encoding",
                        "Assigns a unique integer to each category. Example: " +
                        "High School → 0, Bachelor → 1, Master → 2")
        
        st.subheader(tr('pre.scaling'))
        
        apply_scaling = st.checkbox(
            tr('pre.scaling.checkbox'),
            value=True,
            help="Standardize features by removing the mean and scaling to unit variance"
        )
        
        info_tooltip("StandardScaler / Feature Scaling",
                    "Feature scaling transforms numeric features to have mean=0 and standard deviation=1. " +
                    "This ensures all features contribute equally to the model, especially important for " +
                    "distance-based algorithms like KNN.")
        
        if st.button(tr('btn.apply_preproc'), type="primary"):
            with st.spinner("Processing data..."):
                # Create a copy
                df_processed = df.copy()
                
                # Handle missing values - Numeric
                if numeric_strategy == "Mean Imputation":
                    df_processed['Age'].fillna(df_processed['Age'].mean(), inplace=True)
                    df_processed['Income'].fillna(df_processed['Income'].mean(), inplace=True)
                else:
                    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
                    df_processed['Income'].fillna(df_processed['Income'].median(), inplace=True)
                
                # Handle missing values - Categorical
                if categorical_strategy == "Mode (Most Frequent)":
                    df_processed['Education'].fillna(df_processed['Education'].mode()[0], inplace=True)
                else:
                    df_processed['Education'].fillna('Unknown', inplace=True)
                
                # Encoding
                if encoding_method == "One-Hot Encoding":
                    df_processed = pd.get_dummies(df_processed, columns=['Education', 'Gender'], 
                                                  drop_first=True)
                else:
                    le_education = LabelEncoder()
                    le_gender = LabelEncoder()
                    df_processed['Education'] = le_education.fit_transform(df_processed['Education'])
                    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
                
                # Separate features and target
                X = df_processed.drop('Subscribed', axis=1)
                y = df_processed['Subscribed']
                
                feature_names = X.columns.tolist()
                
                # Scaling
                if apply_scaling:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    X = pd.DataFrame(X_scaled, columns=feature_names)
                
                # Store in session state
                st.session_state.preprocessed_df = df_processed
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = feature_names
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("✅ Preprocessing completed successfully!")
        
        if st.session_state.preprocessed_df is not None:
            st.subheader(tr('pre.beforeafter'))
            
            tab1, tab2 = st.tabs(["Data Comparison", "Visualization"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data (First 5 rows)**")
                    st.dataframe(df.head())
                    st.info(f"Missing values: {df.isnull().sum().sum()}")
                
                with col2:
                    st.markdown("**Preprocessed Data (First 5 rows)**")
                    display_df = st.session_state.X.head().copy()
                    display_df['Subscribed'] = st.session_state.y.head().values
                    st.dataframe(display_df)
                    st.info(f"Missing values: {st.session_state.X.isnull().sum().sum()}")
            
            with tab2:
                if apply_scaling and len(st.session_state.X.columns) >= 2:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Original data
                    axes[0].scatter(df['Age'], df['Income'], 
                                   c=np.where(df['Subscribed']=='Yes', 1, 0),
                                   cmap='RdYlBu', alpha=0.6, edgecolors='k')
                    axes[0].set_xlabel('Age')
                    axes[0].set_ylabel('Income')
                    axes[0].set_title('Before Scaling')
                    
                    # Scaled data
                    if encoding_method == "Label Encoding":
                        age_col = 'Age'
                        income_col = 'Income'
                    else:
                        age_col = st.session_state.X.columns[0]
                        income_col = st.session_state.X.columns[1]
                    
                    axes[1].scatter(st.session_state.X[age_col], 
                                   st.session_state.X[income_col],
                                   c=np.where(st.session_state.y=='Yes', 1, 0),
                                   cmap='RdYlBu', alpha=0.6, edgecolors='k')
                    axes[1].set_xlabel(age_col)
                    axes[1].set_ylabel(income_col)
                    axes[1].set_title('After Scaling')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("""
                    **Notice:** After scaling, both features have similar ranges (centered around 0), 
                    making them comparable in magnitude. This helps models treat all features equally.
                    """)
    
    # ==================== SECTION 4: MODELING ====================
    elif section_key == "model":
        st.header(tr('model.header'))
        
        if st.session_state.X_train is None:
            st.warning(tr('warn.need.preproc'))
            st.info(tr('info.how.preproc'))
            return
        
        info_tooltip("Machine Learning Model",
                    "A machine learning model is an algorithm that learns patterns from data to make predictions. " +
                    "Different models use different strategies to find these patterns.")
        
        st.subheader(tr('model.select'))
        
        model_options = {'lr': tr('model.lr'), 'knn': tr('model.knn')}
        model_label = st.radio(tr('model.radio'), list(model_options.values()))
        model_type = next(k for k, v in model_options.items() if v == model_label)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if model_type == "lr":
                st.markdown("""
                **Logistic Regression**
                
                A <span class="term">linear model</span> that draws a straight line (or hyperplane) 
                to separate classes. It works well when classes are roughly linearly separable.
                
                **Pros:**
                - Simple and interpretable
                - Fast training
                - Works well with linearly separable data
                
                **Cons:**
                - Assumes linear decision boundary
                - May underfit complex patterns
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                **K-Nearest Neighbors (KNN)**
                
                A <span class="term">non-parametric model</span> that classifies based on the 
                majority class of the k nearest neighbors.
                
                **Pros:**
                - No assumptions about data distribution
                - Can capture complex patterns
                - Intuitive concept
                
                **Cons:**
                - Slower prediction
                - Sensitive to feature scaling
                - Affected by irrelevant features
                """, unsafe_allow_html=True)
        
        with col2:
            k_value = 5  # Default value
            if model_type == "knn":
                k_value = st.slider(
                    tr('model.knn.k'),
                    min_value=1,
                    max_value=50,
                    value=5,
                    help="Number of nearest neighbors to consider for classification"
                )
                
                info_tooltip("Hyperparameter (k)",
                            "A hyperparameter is a setting we choose before training. For KNN, 'k' determines " +
                            "how many neighbors vote on the classification. Small k → more complex (may overfit), " +
                            "Large k → smoother decision boundary (may underfit).")
        
        if st.button(tr('btn.train_model'), type="primary"):
            with st.spinner("Training model..."):
                # Train model
                if model_type == "lr":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = KNeighborsClassifier(n_neighbors=k_value)
                
                model.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.model = model
                st.session_state.model_type = model_type
                
                # Make predictions
                y_pred_train = model.predict(st.session_state.X_train)
                y_pred_test = model.predict(st.session_state.X_test)
                
                st.session_state.y_pred_train = y_pred_train
                st.session_state.y_pred_test = y_pred_test
                
                train_acc = accuracy_score(st.session_state.y_train, y_pred_train)
                test_acc = accuracy_score(st.session_state.y_test, y_pred_test)
                
                st.success(f"✅ Model trained! Train Accuracy: {train_acc:.2%} | Test Accuracy: {test_acc:.2%}")
        
        if hasattr(st.session_state, 'model'):
            st.subheader("📊 Model Insights")
            
            if st.session_state.model_type == "lr":
                st.markdown("**Feature Coefficients**")
                
                coefficients = st.session_state.model.coef_[0]
                feature_importance = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Coefficient': coefficients
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(feature_importance)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
                    ax.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
                    ax.set_xlabel('Coefficient Value')
                    ax.set_title('Feature Importance (Logistic Regression)')
                    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    st.pyplot(fig)
                
                st.info("""
                **Interpretation:** Positive coefficients increase the probability of 'Yes' subscription, 
                while negative coefficients decrease it. Larger magnitude = stronger influence.
                """)
            
            st.subheader(tr('model.decision.header'))
            
            st.info(tr('model.decision.info'))
            
            # Use first two features for visualization
            X_train_2d = st.session_state.X_train.iloc[:, :2].values
            X_test_2d = st.session_state.X_test.iloc[:, :2].values
            
            # Train a 2D model for visualization
            if st.session_state.model_type == "lr":
                model_2d = LogisticRegression(random_state=42, max_iter=1000)
            else:
                model_2d = KNeighborsClassifier(n_neighbors=k_value)
            
            model_2d.fit(X_train_2d, st.session_state.y_train)
            
            tab1, tab2 = st.tabs([tr('tab.train'), tr('tab.test')])
            
            with tab1:
                fig = plot_decision_boundary(
                    X_train_2d, 
                    st.session_state.y_train,
                    model_2d,
                    st.session_state.feature_names[:2]
                )
                st.pyplot(fig)
            
            with tab2:
                fig = plot_decision_boundary(
                    X_test_2d,
                    st.session_state.y_test,
                    model_2d,
                    st.session_state.feature_names[:2]
                )
                st.pyplot(fig)
            
            st.markdown("""
            **What you're seeing:**
            - **Colored regions:** The model's prediction for that area of the feature space
            - **Data points:** Actual samples, colored by their true class
            - **Boundary:** Where the model switches its prediction from one class to another
            """)
    
    # ==================== SECTION 5: EVALUATION ====================
    elif section_key == "eval":
        st.header(tr('eval.header'))
        
        if not hasattr(st.session_state, 'model'):
            st.warning("⚠️ Please train a model first!")
            st.info("Navigate to 'Modeling Sandbox' section and train a model.")
            return
        
        st.subheader(tr('eval.split'))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(tr('metric.train.samples'), len(st.session_state.X_train))
            st.metric(tr('metric.train.pct'), "80%")
        
        with col2:
            st.metric(tr('metric.test.samples'), len(st.session_state.X_test))
            st.metric(tr('metric.test.pct'), "20%")
        
        info_tooltip("Train-Test Split",
                    "We split data into training and test sets. The model learns from the training set " +
                    "and is evaluated on the test set (data it hasn't seen). This helps us assess how well " +
                    "the model generalizes to new data.")
        
        st.subheader(tr('eval.performance'))
        
        train_acc = accuracy_score(st.session_state.y_train, st.session_state.y_pred_train)
        test_acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred_test)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(tr('metric.train.acc'), f"{train_acc:.2%}")
        
        with col2:
            st.metric(tr('metric.test.acc'), f"{test_acc:.2%}")
        
        with col3:
            overfitting_gap = train_acc - test_acc
            st.metric(tr('metric.overfit.gap'), f"{overfitting_gap:.2%}",
                     delta=f"{'High' if overfitting_gap > 0.1 else 'Low'} overfitting",
                     delta_color="inverse")
        
        info_tooltip("Accuracy",
                    "Accuracy is the percentage of correct predictions. It's calculated as: " +
                    "(Correct Predictions) / (Total Predictions). While intuitive, accuracy can be " +
                    "misleading with imbalanced datasets.")
        
        st.subheader(tr('cm.header'))
        
        tab1, tab2 = st.tabs(["Test Set", "Training Set"])
        
        with tab1:
            cm_test = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_test,
                                      labels=['No', 'Yes'])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=[tr('label.no'), tr('label.yes')],
                           yticklabels=[tr('label.no'), tr('label.yes')],
                           ax=ax, cbar_kws={'label': 'Count'})
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix (Test Set)')
                st.pyplot(fig)
            
            with col2:
                st.markdown("""
                **How to read the Confusion Matrix:**
                
                - **Top-Left (True Negatives):** Correctly predicted "No"
                - **Top-Right (False Positives):** Incorrectly predicted "Yes" (Type I Error)
                - **Bottom-Left (False Negatives):** Incorrectly predicted "No" (Type II Error)
                - **Bottom-Right (True Positives):** Correctly predicted "Yes"
                
                **Ideal:** High values on the diagonal (correct predictions), 
                low values off-diagonal (errors).
                """)
                
                # Calculate metrics
                tn, fp, fn, tp = cm_test.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                st.metric("Precision (Yes class)", f"{precision:.2%}")
                st.metric("Recall (Yes class)", f"{recall:.2%}")
        
        with tab2:
            cm_train = confusion_matrix(st.session_state.y_train, st.session_state.y_pred_train,
                                       labels=['No', 'Yes'])
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                       xticklabels=[tr('label.no'), tr('label.yes')],
                       yticklabels=[tr('label.no'), tr('label.yes')],
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title('Confusion Matrix (Training Set)')
            st.pyplot(fig)
        
        info_tooltip("Confusion Matrix",
                    "A confusion matrix shows the counts of correct and incorrect predictions for each class. " +
                    "It provides more insight than accuracy alone, especially for imbalanced datasets.")
        
        st.subheader("🧠 Key Concepts in Model Evaluation")
        
        tab1, tab2, tab3 = st.tabs(["Generalization", "Sample Size & Variance", "Overfitting"])
        
        with tab1:
            st.markdown("""
            <div class="info-box">
            <h4>What is Generalization?</h4>
            <p><span class="term">Generalization</span> refers to a model's ability to perform well on new, 
            unseen data—not just the data it was trained on.</p>
            
            <ul>
            <li><strong>Good Generalization:</strong> Test accuracy close to training accuracy</li>
            <li><strong>Poor Generalization:</strong> High training accuracy but low test accuracy (overfitting)</li>
            </ul>
            
            <p><strong>Your Model:</strong></p>
            <ul>
            <li>Training Accuracy: {:.2%}</li>
            <li>Test Accuracy: {:.2%}</li>
            <li>Gap: {:.2%} {}</li>
            </ul>
            </div>
            """.format(
                train_acc, test_acc, abs(train_acc - test_acc),
                "✅ Good generalization!" if abs(train_acc - test_acc) < 0.05 
                else "⚠️ May be overfitting" if train_acc > test_acc + 0.1
                else "⚠️ May be underfitting"
            ), unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="info-box">
            <h4>Sample Size & Variance</h4>
            
            <p><span class="term">Sample Size</span> is the number of data points we use for training. 
            <span class="term">Variance</span> refers to how much our model's predictions vary with different 
            training data.</p>
            
            <ul>
            <li><strong>Small Sample Size:</strong> High variance, unstable models, poor generalization</li>
            <li><strong>Large Sample Size:</strong> Lower variance, more stable models, better generalization</li>
            </ul>
            
            <p><strong>Your Dataset:</strong></p>
            <ul>
            <li>Total samples: 1,000</li>
            <li>Training samples: {}</li>
            <li>Test samples: {}</li>
            <li>Assessment: {} samples is generally good for basic models</li>
            </ul>
            </div>
            """.format(
                len(st.session_state.X_train),
                len(st.session_state.X_test),
                len(st.session_state.X_train)
            ), unsafe_allow_html=True)
            
            # Simulate learning curve
            st.markdown("**Learning Curve Concept:**")
            
            sample_sizes = np.linspace(50, len(st.session_state.X_train), 10, dtype=int)
            train_scores = []
            test_scores = []
            
            for size in sample_sizes:
                X_subset = st.session_state.X_train.iloc[:size]
                y_subset = st.session_state.y_train.iloc[:size]
                
                temp_model = st.session_state.model.__class__(**st.session_state.model.get_params())
                temp_model.fit(X_subset, y_subset)
                
                train_scores.append(accuracy_score(y_subset, temp_model.predict(X_subset)))
                test_scores.append(accuracy_score(st.session_state.y_test, 
                                                  temp_model.predict(st.session_state.X_test)))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(sample_sizes, train_scores, 'o-', label='Training Accuracy', linewidth=2)
            ax.plot(sample_sizes, test_scores, 's-', label='Test Accuracy', linewidth=2)
            ax.set_xlabel('Training Sample Size')
            ax.set_ylabel('Accuracy')
            ax.set_title('Learning Curve: How Sample Size Affects Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.info("""
            **Observation:** As training size increases, test accuracy generally improves and 
            stabilizes, showing the importance of having enough data.
            """)
        
        with tab3:
            st.markdown("""
            <div class="info-box">
            <h4>Understanding Overfitting</h4>
            
            <p><span class="term">Overfitting</span> occurs when a model learns the training data too well, 
            including its noise and peculiarities, leading to poor performance on new data.</p>
            
            <p><strong>Signs of Overfitting:</strong></p>
            <ul>
            <li>High training accuracy but low test accuracy</li>
            <li>Model is too complex relative to the amount of data</li>
            <li>Large gap between training and test performance</li>
            </ul>
            
            <p><strong>How to Prevent Overfitting:</strong></p>
            <ul>
            <li>Use more training data</li>
            <li>Use simpler models or reduce model complexity</li>
            <li>Apply regularization techniques</li>
            <li>Use cross-validation</li>
            <li>For KNN: Increase k (more neighbors = smoother decision boundary)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            **Your Model's Overfitting Assessment:**
            
            - Training Accuracy: {train_acc:.2%}
            - Test Accuracy: {test_acc:.2%}
            - Gap: {abs(train_acc - test_acc):.2%}
            
            **Status:** {
                "✅ No significant overfitting detected" if abs(train_acc - test_acc) < 0.05
                else "⚠️ Moderate overfitting - consider simplifying the model" if abs(train_acc - test_acc) < 0.15
                else "🔴 High overfitting - model needs adjustment"
            }
            """)
            
            if st.session_state.model_type == "K-Nearest Neighbors (KNN)":
                st.info("""
                **For KNN:** Try increasing the number of neighbors (k) to reduce overfitting. 
                A larger k creates a smoother, less complex decision boundary.
                """)

# Run the app
if __name__ == "__main__":
    main()
