import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import random

# ======================
# DATA GENERATION (Simulated)
# ======================
def generate_team_data(team_size=8, days=30):
    """Generate realistic simulated team data"""
    roles = ['Engineer', 'Designer', 'PM', 'QA']
    members = [f"Member_{i}" for i in range(1, team_size+1)]
    
    data = []
    start_date = datetime.now() - timedelta(days=days)
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        weekday = current_date.weekday()  # Mon=0, Sun=6
        
        for member in members:
            # Simulate realistic patterns (weekdays > weekends, etc.)
            is_weekend = weekday >= 5
            base_tasks = 3 if is_weekend else random.randint(4, 8)
            completed = random.randint(0, base_tasks) if not is_weekend else random.randint(0, 2)
            
            # Simulate focus time (higher on weekdays)
            focus_hours = random.uniform(1.5, 4.0) if is_weekend else random.uniform(2.5, 6.5)
            
            # Simulate meetings (more mid-week)
            meetings = 0 if is_weekend else random.randint(1, 4 if weekday in [1,2,3] else 2)
            
            # Simulate collaboration touches
            collab_events = random.randint(0, 3) if is_weekend else random.randint(3, 12)
            
            # Random role assignment
            role = random.choice(roles)
            
            data.append({
                'date': current_date.date(),
                'member': member,
                'role': role,
                'tasks_assigned': base_tasks,
                'tasks_completed': completed,
                'focus_hours': round(focus_hours, 1),
                'meetings': meetings,
                'collab_events': collab_events,
                'context_switches': random.randint(5, 25),
                'after_hours_work': random.uniform(0, 1.5) if random.random() < 0.3 else 0
            })
    
    return pd.DataFrame(data)

# ======================
# ANALYTICS ENGINE
# ======================
class ProductivityAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.df['completion_rate'] = (self.df['tasks_completed'] / self.df['tasks_assigned'].replace(0, 1)) * 100
        self.df['efficiency_score'] = (self.df['tasks_completed'] / (self.df['focus_hours'] + 0.1)) * 10  # tasks per focus hour
    
    def team_health_score(self):
        """Composite health score (0-100)"""
        avg_completion = self.df['completion_rate'].mean()
        avg_focus = self.df['focus_hours'].mean()
        meeting_load = self.df['meetings'].mean()
        after_hours = self.df['after_hours_work'].mean()
        
        # Weighted formula emphasizing outcomes over activity
        score = (
            avg_completion * 0.4 +          # 40% completion rate
            min(avg_focus * 5, 30) +        # 30% focus time (capped)
            max(0, 25 - meeting_load * 5) + # 25% low meeting load
            max(0, 5 - after_hours * 3)     # 5% work-life balance
        )
        return min(100, max(0, score))
    
    def detect_bottlenecks(self):
        """Identify workflow bottlenecks using anomaly detection"""
        features = ['tasks_assigned', 'tasks_completed', 'focus_hours', 'context_switches']
        X = self.df[features].values
        
        # Isolation Forest for anomaly detection
        clf = IsolationForest(contamination=0.1, random_state=42)
        preds = clf.fit_predict(X)
        
        self.df['is_bottleneck'] = preds == -1
        
        # Bottleneck summary
        bottleneck_days = self.df[self.df['is_bottleneck']].groupby('date').size()
        frequent_bottlenecks = bottleneck_days[bottleneck_days > 2].index.tolist()
        
        return {
            'bottleneck_days': frequent_bottlenecks,
            'overloaded_members': self.df[self.df['is_bottleneck']]['member'].value_counts().head(3).index.tolist(),
            'common_pattern': 'High context switching + low completion' if self.df[self.df['is_bottleneck']]['context_switches'].mean() > 15 else 'Task overload'
        }
    
    def burnout_risk(self):
        """Simple burnout risk model"""
        self.df['burnout_risk_score'] = (
            self.df['after_hours_work'] * 30 +
            (self.df['meetings'] > 4) * 20 +
            (self.df['context_switches'] > 20) * 25 +
            (self.df['completion_rate'] < 50) * 25
        )
        high_risk = self.df[self.df['burnout_risk_score'] > 60]
        return {
            'high_risk_members': high_risk['member'].unique().tolist()[:3],
            'avg_risk_score': self.df['burnout_risk_score'].mean(),
            'trend': 'increasing' if self.df.groupby('date')['burnout_risk_score'].mean().iloc[-7:].mean() > 
                                  self.df.groupby('date')['burnout_risk_score'].mean().iloc[:-7].mean() else 'stable'
        }
    
    def effort_vs_impact(self):
        """Classify work as high/low impact based on completion quality vs time spent"""
        self.df['impact_category'] = pd.cut(
            self.df['efficiency_score'],
            bins=[0, 2, 4, 100],
            labels=['Low Impact', 'Medium Impact', 'High Impact']
        )
        return self.df.groupby('impact_category').size().to_dict()

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Team Productivity Analyzer", layout="wide")

st.title("📊 Team Productivity Analyzer")
st.markdown("*Data-driven insights without surveillance — simulated demo*")

# Sidebar controls
st.sidebar.header("Simulation Controls")
team_size = st.sidebar.slider("Team Size", 5, 20, 10)
days = st.sidebar.slider("Historical Data (days)", 14, 60, 30)
if st.sidebar.button("🔄 Regenerate Data"):
    st.cache_data.clear()
    st.rerun()

@st.cache_data
def get_data(team_size, days):
    return generate_team_data(team_size, days)

df = get_data(team_size, days)
analyzer = ProductivityAnalyzer(df)

# ======================
# DASHBOARD
# ======================
col1, col2, col3, col4 = st.columns(4)

with col1:
    health = analyzer.team_health_score()
    st.metric("Team Health Score", f"{health:.0f}/100", 
              delta=f"{health - 72:.0f}" if health > 72 else f"{health - 72:.0f}",
              delta_color="normal")

with col2:
    bottleneck = analyzer.detect_bottlenecks()
    st.metric("Bottleneck Risk", "⚠️ Detected" if bottleneck['bottleneck_days'] else "✅ Stable")

with col3:
    burnout = analyzer.burnout_risk()
    st.metric("Burnout Risk", f"{burnout['avg_risk_score']:.0f}/100", 
              delta=burnout['trend'].upper(), delta_color="inverse")

with col4:
    impact = analyzer.effort_vs_impact()
    high_impact_pct = impact.get('High Impact', 0) / df.shape[0] * 100
    st.metric("High-Impact Work", f"{high_impact_pct:.0f}%")

# Charts row
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    daily = df.groupby('date').agg({
        'tasks_completed': 'sum',
        'focus_hours': 'mean',
        'meetings': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['tasks_completed'], 
                            name='Tasks Completed', line=dict(color='#2ecc71')))
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['focus_hours']*5,  # scaled for visibility
                            name='Focus Hours (scaled)', line=dict(color='#3498db')))
    fig.update_layout(title="Daily Output & Focus Trends", height=300, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    impact_dist = pd.DataFrame(list(analyzer.effort_vs_impact().items()), 
                              columns=['Category', 'Count'])
    fig2 = px.pie(impact_dist, values='Count', names='Category',
                 title="Effort vs. Impact Distribution",
                 color='Category',
                 color_discrete_map={'High Impact':'#2ecc71', 'Medium Impact':'#f39c12', 'Low Impact':'#e74c3c'})
    st.plotly_chart(fig2, use_container_width=True)

# Insights section
st.subheader("🔍 Key Insights & Recommendations")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("### ⚠️ Bottleneck Analysis")
    if bottleneck['bottleneck_days']:
        st.warning(f"Workflow bottlenecks detected on {len(bottleneck['bottleneck_days'])} days")
        st.write(f"**Pattern**: {bottleneck['common_pattern']}")
        st.write(f"**At-risk members**: {', '.join(bottleneck['overloaded_members'][:2])}")
        st.info("💡 *Recommendation*: Reduce context switching for overloaded members by batching similar tasks")
    else:
        st.success("No significant bottlenecks detected in the last 30 days")

with insights_col2:
    st.markdown("### 🧠 Burnout Prevention")
    if burnout['high_risk_members']:
        st.error(f"⚠️ {len(burnout['high_risk_members'])} members show elevated burnout risk")
        st.write(f"**Trend**: Risk is {burnout['trend']}")
        st.info("💡 *Recommendation*: Enable 'focus hours' policy and review after-hours notification settings")
    else:
        st.success("Burnout risk within healthy range")

# Member-level drilldown
st.subheader("👥 Individual Performance (Anonymized)")
member_view = df.groupby('member').agg({
    'tasks_completed': 'mean',
    'focus_hours': 'mean',
    'completion_rate': 'mean',
    'after_hours_work': 'mean'
}).round(1).sort_values('completion_rate', ascending=False).reset_index()

# Add productivity tier
member_view['tier'] = pd.cut(
    member_view['completion_rate'],
    bins=[0, 60, 85, 100],
    labels=['Needs Support', 'Solid Performer', 'Top Performer']
)

st.dataframe(member_view[['member', 'tasks_completed', 'focus_hours', 
                         'completion_rate', 'after_hours_work', 'tier']],
            use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
🔒 **Privacy Note**: This demo uses *simulated data only*. A production version would:
• Never access message content or keystrokes  
• Allow opt-in/opt-out per data category  
• Provide full transparency into scoring algorithms  
• Store data encrypted with strict retention policies
""")

st.caption("💡 *This prototype demonstrates core analytics logic. Production deployment requires API integrations, auth, and infrastructure.*")
