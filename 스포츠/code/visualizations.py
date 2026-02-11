"""
Selected F1 Driver Visualizations
- viz_v5_05: Driver Type Classification Matrix
- viz_v5_07: Investment Opportunity Matrix
- viz_v4_08: Oscar Piastri Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# Color palette
COLORS = {
    'primary': '#E10600',
    'secondary': '#1E1E1E',
    'accent': '#00D2BE',
    'warning': '#FF8700',
    'success': '#00FF00',
    'hidden': '#6B5B95',
    'skill': '#88B04B',
    'overvalued': '#FF6B6B',
    'undervalued': '#4ECDC4',
    'neutral': '#95A5A6',
}

# Driver name to code mapping
name_to_code = {
    'Lewis Hamilton': 'HAM', 'Max Verstappen': 'VER', 'Charles Leclerc': 'LEC',
    'Lando Norris': 'NOR', 'Fernando Alonso': 'ALO', 'George Russell': 'RUS',
    'Sergio Perez': 'PER', 'Carlos Sainz': 'SAI', 'Valtteri Bottas': 'BOT',
    'Daniel Ricciardo': 'RIC', 'Pierre Gasly': 'GAS', 'Esteban Ocon': 'OCO',
    'Oscar Piastri': 'PIA', 'Kevin Magnussen': 'MAG', 'Alexander Albon': 'ALB',
    'Lance Stroll': 'STR', 'Nico Hulkenberg': 'HUL', 'Guanyu Zhou': 'ZHO',
    'Zhou Guanyu': 'ZHO', 'Logan Sargeant': 'SAR', 'Yuki Tsunoda': 'TSU',
    'Sebastian Vettel': 'VET', 'Kimi Räikkönen': 'RAI', 'Mick Schumacher': 'MSC',
    'Nicholas Latifi': 'LAT', 'Antonio Giovinazzi': 'GIO', 'Romain Grosjean': 'GRO',
    'Daniil Kvyat': 'KVY'
}


def load_data():
    """Load all required data"""
    hidden = pd.read_csv('data/main/driver_latent_ability_fe.csv')
    skill = pd.read_csv('data/main/driver_skill_weighted_100.csv')
    salary = pd.read_csv('data/main/f1_driver_salaries_2018_2024.csv')
    
    # Rename columns
    hidden = hidden.rename(columns={'Driver': 'Driver_Code', 'latent_ability_0_10': 'hidden_ability'})
    skill = skill.rename(columns={'Driver': 'Driver_Code', 'skill_score_0_100': 'skill_score'})
    
    # Add skill components
    skill['Pace'] = skill['mean_Pace_0_100']
    skill['Tyre'] = skill['mean_Tyre_0_100']
    skill['Stability'] = skill['mean_Stability_0_100']
    skill['Overtake'] = skill['mean_Overtake_0_100']
    
    salary['Driver_Code'] = salary['Driver'].map(name_to_code)
    salary_2024 = salary[['Driver_Code', '2024']].dropna().rename(columns={'2024': 'salary'})
    
    # Merge
    df = hidden[['Driver_Code', 'hidden_ability']].merge(
        skill[['Driver_Code', 'skill_score', 'Pace', 'Tyre', 'Stability', 'Overtake']], on='Driver_Code'
    ).merge(salary_2024, on='Driver_Code')
    
    return df, skill


# ============================================================================
# VIZ V5-05: Driver Type Classification Matrix
# ============================================================================
def create_viz_v5_05(df):
    """Driver Type Classification Matrix"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize for quadrant analysis
    hidden_median = df['hidden_ability'].median()
    skill_median = df['skill_score'].median()
    
    # Create quadrant colors
    colors = []
    for _, row in df.iterrows():
        if row['hidden_ability'] > hidden_median and row['skill_score'] > skill_median:
            colors.append('#2ECC71')  # Top Right - Complete
        elif row['hidden_ability'] > hidden_median and row['skill_score'] <= skill_median:
            colors.append('#9B59B6')  # Top Left - Raw Talent
        elif row['hidden_ability'] <= hidden_median and row['skill_score'] > skill_median:
            colors.append('#3498DB')  # Bottom Right - Executor
        else:
            colors.append('#E74C3C')  # Bottom Left - Needs Development
    
    # Create scatter with larger points
    scatter = ax.scatter(df['skill_score'], df['hidden_ability'], 
                        c=colors, s=350, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add quadrant lines
    ax.axhline(y=hidden_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=skill_median, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add quadrant labels - repositioned to avoid overlap
    ax.text(skill_median + 10, hidden_median + 3.0, 'COMPLETE\n(Elite)', fontsize=14, 
            fontweight='bold', color='#2ECC71', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(skill_median - 10, hidden_median + 3.0, 'RAW TALENT\n(Hidden Gem)', fontsize=14, 
            fontweight='bold', color='#9B59B6', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(skill_median + 10, hidden_median - 3.0, 'EXECUTOR\n(Technical)', fontsize=14, 
            fontweight='bold', color='#3498DB', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(skill_median - 10, hidden_median - 3.0, 'DEVELOPMENT\n(Needs Work)', fontsize=14, 
            fontweight='bold', color='#E74C3C', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add driver labels with adjustText - larger font
    texts = []
    for _, row in df.iterrows():
        fontweight = 'bold' if row['Driver_Code'] in ['VER', 'HAM', 'OCO', 'PIA', 'RUS', 'SAI'] else 'normal'
        texts.append(ax.text(row['skill_score'], row['hidden_ability'], row['Driver_Code'],
                   fontsize=11, ha='center', va='center', fontweight=fontweight))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand_points=(2.0, 2.0), force_points=(0.5, 0.5))
    
    ax.set_xlabel('Skill Score (0-100) →\n"Technical Ability"', fontsize=12)
    ax.set_ylabel('← Hidden Ability (0-10)\n"Racing Intelligence"', fontsize=12)
    ax.set_title('Driver Type Classification Matrix\n(Based on Hidden Ability vs Skill Score)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('viz_v5_05_driver_type_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: viz_v5_05_driver_type_matrix.png")


# ============================================================================
# VIZ V5-07: Investment Opportunity Matrix
# ============================================================================
def create_viz_v5_07(df):
    """Investment Opportunity Matrix"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    beta0, beta1, beta2 = -58.47, 3.79, 0.79
    df = df.copy()
    df['predicted'] = beta0 + beta1 * df['hidden_ability'] + beta2 * df['skill_score']
    df['residual'] = df['salary'] - df['predicted']
    df['value_ratio'] = df['predicted'] / df['salary']  # >1 means undervalued
    
    # Bubble size = actual salary (larger base size)
    sizes = df['salary'] * 15  # Scale for visibility
    
    # Colors based on valuation
    colors = [COLORS['undervalued'] if r < -5 else COLORS['overvalued'] if r > 5 else COLORS['neutral'] 
              for r in df['residual']]
    
    scatter = ax.scatter(df['predicted'], df['salary'], s=sizes, c=colors, 
                        alpha=0.6, edgecolor='black', linewidth=2)
    
    # Perfect valuation line
    max_val = max(df['salary'].max(), df['predicted'].max()) + 5
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Fair Value Line')
    
    # Add labels with salary info using adjustText
    texts = []
    for _, row in df.iterrows():
        label = f"{row['Driver_Code']}\n${row['salary']:.0f}M"
        fontweight = 'bold' if abs(row['residual']) > 8 else 'normal'
        texts.append(ax.text(row['predicted'], row['salary'], label,
                   fontsize=10, ha='center', va='center', fontweight=fontweight))
    
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                expand_points=(2.0, 2.0), force_points=(0.5, 0.5))
    
    # Highlight zones
    ax.fill_between([0, max_val], [0, max_val], [0, 0], alpha=0.1, color=COLORS['undervalued'], 
                    label='Undervalued Zone (Below Line)')
    ax.fill_between([0, max_val], [max_val, max_val], [0, max_val], alpha=0.1, color=COLORS['overvalued'],
                    label='Overvalued Zone (Above Line)')
    
    ax.set_xlabel('Predicted Salary ($M) - Based on Hidden + Skill', fontsize=15)
    ax.set_ylabel('Actual Salary ($M)', fontsize=15)
    ax.set_title('Investment Opportunity Matrix\n(Bubble Size = Actual Salary)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(-10, max_val)
    ax.set_ylim(-5, 60)
    
    plt.tight_layout()
    plt.savefig('viz_v5_07_investment_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Created: viz_v5_07_investment_matrix.png")


# ============================================================================
# VIZ V4-08: Oscar Piastri Analysis
# ============================================================================
def create_viz_v4_08(skill_data):
    """Oscar Piastri: Future Champion Analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Skill Score breakdown
    ax1 = axes[0]
    
    pia_skill = skill_data[skill_data['Driver_Code'] == 'PIA'].iloc[0]
    
    metrics = ['Pace', 'Tyre Mgmt', 'Stability', 'Overtake', 'Skill Score']
    values = [pia_skill['Pace'], pia_skill['Tyre'], pia_skill['Stability'], 
              pia_skill['Overtake'], pia_skill['skill_score']]
    
    colors = ['#3498DB', '#3498DB', '#3498DB', '#FF6B35', '#27AE60']
    
    y_pos = np.arange(len(metrics))
    bars = ax1.barh(y_pos, values, color=colors, edgecolor='white', linewidth=2)
    
    # Ranks (from skill_df)
    ranks = ['Mid', 'High', 'Mid', '1st!', '3rd!']
    for i, (v, r) in enumerate(zip(values, ranks)):
        ax1.text(v + 1, i, f'{v:.1f} ({r})', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.set_title('PIA Skill Score Breakdown\nOvertake #1 + Skill Score #3', 
                  fontsize=14, fontweight='bold', color='#FF6B35')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Comparison with top drivers
    ax2 = axes[1]
    
    compare_drivers = ['VER', 'HAM', 'PIA', 'LEC', 'NOR']
    compare_data = []
    for d in compare_drivers:
        row = skill_data[skill_data['Driver_Code'] == d]
        if len(row) > 0:
            row = row.iloc[0]
            compare_data.append({
                'Driver': d,
                'Skill': row['skill_score'],
                'Overtake': row['Overtake']
            })
    compare_df = pd.DataFrame(compare_data)
    
    x = np.arange(len(compare_df))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, compare_df['Skill'], width, label='Skill Score', 
                    color='#3498DB', edgecolor='white')
    bars2 = ax2.bar(x + width/2, compare_df['Overtake'], width, label='Overtake', 
                    color='#FF6B35', edgecolor='white')
    
    for i, (s, o) in enumerate(zip(compare_df['Skill'], compare_df['Overtake'])):
        ax2.text(i - width/2, s + 1, f'{s:.0f}', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, o + 1, f'{o:.0f}', ha='center', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(compare_df['Driver'], fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 90)
    ax2.legend(fontsize=11)
    ax2.set_title('PIA vs Top Drivers Comparison\nHighest Overtake Score in F1', 
                  fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Highlight PIA
    ax2.axvspan(1.5, 2.5, alpha=0.2, color='#FF6B35')
    
    fig.suptitle('Oscar Piastri: Future Champion Analysis\nSkill Score 3rd + Overtake 1st + $6M = Historic Undervalue', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('viz_v4_08_piastri_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: viz_v4_08_piastri_analysis.png")


# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("=" * 60)
    print("Creating Selected Visualizations")
    print("  - viz_v5_05: Driver Type Classification Matrix")
    print("  - viz_v5_07: Investment Opportunity Matrix")
    print("  - viz_v4_08: Oscar Piastri Analysis")
    print("=" * 60)
    
    # Load data
    df, skill_data = load_data()
    print(f"\n✓ Loaded data for {len(df)} drivers")
    
    # Create visualizations
    print("\nGenerating visualizations...\n")
    
    create_viz_v5_05(df)
    create_viz_v5_07(df)
    create_viz_v4_08(skill_data)
    
    print("\n" + "=" * 60)
    print("All visualizations created successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. viz_v5_05_driver_type_matrix.png")
    print("  2. viz_v5_07_investment_matrix.png")
    print("  3. viz_v4_08_piastri_analysis.png")


if __name__ == "__main__":
    main()
