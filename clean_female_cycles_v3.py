import pandas as pd
import numpy as np
from datetime import timedelta

# Load data
cycles = pd.read_csv('Juli_March_Imputed_Cycles.csv')
female_users = pd.read_csv('female_users.csv')

# Filter to female users only
cycles = cycles[cycles['userid'].isin(female_users['userid'])].copy()
cycles['period_date'] = pd.to_datetime(cycles['period_date'])
cycles = cycles.sort_values(['userid', 'period_date']).reset_index(drop=True)

print(f"Female users with cycle data: {cycles['userid'].nunique()}")
print(f"Total period records: {len(cycles)}")

# Calculate cycle lengths
cycles['next_period'] = cycles.groupby('userid')['period_date'].shift(-1)
cycles['cycle_length'] = (cycles['next_period'] - cycles['period_date']).dt.days

print(f"\nCycle length distribution:")
print(cycles['cycle_length'].describe())

# Create daily records with cycle day labels
all_records = []

for userid, user_cycles in cycles.groupby('userid'):
    period_dates = user_cycles['period_date'].sort_values().tolist()
    cycle_lengths = user_cycles.set_index('period_date')['cycle_length'].to_dict()

    if len(period_dates) == 0:
        continue

    # Get date range: 14 days before first period to last period
    start_date = period_dates[0] - timedelta(days=14)
    end_date = period_dates[-1]

    # Initialize cycle_day labels
    day_labels = {}

    # First pass: label forward from each period (positive days)
    for i, period_date in enumerate(period_dates):
        cycle_len = cycle_lengths.get(period_date)

        # Determine end of this cycle (next period or end of data)
        if i < len(period_dates) - 1:
            next_period = period_dates[i + 1]
        else:
            next_period = None

        # Label day 0 and forward
        day = 0
        current_date = period_date
        while current_date <= end_date:
            if next_period and current_date >= next_period:
                break
            day_labels[current_date] = (day, cycle_len)
            day += 1
            current_date += timedelta(days=1)

    # Second pass: label backwards from each period (negative days take priority)
    for i, period_date in enumerate(period_dates):
        # Get cycle length of the PREVIOUS cycle (the one we're labeling backwards into)
        if i > 0:
            prev_period = period_dates[i - 1]
            prev_cycle_len = cycle_lengths.get(prev_period)
        else:
            prev_cycle_len = None

        for day_offset in range(1, 15):  # -1 to -14
            label_date = period_date - timedelta(days=day_offset)
            if label_date >= start_date:
                day_labels[label_date] = (-day_offset, prev_cycle_len)  # Overwrites positive values

    # Create records
    for date, (cycle_day, cycle_len) in day_labels.items():
        all_records.append({
            'userid': userid,
            'date': date,
            'cycle_day': cycle_day,
            'cycle_length': cycle_len
        })

# Create dataframe
cycle_days_df = pd.DataFrame(all_records)
cycle_days_df = cycle_days_df.sort_values(['userid', 'date']).reset_index(drop=True)

print(f"\nTotal daily records: {len(cycle_days_df)}")
print(f"Unique users: {cycle_days_df['userid'].nunique()}")
print(f"Date range: {cycle_days_df['date'].min()} to {cycle_days_df['date'].max()}")
print(f"\nCycle day distribution:")
print(cycle_days_df['cycle_day'].value_counts().sort_index())

# Save full output
cycle_days_df.to_csv('female_cycle_days.csv', index=False)
print(f"\nSaved to female_cycle_days.csv")

# Filter to cycle lengths 21-35 days
MIN_CYCLE_LENGTH = 21
MAX_CYCLE_LENGTH = 35

cycle_days_filtered = cycle_days_df[
    (cycle_days_df['cycle_length'] >= MIN_CYCLE_LENGTH) &
    (cycle_days_df['cycle_length'] <= MAX_CYCLE_LENGTH)
].copy()

print(f"\n--- Filtered subset (cycle length {MIN_CYCLE_LENGTH}-{MAX_CYCLE_LENGTH} days) ---")
print(f"Total daily records: {len(cycle_days_filtered)}")
print(f"Unique users: {cycle_days_filtered['userid'].nunique()}")
print(f"Cycle day distribution:")
print(cycle_days_filtered['cycle_day'].value_counts().sort_index())

# Save filtered output
cycle_days_filtered.to_csv('female_cycle_days_filtered.csv', index=False)
print(f"\nSaved to female_cycle_days_filtered.csv")

# Load activity data and create active variable
print(f"\n--- Adding activity data ---")
activities = pd.read_csv('activities_energy_output.csv')
activities = activities.rename(columns={'id': 'userid', 'DATE': 'date'})
activities['date'] = pd.to_datetime(activities['date'])

# Keep only relevant columns
activities = activities[['userid', 'date', 'steps', 'workout_duration_minutes']].copy()

# Aggregate duplicates: sum steps and workout minutes per userid+date
activities = activities.groupby(['userid', 'date'], as_index=False).agg({
    'steps': 'sum',
    'workout_duration_minutes': 'sum'
})

print(f"Activity records (after aggregation): {len(activities)}")
print(f"Users with activity data: {activities['userid'].nunique()}")

# Merge activity data with filtered cycle days
cycle_days_filtered['date'] = pd.to_datetime(cycle_days_filtered['date'])
cycle_days_with_activity = cycle_days_filtered.merge(
    activities,
    on=['userid', 'date'],
    how='left'
)

# Create combined activity minutes: (steps/100) + workout_duration_minutes
# Based on CADENCE-adults study: 100 steps/min = moderate intensity walking
cycle_days_with_activity['activity_minutes'] = (
    cycle_days_with_activity['steps'].fillna(0) / 100 +
    cycle_days_with_activity['workout_duration_minutes'].fillna(0)
)

# Set to NaN where BOTH steps and workout are missing
both_missing = (
    cycle_days_with_activity['steps'].isna() &
    cycle_days_with_activity['workout_duration_minutes'].isna()
)
cycle_days_with_activity.loc[both_missing, 'activity_minutes'] = np.nan

print(f"\nActivity minutes stats:")
print(cycle_days_with_activity['activity_minutes'].describe())

# Create active variable: 1 if activity_minutes > person's 25th percentile (bottom quartile)
# This compares bottom 25% (inactive) vs top 75% (active)
user_q25_activity = cycle_days_with_activity.groupby('userid')['activity_minutes'].transform(lambda x: x.quantile(0.25))
cycle_days_with_activity['user_q25_activity'] = user_q25_activity

# Active = 1 if above personal 25th percentile (top 3 quartiles), 0 if in bottom quartile
cycle_days_with_activity['active'] = (
    cycle_days_with_activity['activity_minutes'] > cycle_days_with_activity['user_q25_activity']
).astype(float)

# Set active to NaN where activity_minutes is missing
cycle_days_with_activity.loc[cycle_days_with_activity['activity_minutes'].isna(), 'active'] = np.nan

print(f"\nMatched records with activity data: {cycle_days_with_activity['activity_minutes'].notna().sum()}")
print(f"Active variable distribution (bottom quartile vs top 3 quartiles):")
print(cycle_days_with_activity['active'].value_counts(dropna=False))
print(f"\nExpected ratio: ~25% inactive, ~75% active")

# Check how many users have both states
has_activity = cycle_days_with_activity[cycle_days_with_activity['active'].notna()]
user_states = has_activity.groupby('userid')['active'].nunique()
print(f"\nUsers with both active/inactive states: {(user_states == 2).sum()}")
print(f"Users with only one state: {(user_states == 1).sum()}")

# Create cycle phase variable
def assign_cycle_phase(cycle_day):
    if pd.isna(cycle_day):
        return np.nan
    elif -14 <= cycle_day <= -8:
        return 'early_luteal'
    elif -7 <= cycle_day <= -1:
        return 'late_luteal'
    elif 0 <= cycle_day <= 5:
        return 'menstruation'
    elif cycle_day >= 6:
        return 'follicular'
    else:
        return np.nan

cycle_days_with_activity['cycle_phase'] = cycle_days_with_activity['cycle_day'].apply(assign_cycle_phase)

print(f"\nCycle phase distribution:")
print(cycle_days_with_activity['cycle_phase'].value_counts(dropna=False))

# Load mood data
print(f"\n--- Adding mood data ---")
mood_data = pd.read_csv('juli_panel_daily.csv', usecols=['userid', 'date', 'mood_score', 'mood_energy'])
mood_data['date'] = pd.to_datetime(mood_data['date'])

print(f"Mood records: {len(mood_data)}")
print(f"Users with mood data: {mood_data['userid'].nunique()}")

# Merge mood data
cycle_days_with_activity = cycle_days_with_activity.merge(
    mood_data,
    on=['userid', 'date'],
    how='left'
)

print(f"Matched records with mood_score: {cycle_days_with_activity['mood_score'].notna().sum()}")
print(f"Matched records with mood_energy: {cycle_days_with_activity['mood_energy'].notna().sum()}")

# Sort by userid and date to ensure correct shifting
cycle_days_with_activity = cycle_days_with_activity.sort_values(['userid', 'date']).reset_index(drop=True)

# Create next-day columns by shifting within each user
# shift(-1) moves the NEXT day's value to the current row
cycle_days_with_activity['next_day_mood'] = cycle_days_with_activity.groupby('userid')['mood_score'].shift(-1)
cycle_days_with_activity['next_day_energy'] = cycle_days_with_activity.groupby('userid')['mood_energy'].shift(-1)
cycle_days_with_activity['next_day_mood_plus_energy'] = (
    cycle_days_with_activity['next_day_mood'] + cycle_days_with_activity['next_day_energy']
)

print(f"\nNext-day mood available: {cycle_days_with_activity['next_day_mood'].notna().sum()}")
print(f"Next-day energy available: {cycle_days_with_activity['next_day_energy'].notna().sum()}")
print(f"Next-day mood+energy available: {cycle_days_with_activity['next_day_mood_plus_energy'].notna().sum()}")

# Normalize outcomes: percentage change from each person's mean
print(f"\n--- Normalizing outcomes to % change from individual mean ---")

outcomes_to_normalize = ['next_day_mood', 'next_day_energy', 'next_day_mood_plus_energy']

for outcome in outcomes_to_normalize:
    # Calculate each person's mean for this outcome
    user_means = cycle_days_with_activity.groupby('userid')[outcome].transform('mean')

    # Calculate percentage change from mean: ((value - mean) / mean) * 100
    # Mean represents 100%, so +10 means 10% above their average
    normalized_col = f'{outcome}_pct'
    cycle_days_with_activity[normalized_col] = (
        (cycle_days_with_activity[outcome] - user_means) / user_means
    ) * 100

    print(f"\n{outcome}_pct:")
    print(f"  Records: {cycle_days_with_activity[normalized_col].notna().sum()}")
    print(f"  Mean: {cycle_days_with_activity[normalized_col].mean():.2f}%")
    print(f"  SD: {cycle_days_with_activity[normalized_col].std():.2f}%")
    print(f"  Range: {cycle_days_with_activity[normalized_col].min():.2f}% to {cycle_days_with_activity[normalized_col].max():.2f}%")

# Save output with activity data
cycle_days_with_activity.to_csv('female_cycle_days_filtered.csv', index=False)
print(f"\nUpdated female_cycle_days_filtered.csv with normalized outcomes")
