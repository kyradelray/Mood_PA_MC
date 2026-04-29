import pandas as pd
import numpy as np
from datetime import timedelta

# Load data
print("Loading data...")
male_users = pd.read_csv('male_users.csv')
activities = pd.read_csv('activities_energy_output.csv')
mood_data = pd.read_csv('juli_panel_daily.csv', usecols=['userid', 'date', 'mood_score', 'mood_energy'])

print(f"Male users: {len(male_users)}")

# Prepare activities data
activities = activities.rename(columns={'id': 'userid', 'DATE': 'date'})
activities['date'] = pd.to_datetime(activities['date'])
activities = activities[['userid', 'date', 'steps', 'workout_duration_minutes']].copy()

# Filter to male users
activities = activities[activities['userid'].isin(male_users['userid'])].copy()

# Aggregate duplicates
activities = activities.groupby(['userid', 'date'], as_index=False).agg({
    'steps': 'sum',
    'workout_duration_minutes': 'sum'
})

print(f"Activity records for male users: {len(activities)}")
print(f"Male users with activity data: {activities['userid'].nunique()}")

# Get date range for each user
user_date_ranges = activities.groupby('userid')['date'].agg(['min', 'max']).reset_index()
user_date_ranges.columns = ['userid', 'first_date', 'last_date']

# Function to find first Monday of a month
def first_monday_of_month(year, month):
    from datetime import timedelta
    first_day = pd.Timestamp(year=year, month=month, day=1)
    # Monday is weekday 0
    days_until_monday = (7 - first_day.weekday()) % 7
    if first_day.weekday() == 0:  # Already Monday
        return first_day
    return first_day + timedelta(days=days_until_monday)

# Create pseudo-cycle day labels for each user
print("\nCreating pseudo-cycle day labels based on first Monday of each month...")

all_records = []

for _, row in user_date_ranges.iterrows():
    userid = row['userid']
    first_date = row['first_date']
    last_date = row['last_date']

    # Get all months in the user's date range
    current_date = first_date

    while current_date <= last_date:
        year = current_date.year
        month = current_date.month

        # Find first Monday of this month (this becomes day 0)
        anchor_date = first_monday_of_month(year, month)

        # Day 0 = first Monday of month (anchor_date)
        # Label days from 0 to 27 (4 full weeks)
        for cycle_day in range(0, 28):
            day_date = anchor_date + timedelta(days=cycle_day)

            # Only include if within user's data range
            if first_date <= day_date <= last_date:
                all_records.append({
                    'userid': userid,
                    'date': day_date,
                    'cycle_day': cycle_day,
                    'anchor_date': anchor_date,
                    'year_month': f"{year}-{month:02d}"
                })

        # Move to next month
        if month == 12:
            current_date = pd.Timestamp(year=year + 1, month=1, day=1)
        else:
            current_date = pd.Timestamp(year=year, month=month + 1, day=1)

# Create dataframe
pseudo_cycles_df = pd.DataFrame(all_records)
pseudo_cycles_df = pseudo_cycles_df.drop_duplicates(subset=['userid', 'date'], keep='first')
pseudo_cycles_df = pseudo_cycles_df.sort_values(['userid', 'date']).reset_index(drop=True)

print(f"Total daily records: {len(pseudo_cycles_df)}")
print(f"Unique users: {pseudo_cycles_df['userid'].nunique()}")
print(f"Date range: {pseudo_cycles_df['date'].min()} to {pseudo_cycles_df['date'].max()}")
print(f"\nCycle day distribution:")
print(pseudo_cycles_df['cycle_day'].value_counts().sort_index())

# Assign pseudo-phases: same intervals as women's menstrual cycle phases
def assign_pseudo_phase(cycle_day):
    if pd.isna(cycle_day):
        return np.nan
    elif 0 <= cycle_day <= 5:
        return 'phase_1'  # Days 0-5 (6 days, like menstruation)
    elif 6 <= cycle_day <= 13:
        return 'phase_2'  # Days 6-13 (8 days, like follicular)
    elif 14 <= cycle_day <= 20:
        return 'phase_3'  # Days 14-20 (7 days, like early luteal)
    elif 21 <= cycle_day <= 27:
        return 'phase_4'  # Days 21-27 (7 days, like late luteal)
    else:
        return np.nan

pseudo_cycles_df['pseudo_phase'] = pseudo_cycles_df['cycle_day'].apply(assign_pseudo_phase)

print(f"\nPseudo-phase distribution:")
print(pseudo_cycles_df['pseudo_phase'].value_counts(dropna=False))

# Merge with activity data
print(f"\n--- Adding activity data ---")
pseudo_cycles_df['date'] = pd.to_datetime(pseudo_cycles_df['date'])
pseudo_cycles_with_activity = pseudo_cycles_df.merge(
    activities,
    on=['userid', 'date'],
    how='left'
)

# Create combined activity minutes: (steps/100) + workout_duration_minutes
pseudo_cycles_with_activity['activity_minutes'] = (
    pseudo_cycles_with_activity['steps'].fillna(0) / 100 +
    pseudo_cycles_with_activity['workout_duration_minutes'].fillna(0)
)

# Set to NaN where BOTH steps and workout are missing
both_missing = (
    pseudo_cycles_with_activity['steps'].isna() &
    pseudo_cycles_with_activity['workout_duration_minutes'].isna()
)
pseudo_cycles_with_activity.loc[both_missing, 'activity_minutes'] = np.nan

print(f"Activity minutes stats:")
print(pseudo_cycles_with_activity['activity_minutes'].describe())

# Create active variable: 1 if activity_minutes > person's median (median split)
user_median_activity = pseudo_cycles_with_activity.groupby('userid')['activity_minutes'].transform('median')
pseudo_cycles_with_activity['user_median_activity'] = user_median_activity

# Active = 1 if above personal median, 0 if below
pseudo_cycles_with_activity['active'] = (
    pseudo_cycles_with_activity['activity_minutes'] > pseudo_cycles_with_activity['user_median_activity']
).astype(float)

# Set active to NaN where activity_minutes is missing
pseudo_cycles_with_activity.loc[pseudo_cycles_with_activity['activity_minutes'].isna(), 'active'] = np.nan

print(f"\nMatched records with activity data: {pseudo_cycles_with_activity['activity_minutes'].notna().sum()}")
print(f"Active variable distribution (below vs above personal median):")
print(pseudo_cycles_with_activity['active'].value_counts(dropna=False))

# Check how many users have both states
has_activity = pseudo_cycles_with_activity[pseudo_cycles_with_activity['active'].notna()]
user_states = has_activity.groupby('userid')['active'].nunique()
print(f"\nUsers with both active/inactive states: {(user_states == 2).sum()}")
print(f"Users with only one state: {(user_states == 1).sum()}")

# Add mood data
print(f"\n--- Adding mood data ---")
mood_data['date'] = pd.to_datetime(mood_data['date'])

pseudo_cycles_with_activity = pseudo_cycles_with_activity.merge(
    mood_data,
    on=['userid', 'date'],
    how='left'
)

print(f"Matched records with mood_score: {pseudo_cycles_with_activity['mood_score'].notna().sum()}")
print(f"Matched records with mood_energy: {pseudo_cycles_with_activity['mood_energy'].notna().sum()}")

# Sort and create next-day columns
pseudo_cycles_with_activity = pseudo_cycles_with_activity.sort_values(['userid', 'date']).reset_index(drop=True)

pseudo_cycles_with_activity['next_day_mood'] = pseudo_cycles_with_activity.groupby('userid')['mood_score'].shift(-1)
pseudo_cycles_with_activity['next_day_energy'] = pseudo_cycles_with_activity.groupby('userid')['mood_energy'].shift(-1)
pseudo_cycles_with_activity['next_day_mood_plus_energy'] = (
    pseudo_cycles_with_activity['next_day_mood'] + pseudo_cycles_with_activity['next_day_energy']
)

print(f"\nNext-day mood available: {pseudo_cycles_with_activity['next_day_mood'].notna().sum()}")
print(f"Next-day energy available: {pseudo_cycles_with_activity['next_day_energy'].notna().sum()}")

# Normalize outcomes: percentage change from each person's mean
print(f"\n--- Normalizing outcomes to % change from individual mean ---")

outcomes_to_normalize = ['next_day_mood', 'next_day_energy', 'next_day_mood_plus_energy']

for outcome in outcomes_to_normalize:
    user_means = pseudo_cycles_with_activity.groupby('userid')[outcome].transform('mean')
    normalized_col = f'{outcome}_pct'
    pseudo_cycles_with_activity[normalized_col] = (
        (pseudo_cycles_with_activity[outcome] - user_means) / user_means
    ) * 100

    print(f"\n{outcome}_pct:")
    print(f"  Records: {pseudo_cycles_with_activity[normalized_col].notna().sum()}")
    print(f"  Mean: {pseudo_cycles_with_activity[normalized_col].mean():.2f}%")
    print(f"  SD: {pseudo_cycles_with_activity[normalized_col].std():.2f}%")

# Save output
pseudo_cycles_with_activity.to_csv('male_pseudo_cycles.csv', index=False)
print(f"\nSaved to male_pseudo_cycles.csv")

# Summary
print(f"\n" + "="*60)
print("SUMMARY: Male Pseudo-Cycle Data")
print("="*60)
print(f"Total observations: {len(pseudo_cycles_with_activity)}")
print(f"Unique users: {pseudo_cycles_with_activity['userid'].nunique()}")
print(f"Users with both active/inactive states: {(user_states == 2).sum()}")
print(f"Observations with next_day_mood_pct: {pseudo_cycles_with_activity['next_day_mood_pct'].notna().sum()}")
print(f"\nPhase mapping (starting from first Monday of month, same intervals as women):")
print(f"  Phase 1: days 0-5   (6 days, like menstruation)")
print(f"  Phase 2: days 6-13  (8 days, like follicular)")
print(f"  Phase 3: days 14-20 (7 days, like early luteal)")
print(f"  Phase 4: days 21-27 (7 days, like late luteal)")
