import streamlit as st
import pandas as pd
import os
import datetime
import holidays
import math
from ortools.sat.python import cp_model

# ----------------------------------------
# 1. INITIALIZE MOCK DATA
# ----------------------------------------
def load_initial_staff():
    staff_data = []
    
    for i in range(1, 38):
        if i <= 7: role = "ANUM"
        elif i <= 25: role = "RN (In Charge)"
        elif i <= 29: role = "RN"
        else: role = "EN/Learner"
        
        staff_data.append({
            "ID": f"Staff_{i:02d}",
            "Role": role,
            "EFT": 1.0,
            "Gender": "Female" if i % 3 != 0 else "Male",
            "Night_Pool": True if i >= 32 else False,
            "Prior_Consecutive_Days": 0,
            "Last_Shift_Type": "None",
            "Approved_Leave_Days": "",
            "Requested_RDOs": "",
            "Unavailable_DOW": "", 
            "Allow_Fragmented_Shifts": False,
            "Entire_Roster_Leave": False,
            "Secondary_Role": "None",
            "Secondary_EFT": 0.0,
            "No_AM_DOW": "",
            "No_PM_DOW": "",
            "Preferred_Shift": "None",
            "Study_Leave_Days": "",
            "External_Working_Days": "" # e.g., CNM shifts, secondments
        })
        
    return pd.DataFrame(staff_data)

# ----------------------------------------
# 2. THE SOLVER ENGINE (PHASE 2 & 3)
# ----------------------------------------
def solve_roster(df, num_days, start_date):
    df = df.reset_index(drop=True)
    
    model = cp_model.CpModel()
    
    num_staff = len(df)
    all_staff = range(num_staff)
    all_days = range(num_days)
    
    shift_names = ["AM", "PM", "Night"]
    
    # Create the boolean matrix
    roster = {}
    role_primary = {}
    role_secondary = {}
    
    for n in all_staff:
        has_secondary = df.iloc[n]["Secondary_Role"] != "None" and df.iloc[n]["Secondary_EFT"] > 0
        
        for d in all_days:
            for s in range(3):
                roster[(n, d, s)] = model.NewBoolVar(f'staff_{n}_day_{d}_shift_{s}')
                role_primary[(n, d, s)] = model.NewBoolVar(f'prim_{n}_{d}_{s}')
                
                if has_secondary:
                    role_secondary[(n, d, s)] = model.NewBoolVar(f'sec_{n}_{d}_{s}')
                    model.Add(roster[(n, d, s)] == role_primary[(n, d, s)] + role_secondary[(n, d, s)])
                else:
                    model.Add(roster[(n, d, s)] == role_primary[(n, d, s)])
                    
    # CONSTRAINT: Maximum 1 shift per day per staff member
    for n in all_staff:
        for d in all_days:
            model.AddAtMostOne(roster[(n, d, s)] for s in range(3))
            
    # PRE-PROCESSING: Genders
    males = df.index[df['Gender'] == 'Male'].tolist()
    females = df.index[df['Gender'] == 'Female'].tolist()

    # CONSTRAINT: Vacancy Tolerances & Calendar Logic
    end_date = start_date + datetime.timedelta(days=num_days)
    vic_holidays = holidays.AU(subdiv='VIC', years=[start_date.year, end_date.year])

    for d in all_days:
        current_date = start_date + datetime.timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        is_friday = current_date.weekday() == 4
        is_pub_hol = current_date in vic_holidays
        
        # 1. AM SHIFT (s=0)
        model.Add(sum(roster[(n, d, 0)] for n in all_staff) == 5)
        
        # 2. PM SHIFT (s=1)
        pm_sum = sum(roster[(n, d, 1)] for n in all_staff)
        if is_weekend or is_friday or is_pub_hol:
            model.Add(pm_sum == 5)
        else:
            model.Add(pm_sum >= 4)
            model.Add(pm_sum <= 5)
            
        # 3. NIGHT SHIFT (s=2)
        night_sum = sum(roster[(n, d, 2)] for n in all_staff)
        if is_weekend or is_pub_hol:
            model.Add(night_sum == 4)
        else:
            model.Add(night_sum >= 3)
            model.Add(night_sum <= 4)
            
    # CONSTRAINT: Skill Mix & Demographics
    for d in all_days:
        for s in range(3):
            model.Add(sum(roster[(n, d, s)] for n in females if (n, d, s) in roster) >= 1)
            
            anum_sum = (
                sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "ANUM" and (n, d, s) in role_primary) + 
                sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "ANUM" and (n, d, s) in role_secondary)
            )
            
            rn_in_charge_sum = (
                sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "RN (In Charge)" and (n, d, s) in role_primary) + 
                sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "RN (In Charge)" and (n, d, s) in role_secondary)
            )
            
            en_sum = (
                sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "EN/Learner" and (n, d, s) in role_primary) + 
                sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "EN/Learner" and (n, d, s) in role_secondary)
            )
            
            model.Add(anum_sum + rn_in_charge_sum >= 2)
            
            if s == 2:
                model.Add(en_sum <= 1)
            else:
                model.Add(en_sum <= 2)
            
    # CONSTRAINT: Night Pool vs Day Pool Separation
    for n in all_staff:
        is_night_pool = df.iloc[n]["Night_Pool"]
        for d in all_days:
            if is_night_pool:
                model.Add(roster[(n, d, 0)] == 0)
                model.Add(roster[(n, d, 1)] == 0)
            else:
                model.Add(roster[(n, d, 2)] == 0)

    # CONSTRAINT: Leave and Availability
    for n in all_staff:
        leave_str = str(df.iloc[n]["Approved_Leave_Days"])
        rdo_str = str(df.iloc[n]["Requested_RDOs"])
        
        def parse_days(day_string):
            if pd.isna(day_string) or day_string.strip() == "" or day_string.strip() == "nan":
                return []
            try:
                return [int(x.strip()) - 1 for x in day_string.split(",")]
            except ValueError:
                return []
                
        study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
        study_days = set(parse_days(study_str))
        
        ext_str = str(df.iloc[n].get("External_Working_Days", ""))
        ext_days = set(parse_days(ext_str))
        
        # Combine all leave, RDOs, study days, and external days into one blocked list for the ward
        unavailable_days = set(parse_days(leave_str) + parse_days(rdo_str)) | study_days | ext_days
                
        for d in unavailable_days:
            if 0 <= d < num_days:
                for s in range(3):
                    if (n, d, s) in roster:
                        model.Add(roster[(n, d, s)] == 0)

        dow_str = str(df.iloc[n]["Unavailable_DOW"]).lower()
        if dow_str and dow_str != "nan":
            for d in all_days:
                current_date = start_date + datetime.timedelta(days=d)
                day_name = current_date.strftime("%A").lower()
                if day_name in dow_str:
                    for s in range(3):
                        model.Add(roster[(n, d, s)] == 0)
# Block specific shift unavailabilities (e.g., "Cannot work Monday AM")
        no_am_str = str(df.iloc[n].get("No_AM_DOW", "")).lower()
        no_pm_str = str(df.iloc[n].get("No_PM_DOW", "")).lower()
        
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            day_name = current_date.strftime("%A").lower()
            
            # If the day name is in their restricted AM list, block the AM shift (s=0)
            if no_am_str and no_am_str != "nan" and day_name in no_am_str:
                if (n, d, 0) in roster:
                    model.Add(roster[(n, d, 0)] == 0)
                    
            # If the day name is in their restricted PM list, block the PM shift (s=1)
            if no_pm_str and no_pm_str != "nan" and day_name in no_pm_str:
                if (n, d, 1) in roster:
                    model.Add(roster[(n, d, 1)] == 0)

    # CONSTRAINT: Hour-Aware EFT Allocation
    for n in all_staff:
        is_fully_absent = df.iloc[n].get("Entire_Roster_Leave", False)
        has_secondary = df.iloc[n]["Secondary_Role"] != "None" and df.iloc[n]["Secondary_EFT"] > 0
        
        if is_fully_absent:
            final_shift_target = 0
            for d in range(num_days):
                for s in range(3):
                    if (n, d, s) in roster:
                        model.Add(roster[(n, d, s)] == 0)
        else:
            primary_eft = df.iloc[n]["EFT"]
            secondary_eft = df.iloc[n]["Secondary_EFT"] if has_secondary else 0.0
            total_eft = primary_eft + secondary_eft
            is_night_pool = df.iloc[n]["Night_Pool"]
            
            target_hours = total_eft * 80.0 * (num_days / 14.0)
            shift_length = 10.0 if is_night_pool else 8.0
            
            leave_str = str(df.iloc[n]["Approved_Leave_Days"])
            valid_leave_days = 0
            if leave_str and leave_str.lower() != 'nan':
                for val in leave_str.split(","):
                    try:
                        d = int(val.strip()) - 1
                        if 0 <= d < num_days:
                            valid_leave_days += 1
                    except ValueError:
                        pass
                        
            # Calculate Standard Leave
            leave_hours_deduction = valid_leave_days * shift_length
            
            # NEW: Calculate Study Leave (Strictly 8 hours per day)
            study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
            # We re-parse here quickly safely
            study_count = 0
            if study_str and study_str.lower() != 'nan':
                for val in study_str.split(","):
                    try:
                        if 0 <= (int(val.strip()) - 1) < num_days: study_count += 1
                    except ValueError: pass
                    
            study_hours_deduction = study_count * 8.0
            
            # Adjust the final targets!
            adjusted_target_hours = max(0, target_hours - leave_hours_deduction - study_hours_deduction)
            final_shift_target = math.ceil(adjusted_target_hours / shift_length)
            
            if has_secondary:
                sec_target_hours = secondary_eft * 80.0 * (num_days / 14.0)
                sec_shift_target = math.ceil(sec_target_hours / shift_length)
                sec_shift_target = min(sec_shift_target, final_shift_target)
                model.Add(sum(role_secondary[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in role_secondary) == sec_shift_target)
            
        model.Add(sum(roster[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in roster) == final_shift_target)

    # CONSTRAINT: Rest and Fatigue Rules
    for n in all_staff:
        raw_prior = df.iloc[n]["Prior_Consecutive_Days"]
        prior_days = int(raw_prior) if pd.notna(raw_prior) else 0
        last_shift = str(df.iloc[n]["Last_Shift_Type"]).strip()
        is_night_pool = df.iloc[n]["Night_Pool"]
        
        # --- DEFINE VIRTUAL DAYS FIRST ---
        study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
        study_days = [int(x.strip()) - 1 for x in study_str.split(",") if x.strip().isdigit()] if study_str and study_str.lower() != 'nan' else []
        
        ext_str = str(df.iloc[n].get("External_Working_Days", ""))
        ext_days = [int(x.strip()) - 1 for x in ext_str.split(",") if x.strip().isdigit()] if ext_str and ext_str.lower() != 'nan' else []
        
        virtual_days = set(study_days + ext_days)
        # ---------------------------------
        
        # 1. Minimum 8-Hour Gap (Rolling)
        for d in range(num_days - 1):
            model.AddImplication(roster[(n, d, 2)], roster[(n, d+1, 0)].Not())
            model.AddImplication(roster[(n, d, 2)], roster[(n, d+1, 1)].Not())
            
        # 1b. Boundary 8-Hour Gap (Day 0)
        if last_shift == "PM":
            model.Add(roster[(n, 0, 0)] == 0) 
        elif last_shift == "Night":
            model.Add(roster[(n, 0, 0)] == 0)
            model.Add(roster[(n, 0, 1)] == 0)
            
        # 2. Maximum Consecutive Shifts (6 days total, 4 nights total)
        for d in range(num_days - 6):
            ward_shifts = sum(roster[(n, d+w, s)] for w in range(7) for s in range(3) if (n, d+w, s) in roster)
            virtual_shifts = sum(1 for w in range(7) if (d+w) in virtual_days)
            model.Add(ward_shifts + virtual_shifts <= 6)
            
        for d in range(num_days - 4):
            model.Add(sum(roster[(n, d+w, 2)] for w in range(5)) <= 4)
            
        # 2b. Boundary Consecutive Shifts
        if prior_days > 0:
            days_to_check = 7 - prior_days
            remaining_allowed = max(0, 6 - prior_days)
            if days_to_check > 0 and num_days >= days_to_check:
                ward_boundary = sum(roster[(n, w, s)] for w in range(days_to_check) for s in range(3) if (n, w, s) in roster)
                virtual_boundary = sum(1 for w in range(days_to_check) if w in virtual_days)
                model.Add(ward_boundary + virtual_boundary <= remaining_allowed)

        # 3. The 48-Hour Transition Rule
        if last_shift == "Night" and not is_night_pool:
            if num_days > 0:
                for s in range(3):
                    model.Add(roster[(n, 0, s)] == 0)
            if num_days > 1:
                for s in range(3):
                    model.Add(roster[(n, 1, s)] == 0)
                    
        # 4. Maximum 2 Blocks of Shifts Per Fortnight
        allow_fragmented = df.iloc[n]["Allow_Fragmented_Shifts"]
        
        for start_d in range(0, num_days, 14):
            end_d = min(start_d + 14, num_days)
            block_ends = []
            
            for d in range(start_d, end_d - 1):
                working_today = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                working_tomorrow = sum(roster[(n, d+1, s)] for s in range(3) if (n, d+1, s) in roster) + (1 if d+1 in virtual_days else 0)
                
                is_end_of_block = model.NewBoolVar(f'block_end_staff_{n}_day_{d}')
                model.Add(working_today - working_tomorrow <= is_end_of_block)
                block_ends.append(is_end_of_block)
                
            if not allow_fragmented:
                model.Add(sum(block_ends) <= 2)
                
        # 5. Minimum 2 Days Off at a Time (Ban single days off)
        if not allow_fragmented:
            for d in range(num_days - 2):
                working_d = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                working_d1 = sum(roster[(n, d+1, s)] for s in range(3) if (n, d+1, s) in roster) + (1 if d+1 in virtual_days else 0)
                working_d2 = sum(roster[(n, d+2, s)] for s in range(3) if (n, d+2, s) in roster) + (1 if d+2 in virtual_days else 0)
                
                model.Add(working_d - working_d1 + working_d2 <= 1)
                
            if prior_days > 0 and num_days > 1:
                working_day_0 = sum(roster[(n, 0, s)] for s in range(3) if (n, 0, s) in roster) + (1 if 0 in virtual_days else 0)
                working_day_1 = sum(roster[(n, 1, s)] for s in range(3) if (n, 1, s) in roster) + (1 if 1 in virtual_days else 0)
                model.Add(working_day_1 <= working_day_0)

    # OPTIMIZATION: Soft Constraints (Fatigue & Fairness)
    penalties = []
    
    for n in all_staff:
        late_earlies = []
        for d in range(num_days - 1):
            is_late_early = model.NewBoolVar(f'late_early_staff_{n}_day_{d}')
            model.Add(roster[(n, d, 1)] + roster[(n, d+1, 0)] - 1 <= is_late_early)
            late_earlies.append(is_late_early)
            
        model.Add(sum(late_earlies) <= (num_days // 14))
        penalties.extend(late_earlies)
        
        is_night_pool = df.iloc[n]["Night_Pool"]
        shift_length = 10.0 if is_night_pool else 8.0
        
        base_target_hours = df.iloc[n]["EFT"] * 80.0 * (num_days / 14.0)
        shifts_per_roster = math.ceil(base_target_hours / shift_length)
        
        ideal_max_consecutive = int((shifts_per_roster / 2) + 1)
        
        if ideal_max_consecutive < 6:
            for d in range(num_days - ideal_max_consecutive):
                over_consecutive = model.NewBoolVar(f'over_consec_{n}_{d}')
                window_sum = sum(roster[(n, d+w, s)] for w in range(ideal_max_consecutive + 1) for s in range(3))
                model.Add(window_sum - ideal_max_consecutive <= over_consecutive)
                penalties.append(over_consecutive * 5)

# 3. Maximize Preferred Shifts
        pref_shift = str(df.iloc[n].get("Preferred_Shift", "None"))
        if pref_shift in ["AM", "PM", "Night"]:
            pref_idx = shift_names.index(pref_shift) # Finds if it's 0, 1, or 2
            
            for d in range(num_days):
                for s in range(3):
                    # If this shift exists, and it is NOT their preferred shift...
                    if s != pref_idx and (n, d, s) in roster:
                        # Add a small penalty (e.g., weight of 2) to discourage the engine.
                        # Because it's a soft constraint, the engine will still break this rule 
                        # if the ward absolutely needs them to fill a quota!
                        penalties.append(roster[(n, d, s)] * 2)

    model.Minimize(sum(penalties))                
    
    # EXECUTE THE SOLVER
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0 
    status = solver.Solve(model)
    
    # FORMAT THE OUTPUT
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        roster_output = []
        
        day_headers = []
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            header = current_date.strftime("%a %d-%b") 
            if current_date in vic_holidays:
                header += " (PH)"
            day_headers.append(header)
            
        for n in all_staff:
            staff_row = {
                "Staff ID": df.iloc[n]["ID"],
                "Role": df.iloc[n]["Role"]
            }
            
            for d in all_days:
                assigned_shift = "" 
                
                # Check if we need to print "Study Leave"
                study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
                study_days = []
                if study_str and study_str.lower() != 'nan':
                    study_days = [int(x.strip()) - 1 for x in study_str.split(",") if x.strip().isdigit()]
                    
                if d in study_days:
                    assigned_shift = "Study Leave"
                elif d in ext_days:
                    assigned_shift = "External/CNM"
                else:
                    for s in range(3):
                        if (n, d, s) in roster and solver.Value(roster[(n, d, s)]) == 1:
                            if (n, d, s) in role_secondary and solver.Value(role_secondary[(n, d, s)]) == 1:
                                assigned_shift = f"{shift_names[s]} ({df.iloc[n]['Secondary_Role']})"
                            else:
                                assigned_shift = shift_names[s]
                
                staff_row[day_headers[d]] = assigned_shift
                
            roster_output.append(staff_row)
            
        return pd.DataFrame(roster_output)
    else:
        return None

# ----------------------------------------
# 3. STREAMLIT USER INTERFACE
# ----------------------------------------
st.set_page_config(layout="wide", page_title="Ward Rostering Engine")

st.title("Automated Ward Rostering Engine")
st.markdown("Adjust the staff parameters below, then click generate.")

with st.sidebar:
    st.header("Roster Settings")
    start_date = st.date_input("Roster Start Date", datetime.date.today())
    roster_days = st.slider("Roster Length (Days)", min_value=14, max_value=182, value=14, step=7)

if "staff_df" not in st.session_state:
    if os.path.exists("staff_profiles.csv"):
        st.session_state.staff_df = pd.read_csv("staff_profiles.csv")
    else:
        st.session_state.staff_df = load_initial_staff()

# --- FORCE CLEAN DATA TYPES (Bypasses Memory Caching Bugs) ---
raw_df = st.session_state.staff_df.copy()

# 1. Inject missing columns
if "Secondary_Role" not in raw_df.columns: raw_df["Secondary_Role"] = "None"
if "Secondary_EFT" not in raw_df.columns: raw_df["Secondary_EFT"] = 0.0
if "No_AM_DOW" not in raw_df.columns: raw_df["No_AM_DOW"] = ""
if "No_PM_DOW" not in raw_df.columns: raw_df["No_PM_DOW"] = ""
if "Preferred_Shift" not in raw_df.columns: raw_df["Preferred_Shift"] = "None"
if "Study_Leave_Days" not in raw_df.columns: raw_df["Study_Leave_Days"] = ""
if "External_Working_Days" not in raw_df.columns: raw_df["External_Working_Days"] = ""

# 2. Force Checkboxes to Boolean
bool_cols = ["Night_Pool", "Allow_Fragmented_Shifts", "Entire_Roster_Leave"]
for col in bool_cols:
    if col in raw_df.columns:
        raw_df[col] = raw_df[col].astype(str).str.lower().isin(['true', '1', 't', 'yes'])

# 3. Force Numbers to Float/Int
raw_df["EFT"] = pd.to_numeric(raw_df["EFT"], errors='coerce').fillna(1.0)
raw_df["Secondary_EFT"] = pd.to_numeric(raw_df["Secondary_EFT"], errors='coerce').fillna(0.0)
raw_df["Prior_Consecutive_Days"] = pd.to_numeric(raw_df["Prior_Consecutive_Days"], errors='coerce').fillna(0).astype(int)

# 4. Force Text to String
for col in raw_df.columns:
    if col not in bool_cols and col not in ["EFT", "Secondary_EFT", "Prior_Consecutive_Days"]:
        raw_df[col] = raw_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')

# Update memory with the scrubbed version
st.session_state.staff_df = raw_df

st.subheader("Staff Pool Management")
edited_df = st.data_editor(
    st.session_state.staff_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Night_Pool": st.column_config.CheckboxColumn("Night Pool?", default=False),
        "Approved_Leave_Days": st.column_config.TextColumn("Approved Leave (e.g. 1, 14)"),
        "Requested_RDOs": st.column_config.TextColumn("Requested RDOs (e.g. 2, 5)"),
        "EFT": st.column_config.NumberColumn("EFT", min_value=0.1, max_value=1.0, step=0.1),
        "Prior_Consecutive_Days": st.column_config.NumberColumn("Prior Consec", min_value=0, max_value=6, step=1),
        "Last_Shift_Type": st.column_config.SelectboxColumn("Last Shift Type", options=["None", "AM", "PM", "Night"]),
        "Unavailable_DOW": st.column_config.TextColumn("Unavailable Days (e.g., Monday, Tuesday)"),
        "Allow_Fragmented_Shifts": st.column_config.CheckboxColumn("Allow Fragmented Shifts", default=False),
        "Entire_Roster_Leave": st.column_config.CheckboxColumn("On Leave (Entire Roster)", default=False),
        "Secondary_Role": st.column_config.SelectboxColumn("Secondary Role", options=["None", "ANUM", "RN (In Charge)", "RN", "EN/Learner"]),
        "Secondary_EFT": st.column_config.NumberColumn("Secondary EFT", min_value=0.0, max_value=1.0, step=0.1),
        "No_AM_DOW": st.column_config.TextColumn("No AM Days (e.g. Monday)"),
        "No_PM_DOW": st.column_config.TextColumn("No PM Days (e.g. Tuesday, Friday)"),
        "Preferred_Shift": st.column_config.SelectboxColumn("Preferred Shift", options=["None", "AM", "PM", "Night"]),
        "Study_Leave_Days": st.column_config.TextColumn("Study Leave (e.g. 5, 12)"),
        "External_Working_Days": st.column_config.TextColumn("External/CNM Days (e.g. 3, 4, 5)")
    }
)

if st.button("Save Staff Profiles"):
    edited_df.to_csv("staff_profiles.csv", index=False)
    st.success("Staff profiles saved successfully! They will load automatically next time.")

if st.button("Generate Roster", type="primary"):
    with st.spinner("Calculating optimal shifts (this may take a few seconds)..."):
        result_df = solve_roster(edited_df, roster_days, start_date)
        
        if result_df is not None:
            st.success("Roster Generated Successfully!")
            st.dataframe(result_df, use_container_width=True)
            
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Roster as CSV",
                data=csv,
                file_name='ward_roster.csv',
                mime='text/csv',
            )
        else:
            st.error("No feasible roster could be generated. Try adjusting leave or adding more staff to the Night Pool.")
