import streamlit as st
import pandas as pd
import os
import datetime
import holidays
import math
from ortools.sat.python import cp_model
from streamlit_gsheets import GSheetsConnection

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
            "ID": f"Staff_{i:02d}", "Role": role, "EFT": 1.0,
            "Gender": "Female" if i % 3 != 0 else "Male",
            "Night_Pool": True if i >= 32 else False,
            "Prior_Consecutive_Days": 0, "Last_Shift_Type": "None",
            "Approved_Leave_Days": "", "Requested_RDOs": "", "Unavailable_DOW": "", 
            "Allow_Fragmented_Shifts": False, "Entire_Roster_Leave": False,
            "Secondary_Role": "None", "Secondary_EFT": 0.0,
            "No_AM_DOW": "", "No_PM_DOW": "", "Preferred_Shift": "None",
            "Study_Leave_Days": "", "External_Working_Days": "",
            "W1_Preferences": "", "W2_Preferences": ""
        })
    return pd.DataFrame(staff_data)

# ----------------------------------------
# 2. THE SOLVER ENGINE 
# ----------------------------------------
def solve_roster(df, num_days, start_date, debug_flags):
    df = df.reset_index(drop=True)
    model = cp_model.CpModel()
    
    num_staff = len(df)
    all_staff = range(num_staff)
    all_days = range(num_days)
    shift_names = ["AM", "PM", "Night"]
    
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
                    
    for n in all_staff:
        for d in all_days:
            model.AddAtMostOne(roster[(n, d, s)] for s in range(3))
            
    females = df.index[df['Gender'] == 'Female'].tolist()

    end_date = start_date + datetime.timedelta(days=num_days)
    vic_holidays = holidays.AU(subdiv='VIC', years=[start_date.year, end_date.year])
    
    staffing_level_penalties = []

    # --- FULLY ELASTIC PRIORITIZED COVERAGE ---
    for d in all_days:
        current_date = start_date + datetime.timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        is_monday = current_date.weekday() == 0
        is_friday = current_date.weekday() == 4
        is_pub_hol = current_date in vic_holidays
        
        am_sum = sum(roster[(n, d, 0)] for n in all_staff)
        pm_sum = sum(roster[(n, d, 1)] for n in all_staff)
        night_sum = sum(roster[(n, d, 2)] for n in all_staff)
        
        if not debug_flags.get("ignore_coverage"):
            # 1. AM SHIFTS: Elastic (Range 4 to 5)
            model.Add(am_sum >= 4)
            model.Add(am_sum <= 5)
            missing_am = model.NewIntVar(0, 1, f'missing_am_d{d}')
            model.Add(missing_am == 5 - am_sum)
            
            if is_weekend or is_monday or is_pub_hol:
                staffing_level_penalties.append(missing_am * 50) # High priority full staffing
            else:
                staffing_level_penalties.append(missing_am * 15) # Mid-week flexible drop to 4
                
            # 2. PM SHIFTS: Elastic (Range 4 to 5)
            model.Add(pm_sum >= 4)
            model.Add(pm_sum <= 5)
            missing_pm = model.NewIntVar(0, 1, f'missing_pm_d{d}')
            model.Add(missing_pm == 5 - pm_sum)
            
            if is_weekend or is_friday or is_pub_hol:
                staffing_level_penalties.append(missing_pm * 50) # High priority full staffing
            else:
                staffing_level_penalties.append(missing_pm * 15) # Mid-week flexible drop to 4
                
            # 3. NIGHT SHIFTS: Elastic (Range 3 to 4)
            model.Add(night_sum >= 3)
            model.Add(night_sum <= 4)
            missing_night = model.NewIntVar(0, 1, f'missing_night_d{d}')
            model.Add(missing_night == 4 - night_sum)
            
            if is_weekend or is_pub_hol:
                staffing_level_penalties.append(missing_night * 50) # Weekend nights priority target 4
            else:
                staffing_level_penalties.append(missing_night * 20) # Weekday nights flexible drop to 3
    # ------------------------------------------
            
    leadership_penalties = []
    for d in all_days:
        for s in range(3):
            if not debug_flags.get("ignore_leadership"):
                model.Add(sum(roster[(n, d, s)] for n in females if (n, d, s) in roster) >= 1)
            
            anum_sum = (
                sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "ANUM" and (n, d, s) in role_primary) + 
                sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "ANUM" and (n, d, s) in role_secondary)
            )
            rn_in_charge_sum = (
                sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "RN (In Charge)" and (n, d, s) in role_primary) + 
                sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "RN (In Charge)" and (n, d, s) in role_secondary)
            )
            
            if not debug_flags.get("ignore_leadership"):
                model.Add(anum_sum <= 1)
                model.Add(anum_sum + rn_in_charge_sum >= 1)
                missing_leader = model.NewIntVar(0, 1, f'missing_leader_d{d}_s{s}')
                model.Add(missing_leader == 2 - (anum_sum + rn_in_charge_sum))
                leadership_penalties.append(missing_leader * 60)
            
            missing_anum = model.NewIntVar(0, 1, f'missing_anum_d{d}_s{s}')
            model.Add(missing_anum == 1 - anum_sum)
            leadership_penalties.append(missing_anum * 15)
            
    if not debug_flags.get("ignore_night_pool"):
        for n in all_staff:
            is_night_pool = df.iloc[n]["Night_Pool"]
            for d in all_days:
                if is_night_pool:
                    model.Add(roster[(n, d, 0)] == 0)
                    model.Add(roster[(n, d, 1)] == 0)
                else:
                    model.Add(roster[(n, d, 2)] == 0)

    for n in all_staff:
        leave_str = str(df.iloc[n]["Approved_Leave_Days"])
        def parse_days(day_string):
            if pd.isna(day_string) or day_string.strip() == "" or day_string.strip() == "nan": return []
            try: return [int(x.strip()) - 1 for x in day_string.split(",")]
            except ValueError: return []
                
        study_days = set(parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))))
        ext_days = set(parse_days(str(df.iloc[n].get("External_Working_Days", ""))))
        unavailable_days = set(parse_days(leave_str)) | study_days | ext_days
                
        if not debug_flags.get("ignore_leave"):
            for d in unavailable_days:
                if 0 <= d < num_days:
                    for s in range(3):
                        if (n, d, s) in roster: model.Add(roster[(n, d, s)] == 0)

        dow_str = str(df.iloc[n]["Unavailable_DOW"]).lower()
        if dow_str and dow_str != "nan" and not debug_flags.get("ignore_leave"):
            for d in all_days:
                current_date = start_date + datetime.timedelta(days=d)
                if current_date.strftime("%A").lower() in dow_str:
                    for s in range(3): model.Add(roster[(n, d, s)] == 0)

    for n in all_staff:
        is_fully_absent = df.iloc[n].get("Entire_Roster_Leave", False)
        has_secondary = df.iloc[n]["Secondary_Role"] != "None" and df.iloc[n]["Secondary_EFT"] > 0
        
        if is_fully_absent:
            final_shift_target = 0
            for d in range(num_days):
                for s in range(3):
                    if (n, d, s) in roster: model.Add(roster[(n, d, s)] == 0)
        else:
            primary_eft = df.iloc[n]["EFT"]
            secondary_eft = df.iloc[n]["Secondary_EFT"] if has_secondary else 0.0
            total_eft = primary_eft + secondary_eft
            is_night_pool = df.iloc[n]["Night_Pool"]
            shift_length = 10.0 if is_night_pool else 8.0
            
            base_shifts = (total_eft * 80.0 * (num_days / 14.0)) / shift_length
            valid_leave_days = len([d for d in parse_days(str(df.iloc[n]["Approved_Leave_Days"])) if 0 <= d < num_days])
            study_count = len([d for d in parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))) if 0 <= d < num_days])
            
            fraction_present = max(0.0, (num_days - valid_leave_days - study_count) / num_days)
            final_shift_target = round(base_shifts * fraction_present)
            
            if has_secondary and not debug_flags.get("ignore_eft"):
                sec_base = (secondary_eft * 80.0 * (num_days / 14.0)) / shift_length
                sec_shift_target = round(sec_base * fraction_present)
                sec_shift_target = min(sec_shift_target, final_shift_target)
                model.Add(sum(role_secondary[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in role_secondary) == sec_shift_target)
            
        if not debug_flags.get("ignore_eft"):
            model.Add(sum(roster[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in roster) == final_shift_target)

    fatigue_penalties = []
    if not debug_flags.get("ignore_fatigue"):
        for n in all_staff:
            raw_prior = df.iloc[n]["Prior_Consecutive_Days"]
            prior_days = int(raw_prior) if pd.notna(raw_prior) else 0
            last_shift = str(df.iloc[n]["Last_Shift_Type"]).strip()
            is_night_pool = df.iloc[n]["Night_Pool"]
            
            study_days = parse_days(str(df.iloc[n].get("Study_Leave_Days", "")))
            ext_days = parse_days(str(df.iloc[n].get("External_Working_Days", "")))
            virtual_days = set(study_days + ext_days)
            
            for d in range(num_days - 1):
                model.AddImplication(roster[(n, d, 2)], roster[(n, d+1, 0)].Not())
                model.AddImplication(roster[(n, d, 2)], roster[(n, d+1, 1)].Not())
                
            if last_shift == "PM": model.Add(roster[(n, 0, 0)] == 0) 
            elif last_shift == "Night":
                model.Add(roster[(n, 0, 0)] == 0)
                model.Add(roster[(n, 0, 1)] == 0)
                
            shift_length = 10.0 if is_night_pool else 8.0
            shifts_per_fortnight = math.ceil((df.iloc[n]["EFT"] * 80.0) / shift_length)
            max_consec = int((shifts_per_fortnight / 2) + 1)
            
            for d in range(num_days - max_consec):
                ward_shifts = sum(roster[(n, d+w, s)] for w in range(max_consec + 1) for s in range(3) if (n, d+w, s) in roster)
                virtual_shifts = sum(1 for w in range(max_consec + 1) if (d+w) in virtual_days)
                model.Add(ward_shifts + virtual_shifts <= max_consec)
                
            for d in range(num_days - 4):
                model.Add(sum(roster[(n, d+w, 2)] for w in range(5)) <= 4)
                
            if prior_days > 0:
                days_to_check = (max_consec + 1) - prior_days
                remaining_allowed = max(0, max_consec - prior_days)
                if days_to_check > 0 and num_days >= days_to_check:
                    ward_boundary = sum(roster[(n, w, s)] for w in range(days_to_check) for s in range(3) if (n, w, s) in roster)
                    virtual_boundary = sum(1 for w in range(days_to_check) if w in virtual_days)
                    model.Add(ward_boundary + virtual_boundary <= remaining_allowed)

            if last_shift == "Night" and not is_night_pool:
                if num_days > 0:
                    for s in range(3): model.Add(roster[(n, 0, s)] == 0)
                if num_days > 1:
                    for s in range(3): model.Add(roster[(n, 1, s)] == 0)
                        
            allow_fragmented = df.iloc[n]["Allow_Fragmented_Shifts"]
            if not allow_fragmented:
                internal_starts = []
                for d in range(1, num_days):
                    w_yest = sum(roster[(n, d-1, s)] for s in range(3) if (n, d-1, s) in roster) + (1 if d-1 in virtual_days else 0)
                    w_tod = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                    is_start = model.NewBoolVar(f'internal_start_{n}_day_{d}')
                    model.Add(w_tod - w_yest <= is_start)
                    internal_starts.append(is_start)
                
                model.Add(sum(internal_starts) <= 3)
                extra_blocks = model.NewIntVar(0, 14, f'extra_blocks_{n}')
                model.Add(sum(internal_starts) - 1 <= extra_blocks)
                fatigue_penalties.append(extra_blocks * 50)
                    
            if not allow_fragmented:
                for d in range(num_days - 2):
                    w0 = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                    w1 = sum(roster[(n, d+1, s)] for s in range(3) if (n, d+1, s) in roster) + (1 if d+1 in virtual_days else 0)
                    w2 = sum(roster[(n, d+2, s)] for s in range(3) if (n, d+2, s) in roster) + (1 if d+2 in virtual_days else 0)
                    model.Add(w0 - w1 + w2 <= 1)
                    model.Add(w1 - w0 - w2 <= 0)
                    
                w0 = sum(roster[(n, 0, s)] for s in range(3) if (n, 0, s) in roster) + (1 if 0 in virtual_days else 0)
                w1 = sum(roster[(n, 1, s)] for s in range(3) if (n, 1, s) in roster) + (1 if 1 in virtual_days else 0)
                if prior_days == 0 and num_days > 1: model.Add(w0 - w1 <= 0)
                if prior_days > 0 and num_days > 1: model.Add(w1 <= w0)

    shift_mix_penalties = []
    granular_penalties = []
    penalties = []
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    shift_map = {"am": 0, "pm": 1, "night": 2}

    for n in all_staff:
        w1_str = str(df.iloc[n].get("W1_Preferences", "")).lower()
        w2_str = str(df.iloc[n].get("W2_Preferences", "")).lower()

        for week_offset, pref_str in [(0, w1_str), (7, w2_str)]:
            if not pref_str or pref_str == 'nan': continue
            for req in [x.strip() for x in pref_str.split(",")]:
                parts = req.split()
                if len(parts) >= 2 and parts[0][:3] in day_map and parts[1] in shift_map:
                    target_day = day_map[parts[0][:3]] + week_offset
                    target_shift = shift_map[parts[1]]
                    if target_day < num_days and (n, target_day, target_shift) in roster:
                        granular_penalties.append(-10 * roster[(n, target_day, target_shift)])

        if not df.iloc[n].get("Night_Pool", False):
            pref = str(df.iloc[n].get("Preferred_Shift", "None")).strip()
            am_shifts = sum(roster[(n, d, 0)] for d in range(num_days) if (n, d, 0) in roster)
            pm_shifts = sum(roster[(n, d, 1)] for d in range(num_days) if (n, d, 1) in roster)
            dev = model.NewIntVar(0, 100, f'shift_balance_dev_{n}')
            if pref.upper() == "PM":
                model.Add(dev >= pm_shifts - (3 * am_shifts))
                model.Add(dev >= -(pm_shifts - (3 * am_shifts)))
                shift_mix_penalties.append(dev * 2) 
            elif pref.upper() == "AM":
                model.Add(dev >= am_shifts - (3 * pm_shifts))
                model.Add(dev >= -(am_shifts - (3 * pm_shifts)))
                shift_mix_penalties.append(dev * 2)
            else:
                model.Add(dev >= am_shifts - pm_shifts)
                model.Add(dev >= -(am_shifts - pm_shifts))
                shift_mix_penalties.append(dev * 4) 

        rdo_str = str(df.iloc[n]["Requested_RDOs"])
        if rdo_str and rdo_str.lower() != 'nan':
            for val in rdo_str.split(","):
                try:
                    d = int(val.strip()) - 1
                    if 0 <= d < num_days:
                        for s in range(3):
                            if (n, d, s) in roster: penalties.append(roster[(n, d, s)] * 30) 
                except ValueError: pass

    model.Minimize(
        sum(shift_mix_penalties) + sum(penalties) + sum(granular_penalties) + 
        sum(leadership_penalties) + sum(staffing_level_penalties) + sum(fatigue_penalties)
    )              
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0 
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        acting_leaders = set()
        for d in all_days:
            for s in range(3):
                anum_present = False
                rn_in_charges = []
                for n in all_staff:
                    if (n, d, s) in roster and solver.Value(roster[(n, d, s)]) == 1:
                        active_role = df.iloc[n]["Role"]
                        if (n, d, s) in role_secondary and solver.Value(role_secondary[(n, d, s)]) == 1:
                            active_role = df.iloc[n]["Secondary_Role"]
                        if active_role == "ANUM": anum_present = True
                        elif active_role == "RN (In Charge)": rn_in_charges.append(n)
                if not anum_present and len(rn_in_charges) > 0:
                    acting_leaders.add((rn_in_charges[0], d, s))

        roster_output = []
        day_headers = []
        
        # Track shift totals for the summary rows
        am_totals = []
        pm_totals = []
        night_totals = []
        
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            header = current_date.strftime("%a %d-%b") 
            if current_date in vic_holidays: header += " (PH)"
            day_headers.append(header)
            
            # Count actual staff assigned
            a_sum = sum(1 for n in all_staff if (n, d, 0) in roster and solver.Value(roster[(n, d, 0)]) == 1)
            p_sum = sum(1 for n in all_staff if (n, d, 1) in roster and solver.Value(roster[(n, d, 1)]) == 1)
            n_sum = sum(1 for n in all_staff if (n, d, 2) in roster and solver.Value(roster[(n, d, 2)]) == 1)
            
            am_totals.append(a_sum)
            pm_totals.append(p_sum)
            night_totals.append(n_sum)
            
        for n in all_staff:
            staff_row = {"Staff ID": df.iloc[n]["ID"], "Role": df.iloc[n]["Role"]}
            for d in all_days:
                assigned_shift = "" 
                study_days = parse_days(str(df.iloc[n].get("Study_Leave_Days", "")))
                ext_days = parse_days(str(df.iloc[n].get("External_Working_Days", "")))
                
                if d in study_days: assigned_shift = "Study Leave"
                elif d in ext_days: assigned_shift = "External/CNM"
                else:
                    for s in range(3):
                        if (n, d, s) in roster and solver.Value(roster[(n, d, s)]) == 1:
                            if (n, d, s) in role_secondary and solver.Value(role_secondary[(n, d, s)]) == 1:
                                assigned_shift = f"{shift_names[s]} ({df.iloc[n]['Secondary_Role']})"
                            else: assigned_shift = shift_names[s]
                            if (n, d, s) in acting_leaders: assigned_shift += " (Shift Leader)"
                staff_row[day_headers[d]] = assigned_shift
            roster_output.append(staff_row)
            
        result_df = pd.DataFrame(roster_output)
        
        # --- ADD AGENCY / SHORTFALL SUMMARY ROWS ---
        agency_am = []
        agency_pm = []
        agency_night = []
        
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            is_weekend = current_date.weekday() >= 5
            is_monday = current_date.weekday() == 0
            is_friday = current_date.weekday() == 4
            is_pub_hol = current_date in vic_holidays
            
            # Target AM is 5
            target_am = 5
            short_am = max(0, target_am - am_totals[d])
            agency_am.append(f"Short {short_am} (Agency)" if short_am > 0 else "Fully Staffed")
            
            # Target PM is 5
            target_pm = 5
            short_pm = max(0, target_pm - pm_totals[d])
            agency_pm.append(f"Short {short_pm} (Agency)" if short_pm > 0 else "Fully Staffed")
            
            # Target Night is 4 on weekends/PH, 3 or 4 on weekdays
            target_night = 4 if (is_weekend or is_pub_hol) else 3
            short_night = max(0, target_night - night_totals[d])
            agency_night.append(f"Short {short_night} (Agency)" if short_night > 0 else "Fully Staffed")
            
        summary_row_am = {"Staff ID": "🚨 AM Shortfall", "Role": "AGENCY CHECK"}
        summary_row_pm = {"Staff ID": "🚨 PM Shortfall", "Role": "AGENCY CHECK"}
        summary_row_night = {"Staff ID": "🚨 Night Shortfall", "Role": "AGENCY CHECK"}
        
        for idx, h in enumerate(day_headers):
            summary_row_am[h] = agency_am[idx]
            summary_row_pm[h] = agency_pm[idx]
            summary_row_night[h] = agency_night[idx]
            
        summary_df = pd.DataFrame([summary_row_am, summary_row_pm, summary_row_night])
        final_output_df = pd.concat([result_df, summary_df], ignore_index=True)
        # -------------------------------------------
        
        return final_output_df
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
    
    st.markdown("---")
    st.header("🛠️ Constraint Troubleshooter")
    debug_flags = {
        "ignore_coverage": st.checkbox("Ignore Minimum Staff Levels"),
        "ignore_leadership": st.checkbox("Ignore Leadership Minimums"),
        "ignore_fatigue": st.checkbox("Ignore Fatigue & Rest Rules"),
        "ignore_leave": st.checkbox("Ignore Approved Leave"),
        "ignore_eft": st.checkbox("Ignore Contract EFT Targets"),
        "ignore_night_pool": st.checkbox("Ignore Night Pool Separation")
    }

conn = st.connection("gsheets", type=GSheetsConnection)
SHEET_URL = "Ward Staff Profiles" 

if "staff_df" not in st.session_state:
    try:
        pulled_df = conn.read(spreadsheet=SHEET_URL, ttl=0).dropna(how="all")
        st.session_state.staff_df = pulled_df
    except:
        st.session_state.staff_df = load_initial_staff()

raw_df = st.session_state.staff_df.copy()
raw_df.columns = raw_df.columns.str.strip()

missing_columns = {
    "Secondary_Role": "None", "Secondary_EFT": 0.0, "No_AM_DOW": "", "No_PM_DOW": "",
    "Preferred_Shift": "None", "Study_Leave_Days": "", "External_Working_Days": "",
    "W1_Preferences": "", "W2_Preferences": ""
}
for col, default_val in missing_columns.items():
    if col not in raw_df.columns: raw_df[col] = default_val

def make_boolean(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    val_str = str(val).strip().upper()
    if val_str in ['TRUE', '1', '1.0', 'T', 'YES', 'Y']: return True
    return False

bool_cols = ["Night_Pool", "Allow_Fragmented_Shifts", "Entire_Roster_Leave"]
for col in bool_cols:
    if col not in raw_df.columns: raw_df[col] = False
    raw_df[col] = raw_df[col].apply(make_boolean).astype(bool)

raw_df["EFT"] = pd.to_numeric(raw_df["EFT"], errors='coerce').fillna(1.0)
raw_df["Secondary_EFT"] = pd.to_numeric(raw_df["Secondary_EFT"], errors='coerce').fillna(0.0)
raw_df["Prior_Consecutive_Days"] = pd.to_numeric(raw_df["Prior_Consecutive_Days"], errors='coerce').fillna(0).astype(int)

for col in raw_df.columns:
    if col not in bool_cols and col not in ["EFT", "Secondary_EFT", "Prior_Consecutive_Days"]:
        raw_df[col] = raw_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')

st.session_state.staff_df = raw_df

st.subheader("Staff Pool Management")
edited_df = st.data_editor(
    st.session_state.staff_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "Night_Pool": st.column_config.CheckboxColumn("Night Pool?", default=False),
        "Approved_Leave_Days": st.column_config.TextColumn("Approved Leave"),
        "Requested_RDOs": st.column_config.TextColumn("Requested RDOs"),
        "EFT": st.column_config.NumberColumn("EFT", min_value=0.1, max_value=1.0, step=0.1),
        "Prior_Consecutive_Days": st.column_config.NumberColumn("Prior Consec", min_value=0, max_value=6, step=1),
        "Last_Shift_Type": st.column_config.SelectboxColumn("Last Shift Type", options=["None", "AM", "PM", "Night"]),
        "Unavailable_DOW": st.column_config.TextColumn("Unavailable Days"),
        "Allow_Fragmented_Shifts": st.column_config.CheckboxColumn("Allow Fragmented Shifts", default=False),
        "Entire_Roster_Leave": st.column_config.CheckboxColumn("On Leave (Entire Roster)", default=False),
        "Secondary_Role": st.column_config.SelectboxColumn("Secondary Role", options=["None", "ANUM", "RN (In Charge)", "RN", "EN/Learner"]),
        "Secondary_EFT": st.column_config.NumberColumn("Secondary EFT", min_value=0.0, max_value=1.0, step=0.1),
        "No_AM_DOW": st.column_config.TextColumn("No AM Days"),
        "No_PM_DOW": st.column_config.TextColumn("No PM Days"),
        "Preferred_Shift": st.column_config.SelectboxColumn("Preferred Shift", options=["None", "AM", "PM", "Night"]),
        "Study_Leave_Days": st.column_config.TextColumn("Study Leave"),
        "External_Working_Days": st.column_config.TextColumn("External/CNM Days"),
        "W1_Preferences": st.column_config.TextColumn("W1 Prefs"),
        "W2_Preferences": st.column_config.TextColumn("W2 Prefs")
    }
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Save Staff Profiles"):
        with st.spinner("Saving securely..."):
            save_df = edited_df.copy()
            for col in bool_cols:
                if col in save_df.columns: save_df[col] = save_df[col].astype(str).str.upper()
            conn.update(spreadsheet=SHEET_URL, data=save_df)
            st.session_state.staff_df = edited_df
        st.success("Saved to Cloud!")

with col2:
    if st.button("🔄 Sync from Google Sheet"):
        if "staff_df" in st.session_state: del st.session_state["staff_df"]
        st.rerun()

if st.button("Generate Roster", type="primary"):
    with st.spinner("Calculating optimal shifts..."):
        result_df = solve_roster(edited_df, roster_days, start_date, debug_flags)
        if result_df is not None:
            st.success("Roster Generated Successfully!")
            st.dataframe(result_df, use_container_width=True)
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Roster as CSV", data=csv, file_name='ward_roster.csv', mime='text/csv')
        else:
            st.error("No feasible roster could be generated. Check your leave balances or turn on Troubleshooter flags.")
