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
            "External_Working_Days": "",
            "W1_Preferences": "",
            "W2_Preferences": ""
        })
        
    return pd.DataFrame(staff_data)

# ----------------------------------------
# 2. THE SOLVER ENGINE (PHASE 2 & 3)
# ----------------------------------------
def solve_roster(df, num_days, start_date, debug_flags):
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
    females = df.index[df['Gender'] == 'Female'].tolist()

    # CONSTRAINT: Vacancy Tolerances & Calendar Logic
    end_date = start_date + datetime.timedelta(days=num_days)
    vic_holidays = holidays.AU(subdiv='VIC', years=[start_date.year, end_date.year])
    
    staffing_level_penalties = []

    for d in all_days:
        current_date = start_date + datetime.timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        is_friday = current_date.weekday() == 4
        is_pub_hol = current_date in vic_holidays
        
        # 1. AM SHIFT (s=0)
        if not debug_flags.get("ignore_coverage"):
            model.Add(sum(roster[(n, d, 0)] for n in all_staff) == 5)
        
        # 2. PM SHIFT (s=1)
        pm_sum = sum(roster[(n, d, 1)] for n in all_staff)
        if not debug_flags.get("ignore_coverage"):
            if is_weekend or is_friday or is_pub_hol:
                model.Add(pm_sum == 5)
            else:
                model.Add(pm_sum >= 4)
                model.Add(pm_sum <= 5)
            
        # 3. NIGHT SHIFT (s=2)
        night_sum = sum(roster[(n, d, 2)] for n in all_staff)
        if not debug_flags.get("ignore_coverage"):
            model.Add(night_sum >= 3)
            model.Add(night_sum <= 4)
            
            if is_weekend or is_pub_hol:
                missing_weekend_night = model.NewIntVar(0, 1, f'missing_wknd_night_d{d}')
                model.Add(missing_weekend_night == 4 - night_sum)
                staffing_level_penalties.append(missing_weekend_night * 10)
            
    # CONSTRAINT: Skill Mix & Demographics
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
                model.Add(anum_sum + rn_in_charge_sum >= 2)
            
            missing_anum = model.NewIntVar(0, 1, f'missing_anum_d{d}_s{s}')
            model.Add(missing_anum == 1 - anum_sum)
            leadership_penalties.append(missing_anum * 15)
            
    # CONSTRAINT: Night Pool vs Day Pool Separation
    if not debug_flags.get("ignore_night_pool"):
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
        
        unavailable_days = set(parse_days(leave_str)) | study_days | ext_days
                
        if not debug_flags.get("ignore_leave"):
            for d in unavailable_days:
                if 0 <= d < num_days:
                    for s in range(3):
                        if (n, d, s) in roster:
                            model.Add(roster[(n, d, s)] == 0)

        dow_str = str(df.iloc[n]["Unavailable_DOW"]).lower()
        if dow_str and dow_str != "nan" and not debug_flags.get("ignore_leave"):
            for d in all_days:
                current_date = start_date + datetime.timedelta(days=d)
                day_name = current_date.strftime("%A").lower()
                if day_name in dow_str:
                    for s in range(3):
                        model.Add(roster[(n, d, s)] == 0)

        no_am_str = str(df.iloc[n].get("No_AM_DOW", "")).lower()
        no_pm_str = str(df.iloc[n].get("No_PM_DOW", "")).lower()
        
        if not debug_flags.get("ignore_leave"):
            for d in all_days:
                current_date = start_date + datetime.timedelta(days=d)
                day_name = current_date.strftime("%A").lower()
                
                if no_am_str and no_am_str != "nan" and day_name in no_am_str:
                    if (n, d, 0) in roster:
                        model.Add(roster[(n, d, 0)] == 0)
                        
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
                        
            leave_hours_deduction = valid_leave_days * shift_length
            
            study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
            study_count = 0
            if study_str and study_str.lower() != 'nan':
                for val in study_str.split(","):
                    try:
                        if 0 <= (int(val.strip()) - 1) < num_days: study_count += 1
                    except ValueError: pass
                    
            study_hours_deduction = study_count * 8.0
            
            adjusted_target_hours = max(0, target_hours - leave_hours_deduction - study_hours_deduction)
            final_shift_target = math.ceil(adjusted_target_hours / shift_length)
            
            if has_secondary:
                sec_target_hours = secondary_eft * 80.0 * (num_days / 14.0)
                sec_shift_target = math.ceil(sec_target_hours / shift_length)
                sec_shift_target = min(sec_shift_target, final_shift_target)
                model.Add(sum(role_secondary[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in role_secondary) == sec_shift_target)
            
        if not debug_flags.get("ignore_eft"):
            model.Add(sum(roster[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in roster) == final_shift_target)

    # CONSTRAINT: Rest and Fatigue Rules
    fatigue_penalties = []
    
    if not debug_flags.get("ignore_fatigue"):
        for n in all_staff:
            raw_prior = df.iloc[n]["Prior_Consecutive_Days"]
            prior_days = int(raw_prior) if pd.notna(raw_prior) else 0
            last_shift = str(df.iloc[n]["Last_Shift_Type"]).strip()
            is_night_pool = df.iloc[n]["Night_Pool"]
            
            study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
            study_days = [int(x.strip()) - 1 for x in study_str.split(",") if x.strip().isdigit()] if study_str and study_str.lower() != 'nan' else []
            
            ext_str = str(df.iloc[n].get("External_Working_Days", ""))
            ext_days = [int(x.strip()) - 1 for x in ext_str.split(",") if x.strip().isdigit()] if ext_str and ext_str.lower() != 'nan' else []
            
            virtual_days = set(study_days + ext_days)
            
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
                
            # --- UPDATED: Dynamic Maximum Consecutive Shifts (SOFT CONSTRAINT) ---
            shift_length = 10.0 if is_night_pool else 8.0
            shifts_per_fortnight = math.ceil((df.iloc[n]["EFT"] * 80.0) / shift_length)
            max_consec = int((shifts_per_fortnight / 2) + 1)
            
            for d in range(num_days - max_consec):
                ward_shifts = sum(roster[(n, d+w, s)] for w in range(max_consec + 1) for s in range(3) if (n, d+w, s) in roster)
                virtual_shifts = sum(1 for w in range(max_consec + 1) if (d+w) in virtual_days)
                
                # Make this a soft constraint. If RDOs force them to work too many days, allow it with a heavy penalty!
                over_limit = model.NewIntVar(0, 14, f'over_consec_{n}_{d}')
                model.Add(ward_shifts + virtual_shifts - max_consec <= over_limit)
                fatigue_penalties.append(over_limit * 50)
                
            # Max 4 Night Shifts in a row safety net (Kept as a HARD constraint for safety)
            for d in range(num_days - 4):
                model.Add(sum(roster[(n, d+w, 2)] for w in range(5)) <= 4)
                
            # Boundary Consecutive Shifts with dynamic math (SOFT CONSTRAINT)
            if prior_days > 0:
                days_to_check = (max_consec + 1) - prior_days
                remaining_allowed = max(0, max_consec - prior_days)
                if days_to_check > 0 and num_days >= days_to_check:
                    ward_boundary = sum(roster[(n, w, s)] for w in range(days_to_check) for s in range(3) if (n, w, s) in roster)
                    virtual_boundary = sum(1 for w in range(days_to_check) if w in virtual_days)
                    
                    over_boundary = model.NewIntVar(0, 14, f'over_bound_{n}')
                    model.Add(ward_boundary + virtual_boundary - remaining_allowed <= over_boundary)
                    fatigue_penalties.append(over_boundary * 50)

            # 3. The 48-Hour Transition Rule

            # 3. The 48-Hour Transition Rule
            if last_shift == "Night" and not is_night_pool:
                if num_days > 0:
                    for s in range(3):
                        model.Add(roster[(n, 0, s)] == 0)
                if num_days > 1:
                    for s in range(3):
                        model.Add(roster[(n, 1, s)] == 0)
                        
            # --- UPDATED: Maximum 3 Blocks of Shifts Per Fortnight ---
            allow_fragmented = df.iloc[n]["Allow_Fragmented_Shifts"]
            
            if not allow_fragmented:
                block_starts = []
                
                for d in range(1, num_days):
                    w_yest = sum(roster[(n, d-1, s)] for s in range(3) if (n, d-1, s) in roster) + (1 if d-1 in virtual_days else 0)
                    w_tod = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                    
                    is_start = model.NewBoolVar(f'block_start_staff_{n}_day_{d}')
                    model.Add(w_tod - w_yest <= is_start)
                    block_starts.append(is_start)
                    
                w0 = sum(roster[(n, 0, s)] for s in range(3) if (n, 0, s) in roster) + (1 if 0 in virtual_days else 0)
                is_start_0 = model.NewBoolVar(f'block_start_staff_{n}_day_0')
                if prior_days == 0:
                    model.Add(w0 <= is_start_0)
                else:
                    model.Add(0 <= is_start_0)
                    
                block_starts.append(is_start_0)
                
                # Hard Constraint: Absolute maximum of 3 blocks
                model.Add(sum(block_starts) <= 3)
                
                # Soft Constraint: Ideally 2 blocks. Penalty if 3.
                third_block_active = model.NewBoolVar(f'third_block_staff_{n}')
                model.Add(sum(block_starts) - 2 <= third_block_active)
                fatigue_penalties.append(third_block_active * 25)
                    
            # 5. Minimum 2 Days Off AND Minimum 2 Working Days (Ban all fragmentation)
            if not allow_fragmented:
                for d in range(num_days - 2):
                    w0 = sum(roster[(n, d, s)] for s in range(3) if (n, d, s) in roster) + (1 if d in virtual_days else 0)
                    w1 = sum(roster[(n, d+1, s)] for s in range(3) if (n, d+1, s) in roster) + (1 if d+1 in virtual_days else 0)
                    w2 = sum(roster[(n, d+2, s)] for s in range(3) if (n, d+2, s) in roster) + (1 if d+2 in virtual_days else 0)
                    
                    model.Add(w0 - w1 + w2 <= 1)
                    model.Add(w1 - w0 - w2 <= 0)
                    
                w0 = sum(roster[(n, 0, s)] for s in range(3) if (n, 0, s) in roster) + (1 if 0 in virtual_days else 0)
                w1 = sum(roster[(n, 1, s)] for s in range(3) if (n, 1, s) in roster) + (1 if 1 in virtual_days else 0)
                
                if prior_days == 0 and num_days > 1:
                    model.Add(w0 - w1 <= 0)
                    
                if prior_days > 0 and num_days > 1:
                    model.Add(w1 <= w0)

    # OPTIMIZATION: Soft Constraints
    shift_mix_penalties = []
    granular_penalties = []
    penalties = []

    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    shift_map = {"am": 0, "pm": 1, "night": 2}

    for n in all_staff:
        # 1. Granular Shift Preferences (Rewards)
        w1_str = str(df.iloc[n].get("W1_Preferences", "")).lower()
        w2_str = str(df.iloc[n].get("W2_Preferences", "")).lower()

        for week_offset, pref_str in [(0, w1_str), (7, w2_str)]:
            if not pref_str or pref_str == 'nan': 
                continue
                
            requests = [x.strip() for x in pref_str.split(",")]
            for req in requests:
                parts = req.split()
                if len(parts) >= 2:
                    d_str = parts[0][:3] 
                    s_str = parts[1]
                    
                    if d_str in day_map and s_str in shift_map:
                        target_day = day_map[d_str] + week_offset
                        target_shift = shift_map[s_str]
                        
                        if target_day < num_days and (n, target_day, target_shift) in roster:
                            granular_penalties.append(-10 * roster[(n, target_day, target_shift)])

        # 2. General AM/PM Balance
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

        # 3. Avoid excessive late-earlies
        late_earlies = []
        for d in range(num_days - 1):
            is_late_early = model.NewBoolVar(f'late_early_staff_{n}_day_{d}')
            model.Add(roster[(n, d, 1)] + roster[(n, d+1, 0)] - 1 <= is_late_early)
            late_earlies.append(is_late_early)
            
        model.Add(sum(late_earlies) <= (num_days // 14))
        penalties.extend(late_earlies)

        # 4. Maximize Preferred Shifts
        pref_shift = str(df.iloc[n].get("Preferred_Shift", "None"))
        if pref_shift in ["AM", "PM", "Night"]:
            pref_idx = shift_names.index(pref_shift)
            for d in range(num_days):
                for s in range(3):
                    if s != pref_idx and (n, d, s) in roster:
                        penalties.append(roster[(n, d, s)] * 2)

        # 5. Requested RDOs (Avoid if possible, but allow if necessary)
        rdo_str = str(df.iloc[n].get("Requested_RDOs", ""))
        if rdo_str and rdo_str.lower() != 'nan':
            for val in rdo_str.split(","):
                try:
                    d = int(val.strip()) - 1
                    if 0 <= d < num_days:
                        for s in range(3):
                            if (n, d, s) in roster:
                                penalties.append(roster[(n, d, s)] * 30) 
                except ValueError:
                    pass

    # Feed ALL penalties and rewards into the final optimization
    model.Minimize(
        sum(shift_mix_penalties) + 
        sum(penalties) + 
        sum(granular_penalties) + 
        sum(leadership_penalties) + 
        sum(staffing_level_penalties) +
        sum(fatigue_penalties)
    )              
    
    # EXECUTE THE SOLVER
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0 
    status = solver.Solve(model)
    
    # FORMAT THE OUTPUT
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # --- PRE-PROCESS: Identify Acting Shift Leaders ---
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
                            
                        if active_role == "ANUM":
                            anum_present = True
                        elif active_role == "RN (In Charge)":
                            rn_in_charges.append(n)
                            
                if not anum_present and len(rn_in_charges) > 0:
                    acting_leaders.add((rn_in_charges[0], d, s))
        # --------------------------------------------------

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
                
                study_str = str(df.iloc[n].get("Study_Leave_Days", ""))
                study_days = [int(x.strip()) - 1 for x in study_str.split(",") if x.strip().isdigit()] if study_str and study_str.lower() != 'nan' else []
                
                ext_str = str(df.iloc[n].get("External_Working_Days", ""))
                ext_days = [int(x.strip()) - 1 for x in ext_str.split(",") if x.strip().isdigit()] if ext_str and ext_str.lower() != 'nan' else []
                
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
                                
                            if (n, d, s) in acting_leaders:
                                assigned_shift += " (Shift Leader)"
                
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
    
    st.markdown("---")
    st.header("🛠️ Constraint Troubleshooter")
    st.warning("If your roster is failing, turn these rules off ONE by ONE. The rule that fixes the roster is your bottleneck.")
    debug_flags = {
        "ignore_coverage": st.checkbox("Ignore Minimum Staff Levels (Allows short-staffed shifts)"),
        "ignore_leadership": st.checkbox("Ignore Leadership Minimums (Allows shifts with 0 leaders)"),
        "ignore_fatigue": st.checkbox("Ignore Fatigue & Rest Rules (Allows illegal consecutive days)"),
        "ignore_leave": st.checkbox("Ignore Approved Leave (Forces staff to work on leave)"),
        "ignore_eft": st.checkbox("Ignore Contract EFT Targets"),
    }

# --- GOOGLE SHEETS CONNECTION ---
conn = st.connection("gsheets", type=GSheetsConnection)
# REPLACE WITH YOUR ACTUAL URL!
SHEET_URL = "Ward Staff Profiles" 

if "staff_df" not in st.session_state:
    try:
        pulled_df = conn.read(spreadsheet=SHEET_URL, ttl=0) 
        pulled_df = pulled_df.dropna(how="all")
        
        if pulled_df.empty or "ID" not in pulled_df.columns:
            st.session_state.staff_df = load_initial_staff()
        else:
            st.session_state.staff_df = pulled_df
            
    except Exception as e:
        st.error(f"Google Sheets Connection Error: {e}")
        st.session_state.staff_df = load_initial_staff()

# --- FORCE CLEAN DATA TYPES ---
raw_df = st.session_state.staff_df.copy()

missing_columns = {
    "Night_Pool": False,
    "Allow_Fragmented_Shifts": False,
    "Entire_Roster_Leave": False,
    "Secondary_Role": "None",
    "Secondary_EFT": 0.0,
    "No_AM_DOW": "",
    "No_PM_DOW": "",
    "Preferred_Shift": "None",
    "Study_Leave_Days": "",
    "External_Working_Days": "",
    "W1_Preferences": "",
    "W2_Preferences": ""
}

for col, default_val in missing_columns.items():
    if col not in raw_df.columns:
        raw_df[col] = default_val

bool_cols = ["Night_Pool", "Allow_Fragmented_Shifts", "Entire_Roster_Leave"]
for col in bool_cols:
    raw_df[col] = raw_df[col].astype(str).str.strip().str.lower().isin(['true', '1', 't', 'yes', 'y'])

raw_df["EFT"] = pd.to_numeric(raw_df["EFT"], errors='coerce').fillna(1.0)
raw_df["Secondary_EFT"] = pd.to_numeric(raw_df["Secondary_EFT"], errors='coerce').fillna(0.0)
raw_df["Prior_Consecutive_Days"] = pd.to_numeric(raw_df["Prior_Consecutive_Days"], errors='coerce').fillna(0).astype(int)

for col in raw_df.columns:
    if col not in bool_cols and col not in ["EFT", "Secondary_EFT", "Prior_Consecutive_Days"]:
        raw_df[col] = raw_df[col].astype(str).str.replace(r'\.0$', '', regex=True).replace('nan', '')

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
        "External_Working_Days": st.column_config.TextColumn("External/CNM Days (e.g. 3, 4, 5)"),
        "W1_Preferences": st.column_config.TextColumn("W1 Prefs (e.g. Mon AM, Tue PM)"),
        "W2_Preferences": st.column_config.TextColumn("W2 Prefs (e.g. Wed Night)")
    }
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Save Staff Profiles"):
        with st.spinner("Saving securely to Google Sheets..."):
            conn.update(spreadsheet=SHEET_URL, data=edited_df)
            st.session_state.staff_df = edited_df
        st.success("Staff profiles permanently saved to the Cloud!")

with col2:
    if st.button("🔄 Sync from Google Sheet"):
        if "staff_df" in st.session_state:
            del st.session_state["staff_df"]
        st.rerun()

if st.button("Generate Roster", type="primary"):
    with st.spinner("Calculating optimal shifts (this may take a few seconds)..."):
        result_df = solve_roster(edited_df, roster_days, start_date, debug_flags)
        
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
            st.error("No feasible roster could be generated. Try relaxing constraints in the Troubleshooter panel.")
