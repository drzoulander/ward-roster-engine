import streamlit as st
import pandas as pd
import os
import datetime
import holidays
import math
import io
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
            "PD_Leave_Days": "", "Study_Leave_Days": "", "External_Working_Days": "",
            "W1_Preferences": "", "W2_Preferences": "", "Prefer_Not_In_Charge": False
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
    
    def parse_days(day_string):
        if pd.isna(day_string) or day_string.strip() == "" or day_string.strip() == "nan": return []
        try: return [int(x.strip()) - 1 for x in day_string.split(",")]
        except ValueError: return []
        
    def get_pref_requests(w1_str, w2_str):
        reqs = []
        shift_map = {"am": 0, "pm": 1, "night": 2}
        for w_idx, p_str in [(0, str(w1_str).lower()), (1, str(w2_str).lower())]:
            if not p_str or p_str == 'nan': continue
            for req in [x.strip() for x in p_str.split(",")]:
                parts = req.split()
                if len(parts) >= 2:
                    day_abbrev = parts[0][:3].lower()
                    shift_val = parts[1].lower()
                    if shift_val in shift_map:
                        for d in range(w_idx * 7, (w_idx + 1) * 7):
                            if d < num_days:
                                curr_date = start_date + datetime.timedelta(days=d)
                                if curr_date.strftime("%a").lower() == day_abbrev:
                                    reqs.append((d, shift_map[shift_val]))
                                    break
        return reqs
    
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
    leadership_penalties = []

    for d in all_days:
        current_date = start_date + datetime.timedelta(days=d)
        is_weekend = current_date.weekday() >= 5
        is_monday = current_date.weekday() == 0
        is_friday = current_date.weekday() == 4
        is_pub_hol = current_date in vic_holidays
        
        am_sum = sum(roster[(n, d, 0)] for n in all_staff)
        pm_sum = sum(roster[(n, d, 1)] for n in all_staff)
        night_sum = sum(roster[(n, d, 2)] for n in all_staff)
        
        model.Add(am_sum <= 5)
        model.Add(pm_sum <= 5)
        model.Add(night_sum <= 4)
        
        if not debug_flags.get("ignore_coverage"):
            model.Add(am_sum >= 4)
            missing_am = model.NewIntVar(0, 1, f'missing_am_d{d}')
            model.Add(missing_am == 5 - am_sum)
            staffing_level_penalties.append(missing_am * (500 if (is_weekend or is_monday or is_pub_hol) else 200))
                
            model.Add(pm_sum >= 4)
            missing_pm = model.NewIntVar(0, 1, f'missing_pm_d{d}')
            model.Add(missing_pm == 5 - pm_sum)
            staffing_level_penalties.append(missing_pm * (500 if (is_weekend or is_friday or is_pub_hol) else 200))
                
            target_night = 4 if (is_weekend or is_pub_hol) else 3
            model.Add(night_sum >= 3)
            missing_night = model.NewIntVar(0, 1, f'missing_night_d{d}')
            model.Add(missing_night == 4 - night_sum)
            staffing_level_penalties.append(missing_night * (500 if (is_weekend or is_pub_hol) else 200))
            
        for s in range(3):
            en_count = sum(roster[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "EN/Learner")
            if not debug_flags.get("ignore_coverage"):
                model.Add(en_count <= 2)
            
    is_leader = {}
    for d in all_days:
        for s in range(3):
            shift_leader_vars = []
            shift_active = model.NewBoolVar(f'shift_active_{d}_{s}')
            
            total_staff_working = sum(roster[(n, d, s)] for n in all_staff)
            model.Add(total_staff_working > 0).OnlyEnforceIf(shift_active)
            model.Add(total_staff_working == 0).OnlyEnforceIf(shift_active.Not())
            
            if not debug_flags.get("ignore_leadership"):
                females_on_shift = sum(roster[(n, d, s)] for n in females if (n, d, s) in roster)
                missing_female = model.NewIntVar(0, 1, f'missing_fem_d{d}_s{s}')
                model.Add(females_on_shift + missing_female >= 1)
                leadership_penalties.append(missing_female * 30)
            
            anum_sum = sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "ANUM") + sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "ANUM")
            rn_in_charge_sum = sum(role_primary[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "RN (In Charge)") + sum(role_secondary[(n, d, s)] for n in all_staff if df.iloc[n]["Secondary_Role"] == "RN (In Charge)")
            
            if not debug_flags.get("ignore_leadership"):
                total_anums_on_shift = sum(roster[(n, d, s)] for n in all_staff if df.iloc[n]["Role"] == "ANUM")
                excess_anum = model.NewIntVar(0, 10, f'excess_anum_d{d}_s{s}')
                model.Add(total_anums_on_shift - 1 <= excess_anum)
                leadership_penalties.append(excess_anum * 800)
                
                missing_anum = model.NewIntVar(0, 1, f'missing_anum_d{d}_s{s}')
                model.Add(missing_anum >= 1 - anum_sum)
                leadership_penalties.append(missing_anum * 40)
                
                missing_any_leader = model.NewIntVar(0, 2, f'missing_any_leader_d{d}_s{s}')
                model.Add(missing_any_leader >= 2 - (anum_sum + rn_in_charge_sum))
                leadership_penalties.append(missing_any_leader * 80)

            for n in all_staff:
                is_ldr = model.NewBoolVar(f'is_leader_{n}_{d}_{s}')
                is_leader[(n, d, s)] = is_ldr
                
                is_prim_ldr = 1 if df.iloc[n]["Role"] in ["ANUM", "RN (In Charge)"] else 0
                is_sec_ldr = 1 if df.iloc[n]["Secondary_Role"] in ["ANUM", "RN (In Charge)"] else 0
                
                can_be_leader = model.NewBoolVar(f'can_be_leader_{n}_{d}_{s}')
                if is_prim_ldr and is_sec_ldr:
                    model.Add(can_be_leader == roster[(n, d, s)])
                elif is_prim_ldr:
                    model.Add(can_be_leader == role_primary[(n, d, s)])
                elif is_sec_ldr:
                    model.Add(can_be_leader == role_secondary[(n, d, s)])
                else:
                    model.Add(can_be_leader == 0)
                    
                model.Add(is_ldr <= can_be_leader)
                shift_leader_vars.append(is_ldr)
                
                if not debug_flags.get("ignore_leadership"):
                    if is_prim_ldr or is_sec_ldr:
                        role_str = df.iloc[n]["Role"]
                        sec_str = df.iloc[n]["Secondary_Role"]
                        if role_str != "ANUM" and sec_str != "ANUM":
                            if df.iloc[n].get("Prefer_Not_In_Charge", False):
                                leadership_penalties.append(is_ldr * 50)
                            else:
                                leadership_penalties.append(is_ldr * 10)
                                
            no_leader = model.NewBoolVar(f'no_leader_{d}_{s}')
            model.Add(sum(shift_leader_vars) + no_leader == 1).OnlyEnforceIf(shift_active)
            model.Add(sum(shift_leader_vars) == 0).OnlyEnforceIf(shift_active.Not())
            
            if not debug_flags.get("ignore_leadership"):
                leadership_penalties.append(no_leader * 1000)

    if not debug_flags.get("ignore_leadership"):
        for n in all_staff:
            if df.iloc[n].get("Prefer_Not_In_Charge", False):
                leader_count = sum(is_leader[(n, d, s)] for d in all_days for s in range(3))
                excess_leader = model.NewIntVar(0, 14, f'excess_leader_{n}')
                model.Add(leader_count - 1 <= excess_leader)
                leadership_penalties.append(excess_leader * 200)
            
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
        pd_days_set = set(parse_days(str(df.iloc[n].get("PD_Leave_Days", ""))))
        study_days_set = set(parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))))
        ext_days_set = set(parse_days(str(df.iloc[n].get("External_Working_Days", ""))))
        
        unavailable_days = set(parse_days(leave_str)) | study_days_set | pd_days_set | ext_days_set
                
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

    fatigue_penalties = []
    
    for n in all_staff:
        is_fully_absent = df.iloc[n].get("Entire_Roster_Leave", False)
        has_secondary = df.iloc[n]["Secondary_Role"] != "None" and df.iloc[n]["Secondary_EFT"] > 0
        
        valid_leave_list = parse_days(str(df.iloc[n]["Approved_Leave_Days"]))
        study_days_set = set(parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))))
        pd_days_set = set(parse_days(str(df.iloc[n].get("PD_Leave_Days", ""))))
        
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
            
            base_shifts_raw = (total_eft * 80.0 * (num_days / 14.0)) / shift_length
            base_shifts_rounded = math.ceil(base_shifts_raw)
            
            valid_leave_days = len([d for d in valid_leave_list if 0 <= d < num_days])
            fraction_present = max(0.0, (num_days - valid_leave_days) / num_days)
            
            clinical_shifts_after_leave = int(math.floor(base_shifts_rounded * fraction_present + 0.5))
            
            study_count = len([d for d in study_days_set if 0 <= d < num_days])
            pd_count = len([d for d in pd_days_set if 0 <= d < num_days])
            final_shift_target = max(0, clinical_shifts_after_leave - pd_count - study_count)
            
            if has_secondary and not debug_flags.get("ignore_eft"):
                sec_base_raw = (secondary_eft * 80.0 * (num_days / 14.0)) / shift_length
                sec_shifts_after_leave = int(math.floor(math.ceil(sec_base_raw) * fraction_present + 0.5))
                sec_shift_target = min(sec_shifts_after_leave, final_shift_target)
                
                actual_sec_shifts = sum(role_secondary[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in role_secondary)
                sec_short = model.NewIntVar(0, 14, f'sec_short_{n}')
                sec_over = model.NewIntVar(0, 14, f'sec_over_{n}')
                model.Add(actual_sec_shifts == sec_shift_target - sec_short + sec_over)
                fatigue_penalties.append(sec_short * 100000)
                fatigue_penalties.append(sec_over * 100000)
            
        if not debug_flags.get("ignore_eft"):
            actual_shifts = sum(roster[(n, d, s)] for d in range(num_days) for s in range(3) if (n, d, s) in roster)
            shortfall = model.NewIntVar(0, 14, f'eft_short_{n}')
            overage = model.NewIntVar(0, 14, f'eft_over_{n}')
            
            model.Add(actual_shifts == final_shift_target - shortfall + overage)
            fatigue_penalties.append(shortfall * 100000)
            fatigue_penalties.append(overage * 100000)

    if not debug_flags.get("ignore_fatigue"):
        for n in all_staff:
            raw_prior = df.iloc[n]["Prior_Consecutive_Days"]
            prior_days = int(raw_prior) if pd.notna(raw_prior) else 0
            last_shift = str(df.iloc[n]["Last_Shift_Type"]).strip().upper()
            is_night_pool = df.iloc[n]["Night_Pool"]
            
            pd_days_set = set(parse_days(str(df.iloc[n].get("PD_Leave_Days", ""))))
            study_days_set = set(parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))))
            ext_days_set = set(parse_days(str(df.iloc[n].get("External_Working_Days", ""))))
            
            # --- FIXED: ACCURATE DAY-TO-NIGHT TRANSITION BANS ---
            for d in range(num_days - 1):
                model.AddImplication(roster[(n, d, 0)], roster[(n, d+1, 2)].Not()) # AM -> Night tomorrow banned
                model.AddImplication(roster[(n, d, 1)], roster[(n, d+1, 2)].Not()) # PM -> Night tomorrow banned
                
            if last_shift in ["AM", "PM"]:
                model.Add(roster[(n, 0, 2)] == 0) # Prior Day Shift -> Day 0 Night banned
                
            if last_shift == "NIGHT":
                model.Add(roster[(n, 0, 0)] == 0)
                model.Add(roster[(n, 0, 1)] == 0)
                # NOTE: We DO NOT ban Day 0 Night here, restoring massive capacity to the Night Pool
            # ----------------------------------------------------
                
            for sd in (pd_days_set | study_days_set):
                if 0 <= sd < num_days:
                    if sd - 1 >= 0: model.Add(roster[(n, sd-1, 2)] == 0)
                    if sd - 2 >= 0: model.Add(roster[(n, sd-2, 2)] == 0)
                
            shift_length = 10.0 if is_night_pool else 8.0
            total_eft_fatigue = df.iloc[n]["EFT"] + (df.iloc[n]["Secondary_EFT"] if df.iloc[n]["Secondary_Role"] != "None" else 0.0)
            base_shifts_fatigue = math.ceil((total_eft_fatigue * 80.0) / shift_length)
            
            ext_count = len([d for d in ext_days_set if 0 <= d < num_days])
            pd_count = len([d for d in pd_days_set if 0 <= d < num_days])
            total_shifts_for_fatigue = base_shifts_fatigue + ext_count + pd_count
            max_consec = int((total_shifts_for_fatigue / 2) + 1)
            
            for d in range(num_days - 4):
                model.Add(sum(roster[(n, d+w, 2)] for w in range(5)) <= 4)
                
            if last_shift == "NIGHT" and not is_night_pool:
                if num_days > 0:
                    for s in range(3): model.Add(roster[(n, 0, s)] == 0)
                if num_days > 1:
                    for s in range(3): model.Add(roster[(n, 1, s)] == 0)
                        
            virtual_work_days_active = pd_days_set | ext_days_set
            
            is_active_vars = []
            is_duty_vars = []
            for d in range(num_days):
                act_var = model.NewBoolVar(f'act_{n}_{d}')
                if d in virtual_work_days_active:
                    model.Add(act_var == 1)
                else:
                    model.Add(act_var == sum(roster[(n, d, s)] for s in range(3)))
                is_active_vars.append(act_var)
                
                duty_var = model.NewBoolVar(f'duty_{n}_{d}')
                if d in study_days_set:
                    model.Add(duty_var == 1)
                else:
                    model.Add(duty_var == act_var)
                is_duty_vars.append(duty_var)
                
            for A in range(num_days):
                for B in range(A + max_consec, num_days):
                    model.AddBoolOr([is_active_vars[A].Not(), is_active_vars[B].Not()] + [is_duty_vars[k].Not() for k in range(A+1, B)])

            if prior_days > 0:
                for B in range(num_days):
                    if B + prior_days >= max_consec:
                        model.AddBoolOr([is_active_vars[B].Not()] + [is_duty_vars[k].Not() for k in range(0, B)])
            
            # --- FIXED: FLAWLESS NEGATIVE PRIOR DAYS MAPPING ---
            w_minus_1 = 1 if prior_days > 0 else 0
            if prior_days > 1:
                w_minus_2 = 1
            elif prior_days == -1:
                w_minus_2 = 1  # Exactly 1 day off means Day -1 was Off, Day -2 was On
            else:
                w_minus_2 = 0  # 0 or <= -2 means Day -2 was Off
            
            def past_active(d_idx):
                if d_idx == -1: return w_minus_1
                if d_idx == -2: return w_minus_2
                return 0
            
            def past_duty(d_idx):
                return past_active(d_idx)
            # ---------------------------------------------------

            allow_fragmented = df.iloc[n]["Allow_Fragmented_Shifts"]
            if not allow_fragmented:
                internal_starts = []
                for d in range(0, num_days):
                    w_yest = past_active(d-1) if d-1 < 0 else is_active_vars[d-1]
                    w_tod = is_active_vars[d]
                    is_start = model.NewBoolVar(f'internal_start_{n}_day_{d}')
                    model.Add(w_tod - w_yest <= is_start)
                    internal_starts.append(is_start)
                
                extra_blocks = model.NewIntVar(0, 14, f'extra_blocks_{n}')
                model.Add(sum(internal_starts) - 1 <= extra_blocks)
                fatigue_penalties.append(extra_blocks * 400)
                    
                for d_idx in range(-1, num_days - 1):
                    w0_duty = past_duty(d_idx-1) if d_idx-1 < 0 else is_duty_vars[d_idx-1]
                    w1_duty = past_duty(d_idx) if d_idx < 0 else is_duty_vars[d_idx]
                    w2_duty = past_duty(d_idx+1) if d_idx+1 < 0 else is_duty_vars[d_idx+1]
                    
                    w1_active = past_active(d_idx) if d_idx < 0 else is_active_vars[d_idx]
                    
                    iso_off = model.NewBoolVar(f'iso_off_n{n}_d{d_idx}')
                    model.Add(w0_duty - w1_duty + w2_duty - 1 <= iso_off)
                    fatigue_penalties.append(iso_off * 20000)
                    
                    iso_on = model.NewBoolVar(f'iso_on_n{n}_d{d_idx}')
                    model.Add(w1_active - w0_duty - w2_duty <= iso_on)
                    fatigue_penalties.append(iso_on * 20000)

            if not is_night_pool:
                for d in range(0, num_days):
                    w_yest = past_active(d-1) if d-1 < 0 else is_active_vars[d-1]
                    am_tod = roster[(n, d, 0)]
                    bad_start = model.NewBoolVar(f'bad_start_n{n}_d{d}')
                    model.Add(am_tod - w_yest <= bad_start)
                    fatigue_penalties.append(bad_start * 8)
                    
                    w_tom = 0 if d+1 >= num_days else is_active_vars[d+1]
                    pm_tod = roster[(n, d, 1)]
                    bad_end = model.NewBoolVar(f'bad_end_n{n}_d{d}')
                    model.Add(pm_tod - w_tom <= bad_end)
                    fatigue_penalties.append(bad_end * 8)

    shift_mix_penalties = []
    granular_penalties = []
    penalties = []
    
    for n in all_staff:
        late_earlies = []
        last_shift = str(df.iloc[n].get("Last_Shift_Type", "")).strip().upper()
        if last_shift == "PM" and (n, 0, 0) in roster:
            late_earlies.append(roster[(n, 0, 0)])

        for d in range(num_days - 1):
            is_late_early = model.NewBoolVar(f'late_early_staff_{n}_day_{d}')
            model.Add(roster[(n, d, 1)] + roster[(n, d+1, 0)] - 1 <= is_late_early)
            late_earlies.append(is_late_early)
            
        pd_days_set = set(parse_days(str(df.iloc[n].get("PD_Leave_Days", ""))))
        study_days_set = set(parse_days(str(df.iloc[n].get("Study_Leave_Days", ""))))
        
        for sd in (pd_days_set | study_days_set):
            if sd - 1 >= 0:
                is_le_study = model.NewBoolVar(f'le_study_{n}_d{sd}')
                model.Add(roster[(n, sd-1, 1)] <= is_le_study)
                late_earlies.append(is_le_study)
            
        excess_le = model.NewIntVar(0, 14, f'excess_le_{n}')
        model.Add(sum(late_earlies) - (num_days // 14) <= excess_le)
        penalties.append(excess_le * 40)
        penalties.extend(late_earlies)

    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    shift_map = {"am": 0, "pm": 1, "night": 2}

    for n in all_staff:
        rdo_str = str(df.iloc[n]["Requested_RDOs"])
        req_rdos = []
        if rdo_str and rdo_str.lower() != 'nan':
            for val in rdo_str.split(","):
                try:
                    d = int(val.strip()) - 1
                    if 0 <= d < num_days: req_rdos.append(d)
                except ValueError: pass
                
        if req_rdos:
            rdo_missed_count = model.NewIntVar(0, len(req_rdos), f'rdo_miss_{n}')
            worked_on_rdos = sum(roster[(n, d, s)] for s in range(3) for d in req_rdos if (n, d, s) in roster)
            model.Add(rdo_missed_count == worked_on_rdos) 
            
            rdo_miss_sq = model.NewIntVar(0, len(req_rdos)**2, f'rdo_sq_{n}')
            model.AddMultiplicationEquality(rdo_miss_sq, [rdo_missed_count, rdo_missed_count])
            granular_penalties.append(rdo_miss_sq * 100)

        pref_reqs = get_pref_requests(df.iloc[n].get("W1_Preferences", ""), df.iloc[n].get("W2_Preferences", ""))
        if pref_reqs:
            pref_missed_count = model.NewIntVar(0, len(pref_reqs), f'pref_miss_{n}')
            granted_prefs = sum(roster[(n, d, s)] for d, s in pref_reqs if (n, d, s) in roster)
            model.Add(pref_missed_count == len(pref_reqs) - granted_prefs)
            
            pref_miss_sq = model.NewIntVar(0, len(pref_reqs)**2, f'pref_sq_{n}')
            model.AddMultiplicationEquality(pref_miss_sq, [pref_missed_count, pref_missed_count])
            granular_penalties.append(pref_miss_sq * 100)

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
                for n in all_staff:
                    if (n, d, s) in is_leader and solver.Value(is_leader[(n, d, s)]) == 1:
                        active_role = df.iloc[n]["Role"]
                        if (n, d, s) in role_secondary and solver.Value(role_secondary[(n, d, s)]) == 1:
                            active_role = df.iloc[n]["Secondary_Role"]
                        if active_role != "ANUM":
                            acting_leaders.add((n, d, s))

        roster_output = []
        tally_output = []
        day_headers = []
        am_totals = []
        pm_totals = []
        night_totals = []
        
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            header = current_date.strftime("%a %d-%b") 
            if current_date in vic_holidays: header += " (PH)"
            day_headers.append(header)
            
            am_totals.append(sum(1 for n in all_staff if (n, d, 0) in roster and solver.Value(roster[(n, d, 0)]) == 1))
            pm_totals.append(sum(1 for n in all_staff if (n, d, 1) in roster and solver.Value(roster[(n, d, 1)]) == 1))
            night_totals.append(sum(1 for n in all_staff if (n, d, 2) in roster and solver.Value(roster[(n, d, 2)]) == 1))
            
        summary_row_am_count = {"Staff ID": "📊 Ward AM Total", "Role": ""}
        summary_row_pm_count = {"Staff ID": "📊 Ward PM Total", "Role": ""}
        summary_row_night_count = {"Staff ID": "📊 Ward Night Total", "Role": ""}
        for idx, h in enumerate(day_headers):
            summary_row_am_count[h] = str(am_totals[idx])
            summary_row_pm_count[h] = str(pm_totals[idx])
            summary_row_night_count[h] = str(night_totals[idx])
        roster_output.extend([summary_row_am_count, summary_row_pm_count, summary_row_night_count])

        for n in all_staff:
            staff_row = {"Staff ID": df.iloc[n]["ID"], "Role": df.iloc[n]["Role"]}
            
            pd_days_list = parse_days(str(df.iloc[n].get("PD_Leave_Days", "")))
            study_days_list = parse_days(str(df.iloc[n].get("Study_Leave_Days", "")))
            ext_days_list = parse_days(str(df.iloc[n].get("External_Working_Days", "")))
            valid_leave_list = parse_days(str(df.iloc[n]["Approved_Leave_Days"]))
            
            actual_shifts_worked = 0
            
            for d in all_days:
                assigned_shift = "" 
                if d in pd_days_list: assigned_shift = "--- PD Leave ---"
                elif d in study_days_list: assigned_shift = "--- Study ---"
                elif d in ext_days_list: assigned_shift = "--- External ---"
                elif df.iloc[n].get("Entire_Roster_Leave", False) or d in valid_leave_list: assigned_shift = "--- Leave ---"
                else:
                    for s in range(3):
                        if (n, d, s) in roster and solver.Value(roster[(n, d, s)]) == 1:
                            actual_shifts_worked += 1
                            if (n, d, s) in role_secondary and solver.Value(role_secondary[(n, d, s)]) == 1:
                                assigned_shift = f"{shift_names[s]} ({df.iloc[n]['Secondary_Role']})"
                            else: assigned_shift = shift_names[s]
                            if (n, d, s) in acting_leaders: assigned_shift += " (Shift Leader)"
                staff_row[day_headers[d]] = assigned_shift
            roster_output.append(staff_row)
            
            rdo_str = str(df.iloc[n]["Requested_RDOs"])
            req_rdos = []
            if rdo_str and rdo_str.lower() != 'nan':
                for val in rdo_str.split(","):
                    try:
                        d_rdo = int(val.strip()) - 1
                        if 0 <= d_rdo < num_days: req_rdos.append(d_rdo)
                    except ValueError: pass
            
            rdos_granted = 0
            granted_rdo_strings = []
            for d_rdo in req_rdos:
                worked = sum(solver.Value(roster[(n, d_rdo, s)]) for s in range(3) if (n, d_rdo, s) in roster)
                if worked == 0:
                    rdos_granted += 1
                    granted_rdo_strings.append(f"Day {d_rdo+1}")
            
            if req_rdos:
                if rdos_granted > 0: rdo_tally = f"{rdos_granted}/{len(req_rdos)} ({', '.join(granted_rdo_strings)})"
                else: rdo_tally = f"0/{len(req_rdos)}"
            else: rdo_tally = "N/A"
            
            pref_reqs = get_pref_requests(df.iloc[n].get("W1_Preferences", ""), df.iloc[n].get("W2_Preferences", ""))
            prefs_granted = 0
            granted_pref_strings = []
            for t_day, t_shift in pref_reqs:
                if (n, t_day, t_shift) in roster and solver.Value(roster[(n, t_day, t_shift)]) == 1:
                    prefs_granted += 1
                    curr_date = start_date + datetime.timedelta(days=t_day)
                    granted_pref_strings.append(f"{curr_date.strftime('%a')} {shift_names[t_shift]}")
            
            if pref_reqs:
                if prefs_granted > 0: pref_tally = f"{prefs_granted}/{len(pref_reqs)} ({', '.join(granted_pref_strings)})"
                else: pref_tally = f"0/{len(pref_reqs)}"
            else: pref_tally = "N/A"

            is_fully_absent = df.iloc[n].get("Entire_Roster_Leave", False)
            if is_fully_absent:
                target_shifts = 0
            else:
                total_eft = df.iloc[n]["EFT"] + (df.iloc[n]["Secondary_EFT"] if df.iloc[n]["Secondary_Role"] != "None" else 0.0)
                shift_length = 10.0 if df.iloc[n]["Night_Pool"] else 8.0
                base_shifts_raw = (total_eft * 80.0 * (num_days / 14.0)) / shift_length
                base_shifts_rounded = math.ceil(base_shifts_raw)
                
                valid_leave_days = len([d for d in valid_leave_list if 0 <= d < num_days])
                fraction_present = max(0.0, (num_days - valid_leave_days) / num_days)
                clinical_shifts = int(math.floor(base_shifts_rounded * fraction_present + 0.5))
                
                pd_count = len([d for d in pd_days_list if 0 <= d < num_days])
                study_count = len([d for d in study_days_list if 0 <= d < num_days])
                target_shifts = max(0, clinical_shifts - pd_count - study_count)
            
            tally_output.append({
                "Staff ID": df.iloc[n]["ID"],
                "EFT": total_eft,
                "Target Shifts": target_shifts,
                "Achieved Shifts": actual_shifts_worked,
                "Variance": actual_shifts_worked - target_shifts,
                "Leave Days": len([d for d in valid_leave_list if 0 <= d < num_days]) if not is_fully_absent else 14,
                "PD Days": len([d for d in pd_days_list if 0 <= d < num_days]),
                "Study Days": len([d for d in study_days_list if 0 <= d < num_days]),
                "External Days": len([d for d in ext_days_list if 0 <= d < num_days]),
                "RDOs Granted": rdo_tally,
                "Prefs Granted": pref_tally
            })
            
        result_df = pd.DataFrame(roster_output)
        tally_df = pd.DataFrame(tally_output)
        
        agency_am = []
        agency_pm = []
        agency_night = []
        
        for d in all_days:
            current_date = start_date + datetime.timedelta(days=d)
            is_weekend = current_date.weekday() >= 5
            is_pub_hol = current_date in vic_holidays
            
            short_am = max(0, 5 - am_totals[d])
            agency_am.append(f"Short {short_am} (Agency)" if short_am > 0 else "Fully Staffed")
            
            short_pm = max(0, 5 - pm_totals[d])
            agency_pm.append(f"Short {short_pm} (Agency)" if short_pm > 0 else "Fully Staffed")
            
            short_night = max(0, 4 - night_totals[d])
            agency_night.append(f"Short {short_night} (Agency)" if short_night > 0 else "Fully Staffed")
            
        summary_row_am = {"Staff ID": "🚨 AM Shortfall", "Role": "AGENCY CHECK"}
        summary_row_pm = {"Staff ID": "🚨 PM Shortfall", "Role": "AGENCY CHECK"}
        summary_row_night = {"Staff ID": "🚨 Night Shortfall", "Role": "AGENCY CHECK"}
        
        for idx, h in enumerate(day_headers):
            summary_row_am[h] = agency_am[idx]
            summary_row_pm[h] = agency_pm[idx]
            summary_row_night[h] = agency_night[idx]
            
        summary_df = pd.DataFrame([summary_row_am, summary_row_pm, summary_row_night])
        return pd.concat([result_df, summary_df], ignore_index=True), tally_df
    else:
        return None, None

# ----------------------------------------
# 3. STREAMLIT USER INTERFACE
# ----------------------------------------
st.set_page_config(layout="wide", page_title="Ward Rostering Engine")
st.title("Automated Ward Rostering Engine")

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
    "Preferred_Shift": "None", "PD_Leave_Days": "", "Study_Leave_Days": "", "External_Working_Days": "",
    "W1_Preferences": "", "W2_Preferences": "", "Prefer_Not_In_Charge": False
}
for col, default_val in missing_columns.items():
    if col not in raw_df.columns: raw_df[col] = default_val

def make_boolean(val):
    if pd.isna(val): return False
    if isinstance(val, bool): return val
    val_str = str(val).strip().upper()
    if val_str in ['TRUE', '1', '1.0', 'T', 'YES', 'Y']: return True
    return False

bool_cols = ["Night_Pool", "Allow_Fragmented_Shifts", "Entire_Roster_Leave", "Prefer_Not_In_Charge"]
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

# --- WARD CAPACITY DASHBOARD ---
def parse_days_length(day_string):
    if pd.isna(day_string) or day_string.strip() == "" or day_string.strip() == "nan": return 0
    try: return len([int(x.strip()) - 1 for x in day_string.split(",")])
    except: return 0

total_day_shifts_needed = 0
total_night_shifts_needed = 0

for _, row in st.session_state.staff_df.iterrows():
    if row["Entire_Roster_Leave"]: continue
    
    total_eft = row["EFT"] + row["Secondary_EFT"]
    shift_len = 10.0 if row["Night_Pool"] else 8.0
    base_shifts_raw = (total_eft * 80.0) / shift_len
    base_shifts_rounded = math.ceil(base_shifts_raw)
    
    valid_leave = parse_days_length(row["Approved_Leave_Days"])
    pd_leave = parse_days_length(row["PD_Leave_Days"])
    study_leave_new = parse_days_length(row["Study_Leave_Days"])
    
    fraction = max(0.0, (14 - valid_leave) / 14)
    clinical = int(math.floor(base_shifts_rounded * fraction + 0.5))
    target = max(0, clinical - pd_leave - study_leave_new)
    
    if row["Night_Pool"]: total_night_shifts_needed += target
    else: total_day_shifts_needed += target

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Day Shifts Contracted", total_day_shifts_needed, delta=f"{140 - total_day_shifts_needed} Empty Ward Slots", delta_color="normal" if 140 >= total_day_shifts_needed else "inverse")
col_b.metric("Total Night Shifts Contracted", total_night_shifts_needed, delta=f"{56 - total_night_shifts_needed} Empty Ward Slots", delta_color="normal" if 56 >= total_night_shifts_needed else "inverse")
if total_day_shifts_needed > 140 or total_night_shifts_needed > 56:
    st.error("🚨 **WARNING: Ward Over-Contracted!** Your staff legally require more shifts than the physical ward can hold. You MUST use the Troubleshooter to bypass the shift ceilings, or the roster will crash.")
# -----------------------------------

st.markdown("---")

with st.sidebar:
    st.header("Roster Settings")
    start_date = st.date_input("Roster Start Date", datetime.date.today())
    roster_days = st.slider("Roster Length (Days)", min_value=14, max_value=182, value=14, step=7)
    
    st.markdown("---")
    st.header("🛠️ Constraint Troubleshooter")
    debug_flags = {
        "ignore_coverage": st.checkbox("Ignore Minimum Staff Levels (Floor Limits)"),
        "ignore_leadership": st.checkbox("Ignore Leadership Minimums"),
        "ignore_fatigue": st.checkbox("Ignore Fatigue & Rest Rules"),
        "ignore_leave": st.checkbox("Ignore Approved Leave"),
        "ignore_eft": st.checkbox("Ignore Contract EFT Targets"),
        "ignore_night_pool": st.checkbox("Ignore Night Pool Separation")
    }

st.subheader("Staff Pool Management")
edited_df = st.data_editor(
    st.session_state.staff_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "Night_Pool": st.column_config.CheckboxColumn("Night Pool?", default=False),
        "Prefer_Not_In_Charge": st.column_config.CheckboxColumn("Prefer Not In Charge", default=False),
        "Approved_Leave_Days": st.column_config.TextColumn("Approved Leave"),
        "Requested_RDOs": st.column_config.TextColumn("Requested RDOs"),
        "EFT": st.column_config.NumberColumn("EFT", min_value=0.1, max_value=1.0, step=0.1),
        "Prior_Consecutive_Days": st.column_config.NumberColumn("Prior Consec (Negative = Days Off)", min_value=-14, max_value=14, step=1),
        "Last_Shift_Type": st.column_config.SelectboxColumn("Last Shift Type", options=["None", "AM", "PM", "Night"]),
        "Unavailable_DOW": st.column_config.TextColumn("Unavailable Days"),
        "Allow_Fragmented_Shifts": st.column_config.CheckboxColumn("Allow Fragmented Shifts", default=False),
        "Entire_Roster_Leave": st.column_config.CheckboxColumn("On Leave (Entire Roster)", default=False),
        "Secondary_Role": st.column_config.SelectboxColumn("Secondary Role", options=["None", "ANUM", "RN (In Charge)", "RN", "EN/Learner"]),
        "Secondary_EFT": st.column_config.NumberColumn("Secondary EFT", min_value=0.0, max_value=1.0, step=0.1),
        "No_AM_DOW": st.column_config.TextColumn("No AM Days"),
        "No_PM_DOW": st.column_config.TextColumn("No PM Days"),
        "Preferred_Shift": st.column_config.SelectboxColumn("Preferred Shift", options=["None", "AM", "PM", "Night"]),
        "PD_Leave_Days": st.column_config.TextColumn("PD Leave Days"),
        "Study_Leave_Days": st.column_config.TextColumn("Study Leave Days"),
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
        result_df, tally_df = solve_roster(edited_df, roster_days, start_date, debug_flags)
        if result_df is not None:
            st.success("Roster Generated Successfully!")
            
            st.subheader("📋 Staff Contract Tally")
            st.dataframe(tally_df, use_container_width=True)
            
            st.subheader("🏥 Final Ward Roster")
            st.dataframe(result_df, use_container_width=True)
            
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, sheet_name='Ward Roster', index=False)
                    tally_df.to_excel(writer, sheet_name='Staff Tally', index=False)
                
                st.download_button(
                    label="📥 Download Complete Spreadsheet (Excel)",
                    data=buffer.getvalue(),
                    file_name='ward_roster_complete.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except ImportError:
                st.info("💡 Tip: Run `pip install xlsxwriter` in your terminal to get a multi-tab Excel file. Falling back to a combined CSV.")
                csv_roster = result_df.to_csv(index=False)
                csv_tally = tally_df.to_csv(index=False)
                combined_csv = f"WARD ROSTER\n{csv_roster}\n\nSTAFF CONTRACT TALLY\n{csv_tally}"
                
                st.download_button(
                    label="📥 Download Combined Spreadsheet (CSV)", 
                    data=combined_csv.encode('utf-8'), 
                    file_name='ward_roster_combined.csv', 
                    mime='text/csv'
                )
        else:
            st.error("Engine failed to generate. A hard mathematical paradox exists (e.g., impossible to meet the safe staffing floors with currently available staff, or a staff member's Approved Leave forces them to break their max-consecutive-shift limits).")
