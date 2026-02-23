import pandas as pd
import numpy as np
import os, sys, glob, re
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count

# ================= Configuration =================
DRIVER_MAPPING = {
    1: 'VER', 33: 'VER', 2: 'SAR', 3: 'RIC', 4: 'NOR', 5: 'VET', 6: 'LAT', 7: 'RAI',
    8: 'GRO', 9: 'MAZ', 10: 'GAS', 11: 'PER', 12: 'ANT', 14: 'ALO', 16: 'LEC', 18: 'STR',
    20: 'MAG', 21: 'DEV', 22: 'TSU', 23: 'ALB', 24: 'ZHO', 26: 'KVY', 27: 'HUL', 28: 'HAR',
    30: 'LAW', 31: 'OCO', 35: 'SIR', 43: 'COL', 44: 'HAM', 47: 'MSC', 55: 'SAI', 63: 'RUS',
    77: 'BOT', 81: 'PIA', 87: 'BEA', 88: 'KUB', 89: 'AIT', 99: 'GIO'
}

def get_driver_name(did, year=None):
    if isinstance(did, str) and not did.isdigit(): return did
    did = int(did)
    if did == 9: return 'ERI' if year == 2018 else 'MAZ'
    return DRIVER_MAPPING.get(did, str(did))

# ================= Telemetry Cleaning =================
class TelemetryCleaner:
    @staticmethod
    def filter_by_proximity(df, tolerance=50.0):
        pos_df = df[df['Source'] == 'pos']
        if pos_df.empty: return df
        
        # Filter static noise
        counts = pos_df.groupby(['X', 'Y']).size()
        valid = pos_df[pos_df.set_index(['X', 'Y']).index.isin(counts[counts < 100].index)]
        if valid.empty: return df

        tree = cKDTree(valid[['X', 'Y']].values)
        dists, _ = tree.query(df[['X', 'Y']].values)
        
        mask = (dists < tolerance) & (df['Source'].isin(['car', 'pos'])) & (df['Source'] != 'interpolation')
        clean_df = df[mask].copy()
        
        if len(df) > 0 and (1 - len(clean_df)/len(df)) > 0.4:
            print(f"  [WARN] High data loss: {1 - len(clean_df)/len(df):.1%} removed.")
        return clean_df

# ================= Feature Engineering =================
class FeatureEngineer:
    @staticmethod
    def process_driver(df, driver_id):
        # Time & Dedupe
        t = df['SessionTime']
        ts = t.dt.total_seconds().values if hasattr(t, 'dt') else (t.astype('int64')/1e9 if pd.api.types.is_datetime64_any_dtype(t) else t.values)
        ts, idx = np.unique(ts, return_index=True)
        
        if len(ts) < 50: return None
        
        # Extract & Scale (Decimeters -> Meters)
        x, y = df['X'].values[idx] * 0.1, df['Y'].values[idx] * 0.1
        sp = df['Speed'].values[idx]
        lap = df['LapNumber'].fillna(0).values[idx] if 'LapNumber' in df else np.zeros_like(x)

        # Resample 10Hz
        new_t = np.arange(ts[0], ts[-1], 0.1)
        if len(new_t) < 50: return None
        
        xr = interp1d(ts, x, kind='linear', fill_value='extrapolate')(new_t)
        yr = interp1d(ts, y, kind='linear', fill_value='extrapolate')(new_t)
        sr = interp1d(ts, sp, kind='linear', fill_value='extrapolate')(new_t)
        lr = interp1d(ts, lap, kind='nearest', fill_value='extrapolate')(new_t)

        # Smooth
        xs = savgol_filter(xr, 21, 3) 
        ys = savgol_filter(yr, 21, 3)
        
        # Curvature & Physics
        dx, dy = np.gradient(xs), np.gradient(ys)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        curv = np.abs(dx*ddy - dy*ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-6)
        rad = 1.0 / (curv + 1e-6)
        
        v_ms = sr / 3.6
        rad_corr = np.maximum(rad, (v_ms**2)/(7.0*9.81)) # Max 7G physics
        g = np.clip((v_ms**2)/(rad_corr*9.81), 0, 20.0)
        gs = savgol_filter(g, 21, 3) # Smooth G

        return pd.DataFrame({'Driver': driver_id, 'SessionTime': new_t, 'LapNumber': lr, 
                           'X_Smooth': xs, 'Y_Smooth': ys, 'Speed': sr, 
                           'Corner_Radius_m': rad, 'Lateral_G': gs})

# ================= Scoring Logic =================
class PerformanceScorer:
    ANCHOR_CORNERS = {
        'São_Paulo_Grand_Prix': (-20, 460), 'Sao_Paulo_Grand_Prix': (-20, 460), 'Brazilian_Grand_Prix': (-20, 460)
    }

    @staticmethod
    def _get_benchmark(df, race_name):
        # unicode robust lookup
        anchor = PerformanceScorer.ANCHOR_CORNERS.get(race_name)
        if not anchor:
            for k, v in PerformanceScorer.ANCHOR_CORNERS.items():
                if k in race_name or race_name in k:
                    try: print(f"  [ANCHOR] Fuzzy matched '{k}'")
                    except: pass
                    anchor = v; break
        
        if anchor:
            # Verify data exists near anchor
            ax, ay = anchor
            if len(df[((df['X_Smooth']-ax)**2 + (df['Y_Smooth']-ay)**2) < 50**2]) > 50:
                try: print(f"  [ANCHOR] Using fixed {anchor}")
                except: pass
                return ax, ay, df[((df['X_Smooth']-ax)**2 + (df['Y_Smooth']-ay)**2) < 50**2]['Lateral_G'].median()

        # Auto-detect
        high_g = df[df['Lateral_G'] > 2.0].copy()
        if high_g.empty: return None, None, 0
        
        high_g['XB'], high_g['YB'] = (high_g['X_Smooth']/20).round()*20, (high_g['Y_Smooth']/20).round()*20
        cands = high_g.groupby(['XB', 'YB']).agg({'Lateral_G': 'median', 'Speed': 'median', 'Driver': 'nunique'}).reset_index()
        valid = cands[(cands['Driver'] > 10) & (cands['Speed'].between(80, 260))].sort_values('Lateral_G', ascending=False)
        
        if valid.empty: valid = cands[cands['Driver'] > 10].sort_values('Lateral_G', ascending=False)
        if valid.empty: return None, None, 0
        
        best = valid.iloc[0]
        try: print(f"  [ANCHOR] Auto-detected at ({best['XB']}, {best['YB']})")
        except: pass
        return best['XB'], best['YB'], best['Lateral_G']

    @staticmethod
    def calculate_scores(df, meta, team_map=None, viz_dir=None):
        tx, ty, bench_g = PerformanceScorer._get_benchmark(df, meta['Race'])
        if tx is None: return pd.DataFrame()

        # Extract Passes
        drivers, candidates, all_speeds = df['Driver'].unique(), {}, []
        for d in drivers:
            ddf = df[df['Driver'] == d]
            zone = ddf[((ddf['X_Smooth']-tx)**2 + (ddf['Y_Smooth']-ty)**2) < 40**2]
            passes = []
            for ln, g_data in zone.groupby('LapNumber'):
                if len(g_data) < 5: continue
                # False positive filter
                idx_min = g_data['Speed'].idxmin()
                if g_data.loc[idx_min, 'Lateral_G'] < 0.5 or g_data['Speed'].min() > 280: continue
                
                idx_max = g_data['Lateral_G'].idxmax()
                passes.append({
                    'LapNumber': int(ln), 'Max_Corner_G': g_data['Lateral_G'].max(),
                    'Apex_Speed': g_data['Speed'].min(), 'Speed_at_Max_G': g_data.loc[idx_max, 'Speed'],
                    'Apex_G': g_data.loc[idx_min, 'Lateral_G'], 'Apex_Time': g_data.loc[idx_min, 'SessionTime']
                })
            if passes:
                candidates[d] = sorted(passes, key=lambda x: x['Max_Corner_G'], reverse=True)
                all_speeds.append(candidates[d][0]['Apex_Speed'])

        if not all_speeds: return pd.DataFrame()
        bench_speed = np.median(all_speeds)
        records, notes = [], []

        # Scoring & Purification
        cleaned_recs = {}
        for d, cands in candidates.items():
            best = cands[0]
            # Swap outlier (<70% metric)
            if best['Apex_Speed'] < bench_speed * 0.7 and len(cands) > 1 and cands[1]['Apex_Speed'] > bench_speed * 0.8:
                notes.append(f"{d}: Swapped Outlier Lap {best['LapNumber']} -> {cands[1]['LapNumber']}")
                best = cands[1]
            
            if best['Apex_Speed'] < 10: continue
            
            # VMax
            v_max = df[(df['Driver']==d) & (df['LapNumber']==best['LapNumber'])]['Speed'].max()
            if pd.isna(v_max): v_max = df[df['Driver']==d]['Speed'].max()

            rec = {
                'Year': meta['Year'], 'Round': meta['Round'], 'Race': meta['Race'],
                'Driver': get_driver_name(d, meta['Year']), 'Source_Driver': d, 'Team': 'Unknown',
                'LapNumber': best['LapNumber'], 'Apex_Time': best['Apex_Time'],
                'V_Max': v_max, 'Apex_Speed': best['Apex_Speed'], 'Max_Corner_G': best['Max_Corner_G'],
                'Speed_at_Max_G': best['Speed_at_Max_G'], 'Apex_G': best['Apex_G'],
                'Benchmark_G': bench_g, 'Benchmark_Apex_Speed': bench_speed,
                'Car_Score_Original': (v_max + best['Apex_Speed']) / 2,
                'Is_Imputed': False, 'Imputation_Reason': ""
            }
            records.append(rec)
            cleaned_recs[rec['Driver']] = rec

        # Imputation
        if team_map:
            # Map teams
            for r in records: r['Team'] = team_map.get(r['Driver'], 'Unknown')
            recs_by_driver = {r['Driver']: r for r in records}
            
            for team in set(team_map.values()) - {'Unknown'}:
                # Drivers in this race for this team
                t_drivers = [get_driver_name(d, meta['Year']) for d in drivers if team_map.get(get_driver_name(d, meta['Year'])) == team]
                if len(t_drivers) < 2: continue
                
                valid = sorted([recs_by_driver[d] for d in t_drivers if d in recs_by_driver], key=lambda x: x['Apex_Speed'], reverse=True)
                if not valid: continue
                best = valid[0]

                # Impute Missing
                for d in t_drivers:
                    if d not in recs_by_driver:
                        new = best.copy()
                        new.update({'Driver': d, 'Is_Imputed': True, 'Imputation_Reason': f"Missing. From {best['Driver']}"})
                        records.append(new); recs_by_driver[d] = new
                        notes.append(f"{d} ({team}): Missing -> Imputed from {best['Driver']}")

                # Impute Slow (>20%)
                for r in valid:
                    if r['Driver'] == best['Driver']: continue
                    if (best['Apex_Speed'] - r['Apex_Speed']) / best['Apex_Speed'] > 0.2:
                        # Attempt Recovery
                        src_id = r.get('Source_Driver', r['Driver'])
                        cands = sorted(candidates.get(src_id, []), key=lambda x: x['Apex_Speed'], reverse=True)
                        found = None
                        
                        # Tiered Search
                        for th in [0.2, 0.3, 0.4, 0.5]:
                            for c in cands:
                                if (best['Apex_Speed'] - c['Apex_Speed'])/best['Apex_Speed'] <= th and c.get('Apex_G',0)>0.5:
                                    found = c; break
                            if found: break
                        if not found: # Tier 5
                             for c in cands:
                                 if c.get('Apex_G',0)>0.5 and c['Apex_Speed']>50: found = c; break

                        if found:
                            r.update({
                                'LapNumber': found['LapNumber'], 'Apex_Time': found['Apex_Time'],
                                'Apex_Speed': found['Apex_Speed'], 'Max_Corner_G': found['Max_Corner_G'],
                                'Car_Score_Original': (r['V_Max'] + found['Apex_Speed'])/2,
                                'Is_Imputed': False, 'Imputation_Reason': ""
                            })
                            notes.append(f"{r['Driver']}: Recovered Lap {found['LapNumber']}")
                        else:
                            # Hard Impute
                            imp = best.copy()
                            imp.update({'Driver': r['Driver'], 'Is_Imputed': True, 'Imputation_Reason': f"Gap > 20%. From {best['Driver']}"})
                            # Update records list (find and allow update? dict reference works)
                            r.update(imp) # In-place update of standard dict
                            notes.append(f"{r['Driver']}: Gap > 20% -> Imputed from {best['Driver']}")

        if not records: return pd.DataFrame()
        final_df = pd.DataFrame(records)
        
        # Scaling
        scale = lambda x: 5.0 if x.max()==x.min() else 1 + 9*(x - x.min())/(x.max()-x.min())
        for c in ['V_Max', 'Apex_Speed', 'Speed_at_Max_G', 'Car_Score_Original']:
            final_df[f'{c}_Scaled'] = scale(final_df[c])
        final_df['Car_Score_Mixed'] = (final_df['V_Max_Scaled'] + final_df['Apex_Speed_Scaled'])/2

        # Viz
        if viz_dir:
            cands_named = {get_driver_name(k, meta['Year']): v for k,v in candidates.items()}
            try: Visualizer.plot_dashboard(df, tx, ty, meta, viz_dir, final_df, cands_named, notes)
            except Exception as e: print(f"  [WARN] Viz failed: {e}")

        return final_df

# ================= Visualization =================
class Visualizer:
    @staticmethod
    def plot_dashboard(feat, tx, ty, meta, out_dir, scores, cands, notes):
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.2])

        # Track Map
        ax = fig.add_subplot(gs[0:2, 0])
        longest = feat[feat['Driver'] == feat['Driver'].value_counts().idxmax()]
        ax.plot(longest['X_Smooth'], longest['Y_Smooth'], 'k-', lw=0.8, alpha=0.3)
        ax.scatter([tx], [ty], c='r', s=150, marker='X')
        ax.set_title(f"Circuit: {meta.get('Race')}", fontsize=14, fontweight='bold')
        ax.axis('equal'); ax.axis('off')

        # Profiles
        ax_s, ax_g = fig.add_subplot(gs[0, 1:]), fig.add_subplot(gs[1, 1:])
        cmap = plt.get_cmap('tab20')
        drivers = scores.sort_values('Apex_Speed', ascending=False)['Driver'].unique()

        for i, d in enumerate(drivers):
            rec = scores[scores['Driver'] == d].iloc[0]
            if d not in cands: continue # Should use ID check really, but cands keys are Names now
            
            # Find data segment by Time (Robust)
            tc = rec['Apex_Time']
            sid = rec.get('Source_Driver') # ID
            mask = (feat['Driver'] == sid) & (feat['SessionTime'].between(tc-10, tc+10))
            seg = feat[mask].copy()
            if seg.empty: continue

            # Center
            apex_idx = ((seg['X_Smooth']-tx)**2 + (seg['Y_Smooth']-ty)**2).idxmin()
            seg['RelT'] = seg['SessionTime'] - seg.loc[apex_idx, 'SessionTime']
            
            ls, alph, lw = ('--', 0.9, 2.0) if rec['Is_Imputed'] else ('-', 0.7, 1.5)
            c = cmap(i % 20)
            ax_s.plot(seg['RelT'], seg['Speed'], label=f"{d}{' (IMP)' if rec['Is_Imputed'] else ''}", color=c, ls=ls, alpha=alph, lw=lw)
            ax_g.plot(seg['RelT'], seg['Lateral_G'], color=c, ls=ls, alpha=alph, lw=lw)

        ax_s.set_title("Cornering Profiles"); ax_s.set_ylabel("Speed (km/h)"); ax_s.grid(True, alpha=0.3)
        ax_s.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
        ax_g.set_ylabel("Lateral G"); ax_g.grid(True, alpha=0.3)
        
        # Table
        ax_t = fig.add_subplot(gs[2, :]); ax_t.axis('off')
        tbl_data = scores.sort_values('Apex_Speed', ascending=False)[['Driver', 'Team', 'LapNumber', 'Apex_Speed', 'Max_Corner_G', 'Car_Score_Scaled', 'Is_Imputed']]
        show = pd.concat([tbl_data.head(5), tbl_data.tail(3)]).drop_duplicates()
        txt = [[str(x) for x in r] for r in show.values]
        tbl = ax_t.table(cellText=txt, colLabels=show.columns, cellLoc='center', loc='center'); tbl.scale(1, 1.2)
        
        fname = re.sub(r'[^\w\-_\. ]', '_', f"{meta['Year']}_{meta['Round']:02d}_{meta['Race']}_dashboard.png")
        try: plt.savefig(os.path.join(out_dir, fname), dpi=100); plt.close()
        except: plt.close()

# ================= Pipeline Runner =================
def process_single_race(args):
    fpath, out_file, vdir = args
    fname = os.path.basename(fpath)
    try:
        m = re.match(r"(\d{4})_(\d{2})_(.+)\.parquet", fname)
        if not m: return {'status': 'skip', 'filename': fname, 'reason': 'Name format'}
        yr, rnd, race = int(m.group(1)), int(m.group(2)), m.group(3)
        
        safe_name = fname.encode('ascii','replace').decode('ascii')
        
        raw = pd.read_parquet(fpath)
        clean = TelemetryCleaner.filter_by_proximity(raw)
        clean = clean[(clean['Source']=='car') & (clean['X']>-15000)]
        
        feats = []
        for d in clean['Driver'].unique():
            df = clean[clean['Driver']==d].sort_values('SessionTime')
            if len(df)<50: continue
            # Segment
            t = df['SessionTime'].dt.total_seconds().values if hasattr(df['SessionTime'], 'dt') else df['SessionTime'].values.astype(float)
            if t[0]>1e10: t/=1e9
            gaps = np.where(np.diff(t)>2.0)[0]
            bounds = np.concatenate(([0], gaps+1, [len(df)]))
            for k in range(len(bounds)-1):
                seg = df.iloc[bounds[k]:bounds[k+1]]
                if len(seg)>50: 
                    res = FeatureEngineer.process_driver(seg, d)
                    if res is not None: feats.append(res)
        
        if not feats: return {'status': 'skip', 'filename': safe_name, 'reason': 'No features'}
        full = pd.concat(feats)
        
        # Teams
        tm = {}
        if 'Team' in raw.columns:
            for _, r in raw[['Driver', 'Team']].dropna().drop_duplicates().iterrows():
                tm[get_driver_name(r['Driver'], yr)] = r['Team']

        sc = PerformanceScorer.calculate_scores(full, {'Year': yr, 'Round': rnd, 'Race': race}, tm, vdir)
        if sc.empty: return {'status': 'skip', 'filename': safe_name, 'reason': 'No scores'}
        
        return {'status': 'success', 'filename': safe_name, 'data': sc, 'rows': len(sc)}
    except Exception as e:
        return {'status': 'error', 'filename': fname, 'error': str(e)}

def audit_results(csv_path):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    issues = []
    print("\n--- Audit ---")
    for (yr, rnd, tm), g in df.groupby(['Year', 'Round', 'Team']):
        if len(g)<2: continue
        mx, mn = g['Apex_Speed'].max(), g['Apex_Speed'].min()
        if mx>0 and (mx-mn)/mx > 0.2:
            issues.append(f"{yr} R{rnd} {tm}: Gap {(mx-mn)/mx:.1%}")
    if issues: 
        print(f"Found {len(issues)} Teammate Gaps > 20%:")
        for i in issues[:10]: # Limit output
            try: print(i.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            except: pass
    else: print("Audit OK.")

def run_pipeline(in_dir, out_file):
    files = sorted([f for f in glob.glob(os.path.join(in_dir, "*.parquet")) if '_backup' not in f])
    processed = set()
    if os.path.exists(out_file):
        try:
            old = pd.read_csv(out_file)
            if 'Car_Score_Scaled' in old.columns:
                processed = set(zip(old['Year'], old['Round']))
                print(f"Resuming ({len(processed)} done)...")
        except: pass
    
    tasks = []
    for f in files:
        m = re.match(r"(\d{4})_(\d{2})_(.+)\.parquet", os.path.basename(f))
        if m and (int(m.group(1)), int(m.group(2))) not in processed:
            tasks.append((f, out_file, os.path.join(os.path.dirname(out_file), 'race_visualizations')))
    
    if not tasks: 
        audit_results(out_file); return

    print(f"Processing {len(tasks)} races...")
    with Pool(max(1, cpu_count()-1)) as p:
        results = p.map(process_single_race, tasks)
    
    h = not os.path.exists(out_file)
    for res in results:
        if res['status'] == 'success':
            res['data'].to_csv(out_file, mode='a', header=h, index=False)
            h = False
            print(f"[OK] {res['filename']}")
        else:
            print(f"[{res['status'].upper()}] {res['filename']}: {res.get('reason') or res.get('error')}")
            
    audit_results(out_file)

if __name__ == "__main__":
    RAW = r'c:\Users\jeff4\OneDrive\personal\school\2\2-겨울\DSL\300. EDA\f1_telemetry_raw'
    OUT = r'c:\Users\jeff4\OneDrive\personal\school\2\2-겨울\DSL\300. EDA\f1_car_performance.csv'
    run_pipeline(RAW, OUT)