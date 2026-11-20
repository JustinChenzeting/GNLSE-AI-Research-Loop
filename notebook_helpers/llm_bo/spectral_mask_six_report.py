"""Plot the complete six-band sweep and selected pulses using saved fields only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .spectral_mask_balanced import load_run
from .spectral_mask_balanced_report import (
    COLORS, STYLE_PATH, export_tables, finish, measured_costs, pulse_comparison, style,
)
from .spectral_mask_gd import save


def sweep_path(run):
    """Show both objective paths, brightness and the six factors in enumeration order."""
    rows = run.result['sweep']
    x = np.arange(1, len(rows)+1)
    feasible = np.array([r['feasible'] for r in rows])
    fig, axes = style.subplots(2, 2, width_mm=270, height_mm=245)
    fig.suptitle('All 729 masks · search grid: Nt = 16384, dz = 5 mm')
    for ax, metric in zip(axes[0], ('D_NL', 'D_IO')):
        scores = np.array([r[metric] for r in rows])
        best = np.minimum.accumulate(np.r_[run.result['baseline'][metric], np.where(feasible, scores, np.inf)])
        ax.scatter(x[~feasible], scores[~feasible], s=9, marker='x', color='#AAAAAA', label='Below brightness floor')
        ax.scatter(x[feasible], scores[feasible], s=14, marker='o', facecolors='none', edgecolors=COLORS[0], label='Feasible')
        ax.step(np.r_[0,x], best, where='post', color=COLORS[1], marker='', label='Best feasible incl. baseline')
        ax.scatter([0,len(rows)], best[[0,-1]], marker='D', s=28, color=COLORS[1], zorder=4)
        ax.set(xlabel='Sweep order (start → end)', ylabel=metric, title=f'LP01 {metric}', ylim=(-.05,1.25))
        ax.legend(loc='upper right')
    ax = axes[1,0]
    ax.plot(x, [r['brightness_ratio'] for r in rows], linestyle='', marker='.', markersize=3, color=COLORS[0])
    ax.axhline(.5, linestyle='--', color='#555555', label='50% floor')
    ax.set(xlabel='Sweep order (1 → 729)', ylabel='All-mode output photon ratio', title='Measured brightness', ylim=(-.03,1.2))
    ax.legend(loc='upper right')
    ax = axes[1,1]
    ax.imshow(np.array([r['point']['factors'] for r in rows]), aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest',
              extent=(.5,6.5,len(rows)+.5,.5))
    ax.set(xlabel='Band (increasing angular frequency)', ylabel='Sweep order (start → end)', title='Six factors · dark = 1, white = 0')
    ax.set_ylim(len(rows)+.5, -70)
    ax._balanced_x_ticks = list(range(1,7))
    ax._balanced_y_ticks = [1,243,486,729]
    return finish(fig, run.directory, 'sweep_path')


def convergence(run):
    """Plot feasible incumbents versus measured elapsed time, with finer checks marked."""
    rows = [run.result['baseline'], *run.result['sweep']]
    times = np.array([r['elapsed_seconds']/60 for r in rows])
    feasible = np.array([r['feasible'] for r in rows])
    fig, axes = style.subplots(1, 2, width_mm=270, height_mm=130)
    exported = []
    for ax, metric in zip(axes[0], ('D_NL','D_IO')):
        scores = np.array([r[metric] for r in rows])
        best = np.minimum.accumulate(np.where(feasible, scores, np.inf))
        ax.scatter(times[~feasible], scores[~feasible], s=9, marker='x', color='#AAAAAA', label='Infeasible trial')
        ax.step(times, best, where='post', color=COLORS[0], marker='', label='Best feasible, search grid')
        for key,label,marker,color in [('selected_fine','1 mm check','s',COLORS[1]),
                                       ('selected_refined','Finer time grid','D',COLORS[2])]:
            row = run.result['validation'][metric].get(key)
            if row:
                ax.scatter(row['elapsed_seconds']/60,row[metric],marker=marker,color=color,s=40,label=label)
        ax.set(xlabel='Active run time (min)', ylabel=metric, title=f'LP01 {metric}')
        ax.legend(loc='best')
        exported.extend({'objective':metric,'elapsed_seconds':float(t*60),'trial_score':float(s),
                         'feasible':bool(f),'best_feasible':float(b)} for t,s,f,b in zip(times,scores,feasible,best))
    pd.DataFrame(exported).to_csv(run.directory/'convergence_elapsed.csv',index=False)
    return finish(fig,run.directory,'convergence_elapsed')


def selected_masks(run):
    """Show the baseline and both selected six-band amplitude masks on physical frequency axes."""
    spec = json.loads((run.directory/'run.json').read_text())
    edges = np.array(spec['input']['band_edges_detuning_rad_ps'])
    cases = [('Unmasked',run.result['baseline']),
             ('D_NL selected',run.result['best_search']['D_NL']),
             ('D_IO selected',run.result['best_search']['D_IO'])]
    fig, axes = style.subplots(1,3,width_mm=330,height_mm=130)
    fig.suptitle('Shared field-amplitude factors · outside the controlled bandwidth: 1')
    for ax,(name,row) in zip(axes[0],cases):
        factors = row['point']['factors']
        ax.stairs(factors,edges,color=COLORS[0],baseline=None)
        ax.scatter((edges[:-1]+edges[1:])/2,factors,color=COLORS[0],marker='o',s=30)
        ax.set(xlabel='Angular-frequency detuning (rad/ps)',ylabel='Field-amplitude factor',title=name,ylim=(-.12,1.35))
        ax._balanced_y_ticks=[0,.5,1]
    pd.DataFrame([{'case':name,**{f'factor_{i+1}':v for i,v in enumerate(row['point']['factors'])}}
                  for name,row in cases]).to_csv(run.directory/'selected_masks.csv',index=False)
    return finish(fig,run.directory,'selected_masks')


def export_sweep(run):
    """Export all 729 rows and separate feasible rankings for the two main-mode objectives."""
    rows=[]
    for r in run.result['sweep']:
        rows.append({'combination':r['sample'],'evaluation':r['evaluation'],
                     **{f'factor_{i+1}':v for i,v in enumerate(r['point']['factors'])},
                     'D_NL':r['D_NL'],'D_IO':r['D_IO'],'brightness_ratio':r['brightness_ratio'],
                     'input_energy_retained':r['diagnostics']['input_energy_retained'],
                     'feasible':r['feasible'],'cached':r['cached'],'elapsed_seconds':r['elapsed_seconds']})
    frame=pd.DataFrame(rows)
    frame.to_csv(run.directory/'sweep_results.csv',index=False)
    for metric in ('D_NL','D_IO'):
        frame.loc[frame.feasible].sort_values([metric,'combination']).to_csv(run.directory/f'ranking_{metric}.csv',index=False)
    counts={'total_combinations':len(frame),'feasible':int(frame.feasible.sum()),'infeasible':int((~frame.feasible).sum()),
            'lowest_brightness':float(frame.brightness_ratio.min()),'highest_brightness':float(frame.brightness_ratio.max()),
            'mean_input_energy_retained':float(frame.input_energy_retained.mean())}
    save(run.directory/'sweep_summary.json',counts)
    spec=json.loads((run.directory/'run.json').read_text())
    edges=np.array(spec['input']['band_edges_detuning_rad_ps'])
    pd.DataFrame({'band':np.arange(1,7),'omega_low_rad_ps':edges[:-1],'omega_high_rad_ps':edges[1:],
                  'original_energy_fraction':spec['input']['band_energy_fractions'],
                  'search_frequency_samples':spec['input']['samples_per_band']}).to_csv(run.directory/'band_definitions.csv',index=False)
    return counts


def resource_figure(run):
    """Use one calendar-time axis for CPU, memory, GPU and completed solves, marking sampling gaps."""
    from datetime import datetime
    rows=[json.loads(line) for line in (run.directory/'cost_usage.jsonl').read_text().splitlines()]
    gpu=[json.loads(line) for line in (run.directory/'resource_usage.jsonl').read_text().splitlines()]
    created=datetime.fromisoformat(json.loads((run.directory/'run.json').read_text())['created_local']).timestamp()
    x=np.array([(r['timestamp_unix']-created)/60 for r in rows])
    gaps=np.flatnonzero(np.diff(x)>10/60)
    fig,axes=style.subplots(2,2,width_mm=270,height_mm=245)
    for ax in axes.flat:
        for gap in gaps:
            ax.axvspan(x[gap],x[gap+1],color='#DDDDDD',alpha=.5)
        ax.set_xlabel('Calendar time since start (min)')
    for ax,key,scale,title,ylabel in [
        (axes[0,0],'CPU_percent_one_core_100',1,'100% = one logical CPU','Process CPU usage (%)'),
        (axes[0,1],'process_tree_RSS_bytes',1e9,'Sampled memory, including Julia','Process-tree RSS (GB)')]:
        y=np.array([np.nan if r[key] is None else r[key]/scale for r in rows])
        y[gaps+1]=np.nan
        ax.plot(x,y,color=COLORS[0],marker='')
        ax.set(title=title,ylabel=ylabel)
    measured=[r for r in gpu if r['status']=='measured']
    gx=np.array([(datetime.fromisoformat(r['timestamp_utc']).timestamp()-created)/60 for r in measured])
    gy=np.array([r['device_utilization_percent'] for r in measured],dtype=float)
    if len(gx)>1:
        gy[np.flatnonzero(np.diff(gx)>15/60)+1]=np.nan
    axes[1,0].plot(gx,gy,color=COLORS[1],marker='')
    axes[1,0].set(title='Device-wide GPU; solver uses CPU',ylabel='GPU activity, including other apps (%)')
    evaluations=sorted((r for r in run.result['observations'] if not r['cached']),key=lambda r:r['evaluation'])
    axes[1,1].step([r['calendar_seconds']/60 for r in evaluations],np.arange(1,len(evaluations)+1),
                   where='post',color=COLORS[0],marker='')
    axes[1,1].set(title='One matched linear solve per pair',ylabel='Completed nonlinear GNLSE solves')
    fig.suptitle('Calendar time · shaded spans have no resource samples')
    save(run.directory/'telemetry_clock_gaps.json',{
        'clock_basis':'Calendar timestamps for all four panels; active solver/history timers use monotonic time.',
        'gaps_over_10_seconds':[{'start_min':float(x[i]),'end_min':float(x[i+1]),'seconds':float(60*(x[i+1]-x[i]))} for i in gaps],
        'interpretation':'The samples identify gaps; their cause (for example suspension or a clock change) was not measured.'})
    return finish(fig,run.directory,'measured_resources')


def annotate_cost_clocks(cost):
    """Keep elapsed timers distinct and avoid deriving overhead from incompatible clocks."""
    cost['elapsed_clock_bases']={
        'Python_monotonic':['active_runtime_seconds','pair_wall_seconds','search_pair_wall_seconds',
                            'baseline_validation_pair_wall_seconds','controller_other_seconds'],
        'calendar_timestamps':['calendar_elapsed_seconds'],
        'Julia_timed_reported_elapsed':['solver_linear_seconds','solver_nonlinear_seconds'],
        'OS_CPU_counters':['observed_CPU_seconds','observed_CPU_core_hours'],
    }
    cost['pair_setup_compilation_IO_seconds']=None
    cost['pair_setup_compilation_IO_basis']='Not separately measured. Julia elapsed totals exceeded Python monotonic pair durations across the recorded clock gaps, so subtracting them does not measure setup overhead.'
    return cost


def build_report(directory):
    """Write tables, cost records and all required figures from the completed sweep."""
    started,cpu_started=time.monotonic(),time.process_time()
    run=load_run(directory)
    if len(run.result['sweep']) != 729:
        raise ValueError('The full enumeration has not finished.')
    style.configure()
    counts=export_sweep(run)
    export_tables(run)
    cost=annotate_cost_clocks(measured_costs(run))
    cost['calendar_minus_active_seconds']=cost['calendar_elapsed_seconds']-cost['active_runtime_seconds']
    cost['clock_basis_note']='Active duration uses the monotonic timer; calendar duration uses wall-clock timestamps. The difference is recorded, but its cause was not measured.'
    save(run.directory/'run_costs.json',cost)
    figures={'sweep_path':sweep_path(run),'convergence':convergence(run),'selected_masks':selected_masks(run),
             'resources':resource_figure(run)}
    cases=[('Unmasked baseline',dict(run.result['baseline_refined'],presentation_status='Reference'))]
    for metric in ('D_NL','D_IO'):
        validation=run.result['validation'][metric]
        row=validation.get('selected_refined')
        if row is None:
            raise ValueError(f'Missing refined output for {metric}; no substitute curve is allowed.')
        label=f'{metric} checks passed' if validation['status']=='pass' else 'Preliminary'
        cases.append((f'{metric} selected',dict(row,presentation_status=label)))
    for mode,name in [(0,'lp01'),(1,'lp11_cos'),(2,'lp11_sin'),(None,'all_modes')]:
        figures[f'pulses_{name}']=pulse_comparison(run,cases,f'pulse_comparison_{name}',mode=mode)
    figures['pulse_tails']=pulse_comparison(run,cases,'pulse_comparison_lp01_full_window',mode=0,full_time=True)
    source_paths=[Path(__file__),Path(__file__).with_name('spectral_mask_balanced_report.py'),STYLE_PATH]
    for p in source_paths:
        shutil.copy2(p,run.directory/'sources'/p.name)
    manifest={'figures':figures,'new_objective_pairs':0,
              'report_generation_seconds':time.monotonic()-started,
              'report_generation_CPU_seconds':time.process_time()-cpu_started,
              'report_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    old=run.directory/'report_manifest.json'
    if old.exists():
        previous=json.loads(old.read_text())
        passes=previous.get('report_generation_passes',[{'seconds':previous['report_generation_seconds'],
                                                        'CPU_seconds':previous['report_generation_CPU_seconds']}])
    else:
        passes=[]
    passes.append({'seconds':manifest['report_generation_seconds'],'CPU_seconds':manifest['report_generation_CPU_seconds']})
    manifest['report_generation_passes']=passes
    manifest['report_generation_seconds']=sum(p['seconds'] for p in passes)
    manifest['report_generation_CPU_seconds']=sum(p['CPU_seconds'] for p in passes)
    save(old,manifest)
    readme=(f'# Six-band exhaustive spectral-mask sweep\n\n'
            f'All {counts["total_combinations"]} combinations of six shared field-amplitude factors in {{0, 0.5, 1}} were evaluated. '
            f'{counts["feasible"]} meet the 50% all-mode output-photon floor on the search grid.\n\n'
            f'Both LP01 objectives are ranked independently. The physics, search grid and finer-check procedure follow the preceding spectral-mask run. '
            f'All-zero controlled factors retain the unchanged outside-band tails.\n\n'
            f'See `selected_results.csv`, `validation_checks.csv`, `sweep_results.csv` and the two `ranking_*.csv` files. '
            f'Each case has saved input, matched linear output and nonlinear output, shown in time and wavelength for all three modes and their spatially integrated power.\n\n'
            f'Finite-grid minima are not continuous-space optima. Only the selected candidates receive finer checks; the entire ranking has not been validated at finer resolution. '
            f'D_NL validation: **{run.result["validation"]["D_NL"]["status"]}**; D_IO validation: **{run.result["validation"]["D_IO"]["status"]}**.\n\n'
            f'Active run timer: {cost["active_runtime_seconds"]/60:.2f} min; calendar duration: {cost["calendar_elapsed_seconds"]/60:.2f} min; '
            f'nonlinear GNLSE solves: {cost["nonlinear_GNLSE_completed"]}; matched linear solves: {cost["linear_completed"]}. '
            f'All measured costs and their scope are in `run_costs.json`; report/export costs are added separately.\n')
    (run.directory/'README.md').write_text(readme)
    print(json.dumps({'figures':len(figures),'sweep':counts,'report_seconds':manifest['report_generation_seconds']},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    build_report(parser.parse_args().directory)
