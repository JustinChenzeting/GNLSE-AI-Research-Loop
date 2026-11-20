"""Reports from saved balanced-mask observations; never starts a propagation."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import spectral_mask as sm
from .spectral_mask_balanced import OBJECTIVES, field_scores, load_run
from .spectral_mask_gd import save

from . import plot_style as style

STYLE_PATH = Path(style.__file__)
COLORS = ('#4477AA', '#CC8844', '#8877AA')
FIELD_STYLES = (("input", "Input", '#666666', ':'),
                ("linear", "Linear output", '#CC8844', '--'),
                ("nonlinear", "Nonlinear output", '#4477AA', '-'))


def finish(fig, directory, name):
    """Save the figure in PNG and PDF formats."""
    for index, ax in enumerate(fig.axes):
        if ax.get_label() == '<colorbar>':
            continue
        style.style_axes(ax)
        ax.tick_params(axis='both', pad=6)
        if not ax.images:
            ax.margins(x=.06, y=.14)
        if hasattr(ax, '_balanced_y_ticks'):
            ax.set_yticks(ax._balanced_y_ticks)
        if hasattr(ax, '_balanced_x_ticks'):
            ax.set_xticks(ax._balanced_x_ticks)
        style.panel_label(ax, chr(97+index))
    path = Path(directory) / (name + '.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
    return str(path)


def coverage(run):
    """Export proposed and feasible-subset level counts without conflating them."""
    masks = np.load(run.directory/'balanced_masks.npy')
    ids = np.rint(masks*5).astype(int)
    joint = run.result['joint']
    feasible_ids = ids[[r['sample']-1 for r in joint if r['feasible']]]
    generated = np.array([(ids == k).sum(axis=0) for k in range(6)])
    feasible = np.array([(feasible_ids == k).sum(axis=0) for k in range(6)])
    rows = [{'level': k/5, 'generated_min_per_factor': int(generated[k].min()),
             'generated_max_per_factor': int(generated[k].max()),
             'feasible_min_per_factor': int(feasible[k].min()),
             'feasible_max_per_factor': int(feasible[k].max())} for k in range(6)]
    pd.DataFrame(rows).to_csv(run.directory/'coverage_summary.csv', index=False)
    np.savetxt(run.directory/'feasible_level_counts.csv', feasible, delimiter=',', fmt='%d')
    spec = json.loads((run.directory/'run.json').read_text())['input']
    predicted_energy = spec['outside_band_energy_fraction'] + masks**2 @ np.array(spec['band_energy_fractions'])
    save(run.directory/'sampling_feasibility.json', {
        'evaluated_joint_masks': len(joint), 'feasible_joint_masks': sum(r['feasible'] for r in joint),
        'mean_input_energy_retained_from_design': float(predicted_energy.mean()),
        'measured_max_joint_brightness': max(r['brightness_ratio'] for r in joint),
        'interpretation': 'Amplitude factors are squared in input energy. Exact level balance places the ensemble mean near 36.7% input energy; it does not enforce the 50% output-photon floor. The output floor is checked by simulation, not by this estimate.'})
    return pd.DataFrame(rows)


def measured_costs(run):
    """Separate actual solver work, controller costs, resources and unmeasured costs."""
    records = [json.loads(p.read_text()) for p in sorted((run.directory/'gnlse').glob('evaluation_*/measurement.json'))]
    ok = [r for r in records if r['status'] == 'ok']
    cpu = json.loads((run.directory/'cost_summary.json').read_text())
    gpu = json.loads((run.directory/'resource_summary.json').read_text())
    source_rows = {r['evaluation']: r for r in run.result['observations'] if not r['cached']}
    rows = []
    for r in ok:
        source = source_rows[r['evaluation']]
        rows.append({'evaluation': r['evaluation'], 'role': source['role'], 'search': source['search'],
                     'end_elapsed_seconds': source['elapsed_seconds'], 'pair_wall_seconds': r['wall_seconds'],
                     'linear_seconds': r['metrics']['linear_seconds'],
                     'nonlinear_seconds': r['metrics']['nonlinear_seconds'],
                     'D_NL': source['D_NL'], 'D_IO': source['D_IO'],
                     'brightness_ratio': source['brightness_ratio'], 'feasible': source['feasible']})
    frame = pd.DataFrame(rows)
    frame.to_csv(run.directory/'evaluation_costs.csv', index=False)
    solver = float(frame[['linear_seconds', 'nonlinear_seconds']].sum().sum())
    pair_seconds = float(frame['pair_wall_seconds'].sum())
    result = {
        'completed_objective_pairs': len(ok), 'attempted_pairs': len(records),
        'failed_or_incomplete_pairs': len(records)-len(ok),
        'nonlinear_GNLSE_completed': sum(r['metrics'].get('nonlinear_solves', 0) for r in ok),
        'linear_completed': sum(r['metrics'].get('linear_solves', 0) for r in ok),
        'cache_hits': run.result['cache_hits'], 'search_pairs': run.result['search_pairs'],
        'active_runtime_seconds': run.result['active_runtime_seconds'],
        'calendar_elapsed_seconds': run.result['calendar_elapsed_seconds'],
        'solver_linear_seconds': float(frame['linear_seconds'].sum()),
        'solver_nonlinear_seconds': float(frame['nonlinear_seconds'].sum()),
        'pair_wall_seconds': pair_seconds,
        'pair_setup_compilation_IO_seconds': pair_seconds-solver,
        'controller_other_seconds': run.result['active_runtime_seconds']-pair_seconds,
        'search_pair_wall_seconds': float(frame.loc[frame.search, 'pair_wall_seconds'].sum()),
        'baseline_validation_pair_wall_seconds': float(frame.loc[~frame.search, 'pair_wall_seconds'].sum()),
        'observed_CPU_seconds': cpu['observed_process_CPU_seconds'],
        'observed_CPU_core_hours': cpu['observed_CPU_core_hours'],
        'sampled_CPU_percent_one_core_100_mean': cpu['sampled_CPU_percent_one_core_100_time_weighted_mean'],
        'sampled_CPU_percent_machine_capacity_mean': cpu['sampled_CPU_percent_machine_capacity_time_weighted_mean'],
        'sampled_peak_process_tree_RSS_bytes': cpu['sampled_peak_process_tree_RSS_bytes'],
        'raw_fields_bytes': sum(Path(r['artifacts']['fields']).stat().st_size for r in ok),
        'GPU_solver_compute': 'not used; CPU solver code path',
        'GPU_system_wide_utilization_percent_mean': gpu['device_utilization_percent_mean'],
        'GPU_system_wide_utilization_percent_max': gpu['device_utilization_percent_max'],
        'AI_optimizer_calls': 0, 'AI_optimizer_wait_seconds': 0,
        'electricity_kWh': None, 'monetary_cost': None,
        'cost_scope': 'Measured campaign costs include baseline/setup, search, checks and solver shutdown. Figure/notebook rendering is separate; implementation and conversation time are not included. Electricity and money were not measured. GPU readings are device-wide, not attributable to this run.',
    }
    save(run.directory/'run_costs.json', result)
    return result


def export_tables(run):
    """Write every observation, factor and finalist/validation score to tabular files."""
    flattened = []
    for r in run.result['observations']:
        row = {k: r[k] for k in ('evaluation', 'role', 'D_NL', 'D_IO', 'brightness_ratio',
                                'valid', 'feasible', 'cached', 'search', 'elapsed_seconds', 'request_seconds')}
        row.update({f'factor_{i+1:03d}': v for i, v in enumerate(r['point']['factors'])})
        row.update(r['resolution'])
        flattened.append(row)
    pd.DataFrame(flattened).to_csv(run.directory/'all_observations.csv', index=False)
    summary = []
    for objective in OBJECTIVES:
        validation = run.result['validation'].get(objective, {})
        for resolution, key in [('search', 'selected_search'), ('fine step', 'selected_fine'), ('refined grid', 'selected_refined')]:
            row = validation.get(key, run.result['best_search'][objective] if key == 'selected_search' else None)
            if row is not None:
                summary.append({'optimized_metric': objective, 'resolution': resolution,
                                'evaluation': row['evaluation'], 'D_NL': row['D_NL'], 'D_IO': row['D_IO'],
                                'brightness_ratio': row['brightness_ratio'], 'feasible': row['feasible'],
                                'validation': validation.get('status', 'not_run')})
    pd.DataFrame(summary).to_csv(run.directory/'selected_results.csv', index=False)
    checks = [{'objective': obj, 'check': key, 'status': value,
               'absolute_difference': val.get('absolute_differences', {}).get(key)}
              for obj, val in run.result['validation'].items() for key, value in val['checks'].items()]
    pd.DataFrame(checks).to_csv(run.directory/'validation_checks.csv', index=False)
    cross_checks = []
    for target, val in run.result['validation'].items():
        if 'selected_fine' in val and 'selected_refined' in val:
            for metric in OBJECTIVES:
                delta = abs(val['selected_fine'][metric]-val['selected_refined'][metric])
                cross_checks.append({'optimized_metric': target, 'reported_metric': metric,
                                     'absolute_grid_change': delta, 'within_2e_4': delta <= sm.TOLERANCE,
                                     'used_for_this_branch_acceptance': metric == target})
    pd.DataFrame(cross_checks).to_csv(run.directory/'reported_metric_grid_changes.csv', index=False)
    return pd.DataFrame(summary)


def joint_path(run):
    """Show the sampled mask route, both scores and measured brightness together."""
    joint = run.result['joint']
    fig, axes = style.subplots(2, 2, width_mm=270, height_mm=245)
    fig.suptitle('Search-resolution diagnostics · Nt = 16384 · dz = 5 mm')
    x = np.array([r['sample'] for r in joint])
    feasible = np.array([r['feasible'] for r in joint])
    for ax, objective in zip(axes[0], OBJECTIVES):
        scores = np.array([r[objective] for r in joint])
        best = np.minimum.accumulate(np.r_[run.result['baseline'][objective], np.where(feasible, scores, np.inf)])
        ax.scatter(x[~feasible], scores[~feasible], color='#999999', marker='x', s=12, label='Infeasible')
        ax.scatter(x[feasible], scores[feasible], facecolors='none', edgecolors=COLORS[0], s=22, label='Feasible')
        ax.step(np.r_[0, x], best, where='post', color=COLORS[1], linestyle='-', marker='', label='Best feasible incl. baseline')
        ax.set(xlabel='Sample order (start → end)', ylabel=objective, title=f'Balanced batch · LP01 {objective}')
        ax.legend(loc='best')
    axes[1, 0].scatter(x, [r['brightness_ratio'] for r in joint], s=15, color=COLORS[0], marker='o')
    axes[1, 0].axhline(.5, color='#555555', linestyle='--', label='50% floor')
    axes[1, 0].set(xlabel='Sample order (start → end)', ylabel='All-mode output photon ratio', title='Measured brightness')
    axes[1, 0].legend(loc='upper right')
    mask_image = axes[1, 1].imshow(np.array([r['point']['factors'] for r in joint]), aspect='auto', cmap='Blues',
                                 vmin=0, vmax=1, extent=(.5, 100.5, len(joint)+.5, .5))
    axes[1, 1].set(xlabel='Band index', ylabel='Sample order (start → end)', title='All 100 factors; dark = higher transmission')
    axes[1, 1].set_ylim(len(joint)+.5, .5-max(1, .1*len(joint)))
    axes[1, 1]._balanced_y_ticks = np.unique(np.linspace(1, len(joint), 4).astype(int))
    axes[1, 1]._balanced_x_ticks = [1, 25, 50, 75, 100]
    return finish(fig, run.directory, 'joint_sampling_path')


def local_path(run, objective):
    """Show accepted chain paths, rejected trials, incumbent masks and elapsed time."""
    data = run.result['local'].get(objective)
    if not data:
        return None
    proposals = data['proposals']
    fig, axes = style.subplots(2, 2, width_mm=270, height_mm=245)
    fig.suptitle('Search: Nt = 16384, dz = 5 mm · refinement checks shown separately')
    for chain, start in enumerate(data['starts'], 1):
        accepted = [p for p in proposals if p['chain'] == chain and p['accepted']]
        xs = [0] + [p['iteration'] for p in accepted]
        ys = [start[objective]] + [p[objective] for p in accepted]
        axes[0, 0].step(xs, ys, where='post', color=COLORS[chain-1], marker=['o', 's', '^'][chain-1],
                        linestyle=['-', '--', '-.'][chain-1], label=f'Accepted chain {chain}')
        axes[0, 0].scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=55, color=COLORS[chain-1], marker='D')
    rejected = [p for p in proposals if not p['accepted']]
    axes[0, 0].scatter([p['iteration'] for p in rejected], [p[objective] for p in rejected],
                       s=12, color='#999999', marker='x', label='Rejected')
    axes[0, 0].set(xlabel='Proposal number (start → end)', ylabel=objective, title=f'LP01 {objective} · local route')
    axes[0, 0].legend(loc='best')
    for accepted, label, marker, color in [(True, 'Accepted', 'o', COLORS[0]), (False, 'Rejected', 'x', '#999999')]:
        rows = [p for p in proposals if p['accepted'] == accepted]
        axes[0, 1].scatter([p['iteration'] for p in rows], [p['brightness_ratio'] for p in rows],
                           marker=marker, color=color, s=18, label=label)
    axes[0, 1].axhline(.5, linestyle='--', color='#555555', label='50% floor')
    axes[0, 1].set(xlabel='Proposal number', ylabel='All-mode output photon ratio', title='Acceptance includes brightness')
    axes[0, 1].legend(loc='best')
    start_best = min(data['starts'], key=lambda r: r[objective])
    incumbents = [start_best]
    for p in proposals:
        if p['accepted'] and p[objective] < incumbents[-1][objective]:
            incumbents.append(p)
    axes[1, 0].imshow(np.array([r['point']['factors'] for r in incumbents]), cmap='Blues', vmin=0, vmax=1,
                      aspect='auto', extent=(.5, 100.5, len(incumbents)-.5, -.5))
    axes[1, 0].set(xlabel='Band index', ylabel='Incumbent update (start → end)', title='Best mask across the three chains')
    axes[1, 0].set_ylim(len(incumbents)-.5, -.5-max(.6, .1*len(incumbents)))
    axes[1, 0]._balanced_y_ticks = np.unique(np.linspace(0, len(incumbents)-1, 4).astype(int))
    axes[1, 0]._balanced_x_ticks = [1, 25, 50, 75, 100]
    axes[1, 1].step([r['elapsed_seconds']/60 for r in incumbents], [r[objective] for r in incumbents],
                    where='post', color=COLORS[0], marker='o', label='Search-grid incumbent')
    val = run.result['validation'].get(objective, {})
    for key, label, marker, color in [('selected_fine', '1 mm check', 's', COLORS[1]),
                                       ('selected_refined', 'Finer time grid', 'D', COLORS[2])]:
        if key in val:
            r = val[key]
            axes[1, 1].scatter(r['elapsed_seconds']/60, r[objective], marker=marker, color=color, s=35, label=label)
    axes[1, 1].set(xlabel='Campaign elapsed time (min)', ylabel=objective, title='Measured search and validation values')
    axes[1, 1].legend(loc='best')
    return finish(fig, run.directory, f'{objective.lower()}_local_path')


def pulse_curves(row, mode):
    """Convert saved fields to temporal power and physical spectral energy density."""
    path = Path(row['artifacts']['fields'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row['artifacts']['fields_sha256']:
        raise ValueError(f'Field hash mismatch: {path}')
    with np.load(path) as f:
        t, wavelength = f['time_ps'].copy(), f['wavelength_nm'].copy()
        window = float(f['time_window_ps'])
        valid = np.flatnonzero(np.isfinite(wavelength) & (wavelength > 0))
        order = valid[np.argsort(wavelength[valid])]
        curves = {}
        for key, *_ in FIELD_STYLES:
            field = f[key] if mode is None else f[key][:, mode:mode+1, :]
            power = np.sum(abs(field)**2, axis=(1, 2))/1000
            spectral_power = np.sum(abs(np.fft.fftshift(np.fft.ifft(field, axis=0), axes=0))**2, axis=(1, 2))
            density = spectral_power[order]*window*1e-3/(2*np.pi/window)*(2*np.pi*299792.458/wavelength[order]**2)
            curves[key] = (power, density)
        return t, wavelength[order], curves


def pulse_comparison(run, cases, stem, mode=0, full_time=False):
    """Place cases side by side with input/linear/nonlinear time and wavelength rows."""
    arrays = [(name, row, pulse_curves(row, mode)) for name, row in cases]
    scope = ['LP01', 'LP11-cos', 'LP11-sin'][mode] if mode is not None else 'All modes'
    fig, axes = style.subplots(2, len(cases), width_mm=110*len(cases), height_mm=225)
    resolutions = {tuple(sorted(row['resolution'].items())) for _, row in cases}
    if len(resolutions) != 1:
        raise ValueError('Use matched numerical grids for this side-by-side pulse comparison.')
    resolution = cases[0][1]['resolution']
    label = 'metric-specific status shown in each column'
    if any(row.get('saved_validation_status', 'pass') != 'pass' for _, row in cases):
        label = 'preliminary historical comparison'
    resolution_label = f"Nt = {resolution['Nt']} · dz = {1000*resolution['dz_m']:g} mm · {label}"
    fig.suptitle(resolution_label)
    power_max = max(float(curves[key][0].max()) for _, _, (_, _, curves) in arrays for key, *_ in FIELD_STYLES)
    spectrum_max = max(float(curves[key][1].max()) for _, _, (_, _, curves) in arrays for key, *_ in FIELD_STYLES)
    exported = {}
    def draw(ax, data, name, row, domain):
        t, wavelength, curves = data
        for key, label, color, linestyle in FIELD_STYLES:
            x, y = (t, curves[key][0]) if domain == 0 else (wavelength, curves[key][1])
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker='', linewidth=1.3)
        if domain == 0:
            ax.set(xlabel='Time (ps)', ylabel='Power (kW)', xlim=(t[0], t[-1]) if full_time else (-2, 2))
            if full_time:
                ax.set_yscale('log'); ax.set_ylim(max(1e-10, power_max*1e-10), power_max*50)
            else:
                ax.set_ylim(0, power_max*1.5)
            status = row.get('presentation_status', run.result.get('result_label', 'preliminary'))
            ax.set_title(f'{name} · {scope}\n{status} · brightness {row["brightness_ratio"]:.2%}')
        else:
            ax.set(xlabel='Wavelength (nm)', ylabel='Spectral energy density (nJ/nm)', xlim=(1330, 1430),
                   ylim=(0, spectrum_max*1.5), title=f'{name} · same saved field')
        ax.legend(loc='upper right')
    for col, (name, row, data) in enumerate(arrays):
        for domain in (0, 1):
            draw(axes[domain, col], data, name, row, domain)
            single, one = style.subplots(width_mm=135, height_mm=125)
            single.suptitle(f"Nt = {resolution['Nt']} · dz = {1000*resolution['dz_m']:g} mm")
            draw(one[0, 0], data, name, row, domain)
            finish(single, run.directory/'individual_panels', f'{stem}_{col+1}_{"time" if domain == 0 else "wavelength"}')
        t, wave, curves = data
        exported[f'case_{col+1}_time_ps'] = t
        exported[f'case_{col+1}_wavelength_nm'] = wave
        for key, *_ in FIELD_STYLES:
            exported[f'case_{col+1}_{key}_kW'] = curves[key][0]
            exported[f'case_{col+1}_{key}_nJ_per_nm'] = curves[key][1]
    np.savez_compressed(run.directory/f'{stem}_source_data.npz', **exported)
    save(run.directory/f'{stem}_sources.json', {
        'cases': [{'name': name, 'evaluation': row['evaluation'], 'fields': row['artifacts']['fields'],
                   'sha256': row['artifacts']['fields_sha256'], 'resolution': row['resolution']} for name, row in cases],
        'mode_scope': scope, 'spectral_density': 'T * 1e-3 * |ifft(A)|^2 / delta_omega * |domega/dlambda|, in nJ/nm; no curve normalization',
        'comparison': 'Each candidate has its own input, matched linear output and nonlinear output; no time alignment. Scores use full saved grids, not plotted crops.'})
    return finish(fig, run.directory, stem)


def resource_figure(run):
    """Show sampled resource usage and completed evaluation counts against time."""
    rows = [json.loads(line) for line in (run.directory/'cost_usage.jsonl').read_text().splitlines()]
    gpu = [json.loads(line) for line in (run.directory/'resource_usage.jsonl').read_text().splitlines()]
    costs = pd.read_csv(run.directory/'evaluation_costs.csv')
    x = [(r['timestamp_unix']-rows[0]['timestamp_unix'])/60 for r in rows]
    fig, axes = style.subplots(2, 2, width_mm=270, height_mm=245)
    axes[0, 0].plot(x, [r['CPU_percent_one_core_100'] for r in rows], color=COLORS[0], marker='')
    axes[0, 0].set(xlabel='Elapsed time (min)', ylabel='Process CPU usage (%)', title='100% = one logical CPU')
    axes[0, 1].plot(x, [r['process_tree_RSS_bytes']/1e9 for r in rows], color=COLORS[0], marker='')
    axes[0, 1].set(xlabel='Elapsed time (min)', ylabel='Sampled process-tree RSS (GB)', title='Memory, including Julia')
    measured = [r for r in gpu if r['status'] == 'measured']
    axes[1, 0].plot([r['elapsed_seconds']/60 for r in measured], [r['device_utilization_percent'] for r in measured],
                    color=COLORS[1], marker='')
    axes[1, 0].set(xlabel='Elapsed time (min)', ylabel='Device-wide GPU usage (%)', title='Includes other apps; solver uses CPU')
    axes[1, 1].step(costs.end_elapsed_seconds/60, np.arange(1, len(costs)+1), where='post', color=COLORS[0],
                    marker='', label='Completed nonlinear GNLSE solves')
    axes[1, 1].set(xlabel='Elapsed time (min)', ylabel='Completed simulations', title='One linear solve also saved per pair')
    axes[1, 1].legend(loc='upper left')
    return finish(fig, run.directory, 'measured_resources')


def convergence_figure(run):
    """Show each objective's best feasible score over the entire measured run."""
    fig, axes = style.subplots(1, 2, width_mm=270, height_mm=130)
    exports = []
    for ax, objective in zip(axes[0], OBJECTIVES):
        history = [run.result['baseline']] + [r for r in run.result['observations']
            if r['search'] and (r['role'] == 'balanced joint' or r.get('objective') == objective)]
        times = np.array([r['elapsed_seconds']/60 for r in history])
        scores = np.array([r[objective] for r in history])
        feasible = np.array([r['feasible'] for r in history])
        best = np.minimum.accumulate(np.where(feasible, scores, np.inf))
        ax.scatter(times[~feasible], scores[~feasible], marker='x', color='#AAAAAA', s=9, label='Infeasible trial')
        ax.step(times, best, where='post', color=COLORS[0], marker='', label='Best feasible, search grid')
        for key, label, marker, color in [('selected_fine', '1 mm check', 's', COLORS[1]),
                                           ('selected_refined', 'Finer time grid', 'D', COLORS[2])]:
            row = run.result['validation'].get(objective, {}).get(key)
            if row:
                ax.scatter(row['elapsed_seconds']/60, row[objective], color=color, marker=marker, s=40, label=label)
        ax.set(xlabel='Campaign elapsed time (min)', ylabel=objective, title=f'LP01 {objective} · full search')
        ax.legend(loc='best')
        exports.extend({'objective': objective, 'elapsed_seconds': float(t*60), 'trial_score': float(s),
                        'feasible': bool(f), 'best_feasible': float(b)} for t, s, f, b in zip(times, scores, feasible, best))
    pd.DataFrame(exports).to_csv(run.directory/'convergence_elapsed.csv', index=False)
    return finish(fig, run.directory, 'convergence_elapsed')


def build_report(directory):
    """Generate report data and figures entirely from the completed saved campaign."""
    started = time.monotonic()
    cpu_started = time.process_time()
    run = load_run(directory)
    if run.result['status'] == 'running':
        raise RuntimeError('Wait for the bounded optical run before making the final report.')
    prior_manifest = run.directory/'report_manifest.json'
    previous = json.loads(prior_manifest.read_text()) if prior_manifest.exists() else {}
    passes = previous.get('report_generation_passes', [])
    if previous and not passes:
        passes = [{'seconds': previous['report_generation_seconds'], 'CPU_seconds': previous['report_generation_CPU_seconds']}]
    style.configure()
    save(run.directory/'metric_key_map.json', {
        'state.json and all_observations.csv': {'D_NL': 'LP01 matched linear/nonlinear overlap distortion',
                                             'D_IO': 'LP01 candidate input/nonlinear overlap distortion'},
        'gnlse/evaluation_*/lp01_scores.json': 'Explicit LP01 scores plus total-field diagnostics; recomputed from the saved fields.',
        'gnlse/evaluation_*/measurement.json:D_NL': 'Unchanged legacy worker key: this is D_NL_total_field, not the optimized LP01 score.',
        'brightness': 'All-mode nonlinear output photons divided by the unmasked fine-step reference on the matching time grid.'})
    report_sources = [Path(__file__), STYLE_PATH,
                      Path(__file__).with_name('spectral_mask_cost_monitor.py'),
                      Path(__file__).with_name('spectral_mask_resources.py')]
    source_hashes = {}
    for path in report_sources:
        data = path.read_bytes()
        (run.directory/'sources'/path.name).write_bytes(data)
        source_hashes[str(path)] = hashlib.sha256(data).hexdigest()
    coverage(run)
    costs = measured_costs(run)
    summary = export_tables(run)
    figures = {'joint_path': joint_path(run), 'resources': resource_figure(run),
               'convergence': convergence_figure(run)}
    for objective in OBJECTIVES:
        figures[f'{objective}_path'] = local_path(run, objective)
    cases = [('Unmasked baseline', dict(run.result['baseline_refined'], presentation_status='Reference'))]
    cases += [(f'{obj} selected', dict(run.result['validation'][obj]['selected_refined'],
                  presentation_status=f'{obj} checks passed' if run.result['validation'][obj]['status'] == 'pass' else 'Preliminary'))
              for obj in OBJECTIVES]
    for mode, name in [(0, 'lp01'), (1, 'lp11_cos'), (2, 'lp11_sin'), (None, 'all_modes')]:
        figures[f'pulses_{name}'] = pulse_comparison(run, cases, f'pulse_comparison_{name}', mode)
    figures['pulse_tails'] = pulse_comparison(run, cases, 'pulse_comparison_lp01_full_window', 0, full_time=True)
    passes.append({'seconds': time.monotonic()-started, 'CPU_seconds': time.process_time()-cpu_started})
    save(run.directory/'report_manifest.json', {
        'figures': figures, 'report_generation_seconds': sum(p['seconds'] for p in passes),
        'report_generation_CPU_seconds': sum(p['CPU_seconds'] for p in passes),
        'report_generation_passes': passes,
        'new_objective_pairs': 0, 'style_helper': str(STYLE_PATH),
        'style_helper_sha256': hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest(),
        'report_source_sha256': source_hashes})
    columns = ['optimized_metric', 'resolution', 'D_NL', 'D_IO', 'brightness_ratio', 'validation']
    result_text = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---']*len(columns)) + ' |\n'
    for row in summary.to_dict('records'):
        result_text += '| ' + ' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns) + ' |\n'
    (run.directory/'README.md').write_text(
        '# Balanced spectral-mask sampling and local refinement\n\n'
        f"Result status: **{run.result['result_label']}**. {run.result['stop_reason']}\n\n"
        'The 300 initial masks use each of six levels exactly 50 times per factor. '
        'Both subsequent searches target LP01, with an all-mode 50% output-photon floor. '
        'All fields and rejected measurements are saved. Local restoration is counted within the 150-proposal allowance per objective.\n\n'
        'The unchanged worker stores its raw total-field score under the legacy `D_NL` key in `measurement.json`. '
        'Use `lp01_scores.json`, `state.json` and `all_observations.csv` for the current LP01 scores; see `metric_key_map.json`.\n\n'
        f"Balanced batch: {sum(r['feasible'] for r in run.result['joint'])}/{len(run.result['joint'])} masks meet the output-photon floor. "
        'The equal allocation of amplitude levels gives about 36.7% mean input-energy transmission; feasibility is a separate measured requirement.\n\n'
        + result_text + '\n\n'
        f"Measured: {costs['completed_objective_pairs']} pairs = {costs['nonlinear_GNLSE_completed']} nonlinear GNLSE solves "
        f"plus {costs['linear_completed']} linear solves; {costs['active_runtime_seconds']/60:.2f} active minutes, "
        f"{costs['calendar_elapsed_seconds']/60:.2f} calendar minutes. CPU {costs['observed_CPU_core_hours']:.3f} core-hours; "
        f"sampled peak RSS {costs['sampled_peak_process_tree_RSS_bytes']/1e9:.2f} GB. No optimizer LLM calls.\n\n"
        'Numerical checks are in validation_checks.csv. Any failed or missing check blocks promotion. '
        'Search-grid winners are not certified converged optima. Electricity and monetary cost were not measured. '
        '[Observations](all_observations.csv) · '
        '[Costs](run_costs.json) · [Validation](validation_checks.csv)\n')
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    build_report(parser.parse_args().directory)
