"""Compare successful cold builds without mixing setup time into compile time."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def minutes(start, end):
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 60


def collect(run_id):
    url = f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}/jobs?per_page=100"
    jobs = json.loads(subprocess.check_output(['gh', 'api', url], text=True))['jobs']
    rows = []
    for job in jobs:
        name = job['name']
        if 'onnxruntime-' not in name or 'cuda-' not in name:
            continue
        platform = 'Windows' if '-win-' in name else 'Linux'
        match = re.search(r'-cuda-(baseline-r[23]|contrib-r[23]|all-removable-strict-r[123]|all-removable-r[123]|combined-r[23]|combined-no-quick|combined)(?:,|$)', name)
        if match:
            variant = match[1]
        elif '-cuda-baseline,' in name:
            variant = 'baseline-r1'
        elif '-cuda-contrib,' in name:
            variant = 'contrib-r1'
        else:
            continue
        build = next((step for step in job['steps'] if step['name'] == 'Build ONNX Runtime'), None)
        if not build or not build.get('completed_at') or not job.get('completed_at'):
            continue
        rows.append(dict(run=run_id, platform=platform, variant=variant, result=job['conclusion'],
                         total=minutes(job['started_at'], job['completed_at']),
                         preparation=minutes(job['started_at'], build['started_at']),
                         build=minutes(build['started_at'], build['completed_at']),
                         postprocess=minutes(build['completed_at'], job['completed_at'])))
    return rows


def main():
    rows = collect(40314439974) + collect(40385495477) + collect(int(os.environ['GITHUB_RUN_ID']))
    fields = ['run', 'platform', 'variant', 'result', 'total', 'preparation', 'build', 'postprocess']
    with Path('CUDA_PHASE_TIMINGS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# CUDA repeated cold-build timings', '',
             'All durations are minutes. Preparation includes dependency installation, checkout and CMake/toolchain setup before the build step.',
             'The build step includes both CMake configuration and compilation; postprocess includes packaging and upload.',
             'Successful samples only are included in medians. Sequential samples can use different hosted runner images; these are observed distributions, not a controlled hardware benchmark.', '',
             '| OS | Condition | Result | Total | Preparation | Build step | Postprocess |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append('| ' + ' | '.join([row['platform'], row['variant'], row['result']] +
                                     [f"{row[k]:.1f}" for k in fields[4:]]) + ' |')
    lines += ['', '## Median and range (successful repeat samples)', '',
              '| OS | Condition | Count | Median total | Total range | Median build step | Build range |',
              '| --- | --- | ---: | ---: | --- | ---: | --- |']
    for platform in ['Windows', 'Linux']:
        for condition in ['baseline', 'contrib', 'combined', 'all-removable', 'all-removable-strict']:
            samples = [row for row in rows if row['platform'] == platform and (re.fullmatch(re.escape(condition)+r'-r\d+', row['variant']) or (condition == 'combined' and row['variant'] == 'combined')) and row['result']=='success']
            if not samples:
                continue
            total = [r['total'] for r in samples]
            build = [r['build'] for r in samples]
            lines.append(f'| {platform} | {condition} | {len(samples)} | {statistics.median(total):.1f} | {min(total):.1f}–{max(total):.1f} | {statistics.median(build):.1f} | {min(build):.1f}–{max(build):.1f} |')
    lines += ['', 'Combined-removal candidates require both platforms to pass before adoption. No inference or GPU performance claim follows from build success.']
    report = '\n'.join(lines)+'\n'
    Path('CUDA_PHASE_TIMINGS.md').write_text(report)
    Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(report)


if __name__ == '__main__':
    main()
