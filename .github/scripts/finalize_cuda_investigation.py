"""Collect two bounded experiments and preserve only a successful CUDA configuration."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

REPO = 'Hiroshiba/onnxruntime-builder'
BASE = '147bd8cfd16aac7b0eab494e8d2dcca2f8dc3df9'
RESULT_BRANCH = 'fix/ort-1.31-cuda-contrib'
RUNS = {
    40314439974: 'b697ee8991f8ad4f013b20074e59c7a28cb71482',
    40317058496: '5d0087519a1243634ac2d35ba2250848b5616879',
}


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def api(path):
    return json.loads(command('gh', 'api', f'repos/{REPO}/{path}'))


def enabled_jobs(jobs, suffix):
    expected = {
        f'build-onnxruntime (onnxruntime-{platform}-x64-cuda-{suffix},'
        for platform in ('win', 'linux')
    }
    selected = [job for job in jobs if any(job['name'].startswith(prefix) for prefix in expected)]
    found = {prefix for prefix in expected if any(job['name'].startswith(prefix) for job in selected)}
    return len(selected) == 2 and found == expected and all(job['conclusion'] == 'success' for job in selected)


def main():
    if os.environ.get('GITHUB_REPOSITORY') != REPO:
        raise RuntimeError('This experiment is restricted to the requested fork')
    deadline = time.monotonic() + 345 * 60
    snapshots = {}
    while True:
        for run_id, expected_sha in RUNS.items():
            run = api(f'actions/runs/{run_id}')
            if run['head_sha'] != expected_sha or run['event'] != 'workflow_dispatch':
                raise RuntimeError(f'Unexpected experiment identity: {run_id}')
            snapshots[run_id] = run
        print({run_id: run['status'] for run_id, run in snapshots.items()}, flush=True)
        if all(run['status'] == 'completed' for run in snapshots.values()):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Experiments still running; no options have been accepted')
        time.sleep(60)

    jobs = {run_id: api(f'actions/runs/{run_id}/jobs?per_page=100')['jobs'] for run_id in RUNS}
    no_quick = enabled_jobs(jobs[40317058496], 'contrib-no-quick')
    contrib = enabled_jobs(jobs[40314439974], 'contrib')
    lines = [
        '# ONNX Runtime 1.31 CUDA build investigation', '',
        f'Baseline: VOICEVOX/onnxruntime-builder PR #137 at `{BASE}`.',
        'ONNX Runtime source: `402979ac1bef89dcd55add41eaa7277c407599eb` (v1.31.0).', '',
        '| Run | Job | Result | Started (UTC) | Completed (UTC) |',
        '| --- | --- | --- | --- | --- |',
    ]
    for run_id, entries in jobs.items():
        for job in entries:
            if not job['name'].startswith('build-onnxruntime '):
                continue
            name = job['name'].split(',', 1)[0].removeprefix('build-onnxruntime (')
            lines.append(f"| [{run_id}](https://github.com/{REPO}/actions/runs/{run_id}) | {name} | {job['conclusion']} | {job['started_at']} | {job['completed_at']} |")
    lines += ['', 'All experiments used cold builds, without publishing releases or code signing.',
              'The baseline and enabled configurations use PR #137 toolchains and GPU architecture coverage.',
              'Passing compilation and packaging does not establish CUDA inference correctness or performance on real GPUs.',
              'Non-CUDA jobs and the private voicevox_onnxruntime target were not tested.', '']
    if not (no_quick or contrib):
        lines += ['No candidate passed on both platforms. No fix branch was created.',
                  'Inspect the build logs and diagnostics artifacts before changing more options.']
        report = '\n'.join(lines) + '\n'
        Path(os.environ['RUNNER_TEMP'], 'CUDA_CONTRIB_RESULTS.md').write_text(report)
        Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(report)
        return

    lines += ['Accepted changes:', '', '- Remove `--disable_contrib_ops` on both CUDA targets.',
              '- Remove the patch that only applies when Contrib Ops are disabled.']
    if no_quick:
        lines.append('- Remove `onnxruntime_QUICK_BUILD=ON` on both CUDA targets.')
    else:
        lines.append('- Retain QUICK_BUILD: removing it did not pass on both platforms.')
    lines += ['', 'Other resource limits, feature toggles and architecture restrictions are retained; their removal has not been verified.',
              'This is a candidate configuration verified for CUDA compilation and packaging, not proof that every remaining option is necessary.', '',
              f'Result branch: https://github.com/{REPO}/tree/{RESULT_BRANCH}', '']
    report = '\n'.join(lines)
    # Refuse to overwrite any existing work. A unique result branch starts at the PR baseline.
    if command('git', 'ls-remote', '--heads', 'origin', f'refs/heads/{RESULT_BRANCH}'):
        raise RuntimeError(f'{RESULT_BRANCH} already exists; refusing to overwrite it')
    command('git', 'fetch', 'origin', BASE)
    command('git', 'switch', '-c', RESULT_BRANCH, BASE)
    path = Path('.github/workflows/build.yml')
    source = path.read_text()
    source, count = re.subn(r'^.*--disable_contrib_ops.*\n', '', source, flags=re.MULTILINE)
    if count != 2:
        raise RuntimeError(f'Expected exactly two contrib flags, found {count}')
    if no_quick:
        source, count = re.subn(r'^.*onnxruntime_QUICK_BUILD=ON.*\n', '', source, flags=re.MULTILINE)
        if count != 2:
            raise RuntimeError(f'Expected exactly two QUICK_BUILD flags, found {count}')
    path.write_text(source)
    Path('fix-cuda-disabled-contrib-ops.patch').unlink()
    Path('CUDA_CONTRIB_RESULTS.md').write_text(report)
    Path(os.environ['RUNNER_TEMP'], 'CUDA_CONTRIB_RESULTS.md').write_text(report)
    command('git', 'diff', '--check')
    command('git', 'config', 'user.name', 'github-actions[bot]')
    command('git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    command('git', 'add', '.github/workflows/build.yml', 'fix-cuda-disabled-contrib-ops.patch', 'CUDA_CONTRIB_RESULTS.md')
    command('git', 'commit', '-m', 'build: enable CUDA contrib ops with the verified PR 137 configuration')
    command('git', 'push', f'--force-with-lease=refs/heads/{RESULT_BRANCH}:', 'origin', f'HEAD:refs/heads/{RESULT_BRANCH}')
    Path(os.environ['GITHUB_STEP_SUMMARY']).write_text(report)


if __name__ == '__main__':
    main()
