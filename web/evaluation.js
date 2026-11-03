(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ui = { form: $('run-form'), tasks: $('task-options'), budget: $('budget'), provider: $('provider'), providerStatus: $('provider-status'), refreshProviders: $('refresh-providers'), repeats: $('repeats'), button: $('run-button'), details: $('setup-details'), notice: $('run-message'), progress: $('progress'), progressLabel: $('progress-label'), progressCount: $('progress-count'), progressFill: $('progress-fill'), progressCurrent: $('progress-current'), select: $('run-select'), empty: $('results-empty'), content: $('result-content'), meta: $('run-meta'), body: $('results-body'), episodes: $('episodes'), episodeCount: $('episode-count') };
  const names = { reactive: 'Reactive', history: 'History', graph_reflection: 'Graph + reflection', adaptive: 'Full adaptive' };
  const methodOrder = Object.keys(names);
  let config = null, runs = [], running = false, activeProvider = null, timer = null, polling = false;

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  function message(text, error = false) {
    ui.notice.hidden = !text;
    ui.notice.textContent = text || '';
    ui.notice.classList.toggle('error', error);
  }
  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {cache: 'no-store'} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    let data;
    try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response.'); }
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  function selectedProvider() {
    return config?.providers?.find(provider => provider.id === ui.provider.value);
  }
  function updateProviderStatus() {
    const provider = selectedProvider();
    if (!provider) {
      ui.providerStatus.textContent = 'Provider status is unavailable; check again before starting.';
      ui.providerStatus.classList.add('unavailable');
      return;
    }
    const label = provider.id === 'openai' ? 'Remote · API calls may incur charges' : 'Local · no silent cloud fallback';
    const availability = provider.ready ? (provider.id === 'openai' ? 'Configured; live access not yet verified' : 'Server and model detected; compatibility not guaranteed') : 'Unavailable';
    ui.providerStatus.textContent = [label, availability, provider.model, provider.detail].filter(Boolean).join(' · ');
    ui.providerStatus.classList.toggle('unavailable', !provider.ready);
  }
  function updateControls() {
    ui.button.disabled = running || !config || !selectedProvider()?.ready;
    ui.button.querySelector('span').textContent = running ? 'Comparison running…' : 'Run comparison';
    ui.provider.disabled = running || !config?.providers?.length;
    ui.refreshProviders.disabled = running;
    ui.budget.disabled = running;
    ui.repeats.disabled = running;
    ui.tasks.querySelectorAll('input').forEach(input => { input.disabled = running; });
  }
  function selectedTasks() {
    return [...ui.tasks.querySelectorAll('input:checked')].map(input => input.value);
  }
  async function loadConfig() {
    const previousTasks = selectedTasks(), previousProvider = ui.provider.value, hadConfig = Boolean(config);
    config = await request('/api/evaluation/config');
    if (!Array.isArray(config.tasks) || !config.tasks.length) throw new Error('No evaluation tasks are available.');
    const limits = config.limits || {};
    const tasks = [...config.tasks].sort((a, b) => Number(b.difficulty === 'complex') - Number(a.difficulty === 'complex'));
    const defaultIndex = 0;
    ui.tasks.replaceChildren();
    tasks.forEach((task, index) => {
      const label = el('label', null, 'task-option'), input = el('input'), content = el('span');
      input.type = 'checkbox'; input.value = task.id; input.checked = hadConfig ? previousTasks.includes(task.id) : index === defaultIndex;
      const title = task.label || (task.task || '').split(' Commands:')[0] || task.id;
      const description = task.description || `Scenario: ${String(task.id).replaceAll('_', ' / ')}`;
      content.append(el('strong', title), el('small', description));
      const complexity = [task.difficulty, Number.isFinite(task.min_actions) && task.min_actions > 0 ? `${task.min_actions} reference steps` : null].filter(Boolean).join(' · ');
      if (complexity) content.append(el('em', complexity));
      label.append(input, content); ui.tasks.append(label);
    });
    ui.budget.max = limits.max_steps || config.max_steps || 20;
    if (!hadConfig) ui.budget.value = Math.min(config.default_max_steps || 14, Number(ui.budget.max));
    ui.repeats.max = limits.max_repeats || 3;
    ui.provider.replaceChildren();
    const providers = Array.isArray(config.providers) ? config.providers : [];
    providers.forEach(provider => ui.provider.add(new Option(provider.label || provider.id, provider.id)));
    if (providers.length) {
      if (running && providers.some(provider => provider.id === activeProvider)) ui.provider.value = activeProvider;
      else if (providers.some(provider => provider.id === previousProvider)) ui.provider.value = previousProvider;
      else if (providers.some(provider => provider.id === 'mlx')) ui.provider.value = 'mlx';
      else { ui.provider.add(new Option('Choose a provider', '')); ui.provider.value = ''; }
    } else {
      ui.provider.add(new Option('Provider availability unknown', ''));
    }
    updateProviderStatus();
    ui.details.textContent = `Select up to ${limits.max_tasks || config.max_tasks || config.tasks.length} tasks. Each run uses the same model and decision budget across methods. A rejected decision consumes one opportunity.`;
    updateControls();
  }
  function formatTime(value) {
    return Number.isFinite(value) ? `${value.toFixed(1)} s` : 'Unavailable';
  }
  function tokenText(tokens) {
    if (!tokens || !Number.isFinite(tokens.total)) return 'Unavailable';
    const parts = [`${tokens.total.toLocaleString()} total`];
    if (Number.isFinite(tokens.input) && Number.isFinite(tokens.output)) parts.push(`${tokens.input.toLocaleString()} in / ${tokens.output.toLocaleString()} out`);
    return parts.join(' · ');
  }
  function addCell(row, value) { row.append(el('td', value)); }
  function renderRun(run) {
    if (!run) { ui.empty.hidden = false; ui.content.hidden = true; return; }
    ui.empty.hidden = true; ui.content.hidden = false;
    const episodes = Array.isArray(run.episodes) ? run.episodes : [];
    const settings = run.settings || {};
    ui.meta.replaceChildren();
    [run.provider ? `Provider · ${run.provider}` : 'Provider · not recorded (earlier run)', run.model ? `Model · ${run.model}` : null, `Decision budget · ${run.action_budget ?? 'unavailable'}`, `Recorded episodes · ${episodes.length}`, run.status ? `Run · ${run.status === 'running' ? 'in progress or interrupted' : String(run.status).replaceAll('_', ' ')}` : null, run.seed != null ? `Seed · ${run.seed}` : null, settings.reflection_limit != null ? `Reflection attempts · ${settings.reflection_limit}` : null, settings.reflection_samples != null ? `Reflection votes · ${settings.reflection_samples}` : null, settings.particle_count != null ? `Particles · ${settings.particle_count}` : null, settings.planning_horizon != null ? `Planning horizon · ${settings.planning_horizon}` : null].filter(Boolean).forEach(text => ui.meta.append(el('span', text)));
    ui.body.replaceChildren();
    const methods = Array.isArray(run.methods) && run.methods.length ? run.methods : methodOrder;
    for (const method of methods) {
      const items = episodes.filter(ep => ep.method === method), valid = items.filter(ep => ['completed', 'budget_exhausted'].includes(ep.status) && !ep.error), errors = items.length - valid.length;
      const successes = valid.filter(ep => ep.success === true).length;
      const totalActions = items.length && items.every(ep => Array.isArray(ep.actions)) ? items.reduce((sum, ep) => sum + ep.actions.length, 0) : null;
      const invalid = items.length && items.every(ep => Number.isFinite(ep.invalid_actions)) ? items.reduce((sum, ep) => sum + ep.invalid_actions, 0) : null;
      const planner = items.length && items.every(ep => Number.isFinite(ep.planner_uses)) ? items.reduce((sum, ep) => sum + ep.planner_uses, 0) : null;
      const elapsed = items.length && items.every(ep => Number.isFinite(ep.elapsed_seconds)) ? items.reduce((sum, ep) => sum + ep.elapsed_seconds, 0) : null;
      const tokens = items.length && items.every(ep => ep.tokens && Number.isFinite(ep.tokens.total)) ? items.reduce((sum, ep) => sum + ep.tokens.total, 0) : null;
      const row = el('tr');
      [names[method] || method, `${successes} / ${valid.length}`, errors, totalActions ?? 'Unavailable', invalid ?? 'Unavailable', planner ?? 'Unavailable', formatTime(elapsed), tokens == null ? 'Unavailable' : tokens.toLocaleString()].forEach(value => addCell(row, value));
      ui.body.append(row);
    }
    ui.episodeCount.textContent = `${episodes.length} recorded`;
    ui.episodes.replaceChildren();
    if (!episodes.length) { ui.episodes.append(el('p', 'No episodes have been recorded for this run yet.', 'muted')); return; }
    episodes.forEach((ep, index) => {
      const details = el('details', null, 'episode'), summary = el('summary');
      const failed = ep.error || ep.status === 'error';
      summary.append(el('strong', `${String(index + 1).padStart(2, '0')}  ${names[ep.method] || ep.method || 'Configuration'} · ${ep.task_id || 'Task'}`), el('span', failed ? (ep.success ? 'Error · goal reached' : 'Error') : ep.success ? 'Success' : (ep.status || 'not complete').replaceAll('_', ' '), failed ? 'bad' : ep.success ? 'ok' : ''));
      const content = el('div', null, 'episode-content');
      content.append(el('p', `Task: ${ep.task || ep.task_id || 'Unavailable'}`));
      if (ep.scenario) content.append(el('p', [ep.scenario.difficulty ? `Difficulty: ${ep.scenario.difficulty}` : null, Number.isFinite(ep.scenario.min_actions) && ep.scenario.min_actions > 0 ? `Feasible reference: ${ep.scenario.min_actions} steps (not a proven minimum)` : null].filter(Boolean).join(' · ')));
      if (ep.initial_observation) content.append(el('p', `Initial observation: ${ep.initial_observation}`));
      content.append(el('p', `Decisions: ${Number.isFinite(ep.decision_count) ? ep.decision_count : 'Unavailable'} · Invalid actions: ${Number.isFinite(ep.invalid_actions) ? ep.invalid_actions : 'Unavailable'} · Planner uses: ${Number.isFinite(ep.planner_uses) ? ep.planner_uses : 'Unavailable'} · Elapsed: ${formatTime(ep.elapsed_seconds)} · Tokens: ${tokenText(ep.tokens)}`));
      if (ep.error) content.append(el('p', `Error: ${ep.error}`, 'error'));
      const actions = Array.isArray(ep.actions) ? ep.actions : [];
      if (actions.length) {
        const list = el('ol');
        // A repeated action right after itself usually means the agent is stuck in a loop.
        actions.forEach((action, i) => { const repeat = i > 0 && action.action === actions[i - 1].action; list.append(el('li', `${action.action || 'Action unavailable'}${action.invalid ? ' [invalid]' : ''}${action.planning_used ? ' [planner used]' : ''}${repeat ? ' [repeated]' : ''} → ${action.observation || 'Observation unavailable'}`, repeat ? 'repeat' : null)); });
        content.append(list);
      } else content.append(el('p', 'No environment actions recorded.'));
      if (Number.isFinite(ep.rejected_decisions) && ep.rejected_decisions) content.append(el('p', `Rejected decisions without execution: ${ep.rejected_decisions}`));
      if (Array.isArray(ep.trace) && ep.trace.length) {
        const trace = el('details'), traceSummary = el('summary', 'Show recorded agent events'), list = el('ol');
        ep.trace.forEach(event => list.append(el('li', `${event.stage || 'Event'}: ${event.message || ''}`)));
        trace.append(traceSummary, list); content.append(trace);
      }
      details.append(summary, content); ui.episodes.append(details);
    });
  }
  async function loadResults() {
    const data = await request('/api/evaluation/results');
    runs = Array.isArray(data.runs) ? data.runs : [];
    const previous = ui.select.value;
    ui.select.replaceChildren();
    if (!runs.length) { ui.select.add(new Option('No runs yet', '')); ui.select.disabled = true; renderRun(null); return; }
    runs.forEach((run, index) => {
      const date = run.created_at ? new Date(run.created_at) : null;
      const label = date && !Number.isNaN(date.valueOf()) ? date.toLocaleString() : `Run ${index + 1}`;
      ui.select.add(new Option(`${label} · ${(run.run_id || '').slice(0, 8)}`, run.run_id));
    });
    ui.select.disabled = false;
    ui.select.value = runs.some(run => run.run_id === previous) ? previous : runs[0].run_id;
    renderRun(runs.find(run => run.run_id === ui.select.value));
  }
  function showStatus(status) {
    running = Boolean(status.running);
    activeProvider = status.provider || null;
    if (running && config?.providers?.some(provider => provider.id === activeProvider)) {
      ui.provider.value = activeProvider;
      updateProviderStatus();
    }
    updateControls();
    ui.progress.hidden = !running && !status.total;
    if (ui.progress.hidden) return;
    const completed = Number(status.completed) || 0, total = Number(status.total) || 0;
    ui.progressLabel.textContent = running ? 'Running matched episodes' : status.error ? 'Run stopped' : 'Run finished';
    ui.progressCount.textContent = total ? `${completed} / ${total} episodes` : '';
    ui.progressFill.style.width = `${total ? Math.min(100, completed / total * 100) : 0}%`;
    ui.progressCurrent.textContent = running ? [status.current_task, names[status.current_method] || status.current_method].filter(Boolean).join(' · ') || 'Preparing…' : '';
    if (status.error) message(status.error, true);
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const wasRunning = running, status = await request('/api/evaluation/status');
      showStatus(status);
      if (wasRunning && !running) {
        if (!status.error) message('Run finished. Inspect the recorded episodes below.');
        await loadResults();
      }
    } catch (error) { message(`Could not check progress: ${error.message}`, true); }
    finally { polling = false; }
  }
  // A provider selection or numeric edit must not accidentally start model calls.
  ui.form.addEventListener('keydown', event => {
    if (event.key === 'Enter' && event.target !== ui.button && !ui.button.contains(event.target)) event.preventDefault();
  });
  ui.form.addEventListener('submit', async event => {
    event.preventDefault();
    if (running || !config) return;
    const provider = selectedProvider();
    if (!provider?.ready) { message('The selected provider is unavailable. Check its readiness before starting.', true); return; }
    const ids = selectedTasks(), maxTasks = config.limits?.max_tasks || config.max_tasks || config.tasks.length, budget = Number(ui.budget.value), repeats = Number(ui.repeats.value);
    if (!ids.length || ids.length > maxTasks) { message(`Select between 1 and ${maxTasks} tasks.`, true); return; }
    if (!Number.isInteger(budget) || budget < 1 || budget > Number(ui.budget.max)) { message(`Decision budget must be between 1 and ${ui.budget.max}.`, true); return; }
    if (!Number.isInteger(repeats) || repeats < 1 || repeats > Number(ui.repeats.max)) { message(`Repeats must be between 1 and ${ui.repeats.max}.`, true); return; }
    running = true; updateControls(); message(provider.id === 'openai' ? 'Starting the comparison using remote, paid model calls.' : 'Starting the comparison using the selected local model; no cloud fallback.');
    ui.progress.hidden = false; ui.progressLabel.textContent = 'Starting matched episodes'; ui.progressCount.textContent = ''; ui.progressFill.style.width = '0%';
    try { await request('/api/evaluation/run', {task_ids: ids, repeats, max_steps: budget, provider: provider.id}); await poll(); }
    catch (error) { running = false; updateControls(); message(error.message, true); }
  });
  ui.provider.addEventListener('change', () => { updateProviderStatus(); updateControls(); });
  ui.refreshProviders.addEventListener('click', async () => {
    try { await loadConfig(); message('Provider status refreshed.'); }
    catch (error) { message(`Could not refresh provider status: ${error.message}`, true); }
  });
  ui.select.addEventListener('change', () => renderRun(runs.find(run => run.run_id === ui.select.value)));
  Promise.all([loadConfig(), loadResults(), poll()]).catch(error => message(`Unable to load the comparison lab: ${error.message}`, true));
  timer = setInterval(poll, 2000);
  window.addEventListener('pagehide', () => clearInterval(timer));
})();
