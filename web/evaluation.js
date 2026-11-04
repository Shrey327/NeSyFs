(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const ui = { form: $('run-form'), tasks: $('task-options'), budget: $('budget'), provider: $('provider'), providerStatus: $('provider-status'), cancel: $('cancel-button'), refreshProviders: $('refresh-providers'), repeats: $('repeats'), button: $('run-button'), details: $('setup-details'), notice: $('run-message'), progress: $('progress'), progressLabel: $('progress-label'), progressCount: $('progress-count'), progressFill: $('progress-fill'), progressCurrent: $('progress-current'), live: $('live-panel'), liveContext: $('live-context'), liveEvents: $('live-events'), select: $('run-select'), empty: $('results-empty'), content: $('result-content'), meta: $('run-meta'), body: $('results-body'), episodes: $('episodes'), episodeCount: $('episode-count') };
  const names = { reactive: 'Reactive', history: 'History', react: 'ReAct-style', reflexion: 'Reflexion-style', graph_reflection: 'Graph + reflection', adaptive: 'NeSyFS (full)' };
  const methodOrder = Object.keys(names);
  let config = null, runs = [], running = false, activeProvider = null, timer = null, polling = false, cancelling = false, lastCompleted = -1, lastStatus = null, liveSignature = '', newestRun = null;

  // Old saved records may contain incomplete reasoning tags or raw payloads.
  // Withhold the entire suspicious value instead of attempting partial removal.
  function safeText(value) {
    const text = String(value);
    let normalized = text;
    for (let i = 0; i < 3; i++) normalized = normalized.replace(/&amp;/gi, '&');
    normalized = normalized.replace(/&lt;?/gi, '<').replace(/&gt;?/gi, '>')
      .replace(/&quot;?/gi, '"').replace(/&apos;?/gi, "'")
      .replace(/&#(x[0-9a-f]+|[0-9]+);?/gi, (_, code) => {
        const number = code[0].toLowerCase() === 'x' ? parseInt(code.slice(1), 16) : parseInt(code, 10);
        return number >= 0 && number <= 0xffff ? String.fromCharCode(number) : '';
      }).replace(/\\u([0-9a-f]{4})/gi, (_, code) => String.fromCharCode(parseInt(code, 16)));
    const marker = /<\s*\/?\s*(?:t\s*h\s*i\s*n\s*k|a\s*n\s*a\s*l\s*y\s*s\s*i\s*s|r\s*e\s*a\s*s\s*o\s*n\s*i\s*n\s*g)/i;
    const privateField = /(?:["']|^|[\s,{])(?:raw(?:_\w*)?|reasoning(?:_\w*)?|analysis|thinking|chain_of_thought|internal(?:_\w*)?)(?:["'])?\s*[:=]/i;
    return marker.test(normalized) || privateField.test(normalized)
      ? '[Content withheld: private or reasoning content detected]' : text;
  }
  // Only fields the agent deliberately records for display may appear in JSON.
  const publicFields = new Set(['role', 'content', 'name', 'names', 'subject', 'predicate', 'object', 'id', 'facts', 'new_facts', 'entities', 'superseded', 'superseded_fact_ids', 'action', 'actions', 'rationale', 'feedback', 'votes', 'aligned', 'approved', 'required', 'accepted', 'attempt', 'observation', 'progress', 'lesson', 'planning_used', 'invalid', 'trial', 'decision', 'rollouts', 'weight', 'score', 'evidence', 'observed_at', 'kind', 'memo', 'sampled', 'temperature', 'ms', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'retries']);
  function publicValue(value) {
    if (typeof value === 'string') return safeText(value);
    if (Array.isArray(value)) return value.map(publicValue);
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
      .filter(([key]) => publicFields.has(key)).map(([key, item]) => [key, publicValue(item)]));
    return value;
  }
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text != null) node.textContent = safeText(text);
    if (className) node.className = className;
    return node;
  }
  function message(text, error = false) {
    ui.notice.hidden = !text;
    ui.notice.textContent = text ? safeText(text) : '';
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
    ui.providerStatus.textContent = safeText([label, availability, provider.model, provider.detail].filter(Boolean).join(' · '));
    ui.providerStatus.classList.toggle('unavailable', !provider.ready);
  }
  function updateControls() {
    ui.button.disabled = running || !config || !selectedProvider()?.ready;
    ui.button.querySelector('span').textContent = running ? 'Comparison running…' : 'Run comparison';
    ui.cancel.hidden = !running;
    ui.cancel.disabled = cancelling;
    ui.cancel.textContent = cancelling ? 'Stopping…' : 'Stop comparison';
    ui.provider.disabled = running || !config?.providers?.length;
    ui.refreshProviders.disabled = running;
    ui.budget.disabled = running;
    ui.repeats.disabled = running;
    ui.tasks.querySelectorAll('input').forEach(input => { input.disabled = running; });
  }
  function selectedTasks() {
    return [...ui.tasks.querySelectorAll('input:checked')].map(input => input.value);
  }
  function syncActiveControls(status) {
    if (!status?.running || !config) return;
    if (Array.isArray(status.task_ids)) {
      const activeTasks = new Set(status.task_ids);
      ui.tasks.querySelectorAll('input').forEach(input => { input.checked = activeTasks.has(input.value); });
    }
    if (Number.isInteger(status.action_budget) && status.action_budget > 0) ui.budget.value = String(status.action_budget);
    if (Number.isInteger(status.repeats) && status.repeats > 0) ui.repeats.value = String(status.repeats);
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
    providers.forEach(provider => ui.provider.add(new Option(safeText(provider.label || provider.id), provider.id)));
    if (providers.length) {
      if (running && providers.some(provider => provider.id === activeProvider)) ui.provider.value = activeProvider;
      else if (providers.some(provider => provider.id === previousProvider)) ui.provider.value = previousProvider;
      else if (providers.some(provider => provider.id === 'mlx')) ui.provider.value = 'mlx';
      else { ui.provider.add(new Option('Choose a provider', '')); ui.provider.value = ''; }
    } else {
      ui.provider.add(new Option('Provider availability unknown', ''));
    }
    updateProviderStatus();
    if (lastStatus?.running) { syncActiveControls(lastStatus); renderLive(lastStatus); }
    ui.details.textContent = safeText(`Select up to ${limits.max_tasks || config.max_tasks || config.tasks.length} tasks. Each run uses the same model and decision budget across methods. A rejected decision consumes one opportunity.`);
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
    [run.provider ? `Provider · ${run.provider}` : 'Provider · not recorded (earlier run)', run.model ? `Model · ${run.model}` : null, `Decision budget · ${run.action_budget ?? 'unavailable'}`, `Recorded episodes · ${episodes.length}`, run.status ? `Run · ${run.status === 'running' ? (run.tracking === 'active' ? 'in progress' : 'unmonitored; may be running elsewhere or interrupted') : String(run.status).replaceAll('_', ' ')}` : null, run.seed != null ? `Seed · ${run.seed}` : null, settings.reflection_limit != null ? `Reflection attempts · ${settings.reflection_limit}` : null, settings.reflection_samples != null ? `Reflection votes · ${settings.reflection_samples}` : null, settings.particle_count != null ? `Particles · ${settings.particle_count}` : null, settings.planning_horizon != null ? `Planning horizon · ${settings.planning_horizon}` : null, run.policy_temperature != null ? `Policy temperature · ${run.policy_temperature}` : run.model_temperature != null ? `Model temperature (legacy) · ${run.model_temperature}` : 'Policy temperature · unavailable', run.sample_temperature != null ? `Reflection/planning temperature · ${run.sample_temperature}` : 'Reflection/planning temperature · unavailable'].filter(Boolean).forEach(text => ui.meta.append(el('span', text)));
    if (episodes.some(ep => ep.usage_coverage?.calls_started === 0 && Array.isArray(ep.actions) && ep.actions.length)) {
      ui.meta.append(el('span', 'Earlier record: some actions have no reported model calls; time and usage may reflect cached output or incomplete instrumentation.', 'meta-warning'));
    }
    ui.body.replaceChildren();
    const methods = Array.isArray(run.methods) && run.methods.length ? run.methods : methodOrder;
    renderBars(episodes, methods);
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
    episodes.forEach((ep, index) => ui.episodes.append(renderEpisode(ep, index)));
  }
  function labeledText(label, value, kind) {
    const block = el('div', null, `detail-block ${kind || ''}`);
    block.append(el('span', label, 'detail-label'), el('p', value || 'Not recorded', 'detail-value'));
    return block;
  }
  function textFor(value) {
    if (typeof value === 'string') {
      try {
        const parsed = JSON.parse(value);
        if (parsed && typeof parsed === 'object') return JSON.stringify(publicValue(parsed), null, 2);
      } catch { /* Display sanitized plain text below. */ }
      return safeText(value);
    }
    return JSON.stringify(publicValue(value), null, 2);
  }
  function modelOutput(value) {
    if (typeof value === 'string') {
      try {
        const parsed = JSON.parse(value);
        return parsed && typeof parsed === 'object' ? textFor(parsed) : '[Unstructured output withheld]';
      } catch { return '[Unstructured output withheld]'; }
    }
    return value && typeof value === 'object' ? textFor(value) : '[Unstructured output withheld]';
  }
  function renderEpisode(ep, index) {
    const details = el('details', null, 'episode');
    details.dataset.method = ep.method || '';
    const summary = el('summary'), identity = el('div', null, 'episode-identity');
    const failed = ep.error || ep.status === 'error';
    const state = failed ? (ep.success ? 'Error · goal reached' : 'Error') : ep.success ? 'Success' : (ep.status || 'not complete').replaceAll('_', ' ');
    identity.append(el('span', `EPISODE ${String(index + 1).padStart(2, '0')} · ${names[ep.method] || ep.method || 'Configuration'}`, 'episode-kicker'),
      el('strong', ep.scenario?.label || ep.task_id || 'Recorded task'));
    summary.append(identity, el('span', state, `episode-status ${failed ? 'bad' : ep.success ? 'ok' : ''}`));
    const content = el('div', null, 'episode-content');
    const task = ep.task || ep.task_id || '';
    const commandsAt = task.indexOf(' Commands:');
    const objective = commandsAt >= 0 ? task.slice(0, commandsAt) : task;
    content.append(labeledText('Task objective', objective, 'task-detail'));
    if (commandsAt >= 0) {
      const rules = el('details', null, 'full-instructions');
      rules.append(el('summary', 'Show full recorded task and command rules'), el('p', task));
      content.append(rules);
    }
    content.append(labeledText('Initial observation', ep.initial_observation, 'observation-detail'));
    const metrics = el('div', null, 'episode-metrics');
    [Number.isFinite(ep.decision_count) ? `${ep.decision_count} decisions` : 'Decisions unavailable',
     Number.isFinite(ep.invalid_actions) ? `${ep.invalid_actions} invalid actions` : null,
     Number.isFinite(ep.rejected_decisions) && ep.rejected_decisions ? `${ep.rejected_decisions} rejected decisions` : null,
     Number.isFinite(ep.planner_uses) ? `${ep.planner_uses} planner uses` : null,
     formatTime(ep.elapsed_seconds), tokenText(ep.tokens)].filter(Boolean).forEach(v => metrics.append(el('span', v)));
    content.append(metrics);
    if (ep.error) content.append(labeledText('Episode error', String(ep.error), 'error-detail'));
    const coverage = ep.usage_coverage;
    if (coverage && coverage.calls_started === 0 && Array.isArray(ep.actions) && ep.actions.length) {
      content.append(el('p', 'No model calls were reported for this episode. Its time and usage cannot be treated as independent inference; cached output or incomplete instrumentation may be involved.', 'data-caveat'));
    }
    const heading = el('div', null, 'detail-heading');
    heading.append(el('h4', 'Actions & observations'), el('p', 'Actual environment actions in recorded order.'));
    content.append(heading);
    const actions = Array.isArray(ep.actions) ? ep.actions : [];
    if (!actions.length) content.append(el('p', 'No environment actions recorded.', 'detail-empty'));
    const actionList = el('ol', null, 'action-timeline');
    actions.forEach((action, i) => {
      const row = el('li', null, 'action-card'), top = el('div', null, 'action-top'), labels = el('div', null, 'action-badges');
      const repeated = i > 0 && action.action === actions[i - 1].action && action.trial === actions[i - 1].trial;
      top.append(el('span', String(i + 1).padStart(2, '0'), 'action-number'), el('span', 'ACTION', 'detail-label'));
      if (action.trial != null) labels.append(el('span', `Attempt ${action.trial}`));
      if (action.invalid) labels.append(el('span', 'Invalid / no change', 'warn'));
      if (action.planning_used) labels.append(el('span', 'Planner selected', 'plan'));
      if (repeated) labels.append(el('span', 'Repeated', 'warn'));
      top.append(labels); row.append(top, el('p', action.action || 'Action unavailable', 'action-value'), labeledText('Environment outcome', action.observation, 'outcome-detail')); actionList.append(row);
    });
    if (actions.length) content.append(actionList);
    if (Array.isArray(ep.lessons) && ep.lessons.length) {
      const lessons = el('div', null, 'lesson-block');
      lessons.append(el('span', 'MODEL-SUPPLIED LESSONS', 'detail-label'));
      ep.lessons.forEach((lesson, i) => lessons.append(el('p', `${i + 1}. ${lesson}`)));
      content.append(lessons);
    }
    const traceHeading = el('div', null, 'detail-heading trace-section-heading');
    traceHeading.append(el('h4', 'Decision events'), el('p', 'Recorded in order. Model explanations are generated outputs, not hidden reasoning.'));
    content.append(traceHeading);
    const events = Array.isArray(ep.trace) ? ep.trace : [];
    if (!events.length) content.append(el('p', 'No decision events were recorded for this episode.', 'detail-empty'));
    const eventList = el('ol', null, 'event-timeline');
    events.forEach((event, i) => {
      const stage = String(event.stage || 'event');
      const allowed = ['memory', 'proposal', 'reflection', 'planning', 'action', 'observation', 'complete', 'rejection', 'lesson', 'model'];
      const item = el('li', null, `event-card event-${allowed.includes(stage) ? stage : 'other'}`);
      const head = el('div', null, 'event-head');
      const label = ({model: 'Model call', proposal: 'Model proposal', reflection: 'Reflection', memory: 'Graph memory', observation: 'Observation', action: 'Executed action', planning: 'Planner', rejection: 'Rejected proposal', lesson: 'Model-supplied lesson', complete: 'Step complete'})[stage] || stage;
      const when = [event.trial != null ? `Attempt ${event.trial}` : null, event.decision != null ? `Decision ${event.decision}` : null].filter(Boolean).join(' · ');
      head.append(el('span', label, 'event-label'), el('span', when || `Event ${i + 1}`, 'event-when'));
      item.append(head);
      if (event.message) item.append(el('p', event.message, 'event-copy'));
      const data = event.data;
      if (data && typeof data === 'object') {
        if (stage === 'model' && (data.prompt != null || data.output != null)) {
          const payload = el('details', null, 'event-payload'); payload.append(el('summary', 'Inspect recorded prompt and model output'));
          if (data.kind) payload.append(el('span', `Model task · ${data.kind}`, 'payload-kind'));
          if (data.prompt != null) { payload.append(el('span', 'PROMPT', 'detail-label'), el('pre', textFor(data.prompt))); }
          if (data.output != null) { payload.append(el('span', 'MODEL OUTPUT', 'detail-label'), el('pre', modelOutput(data.output))); }
          item.append(payload);
        } else {
          const payload = el('details', null, 'event-payload');
          payload.append(el('summary', stage === 'reflection' ? 'Inspect votes and feedback' : 'Inspect recorded details'), el('pre', textFor(data)));
          item.append(payload);
        }
      }
      eventList.append(item);
    });
    if (events.length) content.append(eventList);
    details.append(summary, content);
    return details;
  }
  function renderBars(episodes, methods) {
    const box = $('score-bars');
    box.replaceChildren(...methods.map(method => {
      const items = episodes.filter(ep => ep.method === method);
      const evaluable = items.filter(ep => ['completed', 'budget_exhausted'].includes(ep.status) && !ep.error);
      const wins = evaluable.filter(ep => ep.success === true).length;
      const pct = evaluable.length ? Math.round(wins / evaluable.length * 100) : null;
      const invalid = items.length && items.every(ep => Number.isFinite(ep.invalid_actions)) ? items.reduce((sum, ep) => sum + ep.invalid_actions, 0) : null;
      const row = el('button', null, `score${method === 'adaptive' ? ' ours' : ''}`); row.type = 'button';
      row.disabled = !items.length;
      row.setAttribute('aria-label', `Inspect ${names[method] || method} episodes`);
      const track = el('div', null, 'score-track'), fill = el('span');
      fill.style.width = `${pct ?? 0}%`; track.append(fill);
      row.append(el('b', names[method] || method), track,
        el('span', pct == null ? 'Unavailable' : `${wins}/${evaluable.length} · ${pct}%`, 'score-value'),
        el('small', items.length ? `${items.length} recorded · ${invalid == null ? 'invalid actions unavailable' : `${invalid} invalid actions`}` : 'No episodes recorded', 'score-note'));
      row.addEventListener('click', () => {
        const target = [...ui.episodes.querySelectorAll('.episode')].find(node => node.dataset.method === method);
        if (target) { target.open = true; target.scrollIntoView({behavior: 'smooth', block: 'start'}); target.querySelector('summary')?.focus(); }
      });
      return row;
    }));
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
      ui.select.add(new Option(safeText(`${label} · ${(run.run_id || '').slice(0, 8)}`), run.run_id));
    });
    ui.select.disabled = false;
    // A run that just appeared (started here or elsewhere) replaces the selection; otherwise keep the user's pick.
    const newest = runs[0].run_id, fresh = newest !== newestRun;
    newestRun = newest;
    ui.select.value = !fresh && runs.some(run => run.run_id === previous) ? previous : newest;
    renderRun(runs.find(run => run.run_id === ui.select.value));
  }
  // Plain-language names for what the agent is waiting on.
  const modelTasks = { entities: 'Picking out objects to look up in memory', memory: 'Writing what it saw into graph memory', propose: 'Choosing the next action', reflect: 'Double-checking the proposed action', predict: 'Simulating possible outcomes (planner)', lesson: 'Writing a lesson from the failed attempt' };
  function nowText(last) {
    if (!last) return 'Starting the episode…';
    const data = last.data || {};
    if (last.stage === 'model') return ({ entities: modelTasks.memory, memory: 'Updating graph memory', predict: modelTasks.predict, lesson: 'Starting a fresh attempt with the lesson' })[data.kind] || 'Waiting for the next model response';
    if (last.stage === 'memory') return Array.isArray(data.facts) ? modelTasks.propose : data.new_facts ? 'Moving on to the next decision' : modelTasks.entities;
    return ({ observation: 'Processing the latest observation', decision: modelTasks.propose, proposal: modelTasks.reflect, reflection: data.accepted ? 'Proposal accepted; executing it' : 'Proposal rejected; trying again or planning', planning: 'Planner is choosing an action', action: 'Recording the result', rejection: 'Moving on to the next decision', lesson: 'Starting a fresh attempt with the lesson', complete: 'Moving on to the next decision', error: 'Episode stopped by an error' })[last.stage] || 'Working…';
  }
  function liveRow(icon, label, text, className, extra) {
    const row = el('li', null, `live-step ${className || ''}`);
    const body = el('div');
    body.append(el('span', label, 'live-step-label'), el('p', text || '—'));
    if (extra) body.append(extra);
    row.append(el('span', icon, 'live-step-icon'), body);
    return row;
  }
  function decisionCard(group, current, last) {
    const card = el('article', null, `live-decision${current ? ' current' : ''}`);
    const head = el('div', null, 'live-event-head');
    const title = group.decision ? `Decision ${group.decision}` : 'Setup';
    head.append(el('strong', title), el('span', [group.trial ? `Attempt ${group.trial}` : null, current ? 'in progress' : null].filter(Boolean).join(' · ')));
    card.append(head);
    const steps = el('ol', null, 'live-steps'), calls = [];
    let acted = false, done = false;
    for (const event of group.events) {
      const data = event.data && typeof event.data === 'object' ? event.data : {};
      if (event.stage === 'model') { calls.push(data); continue; }
      if (event.stage === 'observation') {
        const text = data.observation || event.message;
        steps.append(acted ? liveRow('←', 'Result', text, 'result') : liveRow('👁', 'Sees', text, 'seen'));
      } else if (event.stage === 'memory' && Array.isArray(data.facts)) {
        steps.append(liveRow('◎', 'Remembers', data.facts.length ? data.facts.map(f => `${f.subject} ${f.predicate} ${f.object}`).join(' · ') : 'No relevant facts yet', 'memory'));
      } else if (event.stage === 'proposal') {
        steps.append(liveRow('?', 'Proposes', `${data.action || event.message}${data.rationale ? ` — “${data.rationale}”` : ''}`, 'proposal'));
      } else if (event.stage === 'reflection') {
        const ok = data.accepted;
        const notes = (data.votes || []).filter(v => !v.aligned && v.feedback).map(v => v.feedback).join(' ');
        steps.append(liveRow(ok ? '✓' : '✗', ok ? 'Check passed' : 'Check rejected it', `${data.approved ?? '?'} of ${(data.votes || []).length} votes approved${notes ? ` — ${notes}` : ''}`, ok ? 'ok' : 'bad'));
      } else if (event.stage === 'planning' && data.action) {
        steps.append(liveRow('⌘', 'Planner picks', data.action, 'plan'));
      } else if (event.stage === 'action') {
        acted = done = true;
        steps.append(liveRow('▶', 'Does', data.action || event.message, `act${data.invalid ? ' bad' : ''}`));
      } else if (event.stage === 'rejection') {
        done = true;
        steps.append(liveRow('■', 'Skipped', 'No action executed this decision (proposal rejected, no planner).', 'bad'));
      } else if (event.stage === 'lesson') {
        steps.append(liveRow('✎', 'Lesson for next attempt', event.message.replace(/^Attempt \d+ lesson: /, ''), 'plan'));
      } else if (event.stage === 'error') {
        done = true;
        steps.append(liveRow('!', 'Error', event.message, 'bad'));
      }
    }
    if (current && !done) steps.append(liveRow('…', 'Now', nowText(last), 'pending'));
    card.append(steps);
    if (calls.length) {
      const ms = calls.reduce((sum, call) => sum + (Number(call.ms) || 0), 0);
      const payload = el('details', null, 'event-payload');
      payload.append(el('summary', `${calls.length} model call${calls.length === 1 ? '' : 's'} · ${(ms / 1000).toFixed(1)} s · show raw outputs`));
      calls.forEach(call => payload.append(el('span', `${modelTasks[call.kind] || call.kind || 'Model call'} · ${call.ms ?? '?'} ms`, 'payload-kind'), el('pre', modelOutput(call.output))));
      card.append(payload);
    }
    return card;
  }
  function renderLive(status) {
    ui.live.hidden = !status.running;
    if (!status.running) { liveSignature = ''; return; }
    const events = Array.isArray(status.current_events) ? status.current_events : [];
    const actions = Array.isArray(status.current_actions) ? status.current_actions : [];
    const signature = JSON.stringify([status.current_episode_id, status.current_method, events.length, events.at(-1)?.index, actions.length]);
    if (signature === liveSignature) return;
    liveSignature = signature;
    const scenario = config?.tasks?.find(task => task.id === status.current_task);
    const budget = Array.isArray(status.current_budget) ? status.current_budget.reduce((a, b) => a + b, 0) : status.action_budget;
    ui.liveContext.replaceChildren();
    ui.liveContext.append(el('strong', `${names[status.current_method] || status.current_method || 'Preparing'} · ${scenario?.label || status.current_task || 'episode'}`),
      el('span', [status.current_trial ? `Attempt ${status.current_trial}` : null, status.current_decision ? `Decision ${status.current_decision}${budget ? ` of ${budget}` : ''}` : null, `Episode ${(Number(status.completed) || 0) + 1} of ${status.total || '?'}`].filter(Boolean).join(' · ')));
    if (scenario?.task) {
      const task = String(scenario.task), split = task.indexOf(' Commands:');
      ui.liveContext.append(el('p', `Goal: ${split < 0 ? task : task.slice(0, split)}`));
    }
    // Executed actions so far: the quickest way to follow the episode.
    const log = el('ol', null, 'live-actions');
    actions.forEach((action, i) => {
      const item = el('li', null, action.invalid ? 'bad' : '');
      item.append(el('b', `${i + 1}. ${action.action}`), el('span', `→ ${action.observation}`));
      if (action.planning_used) item.append(el('em', 'planner'));
      if (action.trial > 1 && actions[i - 1]?.trial !== action.trial) item.prepend(el('small', `Attempt ${action.trial} (fresh kitchen)`));
      log.append(item);
    });
    const groups = [];
    events.forEach(event => {
      const key = `${event.trial}:${event.decision}`;
      if (groups.at(-1)?.key !== key) groups.push({ key, trial: event.trial, decision: event.decision, events: [] });
      groups.at(-1).events.push(event);
    });
    const cards = groups.map((group, i) => decisionCard(group, i === groups.length - 1, events.at(-1))).reverse();
    const actionHead = el('h3', actions.length ? `Actions taken (${actions.length})` : 'No actions taken yet', 'live-subhead');
    const decisionHead = el('h3', 'Decisions, newest first', 'live-subhead');
    ui.liveEvents.replaceChildren(actionHead, ...(actions.length ? [log] : []), decisionHead,
      ...(cards.length ? cards : [el('p', 'Waiting for the first recorded event.', 'detail-empty')]));
  }
  function showStatus(status) {
    lastStatus = status;
    running = Boolean(status.running);
    cancelling = running && Boolean(status.cancel_requested);
    activeProvider = status.provider || null;
    if (running && config?.providers?.some(provider => provider.id === activeProvider)) {
      ui.provider.value = activeProvider;
      updateProviderStatus();
    }
    syncActiveControls(status);
    updateControls();
    renderLive(status);
    ui.progress.hidden = !running && !status.total;
    if (ui.progress.hidden) return;
    const completed = Number(status.completed) || 0, total = Number(status.total) || 0;
    ui.progressLabel.textContent = cancelling ? 'Stopping after the current operation' : running ? 'Running matched episodes' : status.error ? 'Run stopped' : 'Run finished';
    ui.progressCount.textContent = total ? `${completed} / ${total} episodes` : '';
    ui.progressFill.style.width = `${total ? Math.min(100, completed / total * 100) : 0}%`;
    ui.progressCurrent.textContent = running ? safeText([status.current_task, names[status.current_method] || status.current_method, status.current_trial ? `Attempt ${status.current_trial}` : null, status.current_decision ? `Decision ${status.current_decision}` : null].filter(Boolean).join(' · ') || 'Preparing…') : '';
    if (status.error) message(status.error, true);
    else if (status.warning) message(status.warning, true);
  }
  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const wasRunning = running, status = await request('/api/evaluation/status');
      showStatus(status);
      if (running && Number.isFinite(status.completed) && status.completed !== lastCompleted) {
        lastCompleted = status.completed;
        await loadResults();
      }
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
  ui.cancel.addEventListener('click', async () => {
    if (!running || cancelling) return;
    cancelling = true; updateControls();
    try { await request('/api/evaluation/cancel', {}); message('Stop requested. The current operation must finish before the run stops.'); await poll(); }
    catch (error) { cancelling = false; updateControls(); message(`Could not stop the run: ${error.message}`, true); }
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
