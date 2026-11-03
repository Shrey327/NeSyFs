(() => {
  'use strict';
  const TASK = 'Put the apple in the fridge';
  const STAGES = ['remember', 'propose', 'reflect', 'plan', 'act', 'update'];
  const $ = id => document.getElementById(id);
  const SVG = 'http://www.w3.org/2000/svg';

  // Small DOM builder; model output only ever reaches the page as text.
  function h(tag, attrs = {}, ...kids) {
    const el = tag.startsWith('svg:') ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.setAttribute('class', v);
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
    return el;
  }

  const ui = { start: $('start'), step: $('step'), auto: $('auto'), notice: $('notice') };
  let provider = null, providers = [], neo4jReady = false;
  let sessionId = null, running = false, autoRun = false, completed = false;
  let history = [], events = [], belief = new Map(), openCalls = new Set(), pollTimer = null;

  async function request(path, body) {
    const res = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    let data;
    try { data = await res.json(); } catch { throw new Error('The server returned an unreadable response.'); }
    if (!res.ok) { const err = new Error(data.error || `Request failed (${res.status}).`); err.data = data; throw err; }
    return data;
  }

  function notice(text, kind = '') { ui.notice.textContent = text || ''; ui.notice.className = `notice ${kind}`; }

  function controls() {
    ui.start.disabled = running || !neo4jReady || !provider;
    ui.step.disabled = running || !sessionId || completed;
    ui.auto.disabled = !sessionId || completed || (running && !autoRun);
    ui.auto.textContent = autoRun ? 'Stop auto-run' : 'Auto-run';
    ui.auto.classList.toggle('active', autoRun);
    ui.step.textContent = running ? 'Thinking…' : 'Run one step';
    document.querySelectorAll('.model').forEach(b => { b.disabled = running; });
  }

  // ---------- Models ----------
  function shortLabel(p) { return p.id === 'openai' ? `OpenAI · ${p.model}` : p.label.replace('Local MLX · ', '').replace(' 4-bit', ''); }
  function renderModels() {
    const box = $('models');
    box.replaceChildren(...providers.map(p => h('button', {
      type: 'button', role: 'radio', class: `model${p.ready ? ' ready' : ''}`, 'aria-checked': String(p.id === provider),
      title: p.detail, onclick: async () => {
        if (!p.ready) { notice(`Checking ${shortLabel(p)}…`); await checkStatus(); const now = providers.find(x => x.id === p.id); if (!now || !now.ready) { notice(p.detail, 'error'); return; } }
        provider = p.id; renderModels(); notice(sessionId ? 'Model changes apply to the next episode.' : `${shortLabel(p)} selected.`);
      }
    }, shortLabel(p))));
    controls();
  }
  async function checkStatus() {
    const conn = $('connection');
    try {
      const status = await request('/api/status');
      providers = status.providers || [];
      neo4jReady = status.ready;
      if (!provider || !providers.some(p => p.id === provider && p.ready)) provider = (providers.find(p => p.id === status.default_provider && p.ready) || providers.find(p => p.ready) || {}).id || null;
      renderModels();
      conn.className = status.ready ? 'online' : 'offline';
      conn.textContent = status.ready ? 'Graph ready' : 'Setup needed';
      if (!status.ready) notice(status.error, 'error');
      else if (!provider) notice('No model is ready. Start a local MLX server or set OPENAI_API_KEY.', 'error');
    } catch (err) {
      conn.className = 'offline'; conn.textContent = 'Server offline'; notice(err.message, 'error');
    }
  }

  // ---------- Kitchen ----------
  function renderKitchen(state, observation) {
    if (typeof observation === 'string') $('observation').textContent = observation;
    if (!state) return;
    $('apple').dataset.at = state.apple;
    $('door').classList.toggle('open', !!state.fridge_open);
  }

  // ---------- Pipeline ----------
  function phases(list) {
    let phase = 'remember', acted = false;
    const reached = new Set(['remember']), order = [];
    const go = p => { phase = p; reached.add(p); order.push(p); };
    for (const e of list) {
      if (e.stage === 'llm') {
        const k = e.data && e.data.kind;
        if (k === 'reflect') go('reflect');
        else if (k === 'predict') go('plan');
        else if (k === 'propose' && phase !== 'plan') go('propose');
      } else if (e.stage === 'proposal') go('propose');
      else if (e.stage === 'reflection') go('reflect');
      else if (e.stage === 'planning') go('plan');
      else if (e.stage === 'action') { acted = true; go('act'); }
      else if (e.stage === 'observation' || (e.stage === 'memory' && acted)) go('update');
      else if (e.stage === 'complete') phase = null;
    }
    return {phase, reached, acted};
  }
  function renderPipeline(list, live) {
    const {phase, reached, acted} = phases(list);
    for (const li of $('pipeline').children) {
      const s = li.dataset.stage;
      let state = '';
      if (!list.length && !live) state = '';
      else if (live && s === phase) state = 'live';
      else if (reached.has(s)) state = 'done';
      else if (s === 'plan' && acted) state = 'skipped';
      li.dataset.state = state;
    }
  }

  // ---------- Decision ----------
  function renderDecision(list, live) {
    const box = $('decision'), out = [];
    const proposals = list.filter(e => e.stage === 'proposal');
    if (!proposals.length) {
      box.replaceChildren(h('p', {class: 'empty'}, live ? 'Reading the observation and updating memory…' : 'Run a step to see proposals, reflection votes and the chosen action.'));
      $('decision-meta').textContent = live ? 'Working' : 'This step';
      return;
    }
    proposals.forEach((p, i) => {
      const start = list.indexOf(p), next = proposals[i + 1] ? list.indexOf(proposals[i + 1]) : list.length;
      const slice = list.slice(start, next);
      const verdict = slice.find(e => e.stage === 'reflection');
      const liveVotes = slice.filter(e => e.stage === 'llm' && e.data.kind === 'reflect').map(e => e.data.output);
      const votes = verdict ? verdict.data.votes : liveVotes;
      const d = p.data || {};
      out.push(h('div', {class: 'attempt'},
        h('div', {class: 'attempt-top'}, h('span', {}, `Proposal ${d.attempt || i + 1}`), h('span', {}, 'fast policy')),
        d.feedback ? h('p', {class: 'feedback-in'}, `Told to fix: ${d.feedback}`) : null,
        h('p', {class: 'attempt-action'}, d.action || '—'),
        h('p', {class: 'attempt-why'}, d.rationale || 'No rationale given.'),
        h('div', {class: 'votes', 'aria-label': 'Reflection votes'},
          votes.map(v => h('span', {class: `vote ${v.aligned ? 'yes' : 'no'}`, title: v.feedback || (v.aligned ? 'Aligned' : 'Not aligned')}, v.aligned ? '✓' : '✗')),
          !verdict && live ? h('span', {class: 'vote pending'}, '·') : null,
          verdict ? h('span', {class: `verdict ${verdict.data.accepted ? 'yes' : 'no'}`},
            `${verdict.data.accepted ? 'Accepted' : 'Rejected'} ${verdict.data.approved}/${verdict.data.votes.length} · needs ${verdict.data.required}`) : null),
        h('ul', {class: 'vote-notes'}, votes.filter(v => !v.aligned && v.feedback).map(v => h('li', {}, v.feedback)))));
    });
    const plan = list.find(e => e.stage === 'planning' && e.data);
    const planStart = list.find(e => e.stage === 'planning');
    if (planStart) {
      const rollouts = (plan && plan.data.rollouts) || [];
      const sims = list.filter(e => e.stage === 'llm' && e.data.kind === 'predict').length;
      out.push(h('div', {class: 'attempt'},
        h('div', {class: 'attempt-top'}, h('span', {}, 'Planner fallback'), h('span', {}, `${sims} simulated transitions`)),
        h('p', {class: 'attempt-why'}, 'Reflection kept rejecting, so particles rolled out possible futures and one surviving plan was sampled.'),
        h('ol', {class: 'rollouts'}, rollouts.map(r => h('li', {class: r.chosen ? 'chosen' : ''}, `${r.chosen ? '→ ' : ''}${r.actions.join('  ›  ')}`)))));
    }
    const act = list.find(e => e.stage === 'action');
    const obs = list.find(e => e.stage === 'observation');
    if (act) out.push(h('div', {class: 'outcome'}, h('span', {class: 'metadata'}, act.data && act.data.planning_used ? 'Executed · from planner' : 'Executed'),
      h('b', {}, act.data ? act.data.action : act.message), obs ? h('p', {}, obs.message) : null));
    box.replaceChildren(...out);
    $('decision-meta').textContent = `${proposals.length} proposal${proposals.length > 1 ? 's' : ''}`;
  }

  // ---------- Belief graph ----------
  const keyOf = f => `${f.subject}|${f.predicate}|${f.object}`;
  function applyBelief(base, list) {
    const facts = new Map([...base].map(([k, f]) => [k, {...f, new: false}]));
    for (const e of list) {
      if (e.stage !== 'memory' || !e.data) continue;
      const drop = new Set(e.data.superseded || []);
      for (const [k, f] of facts) if (f.id && drop.has(f.id)) facts.delete(k);
      for (const f of e.data.facts || []) facts.set(keyOf(f), {...f, new: (facts.get(keyOf(f)) || {}).new || false});
      // Re-observed facts are not news; only triples the graph did not hold get highlighted.
      for (const f of e.data.new_facts || []) if (!facts.has(keyOf(f))) facts.set(keyOf(f), {...f, new: true});
    }
    return facts;
  }
  function renderGraph(facts) {
    const svg = $('graph'), list = [...facts.values()].slice(-14);
    const names = [...new Set(list.flatMap(f => [f.subject, f.object]))].slice(0, 12);
    const hot = new Set(list.filter(f => f.new).flatMap(f => [f.subject, f.object]));
    const pos = new Map(), cx = 200, cy = 150, r = names.length > 1 ? 112 : 0;
    names.forEach((n, i) => { const a = -Math.PI / 2 + i * 2 * Math.PI / names.length; pos.set(n, [cx + r * Math.cos(a) * 1.25, cy + r * Math.sin(a)]); });
    const kids = [], pairCount = new Map();
    for (const f of list) {
      const a = pos.get(f.subject), b = pos.get(f.object);
      if (!a || !b || a === b) continue;
      const pair = [f.subject, f.object].sort().join('|'), n = pairCount.get(pair) || 0; pairCount.set(pair, n + 1);
      const bend = (n % 2 ? -1 : 1) * Math.ceil(n / 2) * 26;
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
      const qx = mx - dy / len * bend, qy = my + dx / len * bend;
      kids.push(h('svg:path', {class: `edge${f.new ? ' new' : ''}`, d: `M${a[0]},${a[1]} Q${qx},${qy} ${b[0]},${b[1]}`}));
      const label = h('svg:text', {class: `edge-label${f.new ? ' new' : ''}`, x: (mx + qx) / 2, y: (my + qy) / 2 - 3});
      label.textContent = f.predicate.length > 18 ? f.predicate.slice(0, 17) + '…' : f.predicate;
      kids.push(label);
    }
    for (const n of names) {
      const [x, y] = pos.get(n);
      const t = h('svg:text', {x, y: y + 25}); t.textContent = n.length > 16 ? n.slice(0, 15) + '…' : n;
      kids.push(h('svg:g', {class: `node${hot.has(n) ? ' hot' : ''}`}, h('svg:circle', {cx: x, cy: y, r: 9}), t));
    }
    if (!names.length) { const t = h('svg:text', {x: 200, y: 154, 'text-anchor': 'middle', fill: 'currentColor', opacity: '.5', 'font-size': '13'}); t.textContent = 'No facts yet'; kids.push(t); }
    svg.replaceChildren(...kids);
    $('fact-list').replaceChildren(...list.map(f => h('li', {class: f.new ? 'new' : ''}, `${f.subject} · ${f.predicate} · ${f.object}`)));
    $('graph-meta').textContent = `${facts.size} fact${facts.size === 1 ? '' : 's'}`;
  }

  // ---------- Inference engine ----------
  const fmt = n => n >= 10000 ? `${(n / 1000).toFixed(1)}k` : String(n);
  function renderEngine(list) {
    const calls = list.filter(e => e.stage === 'llm').map(e => e.data);
    const fresh = calls.filter(c => !c.memo);
    const sum = k => fresh.reduce((t, c) => t + (c[k] || 0), 0);
    const prompt = sum('prompt_tokens'), cached = sum('cached_tokens'), gen = sum('completion_tokens'), ms = sum('ms');
    const hasTokens = fresh.some(c => c.prompt_tokens != null);
    const tiles = [
      ['Model calls', calls.length], ['Reused (memo)', calls.length - fresh.length, calls.length > fresh.length],
      ['GPU time', ms ? `${(ms / 1000).toFixed(1)}s` : '—'],
      ['Prompt tokens', hasTokens ? fmt(prompt) : '—'], ['Generated', hasTokens ? fmt(gen) : '—'],
      ['Tokens / s', hasTokens && ms ? Math.round(gen / (ms / 1000)) : '—']];
    $('metrics').replaceChildren(...tiles.map(([label, value, good]) => h('div', {class: `metric${good ? ' good' : ''}`}, h('b', {}, value), h('span', {}, label))));
    const pct = prompt ? Math.round(cached / prompt * 100) : 0;
    $('kv-fill').style.width = `${pct}%`;
    $('kv-label').textContent = hasTokens ? `${pct}% of prompt tokens came from the KV prefix cache (${fmt(cached)} of ${fmt(prompt)}); only the rest was prefilled.`
      : fresh.length ? 'This provider does not report KV cache use.'
      : calls.length ? 'Every call so far reused an earlier answer; nothing was generated.' : 'KV prefix reuse appears after the first local call.';
    const box = $('calls');
    if (!calls.length) { box.replaceChildren(h('li', {class: 'empty'}, 'No calls yet.')); return; }
    const max = Math.max(...calls.map(c => c.ms), 1);
    box.replaceChildren(...calls.map((c, i) => {
      const open = openCalls.has(i);
      const meta = c.memo ? 'reused · 0 ms' : [`${c.ms} ms`, c.completion_tokens != null ? `${c.completion_tokens} tok` : null,
        c.prompt_tokens ? `KV ${c.cached_tokens || 0}/${c.prompt_tokens}` : null, c.retries ? `${c.retries} retry` : null].filter(Boolean).join(' · ');
      return h('li', {class: `call${c.memo ? ' memo' : ''}${open ? ' open' : ''}`, 'data-kind': c.kind},
        h('button', {type: 'button', 'aria-expanded': String(open), onclick: () => { open ? openCalls.delete(i) : openCalls.add(i); renderEngine(events); }},
          h('span', {class: 'kind'}, c.kind), h('span', {class: 'bar', style: `width:${c.memo ? 100 : Math.max(2, c.ms / max * 100)}%`}), h('span', {class: 'call-meta'}, meta)),
        open ? h('div', {class: 'call-detail'},
          h('span', {class: 'metadata'}, 'Output'), h('pre', {}, JSON.stringify(c.output, null, 2)),
          c.raw ? [h('span', {class: 'metadata'}, 'Raw model text'), h('pre', {}, c.raw)] : null,
          h('span', {class: 'metadata'}, 'Prompt (end)'), h('pre', {}, c.prompt)) : null);
    }));
  }

  // ---------- History ----------
  function renderHistory() {
    const box = $('history');
    if (!history.length) { box.replaceChildren(h('li', {class: 'empty'}, 'No steps yet.')); return; }
    box.replaceChildren(...history.map((s, i) => h('li', {},
      h('span', {class: 'history-n'}, i + 1),
      h('div', {}, h('b', {}, s.action), h('p', {}, s.observation)),
      h('span', {class: `tag${s.planning ? ' plan' : ''}`}, `${s.planning ? 'planner · ' : ''}${s.calls} calls · ${(s.ms / 1000).toFixed(1)}s`))));
  }

  function render(list, live) {
    events = list;
    renderPipeline(list, live);
    renderDecision(list, live);
    renderGraph(applyBelief(belief, list));
    renderEngine(list);
  }

  // ---------- Actions ----------
  async function poll() {
    if (!sessionId) return;
    try { const d = await request(`/api/events?session_id=${encodeURIComponent(sessionId)}`); render(d.events || [], true); renderKitchen(d.state, d.observation); }
    catch { /* The step response reports failures; polling is only a live view. */ }
  }

  async function startEpisode() {
    autoRun = false; running = true; controls(); notice('Connecting to the graph and model…');
    try {
      const d = await request('/api/session', {task: TASK, provider});
      sessionId = d.session_id; completed = false; history = []; belief = new Map(); openCalls.clear();
      $('step-count').textContent = '0'; $('engine-model').textContent = (d.model || '').split('/').pop();
      renderKitchen(d.state, d.observation); render([], false); renderHistory();
      notice('Episode ready. Run one step, or let it auto-run.');
    } catch (err) { notice(err.message, 'error'); }
    finally { running = false; controls(); }
  }

  async function runStep() {
    if (!sessionId || running || completed) return false;
    running = true; openCalls.clear(); controls(); render([], true);
    notice(autoRun ? `Auto-run · step ${history.length + 1}` : 'Thinking. Calls appear on the right as they finish.');
    pollTimer = setInterval(poll, 400);
    let ok = false;
    try {
      const d = await request('/api/step', {session_id: sessionId});
      clearInterval(pollTimer);
      completed = d.completed; render(d.events, false); renderKitchen(d.state, d.observation);
      belief = applyBelief(belief, d.events);
      for (const f of belief.values()) f.new = false;
      const calls = d.events.filter(e => e.stage === 'llm');
      history.push({action: d.action, observation: d.observation, planning: d.events.some(e => e.stage === 'planning'),
        calls: calls.length, ms: calls.reduce((t, e) => t + e.data.ms, 0)});
      $('step-count').textContent = String(history.length); renderHistory();
      notice(completed ? 'Task complete. The apple is in the fridge.' : 'Step done. Tap a model call to see its prompt and output.', completed ? 'done' : '');
      ok = true;
    } catch (err) {
      clearInterval(pollTimer);
      if (err.data && err.data.events) { render(err.data.events, false); renderKitchen(err.data.state, err.data.observation); }
      notice(err.message, 'error');
    } finally { running = false; controls(); }
    return ok;
  }

  ui.start.addEventListener('click', startEpisode);
  ui.step.addEventListener('click', runStep);
  ui.auto.addEventListener('click', async () => {
    if (autoRun) { autoRun = false; controls(); notice('Auto-run stops after this step.'); return; }
    autoRun = true; controls();
    while (autoRun && !completed && history.length < 12) { if (!await runStep()) break; }
    autoRun = false; controls();
  });

  render([], false); renderHistory(); controls(); checkStatus();
})();
