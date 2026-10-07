const $ = id => document.getElementById(id);
let ready = false, busy = false, indexing = false;
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '請檢查輸入內容後重試。');
  return data;
}
function controls() {
  $('send').disabled = busy || !ready;
  $('upload').disabled = busy || indexing;
  $('index').disabled = busy || indexing;
  $('file').disabled = busy || indexing;
}
function notice(text, error = false) {
  $('status').textContent = text;
  $('status').className = error ? 'error' : '';
}
async function refresh() {
  try {
    const s = await api('/api/status');
    ready = s.ready;
    indexing = s.indexing;
    $('model').textContent = s.model;
    $('documentStatus').textContent = s.ready ? `${s.pages} 頁 · ${s.chunks} 個段落 · 索引就緒` : (s.pdf_exists ? 'PDF 已存在，請建立索引' : '尚未匯入文件');
    if (indexing) notice('正在建立向量索引，完成後即可提問…');
    else if (s.empty_pages.length) notice(`第 ${s.empty_pages.join('、')} 頁沒有可讀文字，未納入檢索。`);
    controls();
  } catch (e) { notice(e.message, true); }
}
async function build(upload) {
  if (busy) return;
  if (upload && !$('file').files.length) return notice('請先選擇 PDF。', true);
  busy = true; controls(); notice('正在解析 PDF 並建立向量索引，首次載入模型可能需要數分鐘…');
  try {
    let options = {method: 'POST'};
    if (upload) { const data = new FormData(); data.append('file', $('file').files[0]); options.body = data; }
    await api(upload ? '/api/upload' : '/api/index', options);
    document.querySelectorAll('.message').forEach(node => node.remove());
    $('welcome').hidden = false;
    notice('索引建立完成，可以開始提問。');
  } catch (e) { notice(e.message, true); }
  finally { busy = false; await refresh(); controls(); }
}
$('upload').onclick = () => build(true);
$('index').onclick = () => build(false);
document.querySelectorAll('[data-q]').forEach(button => button.onclick = () => { $('question').value = button.dataset.q; $('question').focus(); });
function message(role, text) {
  const el = document.createElement('article'); el.className = `message ${role}`;
  const title = document.createElement('h3'); title.textContent = role === 'user' ? '你的問題' : '論文助手';
  const content = document.createElement('div'); content.className = 'text'; content.textContent = text;
  el.append(title, content); $('conversation').append(el); return el;
}
$('chat').onsubmit = async event => {
  event.preventDefault();
  const question = $('question').value.trim();
  if (!question || busy || !ready) return;
  busy = true; controls(); $('welcome').hidden = true;
  message('user', question); $('question').value = '';
  const reply = message('assistant', '正在檢索相關段落並產生回答…');
  reply.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  try {
    const data = await api('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question})});
    reply.querySelector('.text').textContent = data.answer;
    if (data.warning) { const p = document.createElement('p'); p.className = 'warning'; p.textContent = data.warning; reply.append(p); }
    const sources = document.createElement('div'); sources.className = 'sources';
    for (const source of data.sources) {
      const detail = document.createElement('details'), summary = document.createElement('summary');
      summary.textContent = `[${source.id}] 第 ${source.page} 頁 · 檢索摘錄`;
      const p = document.createElement('p'); p.textContent = source.text;
      const link = document.createElement('a'); link.href = `/api/pdf#page=${source.page}`; link.target = '_blank'; link.rel = 'noopener'; link.textContent = '開啟原文頁面 ↗';
      detail.append(summary, p, link); sources.append(detail);
    }
    reply.append(sources);
  } catch (e) { reply.querySelector('.text').textContent = e.message; $('question').value = question; }
  finally { busy = false; controls(); $('question').focus(); }
};
refresh();
setInterval(() => { if (!busy) refresh(); }, 5000);
