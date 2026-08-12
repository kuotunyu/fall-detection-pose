"""Approved scientific-dashboard styling for the Gradio demo."""

from __future__ import annotations


APP_HEADER_HTML = """
<div class="fd-shell">
  <nav class="fd-nav" aria-label="專案導覽">
    <div class="fd-brand"><span>FD</span><strong>Fall Detection / Pose</strong></div>
    <div><a href="#fd-method">方法</a><a href="#fd-evaluation">評估</a><a href="#fd-limitations">限制</a><a href="https://github.com/kuotunyu/fall-detection-pose" target="_blank" rel="noopener noreferrer">GitHub ↗</a></div>
  </nav>
  <header class="fd-header" id="fd-method">
    <div>
      <p>POSE TRACKING · RULE-BASED EVENT DETECTION</p>
      <h1>姿態追蹤式跌倒事件偵測</h1>
    </div>
    <div>
      <p>使用 YOLO26-pose 擷取人體姿態，ByteTrack 建立時序軌跡，再由規則式狀態機輸出可追溯的事件判定。</p>
      <ul><li>YOLO26-pose</li><li>ByteTrack</li><li>Finite-state machine</li><li>Event-level evaluation</li></ul>
    </div>
  </header>
</div>
"""


DEMO_CSS = r"""
:root {
  --fd-canvas: #dedfd9; --fd-app: #eeebe4; --fd-paper: #f9f8f4;
  --fd-ink: #24312d; --fd-muted: #5c6964; --fd-sage: #687f74;
  --fd-sage-tint: #e7ebe5; --fd-coral: #c66c5d; --fd-coral-tint: #f0ded9;
  --fd-line: rgba(36, 49, 45, .16);
}
body, .gradio-container { background: var(--fd-canvas) !important; color: var(--fd-ink) !important; }
.gradio-container { max-width: 1740px !important; margin: 0 auto !important; padding: 20px 24px 48px !important;
  font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif !important; }
.contain, .main { max-width: none !important; }
:where(.fd-shell, .fd-evidence, .fd-state) :where(h1,h2,h3,h4,strong,b,dt,dd,li,code,label) { color:var(--fd-ink) !important; }
.fd-shell, .fd-evidence, .fd-state { color:var(--fd-ink) !important; }.fd-metric span { color:var(--fd-ink) !important; }.fd-metric small { color:var(--fd-muted) !important; }
footer[aria-label="Gradio footer navigation"] { display:none !important; }
.fd-shell, #fd-input, #fd-processing, #fd-result, #fd-error { border-radius: 3px !important; }
.fd-shell { overflow: hidden; border: 1px solid var(--fd-line); background: var(--fd-app); box-shadow: 0 16px 38px rgba(43,56,51,.11); }
.fd-nav { min-height: 66px; display:flex; align-items:center; justify-content:space-between; padding: 0 28px; border-bottom:1px solid var(--fd-line); background:var(--fd-paper); }
.fd-brand { display:flex; align-items:center; gap:11px; font-size:19px; }
.fd-brand>span { width:34px; height:34px; display:grid; place-items:center; color:white; background:var(--fd-sage); font:700 15px ui-monospace,monospace; }
.fd-nav a { margin-left:24px; color:var(--fd-ink); font-size:16px; font-weight:700; text-decoration:none; border-bottom:1px solid transparent; }
.fd-nav a:hover { border-color:var(--fd-sage); }
.fd-header { display:grid; grid-template-columns:1.15fr .85fr; gap:26px; padding:27px 32px 22px; }
.fd-header>div:first-child>p { margin:0 0 9px; color:var(--fd-sage); font:700 14px ui-monospace,monospace; letter-spacing:.045em; }
.fd-header h1 { margin:0; font-size:clamp(34px,2.5vw,42px); line-height:1.18; letter-spacing:-.035em; }
.fd-header>div:last-child>p { margin:0; color:var(--fd-muted); font-size:18px; line-height:1.72; }
.fd-header ul { display:flex; flex-wrap:wrap; gap:0; margin:16px 0 0; padding:12px 0 0; border-top:1px solid var(--fd-line); list-style:none; }
.fd-header li { padding:0 13px; border-left:1px solid var(--fd-line); font:700 14px ui-monospace,monospace; }
.fd-header li:first-child { padding-left:0; border-left:0; }
.fd-evidence { display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--fd-line); border-bottom:1px solid var(--fd-line); }
.fd-metric { min-height:76px; display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:18px; padding:13px 20px; border-right:1px solid var(--fd-line); }
.fd-metric:last-child { border-right:0; }.fd-metric strong { font-size:29px; }.fd-metric span { font-size:15px; font-weight:650; }.fd-metric small { color:var(--fd-muted); font-size:14px; }
.fd-state { margin-top:14px !important; border:1px solid var(--fd-line) !important; background:var(--fd-paper) !important; box-shadow:0 8px 20px rgba(50,63,57,.055) !important; }
.fd-state, .fd-state>div, .fd-state .form { border-radius:3px !important; }
.fd-state .styler { background:var(--fd-paper) !important; }.fd-upload, .fd-upload .wrap { border-color:var(--fd-line) !important; background:var(--fd-app) !important; color:var(--fd-ink) !important; }
.fd-upload label[data-testid="block-label"] { border-color:var(--fd-line) !important; background:var(--fd-paper) !important; color:var(--fd-ink) !important; }.fd-state .label-wrap, .fd-state .label-wrap button { background:var(--fd-app) !important; color:var(--fd-ink) !important; }
.fd-section-title { margin:0 !important; padding:16px 19px !important; border-bottom:1px solid var(--fd-line); background:var(--fd-paper); }
.fd-section-title h2 { margin:0; font-size:21px; }.fd-section-title p { margin:2px 0 0; color:var(--fd-muted); font-size:15px; }
.fd-input-grid { padding:16px !important; gap:14px !important; }.fd-upload { min-height:300px; }.fd-actions button { min-height:52px !important; border-radius:2px !important; font-size:17px !important; font-weight:700 !important; }
.fd-input-copy { height:100%; padding:8px 2px; }.fd-input-copy>small { color:var(--fd-sage); font:700 14px ui-monospace,monospace; }.fd-input-copy h3 { margin:5px 0 8px; font-size:25px; }.fd-input-copy p { margin:0 0 18px; color:var(--fd-muted); font-size:17px; line-height:1.65; }.fd-input-copy dl { display:grid; grid-template-columns:1fr 1fr; margin:0; border:1px solid var(--fd-line); }.fd-input-copy dl>div { padding:12px 14px; border-right:1px solid var(--fd-line); }.fd-input-copy dl>div:last-child { border-right:0; }.fd-input-copy dt { color:var(--fd-muted); font-size:14px; }.fd-input-copy dd { margin:3px 0 0; font-size:18px; font-weight:700; }
.fd-primary { background:var(--fd-sage) !important; color:white !important; border:0 !important; }.fd-primary:hover { background:#566d62 !important; }
.fd-progress { padding:22px; background:var(--fd-paper); }.fd-progress header { display:flex; align-items:end; justify-content:space-between; }.fd-progress header small { font:700 14px ui-monospace,monospace; color:var(--fd-sage); }.fd-progress h2 { margin:5px 0 0; font-size:25px; }.fd-progress header>strong { font-size:29px; }.fd-progress-track { height:5px; margin:18px 0; background:#d1d6d0; }.fd-progress-track i { display:block; height:100%; background:var(--fd-sage); transition:width .2s ease; }
.fd-progress ol { margin:0; padding:0; list-style:none; border:1px solid var(--fd-line); }.fd-progress li { display:grid; grid-template-columns:34px 1fr auto; align-items:center; gap:12px; min-height:58px; padding:10px 14px; border-bottom:1px solid var(--fd-line); }.fd-progress li:last-child { border-bottom:0; }.fd-progress li.is-active { background:var(--fd-sage-tint); }.fd-progress li i { width:28px; height:28px; display:grid; place-items:center; border:1px solid var(--fd-sage); border-radius:50%; font-style:normal; }.fd-progress li.is-done i { color:white; background:var(--fd-sage); }.fd-progress li span { font-size:17px; font-weight:700; }.fd-progress li small { font:700 14px ui-monospace,monospace; }.fd-progress footer { padding-top:13px; font:700 15px ui-monospace,monospace; }
.fd-workspace { display:grid !important; grid-template-columns:minmax(0,1.72fr) minmax(520px,.78fr) !important; gap:0 !important; }.fd-video { padding:14px !important; border-right:1px solid var(--fd-line); }.fd-video video { max-height:none !important; object-fit:contain !important; }.fd-result-html { padding:14px !important; }.fd-result-panel, .fd-events { height:100%; }.fd-source { display:flex; justify-content:space-between; gap:12px; padding:0 0 10px; color:var(--fd-muted); font-size:16px; }
.fd-event { height:100%; display:grid; grid-template-rows:auto auto 1fr; border-left:4px solid var(--fd-coral); background:var(--fd-paper); }.fd-event>header { display:flex; align-items:start; justify-content:space-between; gap:12px; padding:16px 18px; background:var(--fd-coral-tint); }.fd-event header small { font-size:15px; font-weight:700; }.fd-event h3 { margin:5px 0 0; font-size:28px; }.fd-event header p { margin:0; color:#875b55; font-size:17px; }.fd-event header>b { font:700 14px ui-monospace,monospace; }
.fd-event-grid { display:grid; grid-template-columns:repeat(4,1fr); margin:0; border-bottom:1px solid var(--fd-line); }.fd-event-grid>div { padding:12px 16px; border-right:1px solid var(--fd-line); }.fd-event-grid>div:last-child { border-right:0; }.fd-event-grid dt { font-size:15px; font-weight:700; }.fd-event-grid dd { margin:5px 0 0; font-size:23px; font-weight:700; }
.fd-rules { min-height:0; display:grid; grid-template-rows:auto 1fr; background:var(--fd-sage-tint); }.fd-rules h4 { margin:0; padding:11px 16px; border-bottom:1px solid var(--fd-line); font-size:17px; }.fd-rules ul { display:grid; grid-auto-rows:1fr; margin:0; padding:0; list-style:none; }.fd-rules li { display:grid; grid-template-columns:minmax(0,1fr) 82px; align-items:center; gap:14px; padding:12px 16px; border-bottom:1px solid var(--fd-line); }.fd-rules li:last-child { border-bottom:0; }.fd-rules li>div { min-width:0; }.fd-rules code { display:block; color:var(--fd-ink); font:700 18px ui-monospace,monospace; white-space:normal; }.fd-rules li span { display:block; margin-top:4px; color:var(--fd-muted); font-size:15px; }.fd-rules li>b { align-self:stretch; display:grid; place-items:center; border-left:1px solid var(--fd-line); font:700 16px ui-monospace,monospace; }
.fd-pipeline { display:grid; grid-template-columns:repeat(4,1fr); margin:10px 0 0; padding:0; border:1px solid var(--fd-line); list-style:none; }.fd-pipeline li { padding:10px 8px; border-right:1px solid var(--fd-line); background:var(--fd-sage-tint); text-align:center; font:700 14px ui-monospace,monospace; }.fd-pipeline li:last-child { border-right:0; }
.fd-no-event { padding:24px; }.fd-no-event header { display:flex; align-items:center; gap:14px; }.fd-no-event header>span { width:36px; height:36px; display:grid; place-items:center; border:2px solid var(--fd-sage); color:var(--fd-sage); font-weight:800; }.fd-no-event small { font:700 14px ui-monospace,monospace; }.fd-no-event h2 { margin:3px 0 0; font-size:28px; }.fd-no-event>p { color:var(--fd-muted); font-size:17px; }.fd-no-event dl { display:grid; grid-template-columns:repeat(3,1fr); margin:20px 0 0; border:1px solid var(--fd-line); }.fd-no-event dl>div { padding:14px 16px; border-right:1px solid var(--fd-line); }.fd-no-event dl>div:last-child { border-right:0; }.fd-no-event dt { font-size:15px; }.fd-no-event dd { margin:4px 0 0; font-size:23px; font-weight:700; }
.fd-no-event-reason { margin-top:14px; padding:15px 16px; border-left:4px solid var(--fd-sage); background:var(--fd-sage-tint); }.fd-no-event-reason h3 { margin:0 0 5px; font-size:18px; }.fd-no-event-reason p { margin:0; color:var(--fd-muted); font-size:16px; line-height:1.55; }
.fd-output-actions { padding:10px 14px !important; border-top:1px solid var(--fd-line); }.fd-output-actions button { min-height:48px !important; border-radius:2px !important; font-size:17px !important; }.fd-file { min-height:48px; }
.fd-error { padding:28px; border-left:4px solid var(--fd-coral); }.fd-error small { color:#8a4b42; font:700 14px ui-monospace,monospace; }.fd-error h2 { margin:8px 0 5px; font-size:28px; }.fd-error p { color:var(--fd-muted); font-size:17px; }.fd-error details { margin-top:18px; border-top:1px solid var(--fd-line); padding-top:12px; }
.fd-state :focus-visible, .fd-nav a:focus-visible { outline:3px solid #305d9b !important; outline-offset:3px !important; }
@media (max-width: 1100px) { .fd-workspace { grid-template-columns:1fr !important; }.fd-video { border-right:0; border-bottom:1px solid var(--fd-line); }.fd-header { grid-template-columns:1fr; }.fd-evidence { grid-template-columns:1fr 1fr 1fr; } }
@media (max-width: 650px) { html, body, gradio-app, .gradio-container { width:100% !important; min-width:0 !important; max-width:100% !important; overflow-x:hidden !important; }.gradio-container { padding:10px !important; }.gradio-container>.main { width:100% !important; min-width:0 !important; }.fd-nav { padding:0 16px; }.fd-nav>div:last-child { display:none; }.fd-brand { font-size:18px; }.fd-header { padding:22px 17px 17px; gap:16px; }.fd-header h1 { font-size:32px; }.fd-header>div:last-child>p { font-size:17px; }.fd-header li { font-size:13px; }.fd-evidence { grid-template-columns:1fr; }.fd-metric { min-height:72px; grid-template-columns:118px minmax(0,1fr); gap:2px 12px; border-right:0; border-bottom:1px solid var(--fd-line); }.fd-metric:last-child { border-bottom:0; }.fd-metric strong { grid-row:1 / 3; align-self:center; white-space:nowrap; font-size:27px; }.fd-metric span { align-self:end; font-size:15px; }.fd-metric small { align-self:start; }.fd-workspace { display:flex !important; flex-direction:column !important; }.fd-video, .fd-result-html { padding:10px !important; }.fd-event { display:block; }.fd-event-grid { grid-template-columns:1fr 1fr; }.fd-event-grid>div:nth-child(even) { border-right:0; }.fd-event-grid dd { font-size:20px; }.fd-rules code { font-size:16px; }.fd-rules li { grid-template-columns:minmax(0,1fr) 70px; }.fd-no-event dl { grid-template-columns:1fr; }.fd-no-event dl>div { border-right:0; border-bottom:1px solid var(--fd-line); }.fd-no-event dl>div:last-child { border-bottom:0; }.fd-actions button, .fd-output-actions button { min-height:48px !important; width:100%; }.fd-progress { padding:16px; } }
"""
