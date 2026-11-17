# -*- coding: utf-8 -*-
"""
build_frontend_snapshot.py —— 生成离线可用的 ECharts 仪表盘快照（P2-10）
=====================================================================
把后端数据库的数据直接内嵌进 HTML，双击即可在浏览器打开查看（无需启动服务/无需联网）。
用于答辩演示与论文截图取证；所有数据源自 output/wind_ramp.db。

输出: output/dashboard_offline.html（内嵌数据 + 引用本地 echarts.min.js）
用法: python code/build_frontend_snapshot.py
"""
import os
import json
import shutil
import sqlite3

OUT_DIR = "output"
DB = os.path.join(OUT_DIR, "wind_ramp.db")
FRONT_DIR = os.path.join("code", "frontend")
OUT_HTML = os.path.join(OUT_DIR, "dashboard_offline.html")


def log(m):
    print(f"[snapshot] {m}")


def q(conn, sql, params=()):
    cur = conn.execute(sql, params)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def main():
    if not os.path.exists(DB):
        log(f"数据库不存在: {DB}，请先 python code/database.py")
        return
    conn = sqlite3.connect(DB)
    try:
        meta = {r["key"]: r["value"] for r in q(conn, "SELECT key,value FROM run_meta")}
        total = conn.execute("SELECT COUNT(*) FROM alarm_log").fetchone()[0]
        hit = conn.execute("SELECT COUNT(*) FROM alarm_log WHERE is_hit=1").fetchone()[0]
        ts = q(conn, "SELECT t_idx,p_mw,wspd_mean FROM field_power "
                    "WHERE t_idx BETWEEN 29952 AND 35279 ORDER BY t_idx")
        prob = q(conn, "SELECT t_idx,p_up,p_dn FROM pred_prob "
                       "WHERE t_idx BETWEEN 29952 AND 35279 ORDER BY t_idx")
        events = q(conn, "SELECT event_id,direction,start_tidx FROM ramp_events "
                         "WHERE is_main=1 ORDER BY start_tidx")
        alarms = q(conn, "SELECT alarm_id,t_start,t_end,direction,level,tau,"
                         "matched_event_id,is_hit,lead_steps FROM alarm_log "
                         "ORDER BY t_start LIMIT 200")
    finally:
        conn.close()

    data = dict(meta=meta, total=total, hit=hit, ts=ts, prob=prob,
                events=events, alarms=alarms)
    log(f"内嵌数据: 时序 {len(ts)}、概率 {len(prob)}、事件 {len(events)}、告警 {len(alarms)}")

    local_echarts = os.path.join(FRONT_DIR, "echarts.min.js")
    if os.path.exists(local_echarts):
        shutil.copy(local_echarts, os.path.join(OUT_DIR, "echarts.min.js"))

    payload = json.dumps(data, ensure_ascii=False)
    html = TEMPLATE.replace("__DATA__", payload)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"离线仪表盘已生成: {OUT_HTML}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><title>风电爬坡预警系统 · 离线仪表盘</title>
<script src="echarts.min.js"></script>
<style>
  body{background:#f5f7fa;font-family:'Microsoft YaHei',sans-serif;margin:0;}
  .nav{background:#212529;color:#fff;padding:14px 24px;font-size:1.25rem;font-weight:700;}
  .wrap{padding:18px 24px;}
  .cards{display:flex;gap:16px;margin-bottom:18px;flex-wrap:wrap;}
  .card{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:16px 22px;flex:1;min-width:150px;text-align:center;}
  .card .lbl{color:#888;font-size:.85rem;}
  .card .num{font-size:1.9rem;font-weight:700;margin-top:4px;}
  .panel{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:14px 18px;margin-bottom:18px;}
  .panel h3{margin:2px 0 12px;font-size:1rem;color:#333;}
  #power{width:100%;height:420px;} #prob{width:100%;height:230px;}
  table{width:100%;border-collapse:collapse;font-size:.85rem;}
  th,td{padding:6px 10px;border-bottom:1px solid #eee;text-align:center;}
  th{background:#fafafa;position:sticky;top:0;}
  .up{color:#d62728;font-weight:700;} .down{color:#1f77b4;font-weight:700;}
  .hit{color:#2ca02c;} .miss{color:#bbb;}
  .tblwrap{max-height:340px;overflow:auto;}
</style></head>
<body>
<div class="nav">风电功率爬坡预警系统 · 可视化仪表盘（离线快照）</div>
<div class="wrap">
  <div class="cards" id="cards"></div>
  <div class="panel"><h3>场站功率时序 / 爬坡事件区间 / 上行概率</h3><div id="power"></div></div>
  <div class="panel"><h3>上下行超阈概率（p_up / p_dn）</h3><div id="prob"></div></div>
  <div class="panel"><h3>历史告警记录（前 200 条）</h3>
    <div class="tblwrap"><table><thead><tr>
      <th>ID</th><th>t_start</th><th>t_end</th><th>方向</th><th>等级</th><th>τ</th><th>命中事件</th><th>命中</th><th>提前量</th>
    </tr></thead><tbody id="tbody"></tbody></table></div>
  </div>
</div>
<script>
const DATA = __DATA__;

const hitRate = DATA.total ? (DATA.hit/DATA.total*100).toFixed(1) : '0.0';
const cards = [
  ['告警总数', DATA.total, '#333'],
  ['命中告警', DATA.hit, '#2ca02c'],
  ['命中率', hitRate + '%', '#1f77b4'],
  ['概率源', (DATA.meta.prob_source||'-').toUpperCase(), '#333'],
  ['阈值 τ', DATA.meta.tau||'-', '#333'],
];
document.getElementById('cards').innerHTML = cards.map(c =>
  '<div class="card"><div class="lbl">'+c[0]+'</div><div class="num" style="color:'+c[2]+'">'+c[1]+'</div></div>').join('');

const ts = DATA.ts, prob = DATA.prob, events = DATA.events;
const tIdx = ts.map(d => d.t_idx);
const probMap = {}; prob.forEach(d => probMap[d.t_idx] = d);
const pUp = tIdx.map(t => probMap[t] ? probMap[t].p_up : null);
const evAreas = events.slice(0,120).map(e => ([
  {name:'事件'+e.event_id, xAxis:e.start_tidx,
   itemStyle:{color: e.direction==='up' ? 'rgba(214,39,40,.12)' : 'rgba(31,119,180,.12)'}},
  {xAxis:e.start_tidx+6}
]));
const powerChart = echarts.init(document.getElementById('power'));
powerChart.setOption({
  tooltip:{trigger:'axis'},
  legend:{data:['功率(MW)','风速(m/s)','p_up']},
  grid:{left:64,right:64,top:44,bottom:56},
  xAxis:{type:'category',data:tIdx},
  yAxis:[{type:'value',name:'功率/风速',scale:true},{type:'value',name:'概率',min:0,max:1}],
  dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:8}],
  series:[
    {name:'功率(MW)',type:'line',data:ts.map(d=>d.p_mw),showSymbol:false,smooth:true,lineStyle:{width:1.4},color:'#2ca02c',markArea:{silent:true,data:evAreas}},
    {name:'风速(m/s)',type:'line',data:ts.map(d=>d.wspd_mean),showSymbol:false,smooth:true,lineStyle:{width:1},color:'#ff7f0e'},
    {name:'p_up',type:'line',yAxisIndex:1,data:pUp,showSymbol:false,lineStyle:{width:1,type:'dashed'},color:'#d62728'}
  ]
});

const probChart = echarts.init(document.getElementById('prob'));
probChart.setOption({
  tooltip:{trigger:'axis'},
  legend:{data:['p_up','p_dn']},
  grid:{left:54,right:24,top:34,bottom:44},
  xAxis:{type:'category',data:prob.map(d=>d.t_idx)},
  yAxis:{type:'value',min:0,max:1,name:'概率'},
  dataZoom:[{type:'inside'},{type:'slider',height:14,bottom:6}],
  series:[
    {name:'p_up',type:'line',data:prob.map(d=>d.p_up),showSymbol:false,color:'#d62728'},
    {name:'p_dn',type:'line',data:prob.map(d=>d.p_dn),showSymbol:false,color:'#1f77b4'}
  ]
});

document.getElementById('tbody').innerHTML = DATA.alarms.map(a => {
  const dir = a.direction==='up'
    ? '<span class="up">上爬坡</span>' : '<span class="down">下爬坡</span>';
  const h = a.is_hit ? '<span class="hit">✓</span>' : '<span class="miss">✗</span>';
  return '<tr><td>'+a.alarm_id+'</td><td>'+a.t_start+'</td><td>'+a.t_end+'</td>'+
    '<td>'+dir+'</td><td>'+a.level+'</td><td>'+a.tau+'</td>'+
    '<td>'+(a.matched_event_id==null?'-':a.matched_event_id)+'</td><td>'+h+'</td><td>'+
    (a.lead_steps==null?'-':a.lead_steps)+'</td></tr>';
}).join('');

window.addEventListener('resize', ()=>{powerChart.resize();probChart.resize();});
</script>
</body></html>"""


if __name__ == "__main__":
    main()