'use strict';
const D = window.PROBE_REPORT;
const app = document.getElementById('app');
const E = v => String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const F = v => v == null || !Number.isFinite(v) ? '—' : v.toFixed(3);
const P = v => v == null ? '—' : `${(100*v).toFixed(1)}%`;
const short = {general_baseline:'General baseline',pv_explicit:'Position V · explicit',pv_implicit:'Position V · implicit',ps_explicit:'Position S · explicit',ps_implicit:'Position S · implicit',pt_explicit:'Traits · explicit',pt_implicit:'Traits · implicit',pe_explicit:'Emotion · explicit',pe_implicit:'Emotion · implicit'};
const label = n => short[n] || n.replace('ctrl_','').replaceAll('_',' ');
const bname = b => ({sypr:'SyPR',moral:'AITA moral',elephant:'ELEPHANT',cross_cell:'Cross-system prompts',within_prompt:'Own system-prompt contrast'}[b] || b.replace('syconbench:','SYCON · ').replaceAll('_',' '));
const posname = p => ({response:'Full-response average',first5:'First five response tokens',last_prompt:'Last prompt token'}[p]);
const fieldname = f => f.replaceAll('_',' ');
const url = k => `${k}.html`;
const options = (values, current, text=x=>x) => values.map(v=>`<option value="${E(v)}" ${v===current?'selected':''}>${E(text(v))}</option>`).join('');
const primary = base => base.startsWith('syconbench:') ? `syconbench:${base.split(':')[1]}_failure` : ({sypr:'sypr:sycophantic_praise',moral:'moral:nta_when_yta',elephant:'elephant:validation',cross_cell:'cross_cell:sycophantic'}[base]);
const auc = (key,pos,layer,name,split='eval') => D.targets[key]?.values[pos]?.[split]?.[D.names.indexOf(name)]?.[D.layers.indexOf(layer)] ?? null;
const winner = (key,pos,layer) => D.names.filter(n=>auc(key,pos,layer,n,'selection')!=null).sort((a,b)=>auc(key,pos,layer,b,'selection')-auc(key,pos,layer,a,'selection'))[0];
function table(head,rows){return `<div class="table-wrap"><table><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
function mix(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
function color(v,kind){if(v==null)return '#e4e8ed';const midpoint=kind==='correlation'?0:.5;const t=Math.min(1,Math.abs(v-midpoint)/(kind==='correlation'?1:.5));const end=v<midpoint?(kind==='correlation'?[108,166,222]:[222,156,173]):(kind==='correlation'?[222,136,128]:[83,181,164]);return `rgb(${mix([248,249,250],end,t).join(',')})`;}
function thumbnail(m){
  const order=m.clusters?m.clusters.flat():m.names,columns=m.columns||order;
  const cellX=320/columns.length,cellY=240/order.length;
  let svg=`<svg class="heatmap-preview" viewBox="0 0 320 240" role="img" aria-label="${E(m.title)}; heatmap preview. Open for probe labels and exact values.">`;
  order.forEach((name,i)=>columns.forEach((column,j)=>{
    const value=m.values[m.names.indexOf(name)][m.columns?m.columns.indexOf(column):m.names.indexOf(column)];
    svg+=`<rect x="${j*cellX}" y="${i*cellY}" width="${cellX}" height="${cellY}" fill="${color(value,m.kind)}"><title>${E(label(name))} × ${E(m.columns?'L'+column:label(column))}: ${F(value)}</title></rect>`;
  }));
  if(m.clusters){let edge=0;for(const group of m.clusters.slice(0,-1)){edge+=group.length;svg+=`<path d="M0 ${edge*cellY}H320 M${edge*cellX} 0V240" fill="none" stroke="#172b42" stroke-width="1.5"/>`;}}
  return svg+'</svg>';
}
function heatmapCard(key,m){
  const metric=m.kind==='correlation'?'Pearson r':m.kind==='consensus'?'Co-clustering':'AUROC';
  const target=m.target||primary(m.base),isOOD=target&&!target.startsWith('cross_cell:');
  let note=m.clusters?`Groups: ${m.clusters.map(g=>g.length).join(' + ')}. `:'';
  if(m.kind==='auc'){
    const candidates=m.values.flatMap((row,i)=>row.map((v,j)=>({v,n:m.names[i],layer:m.columns[j]}))).filter(x=>x.v!=null).sort((a,b)=>b.v-a.v);
    const top=candidates[0];note+=`Report peak: ${F(top.v)}, ${label(top.n)}, L${top.layer} (exploratory).`;
  }else if(isOOD&&m.layer!=null){const n=winner(target,m.position,m.layer);note+=`Selection-chosen probe: ${label(n)}; report AUROC ${F(auc(target,m.position,m.layer,n))}.`;}
  else if(m.kind==='transfer')note+='Train prompt pair ↓; test prompt pair →. Diagonal is own-cell.';
  else if(m.kind==='id')note+='Positive versus negative instructions within each probe’s own prompt pair.';
  return `<a class="heatmap-card" href="${url(key)}"><div class="heatmap-caption"><h3>${E(m.title)}</h3><span class="badge">${metric}</span></div>${thumbnail(m)}<div class="preview-scale"><span>${m.kind==='correlation'?'−1':'0'}</span><span class="gradient ${m.kind==='correlation'?'':'auc'}"></span><span>1</span></div><p>${E(note)}</p><span class="open-map">Open labeled heatmap & groups</span></a>`;
}
function allHeatmaps(){
  document.title='All heatmaps · SycoScope';
  const params=new URLSearchParams(window.location.search);
  const bases=['syconbench:debate','syconbench:ethical','syconbench:false_presupposition','sypr','moral','elephant','cross_cell','within_prompt'];
  const kinds=['all','auroc','auc','transfer','id','correlation','consensus'];
  const titles={all:'All heatmaps',auroc:'All AUROC heatmaps',auc:'Natural OOD AUROC',transfer:'Between-system-prompt AUROC',id:'Own-prompt AUROC',correlation:'Score correlation',consensus:'Bootstrap co-clustering'};
  app.innerHTML=`<div class="eyebrow">Heatmap gallery</div><h1>All heatmaps, by benchmark</h1><p>Actual maps, not just links. Open any map for probe labels, cluster membership, exact values and layer-wise AUROC.</p><div class="controls"><label>Dataset<select id="gallery-base">${options(['all',...bases],bases.includes(params.get('dataset'))?params.get('dataset'):'all',b=>b==='all'?'All datasets':bname(b))}</select></label><label>Measurement<select id="gallery-kind">${options(kinds,kinds.includes(params.get('type'))?params.get('type'):'all',k=>titles[k])}</select></label><label>Activations<select id="gallery-position">${options(['all',...D.positions],'response',p=>p==='all'?'All positions':posname(p))}</select></label><label>Layers<select id="gallery-layers"><option value="all">All layers</option><option value="middle">Middle layers: L12 / L16 / L20</option></select></label></div><p class="muted">AUROC maps: 0.5 = chance. Correlation maps: 0 = no correlation. Bootstrap maps: 0.5 = co-clustered in half the resamples. Their color scales describe different quantities.</p><p id="gallery-count" aria-live="polite"></p><div id="gallery-jumps" class="links"></div><div id="heatmap-groups"></div>`;
  const base=document.getElementById('gallery-base'),kind=document.getElementById('gallery-kind'),pos=document.getElementById('gallery-position'),layers=document.getElementById('gallery-layers');
  const render=()=>{
    const entries=Object.entries(D.maps).filter(([key,m])=>(base.value==='all'||m.base===base.value)&&(pos.value==='all'||m.position===pos.value)&&(kind.value==='all'||kind.value===m.kind||(kind.value==='auroc'&&['auc','transfer','id'].includes(m.kind)))&&(layers.value==='all'||m.layer==null||[12,16,20].includes(m.layer)));
    const shown=bases.filter(b=>entries.some(([k,m])=>m.base===b));
    document.getElementById('gallery-count').textContent=`Showing ${entries.length} of ${Object.keys(D.maps).length} saved heatmaps. ${layers.value==='middle'?'Probe × layer AUROC maps retain all eight columns for comparison.':''}`;
    document.getElementById('gallery-jumps').innerHTML=shown.map(b=>`<a href="#maps-${E(b.replaceAll(':','-'))}">${E(bname(b))}</a>`).join('');
    document.getElementById('heatmap-groups').innerHTML=shown.map(b=>{
      const items=entries.filter(([k,m])=>m.base===b);
      return `<section id="maps-${E(b.replaceAll(':','-'))}" class="benchmark-section"><h2>${E(bname(b))}</h2>${['auc','transfer','id','correlation','consensus'].map(type=>{const group=items.filter(([k,m])=>m.kind===type).sort((a,b)=>D.positions.indexOf(a[1].position)-D.positions.indexOf(b[1].position)||(a[1].layer??0)-(b[1].layer??0));return group.length?`<h3 class="measurement-heading">${E(titles[type])}</h3><div class="heatmap-gallery">${group.map(([k,m])=>heatmapCard(k,m)).join('')}</div>`:'';}).join('')}</section>`;
    }).join('')||'<p>No saved heatmaps match these filters. SYCON correlation and bootstrap maps use full-response averages.</p>';
  };
  [base,kind,pos,layers].forEach(el=>el.addEventListener('change',render));render();
}
function drawHeatmap(container,m,clusterOrder=true,context){
  const isLayers=!!m.columns;
  const order=clusterOrder && m.clusters ? m.clusters.flat() : m.names;
  const cols=isLayers?m.columns:order;
  const groupOf=n=>m.clusters?.findIndex(g=>g.includes(n));
  const metric=m.kind==='correlation'?'Pearson r':m.kind==='consensus'?'Co-clustering frequency':'AUROC';
  const boundaries=new Set(m.clusters?.map(g=>g[0])||[]);
  let html=`<div class="table-wrap"><table class="gridmap" aria-label="${E(m.title)}"><thead><tr><th class="row">${m.kind==='transfer'?'Probe ↓ / source prompt →':'Probe ↓'}</th>`;
  html+=cols.map(c=>`<th class="${isLayers?'':'vertical'} ${isLayers&&[12,16,20].includes(c)?'middle':''}"><span>${E(isLayers?'L'+c:label(c))}</span></th>`).join('')+'</tr></thead><tbody>';
  for(const row of order){
    html+=`<tr><th scope="row" class="row ${clusterOrder&&boundaries.has(row)?'boundary-top':''}">${m.clusters?`<span class="badge">G${groupOf(row)+1}</span> `:''}${E(label(row))}</th>`;
    for(const col of cols){const ri=m.names.indexOf(row),ci=isLayers?m.columns.indexOf(col):m.names.indexOf(col),v=m.values[ri][ci];
      html+=`<td style="background:${color(v,m.kind)}" class="${clusterOrder&&boundaries.has(row)?'boundary-top ':''}${clusterOrder&&boundaries.has(col)?'boundary-left ':''}${isLayers&&[12,16,20].includes(col)?'middle':''}"><button type="button" data-row="${E(row)}" data-col="${E(col)}" aria-label="${E(label(row))}, ${E(isLayers?'L'+col:label(col))}: ${metric} ${F(v)}">${v==null?'—':v.toFixed(2)}</button></td>`;
    }html+='</tr>';
  }
  html+=`</tbody></table></div><div class="legend"><span>${m.kind==='correlation'?'−1':0}</span><span class="gradient ${m.kind==='correlation'?'':'auc'}"></span><span>1</span><span>${metric} · ${m.kind==='correlation'?'white = zero correlation':m.kind==='consensus'?'white = 50% co-clustering':'white = chance (0.5)'}</span></div><div class="selected" aria-live="polite">Select a cell for its exact value and label.</div>`;
  container.innerHTML=html;
  container.querySelectorAll('button[data-row]').forEach(button=>button.addEventListener('click',()=>{
    const row=button.dataset.row,col=isLayers?Number(button.dataset.col):button.dataset.col;
    const v=m.values[m.names.indexOf(row)][isLayers?m.columns.indexOf(col):m.names.indexOf(col)];
    let s=`${label(row)} × ${isLayers?'L'+col:label(col)}: ${metric} ${F(v)}.`;
    if(context && !isLayers && m.kind!=='transfer')s+=` On ${fieldname(D.targets[context].field)}, report AUROC: ${F(auc(context,m.position,m.layer,row))} / ${F(auc(context,m.position,m.layer,col))}.`;
    if(m.kind==='transfer'&&row===col)s+=' Own-cell diagonal: this is not cross-prompt transfer.';
    container.querySelector('.selected').textContent=s;
  }));
}
function lineChart(host,series){
  const width=Math.max(320,host.clientWidth),height=280,L=58,R=30,T=20,B=52,w=width-L-R,h=height-T-B;
  const x=l=>L+w*l/28,y=a=>T+h*(1-a);
  let s=`<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="AUROC by transformer layer. Chance is 0.5; full y-axis from zero to one.">`;
  for(let a=0;a<=1.001;a+=.25)s+=`<line class="${a===.5?'chance':'grid'}" x1="${L}" x2="${width-R}" y1="${y(a)}" y2="${y(a)}"/><text x="${L-9}" y="${y(a)+5}" text-anchor="end">${a.toFixed(2)}</text>`;
  const ticks=width<500?[0,12,20,28]:D.layers;
  ticks.forEach(l=>s+=`<text x="${x(l)}" y="${height-27}" text-anchor="middle">${l}</text>`);
  s+=`<text x="${L+w/2}" y="${height-6}" text-anchor="middle">Transformer layer</text><text transform="translate(15 ${T+h/2}) rotate(-90)" text-anchor="middle">AUROC</text>`;
  series.forEach(ser=>{
    let pen=false,path='';ser.values.forEach((v,i)=>{if(v==null){pen=false;return;}path+=`${pen?'L':'M'}${x(D.layers[i])},${y(v)} `;pen=true;});
    s+=`<path d="${path}" fill="none" stroke="${ser.color}" stroke-width="2.5" ${ser.dash?'stroke-dasharray="5 4"':''}/>`;
    ser.values.forEach((v,i)=>{if(v!=null)s+=`<circle cx="${x(D.layers[i])}" cy="${y(v)}" r="4" fill="${ser.color}"><title>${E(ser.name)} · L${D.layers[i]}: ${F(v)}</title></circle>`;});
  });
  s+='</svg><div class="legend">'+series.map(s=>`<span><i class="legend-line" style="background:${s.color}"></i>${E(s.name)}</span>`).join('')+'</div>';host.innerHTML=s;
}
function renderCurve(id,key,pos,probe){
  const host=document.getElementById(id),target=D.targets[key],ni=D.names.indexOf(probe);
  const values=target.values[pos].eval[ni],selection=target.values[pos].selection[ni];
  lineChart(host,[{name:'OOD report',values,color:'#007c83'},{name:'OOD selection',values:selection,color:'#7355ad',dash:true},{name:'Own system-prompt contrast',values:D.id[pos][ni],color:'#bd742c',dash:true}]);
  const valid=values.map((v,i)=>({v,i})).filter(x=>x.v!=null).sort((a,b)=>b.v-a.v);
  const peak=valid[0],selPeak=selection.map((v,i)=>({v,i})).filter(x=>x.v!=null).sort((a,b)=>b.v-a.v)[0];
  const jumps=values.slice(1).map((v,i)=>({delta:v!=null&&values[i]!=null?v-values[i]:null,i})).filter(x=>x.delta!=null).sort((a,b)=>b.delta-a.delta);
  const jump=jumps[0];
  document.getElementById(id+'-detail').innerHTML=`<p><strong>${E(label(probe))}:</strong> largest displayed report AUROC ${F(peak?.v)} at L${peak?D.layers[peak.i]:'—'}; selection-optimal layer ${selPeak?'L'+D.layers[selPeak.i]:'—'} has report AUROC ${F(selPeak?values[selPeak.i]:null)}. ${jump&&jump.delta>0?`Largest adjacent increase: +${F(jump.delta)} from L${D.layers[jump.i]} to L${D.layers[jump.i+1]}.`:''}</p><p class="muted">The report-set maximum and adjacent increase are exploratory, not significance tests. Do not choose a layer on this maximum and call its AUROC an unbiased estimate. Layer traces reuse questions; middle layers are L12/L16/L20.</p>`;
}
function middleTable(key,pos){return table(['Fixed layer','Probe chosen on selection','Selection AUROC','Report AUROC'],[12,16,20].map(l=>{const n=winner(key,pos,l);return ['L'+l,E(label(n)),F(auc(key,pos,l,n,'selection')),F(auc(key,pos,l,n))];}));}
function headlineTable(){const rows=[];for(const [base,fields] of Object.entries(D.headlines)){for(const [field,result] of Object.entries(fields)){if(!result?.best_single_probe)continue;const n=result.best_single_probe,r=result.probes[n];rows.push([`${E(bname(base))} · ${E(fieldname(field))}`,E(label(n)),`${E(posname(r.position))} L${r.layer}`,`<strong>${F(r.auc)}</strong>`,`${F(r.ci_lo)}–${F(r.ci_hi)}`]);}}return table(['Natural OOD target','Selection-chosen probe','Selection-chosen configuration','Report AUROC','95% CI'],rows);}
function overview(){
  document.title='SycoScope · Clusters, OOD and system prompts';
  app.innerHTML=`<div class="eyebrow">Frozen probes / comparative readout</div><h1>Clusters are not failure labels.</h1><p>Explore which probes move together, where their AUROC rises, and whether that signal transfers beyond the system prompts used to train them.</p>
  <div class="stats"><div><span class="stat">20</span><small>frozen definition probes</small></div><div><span class="stat">8</span><small>layers, including L12/16/20</small></div><div><span class="stat">21 / 24</span><small>SYCON response views select k=2</small></div><div><span class="stat">11 / 24</span><small>are hedging versus the other 19</small></div></div>
  <section id="ood"><h2>Where does OOD AUROC rise?</h2><div class="controls"><label>Target<select id="target"></select></label><label>Activation pooling<select id="position"></select></label><label>Probe<select id="probe"></select></label></div><div id="curve"></div><div id="curve-detail"></div><div id="target-note" class="note"></div><h3>Middle layers: choose the probe on selection, report on different questions</h3><div id="middle"></div><p id="auc-link"></p></section>
  <section><h2>Selection-chosen OOD results</h2><p class="muted">Each configuration and winning probe was chosen on selection data. These are the saved matrix estimates—not the largest values found while browsing this report. SYCON intervals bootstrap questions; older benchmark intervals retain their original saved methodology.</p>${headlineTable()}</section>
  <section id="atlas"><h2>Heatmap atlas</h2><p>Each heatmap has its own HTML page. Correlation pages list exact groups; SYCON middle-layer pages also link to bootstrap co-clustering. Numbers identify local groups, not shared concepts across pages.</p><div class="controls"><label>Dataset<select id="base"></select></label><label>Heatmap type<select id="kind"></select></label><label>Position<select id="map-pos"></select></label></div><div id="catalog" class="catalog"></div></section>
  <section id="prompts"><h2>What is “between system prompts”?</h2>${table(['Measurement','What the positive label means','What it can establish'],[['Own contrast','Generated under the named positive instruction, versus its paired negative instruction.','In-distribution instruction-condition separability. Not independently verified sycophantic behavior.'],['Cross-system transfer','A different cell’s positive versus negative generation instructions.','Transfer between instruction contrasts. Inducing system prompts remain present; off-diagonal entries are the relevant transfer test.'],['Natural OOD','Benchmark-specific response behavior or judge label.','Behavioral generalization under a dataset shift. Still dependent on target quality and evaluation design.']])}<p class="note warning">The own-cell diagonal and pooled cross-cell “all” result are not clean OOD tests. Benign control “positive” poles include warranted praise, genuine agreement and calibrated uncertainty; their labels do not all mean bad behavior.</p><div id="id-summary"></div><div class="links">${D.positions.map(p=>`<a href="within_prompt_${p}.html">Own contrast · ${E(posname(p))}</a>`).join('')}</div><h3 style="margin-top:24px">Cross-system AUROC matrices · full-response average</h3><div class="links">${D.layers.map(l=>`<a href="transfer_response_L${String(l).padStart(2,'0')}.html">L${l}</a>`).join('')}</div></section>
  <section><h2>Reading guide</h2><p>AUROC 0.5 is chance ranking; below 0.5 is a reversed association—not automatically no signal. Scores are not polarity-flipped after viewing OOD labels. Full-response averages measure current-answer detection, not advance prediction. First-failure SYCON targets include only turns 2–5 after an entirely successful prior history.</p><p>Cluster count is selected among 2–9, so the method always returns groups. Co-clustering frequency is not AUROC or correlation. Turn/length-adjusted clusters are sensitivity checks, not causal identification. No untouched test establishes a universal concept count.</p><p>SYCON debate failure means neutral/opposed to the assigned stance. Ethical failure means accepting or ignoring the harmful stereotype. False-presupposition failure means accepting the false premise. The task meanings are not interchangeable.</p><p class="muted">Source hashes: <a href="provenance.json">provenance.json</a>. No generation or paid judging was used to make this report.</p></section>`;
  const target=document.getElementById('target'),position=document.getElementById('position'),probe=document.getElementById('probe');
  target.innerHTML=options(Object.keys(D.targets).filter(k=>!k.startsWith('cross_cell:')),'syconbench:ethical_failure',k=>`${bname(D.targets[k].base)} · ${fieldname(D.targets[k].field)}`);
  position.innerHTML=options(D.positions,'response',posname);probe.innerHTML=options(D.names,'pt_implicit',label);
  const update=()=>{renderCurve('curve',target.value,position.value,probe.value);document.getElementById('middle').innerHTML=middleTable(target.value,position.value);const t=D.targets[target.value],c=t.counts.eval;document.getElementById('target-note').textContent=`Report: ${c.n} scored items, ${c.n_pos} positive and ${c.n_neg} negative. ${t.note}`;document.getElementById('auc-link').innerHTML=`<a href="auc_${t.base}_${t.field}_${position.value}.html">Open all 20 probes × 8 layers for this target</a>`;};
  [target,position,probe].forEach(el=>el.addEventListener('change',update));update();
  new ResizeObserver(()=>renderCurve('curve',target.value,position.value,probe.value)).observe(document.getElementById('curve'));
  const base=document.getElementById('base'),kind=document.getElementById('kind'),mp=document.getElementById('map-pos');
  base.innerHTML=options(['all',...new Set(Object.values(D.maps).map(m=>m.base))],'syconbench:ethical',b=>b==='all'?'All datasets':bname(b));
  kind.innerHTML=options(['all','correlation','consensus','auc','transfer','id'],'correlation',k=>({all:'All types',correlation:'Score correlation',consensus:'Bootstrap co-clustering',auc:'OOD AUROC',transfer:'Cross-system AUROC',id:'Own-contrast AUROC'}[k]));
  mp.innerHTML=options(['all',...D.positions],'response',p=>p==='all'?'All positions':posname(p));
  const gallery=()=>{const entries=Object.entries(D.maps).filter(([k,m])=>(base.value==='all'||m.base===base.value)&&(kind.value==='all'||m.kind===kind.value)&&(mp.value==='all'||m.position===mp.value));document.getElementById('catalog').innerHTML=entries.length?entries.map(([k,m])=>`<a href="${url(k)}">${E(m.title)}<small>${m.clusters?'Groups: '+m.clusters.map(g=>g.length).join(' + '):m.kind==='transfer'?'Rows = probes; columns = source prompt cells':'20 probes × 8 layers'}${m.stability?' · question-bootstrap checked':''}</small></a>`).join(''):'<p>No saved heatmaps for this combination. SYCON clustering is available for full-response averages.</p>';};[base,kind,mp].forEach(el=>el.addEventListener('change',gallery));gallery();
  document.getElementById('id-summary').innerHTML=table(['Own-contrast pooling','L12 AUROC range','L16 AUROC range','L20 AUROC range'],D.positions.map(p=>[E(posname(p)),...[12,16,20].map(l=>{const a=D.id[p].map(r=>r[D.layers.indexOf(l)]).filter(v=>v!=null);return `${Math.min(...a).toFixed(6)}–${Math.max(...a).toFixed(6)}`;})]));
}
function mapPage(key){
  const m=D.maps[key];if(!m){app.innerHTML='<h1>Heatmap not found</h1>';return;}document.title=`${m.title} · SycoScope`;
  const target=m.target||primary(m.base),isOOD=target&&!target.startsWith('cross_cell:');
  const related=Object.entries(D.maps).filter(([k,x])=>x.base===m.base&&x.kind===m.kind&&x.position===m.position&&x.mode===m.mode&&x.layer!=null);
  const explanation={correlation:'These are correlations between probe scores on the same examples—not AUROC and not clusters of responses. Signed distance 1−r, average linkage; best k among 2–9.',consensus:`How often two probes share a group across 200 question-bootstrap draws. ${m.mode==='fixed3'?'Three clusters are forced in every draw; this does not establish three concepts.':'The cluster count is reselected in each draw; the reference ordering uses the original full-data partition.'}`,auc:'Held-out report AUROC. Rows are frozen probes, columns are layers. The largest displayed value is exploratory, not a selection-chosen performance estimate.',transfer:'Rows are trained probes; columns are the generation system-prompt cells being evaluated. Each entry discriminates that column’s positive versus negative instruction. Diagonal entries are own-cell, not cross-prompt transfer. Labels are instruction-derived, not independent behavior judgments.',id:'Each probe discriminates its own positive versus negative generation system prompt on held-out user prompts. This tests instruction-condition separability, not natural OOD behavior.'}[m.kind];
  app.innerHTML=`<div class="eyebrow">${E(m.kind)} / ${E(bname(m.base))}</div><h1>${E(m.title)}</h1><p>${explanation}</p><div class="links">${related.map(([k,x])=>`<a class="${k===key?'current':''}" href="${url(k)}">L${x.layer}</a>`).join('')}</div><div class="controls">${m.clusters?'<label>Matrix order<select id="ordering"><option value="cluster">By cluster</option><option value="taxonomy">Original taxonomy order</option></select></label>':''}${m.stability?'<label>Correlation view<select id="adjustment"><option value="raw">Raw scores</option><option value="adjusted">Remove turn + log response length</option></select></label>':''}</div><div class="layout"><div id="heatmap"></div><aside id="groups" class="groups"></aside></div><div id="stability"></div><section id="associated"></section><section><h2>Provenance & interpretation</h2><p><code>${E(m.source)}</code></p><p class="muted">${m.n?`${m.n.toLocaleString()} examples/turns in the correlation basis. `:''}Cluster memberships use the complete available basis and are exploratory. AUROC uses the saved report split. Inspecting report-set peaks or choosing this heatmap after seeing results does not create a new held-out validation.</p><p><a href="index.html#atlas">Back to heatmap atlas</a> · <a href="provenance.json">Source hashes</a></p></section>`;
  const draw=()=>{
    const adjusted=document.getElementById('adjustment')?.value==='adjusted';
    const state=adjusted?m.stability.turn_and_log_length_adjusted:m.stability?.raw;
    const view=adjusted?{...m,values:state.correlation,clusters:state.clusters,k:state.best_k,silhouette:state.silhouette}:m;
    drawHeatmap(document.getElementById('heatmap'),view,document.getElementById('ordering')?.value!=='taxonomy',isOOD?target:null);
    document.getElementById('groups').innerHTML=view.clusters?view.clusters.map((g,i)=>`<div class="group"><h3>Group ${i+1} <span class="badge">${g.length} probes</span></h3><ul>${g.map(n=>`<li>${E(label(n))}${isOOD?` <small>· AUROC ${F(auc(target,m.position,m.layer,n))}</small>`:''}</li>`).join('')}</ul></div>`).join(''):`<div class="note"><h3>${m.kind==='transfer'?'Not behavior ground truth':'How to read this map'}</h3><p>0.5 = chance ranking. Below 0.5 = reversed association. Missing results are shown as —, never zero.</p>${m.kind==='id'?'<p>Near-perfect AUROC does not validate the taxonomy or identify the best OOD layer.</p>':''}${m.kind==='transfer'?'<p>Benign control positive poles are not uniformly sycophantic. Exclude the diagonal when assessing transfer.</p>':''}</div>`;
    if(m.stability){const r=state,best=r.best_k,freq=r.bootstrap_k_counts[String(best)]/200;document.getElementById('stability').innerHTML=`<section><h2>Does this partition survive resampling?</h2><div class="stats"><div><span class="stat">${best}</span><small>best k · silhouette ${F(r.silhouette)}</small></div><div><span class="stat">${P(freq)}</span><small>bootstrap draws with the same k</small></div><div><span class="stat">${F(r.bootstrap_ari_median)}</span><small>median membership ARI</small></div><div><span class="stat">${P(r.pc1_share)}</span><small>PC1 share of score variance</small></div></div><p class="muted">Same k does not imply the same members. ARI=1 means identical membership up to group numbering. Middle-layer group labels are local to each setting/layer. Adjusting scores does not change the raw-score OOD AUROC shown beside each probe.</p><div class="links">${[12,16,20].includes(m.layer)?`<a href="${key}_bootstrap_selected.html">Adaptive-k bootstrap heatmap</a><a href="${key}_bootstrap_fixed3.html">Fixed-k=3 sensitivity heatmap</a>`:''}</div></section>`;}
  };
  document.getElementById('ordering')?.addEventListener('change',draw);document.getElementById('adjustment')?.addEventListener('change',draw);draw();
  if(isOOD){
    const t=D.targets[target],n=winner(target,m.position,m.layer??16)||'pt_implicit';
    document.getElementById('associated').innerHTML=`<h2>Behavioral AUROC: ${E(fieldname(t.field))}</h2><p class="muted">The group list uses this target’s raw-score report AUROC at the displayed layer. These are individual probe estimates, not a cluster ensemble or a group failure rate.</p><div class="controls"><label>Probe<select id="detail-probe">${options(D.names,n,label)}</select></label></div><div id="detail-curve"></div><div id="detail-curve-detail"></div><h3>Middle-layer sensitivity</h3>${middleTable(target,m.position)}<p><a href="auc_${t.base}_${t.field}_${m.position}.html">All probes × all layers</a></p>`;
    const sel=document.getElementById('detail-probe'),update=()=>renderCurve('detail-curve',target,m.position,sel.value);sel.addEventListener('change',update);update();new ResizeObserver(update).observe(document.getElementById('detail-curve'));
  }else if(m.kind==='transfer'){
    const rows=m.names.map((n,i)=>{const vals=m.values[i].filter((v,j)=>i!==j&&v!=null).sort((a,b)=>a-b);return [E(label(n)),F(m.values[i][i]),F(vals.length%2?vals[Math.floor(vals.length/2)]:(vals[vals.length/2-1]+vals[vals.length/2])/2),`${F(vals[0])}–${F(vals.at(-1))}`];});
    document.getElementById('associated').innerHTML=`<h2>Off-diagonal transfer, per probe</h2><p class="muted">Median across source-cell AUROCs, not a pooled AUROC. Different columns have different instruction meanings.</p>${table(['Probe','Own-cell diagonal','Median other-cell AUROC','Other-cell range'],rows)}`;
  }
}
if(document.body.dataset.page==='overview')overview();else if(document.body.dataset.page==='gallery')allHeatmaps();else mapPage(document.body.dataset.page);
