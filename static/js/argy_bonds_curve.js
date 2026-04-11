// argy_bonds_curve.js
// IIFE — sin conflictos de scope con otros archivos.

(function () {
  'use strict';

  var _charts     = { sov: null, lc: null, on: null };
  var _modalChart = null;
  var _modalKey   = null;
  var _onModalSector = 'ALL';  // sector seleccionado en modal ONs

  var _prevRefresh = typeof window.refreshAll === 'function' ? window.refreshAll : function(){};
  window.refreshAll = async function () {
    await _prevRefresh();
    if (document.getElementById('tab-curve')?.style.display !== 'none') {
      await window.loadAndRenderCurve();
    }
  };

  window.onCurveTabClick = async function () {
    await new Promise(function(r) { setTimeout(r, 50); });
    await window.loadAndRenderCurve();
  };

  window.loadAndRenderCurve = async function () {
    _setAllLoading(true);
    try {
      await _ensureAllData();
      _renderSovChart();
      _renderLecapChart();
      _renderOnsChart();
    } catch(e) {
      _showPageError('Error cargando curvas: ' + e.message);
      console.error('[Curve]', e);
    } finally {
      _setAllLoading(false);
    }
  };

  window.openCurveModal = function(key) {
    _modalKey = key;
    _onModalSector = 'ALL';

    var modal = document.getElementById('curveModalBackdrop');
    var title = document.getElementById('curveModalTitle');
    var sub   = document.getElementById('curveModalSub');
    if (!modal) return;

    var titles = {
      sov: { t: '🏛 Soberanos USD',        s: 'TIR vs Duration · Local / NY · con curva teórica' },
      lc:  { t: '📋 LECAPs & BONCAPs ARS', s: 'TIR anual vs Plazo · Zero coupon · con curva teórica' },
      on:  { t: '🏢 ONs Corporativas USD',  s: 'TIR vs Duration · Curva por sector' },
    };
    if (title) title.textContent = titles[key]?.t || '';
    if (sub)   sub.textContent   = titles[key]?.s || '';

    // Mostrar/ocultar selector de sector
    var sel = document.getElementById('curveOnSectorWrap');
    if (sel) sel.style.display = key === 'on' ? 'flex' : 'none';

    // Poblar el selector con los sectores disponibles
    if (key === 'on') {
      var sectors = _getOnSectors();
      var select  = document.getElementById('curveOnSectorSelect');
      if (select) {
        select.innerHTML = '<option value="ALL">Todos los sectores</option>' +
          sectors.map(function(s) { return '<option value="'+s+'">'+s+'</option>'; }).join('');
        select.value = 'ALL';
      }
    }

    modal.style.display = 'flex';
    setTimeout(function() {
      modal.classList.add('open');
      _renderModalChart(key);
    }, 30);
  };

  window.closeCurveModal = function() {
    var modal = document.getElementById('curveModalBackdrop');
    if (!modal) return;
    modal.classList.remove('open');
    setTimeout(function() {
      modal.style.display = 'none';
      if (_modalChart) { _modalChart.destroy(); _modalChart = null; }
    }, 200);
  };

  // Llamado desde el select del modal ONs
  window.onCurveOnSectorChange = function(val) {
    _onModalSector = val;
    _renderModalChart('on');
  };

  // ════════════════════════════════════════════════════
  // DATA
  // ════════════════════════════════════════════════════

  async function _ensureAllData() {
    var jobs = [];
    if (!window._enriched?.length)  jobs.push(_fetchSov());
    if (!window._lcData?.length)    jobs.push(_fetchLc());
    if (!window._onData?.length)    jobs.push(_fetchOn());
    if (jobs.length) await Promise.all(jobs);
  }

  async function _fetchSov() {
    try {
      if (typeof window.BOND_META === 'object' && !Object.keys(window.BOND_META).length)
        if (typeof window.loadBondConfig === 'function') await window.loadBondConfig();
      var d = await window.apiFetch('/argy_bonds/live');
      if (typeof window.enrichBond === 'function')
        window._enriched = (d.bonds || []).map(function(b) { return window.enrichBond(b); });
    } catch(e) { console.warn('[Curve] sov:', e); }
  }

  async function _fetchLc() {
    try {
      var d = await window.apiFetch('/lecap/live');
      window._lcData = d.data || [];
    } catch(e) { console.warn('[Curve] lc:', e); }
  }

  async function _fetchOn() {
    try {
      var d = await window.apiFetch('/ons/live');
      window._onData = d.bonds || [];
    } catch(e) { console.warn('[Curve] on:', e); }
  }

  // ════════════════════════════════════════════════════
  // SECTOR HELPERS
  // ════════════════════════════════════════════════════

  var _sectorColors = {
    'Energy':         'rgba(248,81,73,0.85)',
    'Banking':        'rgba(188,140,255,0.85)',
    'Agro':           'rgba(63,210,100,0.85)',
    'Infrastructure': 'rgba(88,200,255,0.85)',
    'Real Estate':    'rgba(255,180,50,0.85)',
    'Industry':       'rgba(160,160,160,0.85)',
    'Consumer':       'rgba(255,120,190,0.85)',
    'Construction':   'rgba(255,140,80,0.85)',
    'Other':          'rgba(120,120,120,0.85)',
  };

  function _getOnSectors() {
    var seen = {};
    (window._onData || []).forEach(function(b) {
      if (b.tir != null && b.duration != null) seen[b.sector || 'Other'] = 1;
    });
    return Object.keys(seen).sort();
  }

  // ════════════════════════════════════════════════════
  // BUILD DATASETS
  // ════════════════════════════════════════════════════

  function _buildSovDatasets(withCurve) {
    var local = (window._enriched || [])
      .filter(function(b) { return b.tir!=null && b.duration!=null && b.law==='Local'; })
      .map(function(b) { return _pt(b.duration, b.tir*100, b.symbol, 'Precio: US$'+(b.price_usd?.toFixed(2)??'—')+' · Vto: '+(b.maturity??'—')); });
    var ny = (window._enriched || [])
      .filter(function(b) { return b.tir!=null && b.duration!=null && b.law==='NY'; })
      .map(function(b) { return _pt(b.duration, b.tir*100, b.symbol, 'Precio: US$'+(b.price_usd?.toFixed(2)??'—')+' · Vto: '+(b.maturity??'—')); });

    var ds = [];
    if (local.length) ds.push({ label:'Local', data:local, backgroundColor:'rgba(88,166,255,0.85)', borderColor:'#58A6FF', pointRadius:7, pointHoverRadius:11, pointStyle:'circle', showLine:false });
    if (ny.length)    ds.push({ label:'NY',    data:ny,    backgroundColor:'rgba(63,185,80,0.85)',  borderColor:'#3fb950', pointRadius:7, pointHoverRadius:11, pointStyle:'rectRot', showLine:false });

    if (withCurve) {
      var all = local.concat(ny).sort(function(a,b){return a.x-b.x;});
      if (all.length >= 2) ds.push(_curveDataset(_smoothCurve(all, 60)));
    }
    return ds;
  }

  function _buildLcDatasets(withCurve) {
    var lecaps = (window._lcData || [])
      .filter(function(l) { return l.tir!=null && !l.is_expired && l.days_to_maturity>0 && l.security_type==='LECAP'; })
      .map(function(l) { return _pt(l.days_to_maturity/365, l.tir*100, l.symbol, l.security_type+' · $'+(l.price?.toFixed(2)??'—')+' · Vto: '+(l.maturity_date??'—')); });
    var boncaps = (window._lcData || [])
      .filter(function(l) { return l.tir!=null && !l.is_expired && l.days_to_maturity>0 && l.security_type==='BONCAP'; })
      .map(function(l) { return _pt(l.days_to_maturity/365, l.tir*100, l.symbol, l.security_type+' · $'+(l.price?.toFixed(2)??'—')+' · Vto: '+(l.maturity_date??'—')); });

    var ds = [];
    if (lecaps.length)  ds.push({ label:'LECAP',  data:lecaps,  backgroundColor:'rgba(210,153,34,0.85)', borderColor:'#d2991a', pointRadius:6, pointHoverRadius:10, pointStyle:'circle',  showLine:false });
    if (boncaps.length) ds.push({ label:'BONCAP', data:boncaps, backgroundColor:'rgba(188,140,255,0.85)',borderColor:'#bc8cff', pointRadius:6, pointHoverRadius:10, pointStyle:'rectRot', showLine:false });

    if (withCurve) {
      var all = lecaps.concat(boncaps).sort(function(a,b){return a.x-b.x;});
      if (all.length >= 3) ds.push(_curveDataset(_smoothCurve(all, 60)));
    }
    return ds;
  }

  // ONs panel — todos los sectores, sin curva
  function _buildOnDatasets() {
    var bySector = {};
    (window._onData || [])
      .filter(function(b) { return b.tir!=null && b.duration!=null; })
      .forEach(function(b) {
        var s = b.sector || 'Other';
        if (!bySector[s]) bySector[s] = [];
        bySector[s].push(_pt(b.duration, b.tir*100, b.symbol, (b.issuer??'—')+' · US$'+(b.price_usd?.toFixed(2)??'—')+' · Vto: '+(b.maturity??'—')));
      });
    return Object.entries(bySector).map(function(entry) {
      var sector=entry[0], pts=entry[1], c=_sectorColors[sector]||_sectorColors['Other'];
      return { label:sector, data:pts, backgroundColor:c, borderColor:c.replace('0.85','1'), pointRadius:6, pointHoverRadius:10, pointStyle:'circle', showLine:false };
    });
  }

  // ONs modal — sector elegido + curva teórica de ese sector
  function _buildOnModalDatasets(sector) {
    var bySector = {};
    (window._onData || [])
      .filter(function(b) { return b.tir!=null && b.duration!=null; })
      .forEach(function(b) {
        var s = b.sector || 'Other';
        if (!bySector[s]) bySector[s] = [];
        bySector[s].push(_pt(b.duration, b.tir*100, b.symbol, (b.issuer??'—')+' · US$'+(b.price_usd?.toFixed(2)??'—')+' · Vto: '+(b.maturity??'—')));
      });

    var sectorsToShow = sector === 'ALL' ? Object.keys(bySector) : [sector];
    var ds = [];

    sectorsToShow.forEach(function(s) {
      var pts = bySector[s];
      if (!pts || !pts.length) return;
      var c = _sectorColors[s] || _sectorColors['Other'];
      ds.push({ label:s, data:pts, backgroundColor:c, borderColor:c.replace('0.85','1'), pointRadius:7, pointHoverRadius:11, pointStyle:'circle', showLine:false });

      // Curva por sector si hay >= 3 puntos
      if (pts.length >= 3) {
        var sorted = pts.slice().sort(function(a,b){return a.x-b.x;});
        var curve  = _smoothCurve(sorted, 50);
        ds.push(_curveDataset(curve, c.replace('0.85','0.6')));
      }
    });

    return ds;
  }

  // ════════════════════════════════════════════════════
  // CURVA TEÓRICA — regresión logarítmica OLS
  // ════════════════════════════════════════════════════

  function _smoothCurve(pts, nSamples) {
    var validPts = pts.filter(function(p){ return p.x > 0; });
    if (validPts.length < 2) return [];

    var n=validPts.length, sx=0, sy=0, sxx=0, sxy=0;
    for (var i=0; i<n; i++) {
      var lx = Math.log(validPts[i].x);
      sx+=lx; sy+=validPts[i].y; sxx+=lx*lx; sxy+=lx*validPts[i].y;
    }
    var denom = n*sxx - sx*sx;
    if (Math.abs(denom) < 1e-10) return [];

    var b = (n*sxy - sx*sy) / denom;
    var a = (sy - b*sx) / n;

    var xMin=validPts[0].x, xMax=validPts[validPts.length-1].x;
    var result=[];
    for (var j=0; j<=nSamples; j++) {
      var x = xMin + (xMax-xMin)*j/nSamples;
      if (x<=0) continue;
      result.push({ x:parseFloat(x.toFixed(3)), y:parseFloat((a+b*Math.log(x)).toFixed(3)) });
    }
    return result;
  }

  function _curveDataset(data, color) {
    return {
      label: 'Curva teórica',
      data: data,
      borderColor: color || 'rgba(255,255,100,0.55)',
      backgroundColor: 'transparent',
      pointRadius: 0, pointHoverRadius: 0,
      borderWidth: 2, borderDash: [5,4],
      showLine: true, tension: 0.4, type: 'line',
    };
  }

  // ════════════════════════════════════════════════════
  // RENDER — PANELS PEQUEÑOS
  // ════════════════════════════════════════════════════

  function _renderSovChart() {
    var canvas = document.getElementById('curveSovCanvas');
    if (!canvas) return;
    var ds = _buildSovDatasets(false);
    _renderChart('sov', canvas, ds, 'Duration (años)', 'TIR (%)');
    _setStats('Sov', _flatY(ds));
  }

  function _renderLecapChart() {
    var canvas = document.getElementById('curveLcCanvas');
    if (!canvas) return;
    var ds = _buildLcDatasets(false);
    _renderChart('lc', canvas, ds, 'Plazo (años)', 'TIR anual (%)');
    _setStats('Lc', _flatY(ds));
  }

  function _renderOnsChart() {
    var canvas = document.getElementById('curveOnCanvas');
    if (!canvas) return;
    var ds = _buildOnDatasets();
    _renderChart('on', canvas, ds, 'Duration (años)', 'TIR (%)');
    _setStats('On', _flatY(ds));
  }

  // ════════════════════════════════════════════════════
  // RENDER — MODAL
  // ════════════════════════════════════════════════════

  function _renderModalChart(key) {
    var canvas = document.getElementById('curveModalCanvas');
    if (!canvas || typeof Chart === 'undefined') return;
    if (_modalChart) { _modalChart.destroy(); _modalChart = null; }

    var ds, xLabel, yLabel;
    if (key === 'sov') {
      ds = _buildSovDatasets(true); xLabel='Duration (años)'; yLabel='TIR (%)';
    } else if (key === 'lc') {
      ds = _buildLcDatasets(true); xLabel='Plazo (años)'; yLabel='TIR anual (%)';
    } else {
      ds = _buildOnModalDatasets(_onModalSector); xLabel='Duration (años)'; yLabel='TIR (%)';
    }

    if (!ds.length) return;

    var labelPlugin = {
      id: 'modalLabels',
      afterDatasetsDraw: function(chart) {
        var c = chart.ctx;
        chart.data.datasets.forEach(function(ds, di) {
          var meta = chart.getDatasetMeta(di);
          if (meta.hidden || ds.label === 'Curva teórica') return;
          meta.data.forEach(function(pt, i) {
            var item = ds.data[i];
            if (!item || !item.label) return;
            c.save();
            c.font = 'bold 10px monospace';
            c.fillStyle = 'rgba(200,210,220,0.9)';
            c.textAlign = 'center';
            c.fillText(item.label, pt.x, pt.y - 12);
            c.restore();
          });
        });
      },
    };

    try {
      _modalChart = new Chart(canvas.getContext('2d'), {
        type: 'scatter',
        data: { datasets: ds },
        plugins: [labelPlugin],
        options: {
          responsive:true, maintainAspectRatio:false,
          animation:{ duration:250 },
          layout:{ padding:{top:28,right:24,bottom:10,left:10} },
          scales: {
            x: { title:{display:true,text:xLabel,color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:11}}, grid:{color:'rgba(48,54,61,0.5)'}, ticks:{color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:11}}, min:0 },
            y: { title:{display:true,text:yLabel,color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:11}}, grid:{color:'rgba(48,54,61,0.5)'}, ticks:{color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:11},callback:function(v){return v.toFixed(1)+'%';}} },
          },
          plugins: {
            legend:{ position:'bottom', labels:{color:'rgba(139,148,158,0.9)',font:{family:'monospace',size:10},boxWidth:10,padding:14,usePointStyle:true} },
            tooltip:{
              backgroundColor:'rgba(22,27,34,0.97)', borderColor:'rgba(48,54,61,0.8)', borderWidth:1,
              titleColor:'#E6EDF3', bodyColor:'rgba(139,148,158,0.9)',
              titleFont:{family:'monospace',size:12,weight:'bold'}, bodyFont:{family:'monospace',size:11},
              padding:12,
              callbacks:{
                title:function(items){return items[0]?.raw?.label??'';},
                label:function(item){
                  var d=item.raw;
                  if(!d.label) return '';
                  return [yLabel.split(' ')[0]+': '+d.y.toFixed(2)+'%', xLabel.split(' ')[0]+': '+d.x.toFixed(2), d.extra||''];
                },
              },
            },
          },
        },
      });
    } catch(e) { console.error('[Curve] modal error:', e); }
  }

  // ════════════════════════════════════════════════════
  // GENERIC PANEL RENDERER
  // ════════════════════════════════════════════════════

  function _renderChart(key, canvas, datasets, xLabel, yLabel) {
    if (_charts[key]) { _charts[key].destroy(); _charts[key] = null; }
    var emptyEl  = canvas.parentElement.querySelector('.curve-empty');
    var totalPts = datasets.reduce(function(s,d){return s+d.data.length;},0);

    if (!totalPts) {
      canvas.style.display='none';
      if (emptyEl) emptyEl.style.display='flex';
      return;
    }
    canvas.style.display='block';
    if (emptyEl) emptyEl.style.display='none';
    if (typeof Chart==='undefined') { console.error('[Curve] Chart.js no cargado'); return; }

    var labelPlugin = {
      id:'ptLabels_'+key,
      afterDatasetsDraw:function(chart){
        var c=chart.ctx;
        chart.data.datasets.forEach(function(ds,di){
          var meta=chart.getDatasetMeta(di);
          if(meta.hidden||ds.label==='Curva teórica') return;
          meta.data.forEach(function(pt,i){
            var item=ds.data[i];
            if(!item||!item.label) return;
            c.save(); c.font='bold 8px monospace'; c.fillStyle='rgba(200,210,220,0.8)'; c.textAlign='center';
            c.fillText(item.label,pt.x,pt.y-10); c.restore();
          });
        });
      },
    };

    try {
      _charts[key] = new Chart(canvas.getContext('2d'), {
        type:'scatter', data:{datasets:datasets}, plugins:[labelPlugin],
        options:{
          responsive:true, maintainAspectRatio:false, animation:{duration:200},
          layout:{padding:{top:20,right:16,bottom:8,left:8}},
          onClick:function(){ window.openCurveModal(key); },
          scales:{
            x:{title:{display:true,text:xLabel,color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:9}},grid:{color:'rgba(48,54,61,0.5)'},ticks:{color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:9}},min:0},
            y:{title:{display:true,text:yLabel,color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:9}},grid:{color:'rgba(48,54,61,0.5)'},ticks:{color:'rgba(139,148,158,0.8)',font:{family:'monospace',size:9},callback:function(v){return v.toFixed(1)+'%';}}},
          },
          plugins:{
            legend:{position:'bottom',labels:{color:'rgba(139,148,158,0.9)',font:{family:'monospace',size:9},boxWidth:9,padding:10,usePointStyle:true}},
            tooltip:{backgroundColor:'rgba(22,27,34,0.97)',borderColor:'rgba(48,54,61,0.8)',borderWidth:1,titleColor:'#E6EDF3',bodyColor:'rgba(139,148,158,0.9)',titleFont:{family:'monospace',size:11,weight:'bold'},bodyFont:{family:'monospace',size:10},padding:10,
              callbacks:{title:function(items){return items[0]?.raw?.label??'';},label:function(item){var d=item.raw;return[yLabel.split(' ')[0]+': '+d.y.toFixed(2)+'%',xLabel.split(' ')[0]+': '+d.x.toFixed(2),d.extra||''];}}},
          },
        },
      });
    } catch(e) { console.error('[Curve] panel error ('+key+'):', e); }
  }

  // ════════════════════════════════════════════════════
  // HELPERS
  // ════════════════════════════════════════════════════

  function _pt(x,y,label,extra){ return {x:parseFloat(Number(x).toFixed(2)),y:parseFloat(Number(y).toFixed(2)),label:label,extra:extra}; }

  function _flatY(ds) {
    return ds.filter(function(d){return d.label!=='Curva teórica';})
             .reduce(function(acc,d){return acc.concat(d.data.map(function(p){return p.y;}));}, []);
  }

  function _setStats(suffix, ys) {
    if (!ys.length) { ['Min','Avg','Max','Count'].forEach(function(k){var el=document.getElementById('curveStat'+suffix+k);if(el)el.textContent='—';}); return; }
    var min=Math.min.apply(null,ys), max=Math.max.apply(null,ys), avg=ys.reduce(function(a,b){return a+b;},0)/ys.length;
    function _s(id,v){var el=document.getElementById(id);if(el)el.textContent=v;}
    _s('curveStat'+suffix+'Min', min.toFixed(2)+'%');
    _s('curveStat'+suffix+'Avg', avg.toFixed(2)+'%');
    _s('curveStat'+suffix+'Max', max.toFixed(2)+'%');
    _s('curveStat'+suffix+'Count', ys.length+(suffix==='Lc'?' letras':' bonos'));
  }

  function _setAllLoading(show) {
    ['curveSovLoading','curveLcLoading','curveOnLoading'].forEach(function(id){var el=document.getElementById(id);if(el)el.style.display=show?'flex':'none';});
  }

  function _showPageError(msg) {
    var el=document.getElementById('curvePageError');
    if(!el) return;
    el.textContent='⚠ '+msg; el.style.display='block';
    setTimeout(function(){el.style.display='none';},6000);
  }

})();