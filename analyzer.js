/* Chord Nook 音源解析エンジン
   - Worker として動くとき: {channels:[Float32Array...], sr} を受け取り特徴量を返す
   - ページ/Node から読むとき: ChordAnalyzer.{features, estimateBeats, label} を使える */
(function (G) {
  'use strict';

  // ---------- FFT ----------
  const tables = {};
  function fftTables(n) {
    if (tables[n]) return tables[n];
    const c = new Float64Array(n / 2), s = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { c[i] = Math.cos(-2 * Math.PI * i / n); s[i] = Math.sin(-2 * Math.PI * i / n); }
    const rev = new Uint32Array(n);
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; rev[i] = j; }
    return (tables[n] = { c, s, rev });
  }
  function fft(re, im) {
    const n = re.length, T = fftTables(n);
    for (let i = 1; i < n; i++) { const j = T.rev[i]; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0, k = 0; j < half; j++, k += step) {
          const a = i + j, b = a + half, wr = T.c[k], wi = T.s[k];
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  }
  const hann = n => { const w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); return w; };

  // ---------- 特徴量 ----------
  function mixDown(chs, sr) {
    const f = Math.max(1, Math.floor(sr / 20000));
    const n = Math.floor(chs[0].length / f), out = new Float32Array(n), k = 1 / (f * chs.length);
    for (let i = 0; i < n; i++) {
      let s = 0; const o = i * f;
      for (let c = 0; c < chs.length; c++) { const ch = chs[c]; for (let j = 0; j < f; j++) s += ch[o + j]; }
      out[i] = s * k;
    }
    return { x: out, sr: sr / f };
  }

  function features(chs, srIn, progress) {
    const { x, sr } = mixDown(chs, srIn);
    const len = x.length;
    // クロマ用（長い窓）
    const N = sr > 26000 ? 16384 : 8192, hop = N / 4, win = hann(N);
    const tb = [], hb = [], bb = [];
    for (let k = 1; k < N / 2; k++) {
      const f = k * sr / N, p = 12 * Math.log2(f / 440) + 69, m = Math.round(p);
      const w = Math.pow(Math.cos(Math.PI * (p - m)), 2), pc = ((m % 12) + 12) % 12;
      if (f >= 65 && f < 520) tb.push(k, pc, w);
      else if (f >= 520 && f <= 2100) hb.push(k, pc, w);
      if (f >= 38 && f < 260) bb.push(k, pc, w);
    }
    const nF = Math.max(1, Math.floor((len - N) / hop) + 1);
    const chroma = new Float32Array(nF * 12), chromaHi = new Float32Array(nF * 12), bass = new Float32Array(nF * 12), rms = new Float32Array(nF);
    const re = new Float64Array(N), im = new Float64Array(N), mag = new Float64Array(N / 2);
    for (let fi = 0; fi < nF; fi++) {
      const st = fi * hop; let e = 0;
      for (let i = 0; i < N; i++) { const v = st + i < len ? x[st + i] : 0; re[i] = v * win[i]; im[i] = 0; e += v * v; }
      rms[fi] = Math.sqrt(e / N);
      fft(re, im);
      for (let k = 0; k < N / 2; k++) mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      const o = fi * 12;
      for (let i = 0; i < tb.length; i += 3) chroma[o + tb[i + 1]] += mag[tb[i]] * tb[i + 2];
      for (let i = 0; i < hb.length; i += 3) chromaHi[o + hb[i + 1]] += mag[hb[i]] * hb[i + 2];
      for (let i = 0; i < bb.length; i += 3) bass[o + bb[i + 1]] += mag[bb[i]] * bb[i + 2];
      if (progress && fi % 64 === 0) progress(0.75 * fi / nF);
    }
    // 拍取り用（短い窓のスペクトルフラックス）
    const N2 = 1024, hop2 = 256, win2 = hann(N2);
    const nO = Math.max(1, Math.floor((len - N2) / hop2) + 1);
    const onset = new Float32Array(nO);
    const r2 = new Float64Array(N2), i2 = new Float64Array(N2), prev = new Float64Array(N2 / 2);
    for (let fi = 0; fi < nO; fi++) {
      const st = fi * hop2;
      for (let i = 0; i < N2; i++) { r2[i] = (st + i < len ? x[st + i] : 0) * win2[i]; i2[i] = 0; }
      fft(r2, i2);
      let s = 0;
      for (let k = 1; k < N2 / 2; k++) {
        const L = Math.log1p(100 * Math.sqrt(r2[k] * r2[k] + i2[k] * i2[k]));
        const d = L - prev[k]; if (d > 0) s += d; prev[k] = L;
      }
      onset[fi] = s;
      if (progress && fi % 512 === 0) progress(0.75 + 0.25 * fi / nO);
    }
    return { chroma, chromaHi, bass, rms, nF, hopSec: hop / sr, winSec: N / sr, onset, onHopSec: hop2 / sr, onOff: N2 / 2 / sr, dur: len / sr };
  }

  // ---------- テンポ・拍 ----------
  function estimateBeats(F, mult) {
    const on = F.onset, n = on.length, h = F.onHopSec;
    const W = Math.max(1, Math.round(0.4 / h)), cs = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) cs[i + 1] = cs[i] + on[i];
    const o = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - W), b = Math.min(n, i + W + 1);
      o[i] = Math.max(0, on[i] - (cs[b] - cs[a]) / (b - a));
    }
    const minL = Math.floor(60 / 200 / h), maxL = Math.ceil(60 / 55 / h);
    let best = -Infinity, bestL = Math.round(0.5 / h);
    for (let L = minL; L <= maxL && L < n - 1; L++) {
      let s = 0; for (let i = L; i < n; i++) s += o[i] * o[i - L];
      s /= (n - L);
      const bpm = 60 / (L * h), w = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 115) / 0.9, 2));
      if (s * w > best) { best = s * w; bestL = L; }
    }
    const ac = L => { let s = 0; for (let i = L; i < n; i++) s += o[i] * o[i - L]; return s / (n - L); };
    const base = ac(bestL);
    let bpm0 = 60 / (bestL * h);
    if (bpm0 < 72 && Math.round(bestL / 2) >= 2 && ac(Math.round(bestL / 2)) > 0.35 * base) bestL = bestL / 2;
    else if (bpm0 >= 165 && ac(bestL * 2) > 0.35 * base) bestL = bestL * 2;
    let L0 = bestL * (mult || 1);
    const comb = (P, phi) => { let s = 0, c = 0; for (let t = phi; t < n - 1; t += P) { const i = t | 0, f = t - i; s += o[i] * (1 - f) + o[i + 1] * f; c++; } return c ? s / c : 0; };
    let bp = L0, bphi = 0, bs = -1;
    for (let P = L0 * 0.96; P <= L0 * 1.04; P += 0.02) {
      for (let phi = 0; phi < P; phi += 0.5) { const s = comb(P, phi); if (s > bs) { bs = s; bp = P; bphi = phi; } }
    }
    // 位相の微調整
    for (let phi = Math.max(0, bphi - 0.5); phi <= bphi + 0.5; phi += 0.1) { const s = comb(bp, phi); if (s > bs) { bs = s; bphi = phi; } }
    return { period: bp * h, phase: bphi * h + (F.onOff || 0) };
  }
  function makeBeats(period, phase, dur) {
    const b = []; for (let t = phase; t < dur; t += period) b.push(t); return b;
  }

  // ---------- コード判定 ----------
  const QUALS = [['', [0, 4, 7]], ['m', [0, 3, 7]], ['7', [0, 4, 7, 10]], ['m7', [0, 3, 7, 10]], ['M7', [0, 4, 7, 11]]];
  const HARM = [[0, 1], [0, 0.6], [7, 0.36], [0, 0.22], [4, 0.13]];
  const TEMPL = [];
  (function () {
    for (let qi = 0; qi < QUALS.length; qi++) for (let r = 0; r < 12; r++) {
      const t = new Float64Array(12);
      QUALS[qi][1].forEach((iv, n) => { const w = n === 3 ? 0.8 : 1; HARM.forEach(([o, a]) => { t[(r + iv + o) % 12] += w * a; }); });
      const m = t.reduce((a, b) => a + b) / 12; let nrm = 0;
      for (let i = 0; i < 12; i++) { t[i] -= m; nrm += t[i] * t[i]; }
      nrm = Math.sqrt(nrm); for (let i = 0; i < 12; i++) t[i] /= nrm;
      TEMPL.push({ r, q: qi, t, sb: null });
    }
    const n0 = TEMPL.length;
    for (let x = 0; x < n0; x++) {
      const tp = TEMPL[x];
      if (tp.q === 0) TEMPL.push({ r: tp.r, q: 0, t: tp.t, sb: (tp.r + 4) % 12 });
      if (tp.q === 1) TEMPL.push({ r: tp.r, q: 1, t: tp.t, sb: (tp.r + 10) % 12 });
    }
  })();
  const KMAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KMIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
  const corr = (a, b) => {
    const ma = a.reduce((x, y) => x + y) / 12, mb = b.reduce((x, y) => x + y) / 12; let n = 0, da = 0, db = 0;
    for (let i = 0; i < 12; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
    return n / Math.sqrt(da * db + 1e-12);
  };
  function estimateKey(gc) {
    let best = -2, k = 0, minor = false;
    for (let r = 0; r < 12; r++) for (const mn of [false, true]) {
      const p = mn ? KMIN : KMAJ, rot = Array.from({ length: 12 }, (_, i) => p[(i - r + 12) % 12]);
      const c = corr(gc, rot); if (c > best) { best = c; k = r; minor = mn; }
    }
    return { root: k, minor };
  }
  function diatonic(key) {
    const rel = key.minor ? (key.root + 3) % 12 : key.root, set = new Set();
    [[0, 0], [2, 1], [4, 1], [5, 0], [7, 0], [9, 1], [0, 4], [2, 3], [4, 3], [5, 4], [7, 2], [9, 3]].forEach(([o, q]) => set.add(((rel + o) % 12) * 10 + q));
    if (key.minor) { set.add(((key.root + 7) % 12) * 10 + 0); set.add(((key.root + 7) % 12) * 10 + 2); }
    return set;
  }

  function beatFeatures(F, beats, wh) {
    const T = beats.length, C = [], B = [], E = new Float64Array(T);
    const per = T > 1 ? beats[1] - beats[0] : 0.5;
    for (let i = 0; i < T; i++) {
      const t0 = beats[i], t1 = i + 1 < T ? beats[i + 1] : t0 + per;
      const c = new Float64Array(12), b = new Float64Array(12); let cnt = 0, e = 0;
      let j0 = Math.max(0, Math.floor((t0 - F.winSec / 2) / F.hopSec)), j1 = Math.min(F.nF - 1, Math.ceil((t1 - F.winSec / 2) / F.hopSec));
      for (let j = j0; j <= j1; j++) {
        const ct = j * F.hopSec + F.winSec / 2; if (ct < t0 || ct >= t1) continue;
        for (let k = 0; k < 12; k++) { c[k] += F.chroma[j * 12 + k] + wh * F.chromaHi[j * 12 + k]; b[k] += F.bass[j * 12 + k]; }
        e += F.rms[j]; cnt++;
      }
      if (!cnt) {
        const j = Math.max(0, Math.min(F.nF - 1, Math.round(((t0 + t1) / 2 - F.winSec / 2) / F.hopSec)));
        for (let k = 0; k < 12; k++) { c[k] = F.chroma[j * 12 + k] + wh * F.chromaHi[j * 12 + k]; b[k] = F.bass[j * 12 + k]; } e = F.rms[j]; cnt = 1;
      }
      const mx = Math.max(...c) + 1e-12, bx = Math.max(...b) + 1e-12;
      for (let k = 0; k < 12; k++) { c[k] = Math.log1p(5 * c[k] / mx) / Math.log1p(5); b[k] /= bx; }
      C.push(c); B.push(b); E[i] = e / cnt;
    }
    return { C, B, E };
  }

  function viterbi(em, pen) {
    const T = em.length, S = em[0].length, bp = [];
    let prev = Float64Array.from(em[0]);
    for (let i = 1; i < T; i++) {
      let a1 = -Infinity, i1 = 0, a2 = -Infinity;
      for (let s = 0; s < S; s++) { const v = prev[s]; if (v > a1) { a2 = a1; a1 = v; i1 = s; } else if (v > a2) a2 = v; }
      const cur = new Float64Array(S), back = new Int16Array(S), p = pen(i);
      for (let s = 0; s < S; s++) {
        const other = s === i1 ? a2 : a1;
        if (prev[s] >= other - p) { cur[s] = em[i][s] + prev[s]; back[s] = s; }
        else { cur[s] = em[i][s] + other - p; back[s] = i1; }
      }
      bp.push(back); prev = cur;
    }
    let s = 0; for (let k = 1; k < S; k++) if (prev[k] > prev[s]) s = k;
    const path = new Int16Array(T); path[T - 1] = s;
    for (let i = T - 1; i > 0; i--) { s = bp[i - 1][s]; path[i - 1] = s; }
    return path;
  }

  function label(F, beats, opt) {
    opt = opt || {};
    const P = Object.assign({ wh: 0.15, wb: 0.4, wd: 0.06, w7: 0.04, ws: 0.06, lam: 0.22 }, opt.p || {});
    const { C, B, E } = beatFeatures(F, beats, P.wh), T = beats.length;
    const sortedE = Array.from(E).sort((a, b) => a - b), med = sortedE[Math.floor(T / 2)] || 1e-9;
    const gc = new Array(12).fill(0);
    for (let i = 0; i < T; i++) if (E[i] > 0.12 * med) for (let k = 0; k < 12; k++) gc[k] += C[i][k] * Math.min(1, E[i] / med);
    const key = opt.key || estimateKey(gc), dia = diatonic(key);
    const S = TEMPL.length, NC = S;
    const em = [];
    for (let i = 0; i < T; i++) {
      const row = new Float64Array(S + 1), silent = E[i] < 0.12 * med;
      const c = C[i], m = c.reduce((a, b) => a + b) / 12; let nrm = 0; const cc = new Float64Array(12);
      for (let k = 0; k < 12; k++) { cc[k] = c[k] - m; nrm += cc[k] * cc[k]; }
      nrm = Math.sqrt(nrm) + 1e-12;
      for (let s = 0; s < S; s++) {
        const tp = TEMPL[s]; let d = 0; for (let k = 0; k < 12; k++) d += cc[k] * tp.t[k];
        let v = d / nrm + P.wb * B[i][tp.sb == null ? tp.r : tp.sb];
        if (tp.sb != null) v -= P.ws;
        if (dia.has(tp.r * 10 + tp.q)) v += P.wd;
        if (tp.q >= 2) v -= P.w7;
        row[s] = silent ? -1 : v;
      }
      row[NC] = silent ? 1 : -1;
      em.push(row);
    }
    const lam = P.lam;
    let down = opt.down;
    if (down == null) {
      const p0 = viterbi(em, () => lam), cnt = [0, 0, 0, 0];
      for (let i = 1; i < T; i++) if (p0[i] !== p0[i - 1]) cnt[i % 4]++;
      down = cnt.indexOf(Math.max(...cnt));
    }
    const pf = i => { const ph = ((i - down) % 4 + 4) % 4; return ph === 0 ? 0.5 : ph === 2 ? 0.8 : 1.7; };
    const path = viterbi(em, i => lam * pf(i));
    const segs = [];
    for (let i = 0; i < T;) {
      let j = i; while (j + 1 < T && path[j + 1] === path[i]) j++;
      const s = path[i], seg = { b0: i, b1: j + 1, t0: beats[i], t1: j + 1 < T ? beats[j + 1] : F.dur };
      if (s === NC) seg.nc = 1;
      else {
        const tp = TEMPL[s]; seg.r = tp.r; seg.q = tp.q; if (tp.sb != null) seg.b = tp.sb;
        const avg = new Float64Array(S);
        for (let k = i; k <= j; k++) for (let x = 0; x < S; x++) avg[x] += em[k][x];
        const seen = new Set([s]), alts = [];
        for (const x of Array.from({ length: S }, (_, x) => x).sort((a, b) => avg[b] - avg[a])) {
          if (seen.has(x)) continue; seen.add(x); const t = TEMPL[x];
          alts.push(t.sb == null ? [t.r, t.q] : [t.r, t.q, t.sb]); if (alts.length >= 5) break;
        }
        seg.alts = alts;
      }
      segs.push(seg); i = j + 1;
    }
    return { key, down, segs };
  }

  const API = { features, estimateBeats, makeBeats, label, QUALS: QUALS.map(q => q[0]) };

  const isWorker = typeof WorkerGlobalScope !== 'undefined' && G instanceof WorkerGlobalScope;
  if (isWorker) {
    G.onmessage = e => {
      try {
        const F = features(e.data.channels, e.data.sr, p => G.postMessage({ type: 'progress', p }));
        G.postMessage({ type: 'done', F }, [F.chroma.buffer, F.chromaHi.buffer, F.bass.buffer, F.rms.buffer, F.onset.buffer]);
      } catch (err) { G.postMessage({ type: 'error', message: String(err && err.message || err) }); }
    };
  } else {
    G.ChordAnalyzer = API;
    if (typeof module !== 'undefined' && module.exports) module.exports = API;
  }
})(typeof self !== 'undefined' ? self : globalThis);
