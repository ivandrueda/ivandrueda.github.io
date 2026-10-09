/* Public demonstration: synthetic inputs and fixed illustrative coefficients.
   Preserves the original model's factor-to-outcome links, not its fitted values. */
const SurgicalNetworkDemo = (() => {
  const names = ['Coordination', 'Scheduling', 'Patient readiness', 'Anesthesia readiness', 'OR team'];
  const edges = [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[3,2],[4,2],[0,3],[1,3],[2,3],[3,3],[4,3]];
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function normal(random) { return Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random()); }
  function gamma(shape, random) {
    const d = shape - 1/3, c = 1 / Math.sqrt(9*d);
    for (;;) { const x = normal(random), v = (1+c*x)**3; if (v <= 0) continue; const u = random(); if (u < 1-0.0331*x**4 || Math.log(u) < 0.5*x*x + d*(1-v+Math.log(v))) return d*v; }
  }
  function predict(f) {
    const [c,s,p,a,t] = f;
    const start = sigmoid(-1.2 + 0.6*c + 0.9*s + 0.7*p + 1.2*a);
    return [sigmoid(-0.4+0.7*c+0.6*s+1.1*p+0.8*a), start,
      sigmoid(-0.8+1.1*a+1.2*t-0.6*start), sigmoid(-0.1+0.4*c+0.5*s+0.3*p+0.6*a+0.8*t)];
  }
  function weightedQuantile(values, weights, q) {
    const order = values.map((v,i)=>[v,weights[i]]).sort((a,b)=>a[0]-b[0]);
    let sum=0; for (const [v,w] of order) { sum+=w; if (sum>=q) return v; } return order[order.length-1][0];
  }
  function summary(values, weights) {
    return {mean:values.reduce((s,v,i)=>s+v*weights[i],0), lo:weightedQuantile(values,weights,0.05), hi:weightedQuantile(values,weights,0.95)};
  }
  function build(count=12000) {
    const random=rng(41872), samples=[];
    for (let i=0;i<count;i++) { const f=names.map((_,k)=>{ const x=gamma(k+2,random),y=gamma(k+2,random); return x/(x+y); }); samples.push({f,p:predict(f)}); }
    // A wholly synthetic batch of 20 sessions. Counts refer to sessions, not cases.
    const evidence={n:20, successes:[17,9,12], efficiencyMean:0.68, efficiencySD:0.18};
    const logs=samples.map(({p})=>{
      let v=0;
      for(let k=0;k<3;k++) { const success=evidence.successes[k]; v+=success*Math.log(p[k])+(evidence.n-success)*Math.log(1-p[k]); }
      return v-evidence.n*(evidence.efficiencyMean-p[3])**2/(2*evidence.efficiencySD**2);
    });
    const max=Math.max(...logs), raw=logs.map(x=>Math.exp(x-max)), total=raw.reduce((a,b)=>a+b,0);
    const posterior=raw.map(x=>x/total), prior=samples.map(()=>1/count), ess=1/posterior.reduce((s,w)=>s+w*w,0);
    const factorSummary=w=>names.map((_,k)=>summary(samples.map(x=>x.f[k]),w));
    const outcomeSummary=w=>[0,1,2,3].map(k=>summary(samples.map(x=>x.p[k]),w));
    const cdf=[]; posterior.reduce((s,w,i)=>{cdf[i]=s+w;return s+w;},0);
    function draw(u) { let lo=0,hi=cdf.length-1; while(lo<hi) {const mid=(lo+hi)>>1; if(cdf[mid]<u)lo=mid+1;else hi=mid;} return samples[lo]; }
    return {samples,prior,posterior,evidence,ess,priorFactors:factorSummary(prior),posteriorFactors:factorSummary(posterior),priorOutcomes:outcomeSummary(prior),posteriorOutcomes:outcomeSummary(posterior),draw};
  }
  return {names,edges,predict,rng,build,summary};
})();
if (typeof module !== 'undefined') module.exports = SurgicalNetworkDemo;
