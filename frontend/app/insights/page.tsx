"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from "chart.js";
import { Bar, Doughnut, Line } from "react-chartjs-2";

// --- MAP IMPORTS ---
import { ComposableMap, Geographies, Geography } from "react-simple-maps";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, ArcElement);
ChartJS.register(PointElement, LineElement);

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const ARTICLES_CACHE_KEY = "ro-mediaintel-cache-v2";

// --- ROBUST MAP URL (Highcharts CDN) ---
const RO_TOPO_URL = "https://code.highcharts.com/mapdata/countries/ro/ro-all.topo.json";

// --- TYPES ---
type GraphNode = { 
  id: string; 
  group: "source" | "person" | "org"; 
  val: number; 
  x?: number; 
  y?: number; 
  fx?: number | null; 
  fy?: number | null;
};

type GraphLink = { 
  source: string | GraphNode; 
  target: string | GraphNode; 
  value: number; 
  type: "source-entity" | "entity-entity" | "conflict" | "alliance"; 
  label?: string; 
};

type ForceGraphMethods = {
  zoomToFit: (ms?: number, padding?: number) => void;
  d3Force: (name: string) => any;
  d3ReheatSimulation: () => void;
  centerAt?: (x: number, y: number, ms?: number) => void;
};

type ForceGraphProps = {
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
  nodeRelSize?: number;
  nodeCanvasObjectMode?: () => string;
  nodeCanvasObject?: (node: GraphNode, ctx: CanvasRenderingContext2D) => void;
  linkColor?: string | ((link: GraphLink) => string);
  linkWidth?: number | ((link: GraphLink) => number);
  linkDirectionalParticles?: number | ((link: GraphLink) => number);
  linkDirectionalArrowLength?: number | ((link: GraphLink) => number);
  linkDirectionalParticleWidth?: number;
  linkDirectionalParticleSpeed?: number;
  linkDirectionalArrowRelPos?: number;
  linkDirectionalParticleColor?: (link: GraphLink) => string;
  linkLabel?: (link: GraphLink) => string;
  backgroundColor?: string;
  ref?: React.Ref<ForceGraphMethods>;
  cooldownTime?: number;
  d3AlphaDecay?: number;    
  d3VelocityDecay?: number; 
  onNodeHover?: (node: GraphNode | null, prevNode?: GraphNode | null) => void;
  onNodeDrag?: (node: GraphNode, translate: { x: number; y: number }) => void;
  onNodeDragEnd?: (node: GraphNode, translate: { x: number; y: number }) => void;
};

const ForceGraph2D = dynamic<ForceGraphProps>(() => import("react-force-graph-2d"), { ssr: false });

interface Article {
  source: string;
  headline: string;
  link: string;
  date: string;
  person: string[];
  org: string[];
  loc: string[];
  values: string[];
  context: string[];
  sentiment: Record<string, number>;
  relationships: { source: string; target: string; type: string; verb: string }[];
  county?: string; 
}

const GEO_NAME_TO_CODE: Record<string, string> = {
  "București": "B", "Bucuresti": "B", "Bucharest": "B",
  "Alba": "AB", 
  "Arad": "AR", 
  "Argeș": "AG", "Arges": "AG",
  "Bacău": "BC", "Bacau": "BC",
  "Bihor": "BH",
  "Bistrița-Năsăud": "BN", "Bistrita-Nasaud": "BN", 
  "Botoșani": "BT", "Botosani": "BT",
  "Brașov": "BV", "Brasov": "BV",
  "Brăila": "BR", "Braila": "BR",
  "Buzău": "BZ", "Buzau": "BZ",
  "Caraș-Severin": "CS", "Caras-Severin": "CS",
  "Călărași": "CL", "Calarasi": "CL",
  "Cluj": "CJ", 
  "Constanța": "CT", "Constanta": "CT",
  "Covasna": "CV",
  "Dâmbovița": "DB", "Dambovita": "DB",
  "Dolj": "DJ", 
  "Galați": "GL", "Galati": "GL",
  "Giurgiu": "GR", 
  "Gorj": "GJ", 
  "Harghita": "HR",
  "Hunedoara": "HD", 
  "Ialomița": "IL", "Ialomita": "IL",
  "Iași": "IS", "Iasi": "IS",
  "Ilfov": "IF", 
  "Maramureș": "MM", "Maramures": "MM",
  "Mehedinți": "MH", "Mehedinti": "MH",
  "Mureș": "MS", "Mures": "MS",
  "Neamț": "NT", "Neamt": "NT",
  "Olt": "OT", 
  "Prahova": "PH", 
  "Satu Mare": "SM",
  "Sălaj": "SJ", "Salaj": "SJ",
  "Sibiu": "SB", 
  "Suceava": "SV", 
  "Teleorman": "TR", 
  "Timiș": "TM", "Timis": "TM",
  "Tulcea": "TL",
  "Vaslui": "VS", 
  "Vâlcea": "VL", "Valcea": "VL",
  "Vrancea": "VN"
};

// Helper to handle "Timis" vs "Timiș"
const normalizeName = (str: string) => str.normalize("NFD").replace(/[\u0300-\u036f]/g, "");

export default function InsightsPage() {
  const fgRef = useRef<ForceGraphMethods | null>(null);     
  const relFgRef = useRef<ForceGraphMethods | null>(null);  
  
  const [data, setData] = useState<Article[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);
  
  // MAP STATE
  const [geoData, setGeoData] = useState<any>(null);
  const [selectedCounty, setSelectedCounty] = useState<string | null>(null);
  const [tooltipContent, setTooltipContent] = useState("");
  
  const todayIso = new Date().toISOString().slice(0, 10);
  const weekAgoIso = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const [dateFrom, setDateFrom] = useState<string>(weekAgoIso);
  const [dateTo, setDateTo] = useState<string>(todayIso);

  // --- ACTIONS ---
  const clearCache = () => {
    if (typeof window !== "undefined") {
      localStorage.removeItem(ARTICLES_CACHE_KEY);
      setData([]);
      window.location.reload(); // Force reload to fetch fresh
    }
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/scrape`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date_from: dateFrom || undefined, date_to: dateTo || undefined }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      const articles = json.articles || [];
      setData(articles);

      if (typeof window !== "undefined") {
        const cached = {
          articles,
          date_from: dateFrom,
          date_to: dateTo,
          cached_at: Date.now(),
        };
        localStorage.setItem(ARTICLES_CACHE_KEY, JSON.stringify(cached));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  // --- FETCH MAP DATA ---
  useEffect(() => {
    fetch(RO_TOPO_URL)
      .then(res => {
        if (!res.ok) throw new Error("Failed to load map");
        return res.json();
      })
      .then(data => setGeoData(data))
      .catch(err => {
        console.error("Map error:", err);
        setError("Could not load map data.");
      });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const cachedRaw = localStorage.getItem(ARTICLES_CACHE_KEY);
    if (cachedRaw) {
      try {
        const cached = JSON.parse(cachedRaw);
        if (cached.date_from) setDateFrom(cached.date_from);
        if (cached.date_to) setDateTo(cached.date_to);
        if (cached.articles) setData(cached.articles);
        return; 
      } catch (e) { console.warn("Failed to read cache", e); }
    }
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- MAP DATA LOGIC ---
  const countyData = useMemo(() => {
    const stats: Record<string, { count: number; sentimentSum: number; sentimentAvg: number }> = {};
    
    data.forEach(article => {
        // Ensure backend actually returned 'county'
        if (article.county) {
            const code = article.county; // Backend sends "CJ", "B", etc.
            if (!stats[code]) {
                stats[code] = { count: 0, sentimentSum: 0, sentimentAvg: 0 };
            }
            stats[code].count += 1;
            
            const sentimentValues = Object.values(article.sentiment);
            if (sentimentValues.length > 0) {
                const avgArtSentiment = sentimentValues.reduce((a, b) => a + b, 0) / sentimentValues.length;
                stats[code].sentimentSum += avgArtSentiment;
            }
        }
    });

    Object.keys(stats).forEach(code => {
        if (stats[code].count > 0) {
            stats[code].sentimentAvg = stats[code].sentimentSum / stats[code].count;
        }
    });

    return stats;
  }, [data]);

  // --- STATS LOGIC ---
  const perSource = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => { counts[a.source] = (counts[a.source] || 0) + 1; });
    return counts;
  }, [data]);

  const entityCounts = useMemo(() => {
    const buckets: Record<string, number> = { person: 0, org: 0, loc: 0, values: 0, context: 0 };
    data.forEach((a) => {
      buckets.person += a.person.length;
      buckets.org += a.org.length;
      buckets.loc += a.loc.length;
      buckets.values += a.values.length;
      buckets.context += a.context.length;
    });
    return buckets;
  }, [data]);

  const perDay = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => {
      const day = new Date(a.date).toISOString().slice(0, 10);
      counts[day] = (counts[day] || 0) + 1;
    });
    return counts;
  }, [data]);

  const topPeople = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => a.person.forEach((p) => (counts[p] = (counts[p] || 0) + 1)));
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  }, [data]);

  const topOrgs = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => a.org.forEach((o) => (counts[o] = (counts[o] || 0) + 1)));
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  }, [data]);

  // --- CHART DATA OBJECTS ---
  const sourceChart = {
    labels: Object.keys(perSource),
    datasets: [{ label: "Articles", data: Object.values(perSource), backgroundColor: "rgba(79, 70, 229, 0.6)", borderColor: "rgba(79, 70, 229, 1)", borderWidth: 1 }],
  };
  const entityChart = {
    labels: ["Person", "Org", "Location", "Values", "Context"],
    datasets: [{ label: "Entities", data: [entityCounts.person, entityCounts.org, entityCounts.loc, entityCounts.values, entityCounts.context], backgroundColor: ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#94a3b8"] }],
  };
  const timelineChart = {
    labels: Object.keys(perDay).sort(),
    datasets: [{ label: "Articles per day", data: Object.keys(perDay).sort().map((d) => perDay[d]), fill: false, borderColor: "rgba(79, 70, 229, 1)", backgroundColor: "rgba(79, 70, 229, 0.2)", tension: 0.25 }],
  };
  const topPeopleChart = {
    labels: topPeople.map(([name]) => name),
    datasets: [{ label: "Mentions", data: topPeople.map(([, count]) => count), backgroundColor: "rgba(99, 102, 241, 0.7)", borderColor: "rgba(99, 102, 241, 1)", borderWidth: 1 }],
  };
  const topOrgsChart = {
    labels: topOrgs.map(([name]) => name),
    datasets: [{ label: "Mentions", data: topOrgs.map(([, count]) => count), backgroundColor: "rgba(16, 185, 129, 0.7)", borderColor: "rgba(16, 185, 129, 1)", borderWidth: 1 }],
  };

  const sentimentChartData = useMemo(() => {
    const topPersonNames = topPeople.map(([name]) => name).slice(0, 10);
    const scores = topPersonNames.map((name) => {
      let total = 0, count = 0;
      data.forEach((article) => {
        if (article.sentiment && typeof article.sentiment[name] === 'number') {
          total += article.sentiment[name]; count++;
        }
      });
      return count > 0 ? (total / count) : 0;
    });
    return {
      labels: topPersonNames,
      datasets: [{ label: 'Sentiment Mediu', data: scores, backgroundColor: scores.map(s => s >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)'), borderColor: scores.map(s => s >= 0 ? 'rgb(22, 163, 74)' : 'rgb(220, 38, 38)'), borderWidth: 1, borderRadius: 4 }],
    };
  }, [data, topPeople]);

  // --- 2. MAIN KNOWLEDGE GRAPH DATA ---
  const knowledgeGraphData = useMemo(() => {
    if (!data.length) return { nodes: [], links: [] };
    
    const sourceCounts = new Map<string, number>();
    const personCounts = new Map<string, number>();
    const orgCounts = new Map<string, number>();

    data.forEach((a) => {
      sourceCounts.set(a.source, (sourceCounts.get(a.source) || 0) + 1);
      a.person.forEach((p) => personCounts.set(p, (personCounts.get(p) || 0) + 1));
      a.org.forEach((o) => orgCounts.set(o, (orgCounts.get(o) || 0) + 1));
    });

    const topPersons = Array.from(personCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([id]) => id);
    const topOrganizations = Array.from(orgCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([id]) => id);
    const selectedEntities = new Set<string>([...topPersons, ...topOrganizations]);

    const sourceEntityLinks: Record<string, number> = {};
    const cooccurrence: Record<string, number> = {};

    data.forEach((a) => {
      const entities = [...a.person.filter((p) => selectedEntities.has(p)), ...a.org.filter((o) => selectedEntities.has(o))];
      entities.forEach((ent) => {
        const key = `${a.source}::${ent}`;
        sourceEntityLinks[key] = (sourceEntityLinks[key] || 0) + 1;
      });
      const uniqueEntities = Array.from(new Set(entities));
      for (let i = 0; i < uniqueEntities.length; i++) {
        for (let j = i + 1; j < uniqueEntities.length; j++) {
          const [aEnt, bEnt] = [uniqueEntities[i], uniqueEntities[j]].sort();
          const key = `${aEnt}||${bEnt}`;
          cooccurrence[key] = (cooccurrence[key] || 0) + 1;
        }
      }
    });

    const nodes: GraphNode[] = [
      ...Array.from(sourceCounts.entries()).map(([id, count]) => ({ id, group: "source" as const, val: count })),
      ...topPersons.map((id) => ({ id, group: "person" as const, val: personCounts.get(id) || 1 })),
      ...topOrganizations.map((id) => ({ id, group: "org" as const, val: orgCounts.get(id) || 1 })),
    ];
    
    nodes.forEach(node => {
        node.x = Math.random() * 800 - 400; 
        node.y = Math.random() * 600 - 300; 
    });

    const links: GraphLink[] = [
      ...Object.entries(sourceEntityLinks).map(([key, value]) => {
        const [source, target] = key.split("::");
        return { source, target, value, type: "source-entity" } as GraphLink;
      }),
      ...Object.entries(cooccurrence).filter(([, value]) => value >= 2).map(([key, value]) => {
        const [aEnt, bEnt] = key.split("||");
        return { source: aEnt, target: bEnt, value, type: "entity-entity" } as GraphLink;
      }),
    ];

    return { nodes, links };
  }, [data]);

  // --- 3. NEW: RELATIONSHIP GRAPH DATA ---
  const relationshipGraphData = useMemo(() => {
    if (!data.length) return { nodes: [], links: [] };

    const nodesMap = new Map<string, GraphNode>();
    const links: GraphLink[] = [];

    data.forEach(article => {
        if (article.relationships) {
            article.relationships.forEach(rel => {
                if (!nodesMap.has(rel.source)) {
                    const isPerson = article.person.includes(rel.source);
                    nodesMap.set(rel.source, { id: rel.source, group: isPerson ? "person" : "org", val: 1 });
                } else {
                    nodesMap.get(rel.source)!.val += 1;
                }

                if (!nodesMap.has(rel.target)) {
                    const isPerson = article.person.includes(rel.target);
                    nodesMap.set(rel.target, { id: rel.target, group: isPerson ? "person" : "org", val: 1 });
                } else {
                    nodesMap.get(rel.target)!.val += 1;
                }

                links.push({
                    source: rel.source,
                    target: rel.target,
                    value: 5,
                    type: rel.type as "conflict" | "alliance",
                    label: rel.verb
                });
            });
        }
    });

    const nodes = Array.from(nodesMap.values());
    nodes.forEach(node => {
        node.x = Math.random() * 800 - 400; 
        node.y = Math.random() * 600 - 300; 
    });

    return { nodes, links };
  }, [data]);


  const setupForceGraph = (fg: any) => {
    if (!fg) return;
    const chargeForce = fg.d3Force("charge");
    const collideForce = fg.d3Force("collide");
    
    if (chargeForce) chargeForce.strength(-200); 
    if (collideForce) {
        collideForce.radius((node: GraphNode) => Math.sqrt(node.val) * 6 + 15);
        collideForce.iterations(3);
    }
    
    fg.d3ReheatSimulation();
    setTimeout(() => fg.d3ReheatSimulation(), 500);
  };

  useEffect(() => {
    setupForceGraph(fgRef.current);
    setupForceGraph(relFgRef.current);
  }, [knowledgeGraphData, relationshipGraphData]);


  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-white">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8 space-y-8">
        
        {/* HEADER */}
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-indigo-500">RO-MediaIntel</p>
            <h1 className="mt-1 text-3xl font-bold text-slate-900">Insights</h1>
            <p className="text-slate-600">Political dynamics & Source Intelligence</p>
          </div>
          <div className="flex gap-2">
            <Link href="/" className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-slate-700 shadow-sm hover:bg-slate-50">Back</Link>
            
            {/* FORCE REFRESH BUTTON */}
            <button onClick={clearCache} className="inline-flex items-center gap-2 rounded-lg border border-red-200 px-4 py-2 text-red-600 shadow-sm hover:bg-red-50 text-xs">
              ⚠️ Force Clean Refresh
            </button>

            <button onClick={fetchData} disabled={loading} className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-white shadow-md hover:bg-indigo-700 disabled:opacity-60">
              {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </header>

        {/* DATE & ERROR */}
        <section className="rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-slate-700">From</label>
              <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">To</label>
              <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2" />
            </div>
          </div>
        </section>
        {error && <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-sm">{error}</div>}

        {/* --- 1. POLITICAL DYNAMICS --- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Political Dynamics Map</h3>
              <p className="text-sm text-slate-500">
                AI-detected relationships: 
                <span className="text-red-600 font-bold ml-1"> Conflict (→)</span> vs 
                <span className="text-green-600 font-bold ml-1"> Alliance (→)</span>
              </p>
            </div>
          </div>
          <div className="mt-2 h-[500px] rounded-xl border border-slate-100 bg-slate-50 relative overflow-hidden">
             {relationshipGraphData.nodes.length ? (
                <ForceGraph2D
                    ref={relFgRef}
                    graphData={relationshipGraphData}
                    nodeRelSize={10} 
                    backgroundColor="rgba(248, 250, 252, 0.95)"
                    cooldownTime={4000}
                    d3AlphaDecay={0.01}
                    
                    linkColor={(link) => link.type === "conflict" ? "#ef4444" : "#22c55e"}
                    linkWidth={4} 
                    linkDirectionalArrowLength={6} 
                    linkDirectionalArrowRelPos={1}
                    linkDirectionalParticles={(link) => (link.type === "conflict" || link.type === "alliance") ? 2 : 0}
                    linkDirectionalParticleSpeed={0.005}
                    linkDirectionalParticleColor={(link) => link.type === "conflict" ? "#ef4444" : "#22c55e"}

                    linkLabel={(link) => {
                         const s = typeof link.source === 'object' ? (link.source as any).id : link.source;
                         const t = typeof link.target === 'object' ? (link.target as any).id : link.target;
                         return link.label ? `${s} [${link.label}] ${t}` : "";
                    }}
                    
                    onNodeDragEnd={(node) => {
                        node.fx = node.x;
                        node.fy = node.y;
                    }}

                    nodeCanvasObject={(node, ctx) => {
                        const radius = Math.min(45, Math.sqrt(node.val) * 6 + 6);
                        
                        ctx.beginPath();
                        ctx.fillStyle = node.group === "person" ? "#0ea5e9" : "#f59e0b";
                        ctx.strokeStyle = "#fff";
                        ctx.lineWidth = 2;
                        ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI, false);
                        ctx.fill(); ctx.stroke();

                        ctx.font = 'bold 12px Inter';
                        ctx.fillStyle = "rgba(255,255,255,0.9)";
                        const label = node.id as string;
                        const w = ctx.measureText(label).width;
                        ctx.fillRect((node.x!) + radius, (node.y!) - 6, w + 4, 14);
                        ctx.fillStyle = "#000";
                        ctx.fillText(label, (node.x!) + radius + 2, (node.y!) + 5);
                    }}
                />
             ) : (
                <div className="flex h-full items-center justify-center text-slate-400">
                    No relationships detected yet. Try scanning more articles.
                </div>
             )}
          </div>
        </section>

        {/* --- STAT CHARTS --- */}
        <section className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-900">Articles per source</h3>
                    <span className="text-xs text-slate-500">Total: {data.length}</span>
                </div>
                <div className="mt-4">
                    {Object.keys(perSource).length ? <Bar data={sourceChart} /> : <p className="text-sm text-slate-500">No data.</p>}
                </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-900">Entity distribution</h3>
                    <span className="text-xs text-slate-500">Across all articles</span>
                </div>
                <div className="mt-4 flex items-center justify-center">
                    {data.length ? <Doughnut data={entityChart} /> : <p className="text-sm text-slate-500">No data.</p>}
                </div>
            </div>
        </section>

        {/* --- TIMELINE & TOP PEOPLE --- */}
        <section className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-900">Timeline</h3>
                    <span className="text-xs text-slate-500">Daily volume</span>
                </div>
                <div className="mt-4">
                    {timelineChart.labels.length ? <Line data={timelineChart} /> : <p className="text-sm text-slate-500">No data.</p>}
                </div>
            </div>

            <div className="grid gap-6 md:grid-cols-2 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div>
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-slate-900">Top people</h3>
                        <span className="text-xs text-slate-500">Mentions</span>
                    </div>
                    <div className="mt-4">
                        {topPeople.length ? <Bar data={topPeopleChart} options={{ indexAxis: "y" }} /> : <p className="text-sm text-slate-500">No data.</p>}
                    </div>
                </div>
                <div>
                    <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold text-slate-900">Top organisations</h3>
                        <span className="text-xs text-slate-500">Mentions</span>
                    </div>
                    <div className="mt-4">
                        {topOrgs.length ? <Bar data={topOrgsChart} options={{ indexAxis: "y" }} /> : <p className="text-sm text-slate-500">No data.</p>}
                    </div>
                </div>
            </div>
        </section>

        {/* --- SENTIMENT BAROMETER --- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
             <h3 className="text-lg font-semibold mb-4">Sentiment Barometer</h3>
             <div className="h-[300px]">
                {topPeople.length ? <Bar data={sentimentChartData} options={{ indexAxis: 'y', maintainAspectRatio: false, plugins: { legend: { display: false } } }} /> : <p className="text-slate-400">No data.</p>}
             </div>
        </section>

        {/* --- 2. OLD GRAPH: KNOWLEDGE GRAPH --- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">General Knowledge Graph</h3>
              <p className="text-sm text-slate-500">Connections between Sources (Blue) and Entities</p>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-600">
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-indigo-600" /> Source</span>
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-sky-500" /> Person</span>
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-500" /> Org</span>
            </div>
          </div>
          <div className="mt-2 min-h-[600px] rounded-xl border border-slate-100 bg-gradient-to-br from-white via-slate-50 to-slate-100">
            {knowledgeGraphData.nodes.length ? (
              <ForceGraph2D
                ref={fgRef}
                graphData={knowledgeGraphData}
                nodeRelSize={7}
                backgroundColor="rgba(248, 250, 252, 0.95)"
                cooldownTime={4000}
                d3AlphaDecay={0.01}
                linkColor={() => "rgba(79, 70, 229, 0.2)"} 
                linkWidth={1}
                linkDirectionalParticles={0} 
                
                onNodeDragEnd={(node) => {
                    node.fx = node.x;
                    node.fy = node.y;
                }}

                nodeCanvasObject={(node, ctx) => {
                    const colors = { source: "#4f46e5", person: "#0ea5e9", org: "#f59e0b" };
                    const radius = Math.min(30, Math.sqrt(node.val) * 4 + 2);
                    ctx.beginPath();
                    ctx.fillStyle = colors[node.group];
                    ctx.strokeStyle = "#fff";
                    ctx.lineWidth = 1.5;
                    ctx.arc(node.x!, node.y!, radius, 0, 2 * Math.PI, false);
                    ctx.fill(); ctx.stroke();

                    if (node.group === "source" || node.val > 10 || hoverNode?.id === node.id) {
                        ctx.font = '12px Inter';
                        ctx.fillStyle = "#000";
                        ctx.fillText(node.id as string, (node.x!) + radius + 4, (node.y!) + 4);
                    }
                }}
                onNodeHover={(node) => setHoverNode((node as GraphNode) || null)}
              />
            ) : <p className="p-4 text-slate-500">No data.</p>}
          </div>
        </section>

        {/* --- 4. NEW: GEO-SPATIAL RISK MAP --- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="mb-6 flex flex-col md:flex-row md:items-center md:justify-between">
               <div>
                   <h3 className="text-lg font-semibold text-slate-900">Geo-Political Risk Map</h3>
                   <p className="text-sm text-slate-500">
                       Regional sentiment analysis. Click a region to filter intelligence.
                   </p>
               </div>
               {selectedCounty && (
                   <button 
                     onClick={() => setSelectedCounty(null)}
                     className="mt-2 md:mt-0 text-sm font-semibold text-indigo-600 hover:text-indigo-800"
                   >
                     Clear Selection ({selectedCounty})
                   </button>
               )}
            </div>

            <div className="grid lg:grid-cols-3 gap-8">
                {/* LEFT: THE MAP (Fixed Height: h-[500px]) */}
                <div className="lg:col-span-2 h-[500px] border border-slate-100 rounded-xl bg-slate-50 relative overflow-hidden">
                    {/* SHOW LOADING STATE */}
                    {!geoData ? (
                        <div className="flex h-full w-full items-center justify-center text-slate-400 font-medium animate-pulse">
                            Loading Map Data...
                        </div>
                    ) : (
                        <ComposableMap 
                           projection="geoMercator" 
                           // UPDATED SCALE to 4500 to fit Romania perfectly
                           projectionConfig={{ center: [25, 46], scale: 4500 }} 
                           width={800} height={600}
                           style={{ width: "100%", height: "100%" }}
                        >
                            <Geographies geography={geoData}>
                                {({ geographies }) =>
                                    geographies.map((geo) => {
                                        const countyName = geo.properties.name; 
                                        const normName = normalizeName(countyName);
                                        const countyCode = GEO_NAME_TO_CODE[countyName] || GEO_NAME_TO_CODE[normName]; 
                                        
                                        // DEBUG LOG: See what the map thinks vs what we have
                                        // console.log(`Map: ${countyName} -> Code: ${countyCode}`);

                                        const stats = countyCode ? countyData[countyCode] : null;

                                        let fillColor = "#e2e8f0"; 
                                        if (stats) {
                                            if (stats.sentimentAvg > 0.05) fillColor = "#86efac"; 
                                            else if (stats.sentimentAvg < -0.05) fillColor = "#fca5a5"; 
                                            else fillColor = "#fcd34d"; 
                                        }
                                        
                                        const isSelected = selectedCounty === countyCode;
                                        if (isSelected) fillColor = "#4f46e5"; 

                                        return (
                                            <Geography
                                                key={geo.rsmKey}
                                                geography={geo}
                                                onClick={() => {
                                                    if(countyCode) setSelectedCounty(isSelected ? null : countyCode);
                                                }}
                                                onMouseEnter={() => {
                                                    const s = stats ? `Sentiment: ${stats.sentimentAvg.toFixed(2)} | Articles: ${stats.count}` : "No data";
                                                    setTooltipContent(`${countyName}: ${s}`);
                                                }}
                                                onMouseLeave={() => setTooltipContent("")}
                                                style={{
                                                    default: { fill: fillColor, stroke: "#fff", strokeWidth: 0.75, outline: "none" },
                                                    hover: { fill: "#cbd5e1", stroke: "#fff", strokeWidth: 1, outline: "none" },
                                                    pressed: { fill: "#4338ca", stroke: "#fff", strokeWidth: 1, outline: "none" },
                                                }}
                                            />
                                        );
                                    })
                                }
                            </Geographies>
                        </ComposableMap>
                    )}
                    
                    {tooltipContent && (
                        <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur px-3 py-2 rounded-lg shadow-lg border border-slate-200 text-xs font-medium pointer-events-none">
                            {tooltipContent}
                        </div>
                    )}
                </div>

                {/* RIGHT: REGIONAL INTEL PANEL */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 h-[500px] overflow-y-auto">
                    <h4 className="font-semibold text-slate-800 mb-4 sticky top-0 bg-white pb-2 border-b border-slate-100">
                        {selectedCounty ? `Intel: ${Object.keys(GEO_NAME_TO_CODE).find(key => GEO_NAME_TO_CODE[key] === selectedCounty)}` : "Select a region"}
                    </h4>
                    
                    {!selectedCounty ? (
                        <div className="flex flex-col items-center justify-center h-40 text-slate-400 text-sm text-center">
                            <span>Click on the map to see local news and key political figures.</span>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {data.filter(a => a.county === selectedCounty).length === 0 ? (
                                <p className="text-sm text-slate-500">No recent data for this county.</p>
                            ) : (
                                data.filter(a => a.county === selectedCounty).map((art, idx) => (
                                    <div key={idx} className="group">
                                        <a href={art.link} target="_blank" className="text-sm font-medium text-indigo-700 hover:underline block leading-tight">
                                            {art.headline}
                                        </a>
                                        <div className="flex flex-wrap gap-1 mt-1">
                                            {art.person.slice(0,3).map(p => <span key={p} className="text-[10px] bg-blue-50 text-blue-600 px-1.5 rounded">{p}</span>)}
                                            {art.sentiment && Object.values(art.sentiment).some(v => v < 0) && (
                                                <span className="text-[10px] bg-red-50 text-red-600 px-1.5 rounded">Negative</span>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    )}
                </div>
            </div>
        </section>

      </div>
    </main>
  );
}