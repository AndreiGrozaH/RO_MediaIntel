"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const ARTICLES_CACHE_KEY = "ro-mediaintel-cache";

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
}

export default function Home() {
  const [sources, setSources] = useState<string[]>([
    "https://www.digi24.ro/stiri/actualitate/politica",
    "https://spotmedia.ro/stiri/politica",
    "https://www.antena3.ro/politica",
    "https://www.stiripesurse.ro/politica",
    "https://www.dcnews.ro/politica",
    "https://www.digi24.ro/stiri/extern",
    "https://adevarul.ro/politica",
    "https://www.libertatea.ro/politica",
    "https://www.g4media.ro/category/politica",
  ]);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Article[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string[]>([]);
  const [entityFilter, setEntityFilter] = useState<{
    person: boolean;
    org: boolean;
    loc: boolean;
  }>({ person: false, org: false, loc: false });
  const [entityQuery, setEntityQuery] = useState<{
    person: string;
    org: string;
    loc: string;
    text: string;
  }>({ person: "", org: "", loc: "", text: "" });
  const [exactMatch, setExactMatch] = useState<boolean>(true);

  const todayIso = new Date().toISOString().slice(0, 10);
  const weekAgoIso = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);
  const [dateFrom, setDateFrom] = useState<string>(weekAgoIso);
  const [dateTo, setDateTo] = useState<string>(todayIso);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/scrape`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sources,
          date_from: dateFrom || undefined,
          date_to: dateTo || undefined,
        }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      const articles = json.articles || [];
      setData(articles);

      if (typeof window !== "undefined") {
        const payload = {
          articles,
          date_from: dateFrom,
          date_to: dateTo,
          sources,
          cached_at: Date.now(),
        };
        localStorage.setItem(ARTICLES_CACHE_KEY, JSON.stringify(payload));
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const cached = localStorage.getItem(ARTICLES_CACHE_KEY);
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (parsed.articles) setData(parsed.articles);
        if (parsed.date_from) setDateFrom(parsed.date_from);
        if (parsed.date_to) setDateTo(parsed.date_to);
        if (
          parsed.sources &&
          Array.isArray(parsed.sources) &&
          parsed.sources.length
        ) {
          setSources(parsed.sources);
        }
      } catch (e) {
        console.warn("Failed to read cache", e);
      }
    }
  }, []);

  const filteredData = useMemo(() => {
    const normalize = (s: string) =>
      s
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();

    const matchesList = (list: string[], rawNeedle: string) => {
      const needle = normalize(rawNeedle);
      if (!needle) return true;
      return list.some((item) => {
        const norm = normalize(item);
        return exactMatch ? norm === needle : norm.includes(needle);
      });
    };

    const matchesBag = (rawNeedle: string, item: Article) => {
      const needle = normalize(rawNeedle);
      if (!needle) return true;
      const bag = [item.headline, ...item.person, ...item.org, ...item.loc]
        .map((x) => normalize(x))
        .join(" ");
      return bag.includes(needle);
    };

    return data.filter((item) => {
      if (sourceFilter.length && !sourceFilter.includes(item.source))
        return false;
      if (entityFilter.person && item.person.length === 0) return false;
      if (entityFilter.org && item.org.length === 0) return false;
      if (entityFilter.loc && item.loc.length === 0) return false;

      if (!matchesList(item.person, entityQuery.person)) return false;
      if (!matchesList(item.org, entityQuery.org)) return false;
      if (!matchesList(item.loc, entityQuery.loc)) return false;
      if (!matchesBag(entityQuery.text, item)) return false;

      return true;
    });
  }, [data, sourceFilter, entityFilter, entityQuery, exactMatch]);

  const allSources = useMemo(() => {
    const set = new Set<string>();
    data.forEach((d) => set.add(d.source));
    return Array.from(set).sort();
  }, [data]);

  const addSource = (value: string) => {
    if (!value) return;
    try {
      const url = new URL(value.trim());
      if (!sources.includes(url.toString()))
        setSources((prev) => [...prev, url.toString()]);
    } catch {
      setError("Invalid URL");
    }
  };

  const sourceBadges = useMemo(
    () => (
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => (
          <span
            key={s}
            className="group inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-sm text-blue-700 border border-blue-200"
            title={s}
          >
            <span className="truncate max-w-[200px] sm:max-w-xs">{s}</span>
            <button
              onClick={() =>
                setSources((prev) => prev.filter((item) => item !== s))
              }
              className="opacity-0 transition group-hover:opacity-100 text-blue-700 hover:text-blue-900"
              aria-label={`Remove ${s}`}
            >
              ×
            </button>
          </span>
        ))}
      </div>
    ),
    [sources]
  );

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-white">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="mb-10 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-indigo-500">
              RO-MediaIntel
            </p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900 sm:text-4xl">
              Thematic News Aggregation & Intelligence
            </h1>
          </div>
          <div className="flex gap-2">
            <Link
              href="/insights"
              className="inline-flex items-center gap-2 self-start rounded-lg border border-indigo-200 px-4 py-2 text-indigo-700 bg-indigo-50 shadow-sm transition hover:-translate-y-[1px] hover:bg-indigo-100"
            >
              Insights
            </Link>
            <button
              onClick={fetchData}
              disabled={loading}
              className="inline-flex items-center gap-2 self-start rounded-lg bg-indigo-600 px-4 py-2 text-white shadow-md transition hover:-translate-y-[1px] hover:bg-indigo-700 disabled:translate-y-0 disabled:opacity-60"
            >
              {loading ? "Scanning..." : "Launch Scraper"}
            </button>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-3">
          <section className="lg:col-span-2 rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-xl font-semibold text-slate-900">Sources</h2>
              <span className="text-xs font-medium text-slate-500">
                {sources.length} active
              </span>
            </div>
            <div className="mt-4 space-y-2">{sourceBadges}</div>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              <input
                type="url"
                placeholder="https://example.com"
                className="flex-1 rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    addSource((e.target as HTMLInputElement).value);
                    (e.target as HTMLInputElement).value = "";
                  }
                }}
              />
              <button
                onClick={() => {
                  const input =
                    document.querySelector<HTMLInputElement>("input[type=url]");
                  if (input) {
                    addSource(input.value);
                    input.value = "";
                  }
                }}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
              >
                Add source
              </button>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur space-y-4">
            <div>
              <h2 className="text-xl font-semibold text-slate-900">
                Date interval
              </h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-slate-700">
                    From
                  </label>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700">
                    To
                  </label>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                  />
                </div>
              </div>
            </div>

            <div className="border-t border-slate-200 pt-4">
              <h3 className="text-sm font-semibold text-slate-900">Filters</h3>
              <p className="text-xs text-slate-500">
                Narrow results by source and entity types/text.
              </p>
              {(sourceFilter.length > 0 ||
                Object.values(entityFilter).some(Boolean) ||
                entityQuery.person ||
                entityQuery.org ||
                entityQuery.loc ||
                entityQuery.text) && (
                <div className="mt-2">
                  <button
                    onClick={() => {
                      setSourceFilter([]);
                      setEntityFilter({
                        person: false,
                        org: false,
                        loc: false,
                      });
                      setEntityQuery({
                        person: "",
                        org: "",
                        loc: "",
                        text: "",
                      });
                    }}
                    className="text-xs font-semibold text-indigo-700 hover:text-indigo-900"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
              <div className="mt-3 space-y-3">
                <div>
                  <span className="text-xs font-medium text-slate-600">
                    Sources
                  </span>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {allSources.length === 0 && (
                      <span className="text-xs text-slate-400">(none yet)</span>
                    )}
                    {allSources.map((s) => {
                      const active = sourceFilter.includes(s);
                      return (
                        <button
                          key={s}
                          onClick={() =>
                            setSourceFilter((prev) =>
                              prev.includes(s)
                                ? prev.filter((x) => x !== s)
                                : [...prev, s]
                            )
                          }
                          className={`rounded-full border px-3 py-1 text-xs transition ${
                            active
                              ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                              : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                          }`}
                          title={s}
                        >
                          {s.replace(/^https?:\/\//, "").replace(/\/$/, "")}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  {["person", "org", "loc"].map((key) => {
                    const k = key as keyof typeof entityFilter;
                    const active = entityFilter[k];
                    const labels: Record<typeof k, string> = {
                      person: "Has people",
                      org: "Has orgs",
                      loc: "Has locations",
                    } as const;
                    return (
                      <button
                        key={key}
                        onClick={() =>
                          setEntityFilter((prev) => ({
                            ...prev,
                            [k]: !prev[k],
                          }))
                        }
                        className={`rounded-full border px-3 py-1 text-xs transition ${
                          active
                            ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                            : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                        }`}
                      >
                        {labels[k]}
                      </button>
                    );
                  })}
                </div>

                {(sourceFilter.length > 0 ||
                  Object.values(entityFilter).some(Boolean)) && (
                  <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                    <span>
                      Active: {sourceFilter.length} source filter(s),{" "}
                      {Object.values(entityFilter).filter(Boolean).length}{" "}
                      entity toggle(s)
                    </span>
                  </div>
                )}

                <div className="grid gap-2 sm:grid-cols-2">
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Person contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.person}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          person: e.target.value,
                        }))
                      }
                      placeholder="e.g. Iohannis"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Org contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.org}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          org: e.target.value,
                        }))
                      }
                      placeholder="e.g. PSD"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Location contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.loc}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          loc: e.target.value,
                        }))
                      }
                      placeholder="e.g. Bucuresti"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Headline/entities text search
                    </label>
                    <input
                      type="text"
                      value={entityQuery.text}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          text: e.target.value,
                        }))
                      }
                      placeholder="keyword"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-700">
                    <input
                      type="checkbox"
                      checked={exactMatch}
                      onChange={(e) => setExactMatch(e.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-200"
                    />
                    <span>Exact match for person/org/location</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-sm">
            {error}
          </div>
        )}

        <section className="mt-8 space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-xl font-semibold text-slate-900">
              Articles ({filteredData.length})
            </h2>
          </div>
          {loading && <p className="text-slate-500">Loading…</p>}
          {!loading && filteredData.length === 0 && (
            <p className="text-slate-500">No data yet.</p>
          )}
          <div className="grid gap-4">
            {filteredData.map((item) => (
              <article
                key={item.link}
                className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
              >
                <div className="flex flex-col gap-1 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
                  <span className="font-semibold text-slate-700">
                    {item.source}
                  </span>
                  <span>{new Date(item.date).toLocaleString()}</span>
                </div>
                <a
                  href={item.link}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 block text-lg font-semibold text-indigo-700 hover:text-indigo-900"
                >
                  {item.headline}
                </a>
                <div className="mt-3 flex flex-wrap gap-2 text-sm">
                  {item.person.map((p) => (
                    <span
                      key={p}
                      className="rounded-full bg-blue-50 px-2 py-1 text-blue-700"
                    >
                      👤 {p}
                    </span>
                  ))}
                  {item.org.map((o) => (
                    <span
                      key={o}
                      className="rounded-full bg-amber-50 px-2 py-1 text-amber-700"
                    >
                      🏢 {o}
                    </span>
                  ))}
                  {item.loc.map((l) => (
                    <span
                      key={l}
                      className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700"
                    >
                      📍 {l}
                    </span>
                  ))}
                  {item.values.map((v) => (
                    <span
                      key={v}
                      className="rounded-full bg-rose-50 px-2 py-1 text-rose-700"
                    >
                      💰 {v}
                    </span>
                  ))}
                  {item.context.map((c) => (
                    <span
                      key={c}
                      className="rounded-full bg-slate-100 px-2 py-1 text-slate-700"
                    >
                      🧭 {c}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
