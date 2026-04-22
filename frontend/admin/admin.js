(function () {
    const { createApp, ref, reactive, computed, onMounted, watch, defineComponent, h } = Vue;

    const TOKEN_KEY = "aicis_admin_token";

    // ---------- RagTraceViewer 子组件 ----------
    const RagTraceViewer = defineComponent({
        name: "RagTraceViewer",
        props: { trace: { type: Object, required: true } },
        setup(props) {
            const t = props.trace;
            const section = (title, children) =>
                h("div", { class: "trace-section" }, [h("h4", title), children]);
            const kv = (pairs) =>
                h(
                    "div",
                    { class: "trace-kv" },
                    pairs
                        .filter(([, v]) => v !== undefined && v !== null && v !== "")
                        .map(([k, v]) =>
                            h("span", {}, [
                                h("span", { class: "k" }, k + "："),
                                h("span", { class: "v" }, String(v)),
                            ])
                        )
                );
            const chunkList = (items) =>
                h(
                    "div",
                    { class: "chunk-list" },
                    items.map((c, idx) =>
                        h("div", { class: "chunk-item" }, [
                            h("div", { class: "chunk-head" }, [
                                h("strong", {}, `#${idx + 1}`),
                                c.filename ? h("span", {}, c.filename) : null,
                                c.page_number != null ? h("span", {}, `p.${c.page_number}`) : null,
                                c.score != null ? h("span", {}, `score=${Number(c.score).toFixed(4)}`) : null,
                                c.rerank_score != null ? h("span", {}, `rerank=${Number(c.rerank_score).toFixed(4)}`) : null,
                                c.rrf_rank != null ? h("span", {}, `rrf=${c.rrf_rank}`) : null,
                            ]),
                            h("div", { class: "chunk-text" }, c.text || "—"),
                        ])
                    )
                );
            return () => {
                const blocks = [];
                blocks.push(
                    section(
                        "基本",
                        kv([
                            ["tool_used", t.tool_used],
                            ["tool_name", t.tool_name],
                            ["retrieval_mode", t.retrieval_mode],
                            ["retrieval_stage", t.retrieval_stage],
                            ["candidate_k", t.candidate_k],
                        ])
                    )
                );
                const queryRows = [
                    ["query", t.query],
                    ["expansion_type", t.expansion_type],
                    ["expanded_query", t.expanded_query],
                    ["step_back_question", t.step_back_question],
                    ["step_back_answer", t.step_back_answer],
                    ["hypothetical_doc", t.hypothetical_doc],
                    ["rewrite_needed", t.rewrite_needed],
                    ["rewrite_strategy", t.rewrite_strategy],
                    ["rewrite_query", t.rewrite_query],
                    ["grade_score", t.grade_score],
                    ["grade_route", t.grade_route],
                ];
                blocks.push(section("改写 / 评分", kv(queryRows)));

                blocks.push(
                    section(
                        "重排",
                        kv([
                            ["rerank_enabled", t.rerank_enabled],
                            ["rerank_applied", t.rerank_applied],
                            ["rerank_model", t.rerank_model],
                            ["rerank_endpoint", t.rerank_endpoint],
                            ["rerank_error", t.rerank_error],
                        ])
                    )
                );

                blocks.push(
                    section(
                        "Auto-Merge",
                        kv([
                            ["auto_merge_enabled", t.auto_merge_enabled],
                            ["auto_merge_applied", t.auto_merge_applied],
                            ["auto_merge_threshold", t.auto_merge_threshold],
                            ["auto_merge_replaced_chunks", t.auto_merge_replaced_chunks],
                            ["auto_merge_steps", t.auto_merge_steps],
                            ["leaf_retrieve_level", t.leaf_retrieve_level],
                        ])
                    )
                );

                if (t.initial_retrieved_chunks && t.initial_retrieved_chunks.length) {
                    blocks.push(section(`初次召回（${t.initial_retrieved_chunks.length}）`, chunkList(t.initial_retrieved_chunks)));
                }
                if (t.expanded_retrieved_chunks && t.expanded_retrieved_chunks.length) {
                    blocks.push(section(`扩展召回（${t.expanded_retrieved_chunks.length}）`, chunkList(t.expanded_retrieved_chunks)));
                }
                if (t.retrieved_chunks && t.retrieved_chunks.length) {
                    blocks.push(section(`最终 chunks（${t.retrieved_chunks.length}）`, chunkList(t.retrieved_chunks)));
                }

                return h("div", {}, blocks);
            };
        },
    });

    // ---------- 主应用 ----------
    const App = {
        components: { RagTraceViewer },
        setup() {
            const token = ref(localStorage.getItem(TOKEN_KEY) || "");
            const authed = ref(!!token.value);
            const tokenInput = ref("");
            const loggingIn = ref(false);
            const loginErr = ref("");

            const tab = ref("logs");

            const defaultLogsFilters = () => ({
                q: "",
                user_id: "",
                session_id: "",
                model: "",
                role: "",
                status: "",
                range: "all",
                page: 1,
                page_size: 20,
            });
            const defaultSessionsFilters = () => ({
                q: "",
                user_id: "",
                range: "all",
                page: 1,
                page_size: 20,
            });

            const logsFilters = reactive(defaultLogsFilters());
            const sessionsFilters = reactive(defaultSessionsFilters());

            const logsData = reactive({ total: 0, page: 1, page_size: 20, items: [] });
            const sessionsData = reactive({ total: 0, page: 1, page_size: 20, items: [] });
            const logsLoading = ref(false);
            const sessionsLoading = ref(false);

            const stats = ref(null);
            const statsLoading = ref(false);
            const maxDaily = computed(() => {
                if (!stats.value || !stats.value.daily_messages) return 1;
                return stats.value.daily_messages.reduce((m, d) => Math.max(m, d.count), 0) || 1;
            });

            const logDetail = ref(null);
            const sessionDetail = ref(null);

            const viewingUser = computed(() => logsFilters.user_id || "");

            // ---------- fetch 工具 ----------
            const authFetch = async (path, params, method = "GET") => {
                let url = path;
                if (method === "GET" && params) {
                    const q = new URLSearchParams();
                    Object.entries(params).forEach(([k, v]) => {
                        if (v !== undefined && v !== null && v !== "") q.append(k, v);
                    });
                    const s = q.toString();
                    if (s) url += "?" + s;
                }
                const resp = await fetch(url, {
                    method,
                    headers: {
                        Authorization: "Bearer " + token.value,
                    },
                });
                if (resp.status === 401 || resp.status === 403) {
                    logout();
                    throw new Error("鉴权失败，请重新登录");
                }
                if (!resp.ok) {
                    let detail = resp.statusText;
                    try {
                        const j = await resp.json();
                        detail = j.detail || JSON.stringify(j);
                    } catch (_) {
                        /* ignore */
                    }
                    throw new Error(detail);
                }
                return resp.json();
            };

            // ---------- 登录 ----------
            const doLogin = async () => {
                loginErr.value = "";
                if (!tokenInput.value.trim()) {
                    loginErr.value = "请输入 Token";
                    return;
                }
                loggingIn.value = true;
                token.value = tokenInput.value.trim();
                try {
                    await authFetch("/admin/ping");
                    localStorage.setItem(TOKEN_KEY, token.value);
                    authed.value = true;
                    tokenInput.value = "";
                    reloadLogs();
                } catch (e) {
                    loginErr.value = e.message || "登录失败";
                    token.value = "";
                } finally {
                    loggingIn.value = false;
                }
            };

            const logout = () => {
                token.value = "";
                localStorage.removeItem(TOKEN_KEY);
                authed.value = false;
            };

            // ---------- tabs ----------
            const switchTab = (t) => {
                tab.value = t;
                if (t === "logs" && logsData.items.length === 0) reloadLogs();
                if (t === "sessions" && sessionsData.items.length === 0) reloadSessions();
                if (t === "stats") reloadStats();
            };

            // ---------- 日志 ----------
            const reloadLogs = async () => {
                logsLoading.value = true;
                try {
                    const data = await authFetch("/admin/logs", toApiParams(logsFilters));
                    Object.assign(logsData, data);
                } catch (e) {
                    alert("加载失败：" + e.message);
                } finally {
                    logsLoading.value = false;
                }
            };
            const gotoLogsPage = (p) => {
                const max = totalPages(logsData);
                if (p < 1 || p > max) return;
                logsFilters.page = p;
                reloadLogs();
            };
            const resetLogsFilters = () => {
                Object.assign(logsFilters, defaultLogsFilters());
                reloadLogs();
            };
            const openLogDetail = async (row) => {
                try {
                    logDetail.value = await authFetch(`/admin/logs/${row.id}`);
                } catch (e) {
                    alert("加载详情失败：" + e.message);
                }
            };
            const closeLogDetail = () => {
                logDetail.value = null;
            };

            // ---------- 会话 ----------
            const reloadSessions = async () => {
                sessionsLoading.value = true;
                try {
                    const data = await authFetch("/admin/sessions", toApiParams(sessionsFilters));
                    Object.assign(sessionsData, data);
                } catch (e) {
                    alert("加载失败：" + e.message);
                } finally {
                    sessionsLoading.value = false;
                }
            };
            const gotoSessionsPage = (p) => {
                const max = totalPages(sessionsData);
                if (p < 1 || p > max) return;
                sessionsFilters.page = p;
                reloadSessions();
            };
            const resetSessionsFilters = () => {
                Object.assign(sessionsFilters, defaultSessionsFilters());
                reloadSessions();
            };
            const openSessionDetail = async (row) => {
                try {
                    sessionDetail.value = await authFetch(`/admin/sessions/${row.session_id}`);
                } catch (e) {
                    alert("加载详情失败：" + e.message);
                }
            };
            const openSessionById = async (sid) => {
                closeLogDetail();
                try {
                    sessionDetail.value = await authFetch(`/admin/sessions/${sid}`);
                } catch (e) {
                    alert("加载详情失败：" + e.message);
                }
            };
            const closeSessionDetail = () => {
                sessionDetail.value = null;
            };

            // ---------- 总览 ----------
            const reloadStats = async () => {
                statsLoading.value = true;
                try {
                    stats.value = await authFetch("/admin/stats", { days: 7 });
                } catch (e) {
                    alert("加载失败：" + e.message);
                } finally {
                    statsLoading.value = false;
                }
            };

            // ---------- 工具 ----------
            const rangeToStart = (range) => {
                if (!range || range === "all") return "";
                const d = new Date();
                if (range === "7d") d.setDate(d.getDate() - 7);
                else if (range === "4w") d.setDate(d.getDate() - 28);
                else if (range === "6m") d.setMonth(d.getMonth() - 6);
                else return "";
                const pad = (n) => String(n).padStart(2, "0");
                return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
            };
            const toApiParams = (f) => {
                const { range, start, end, ...rest } = f;
                return { ...rest, start: rangeToStart(range) };
            };
            const totalPages = (d) => Math.max(1, Math.ceil((d.total || 0) / (d.page_size || 20)));
            const shortId = (s) => (s ? (s.length > 10 ? s.slice(0, 8) + "…" : s) : "—");
            const roleLabel = (r) => ({ user: "用户", assistant: "AI", system: "system" }[r] || r);
            const previewText = (s) => (s || "").trim().replace(/\s+/g, " ");
            const formatTime = (s) => {
                if (!s) return "—";
                const d = new Date(s);
                if (Number.isNaN(d.getTime())) return s;
                const pad = (n) => String(n).padStart(2, "0");
                return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
            };
            const renderMarkdown = (s) => {
                if (!s) return "";
                try {
                    if (window.marked && window.hljs) {
                        window.marked.setOptions({
                            breaks: true,
                            highlight(code, lang) {
                                try {
                                    if (lang && window.hljs.getLanguage(lang)) {
                                        return window.hljs.highlight(code, { language: lang }).value;
                                    }
                                    return window.hljs.highlightAuto(code).value;
                                } catch (_) {
                                    return code;
                                }
                            },
                        });
                    }
                    return window.marked ? window.marked.parse(s) : escapeHtml(s);
                } catch (_) {
                    return escapeHtml(s);
                }
            };
            const escapeHtml = (s) =>
                String(s).replace(/[&<>"']/g, (c) => ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;",
                }[c]));

            const filterByUser = (uid) => {
                closeSessionDetail();
                logsFilters.user_id = uid;
                logsFilters.page = 1;
                tab.value = "logs";
                reloadLogs();
            };
            const clearUserFilter = () => {
                logsFilters.user_id = "";
                reloadLogs();
            };

            onMounted(() => {
                if (authed.value) reloadLogs();
            });

            return {
                // state
                authed,
                tokenInput,
                loggingIn,
                loginErr,
                tab,
                logsFilters,
                sessionsFilters,
                logsData,
                sessionsData,
                logsLoading,
                sessionsLoading,
                stats,
                statsLoading,
                maxDaily,
                logDetail,
                sessionDetail,
                viewingUser,
                // actions
                doLogin,
                logout,
                switchTab,
                reloadLogs,
                gotoLogsPage,
                resetLogsFilters,
                openLogDetail,
                closeLogDetail,
                reloadSessions,
                gotoSessionsPage,
                resetSessionsFilters,
                openSessionDetail,
                openSessionById,
                closeSessionDetail,
                filterByUser,
                clearUserFilter,
                // helpers
                totalPages,
                shortId,
                roleLabel,
                previewText,
                formatTime,
                renderMarkdown,
            };
        },
    };

    createApp(App).mount("#app");
})();
