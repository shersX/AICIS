(() => {
    if (window.__AICIS_CHATBOT_EMBEDDED__) return;
    window.__AICIS_CHATBOT_EMBEDDED__ = true;

    const defaults = {
        baseUrl: '',
        iframePath: '/',
        iframeQuery: 'embed=1',
        title: 'AICIS 智能助手（试用版）',
        icon: '🧠',
        width: 380,
        height: 640,
        right: 20,
        bottom: 20,
        zIndex: 2147483000,
        startOpen: false,
        // OA 单点登录使用的 cookie 名；如运维改名可在 window.AICIS_CHATBOT_CONFIG 里覆盖
        ssoCookieName: 'oassotoken',
    };

    /** 读取一个 cookie。注意：HttpOnly 的 cookie 永远读不到，调用方需要自己兜底。 */
    const readCookie = (name) => {
        const re = new RegExp('(?:^|;\\s*)' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '=([^;]*)');
        const m = document.cookie.match(re);
        return m ? decodeURIComponent(m[1]) : '';
    };

    const mount = () => {
        const userConfig = window.AICIS_CHATBOT_CONFIG || {};
        const cfg = { ...defaults, ...userConfig };

        // baseUrl 支持两种形态：
        //   1) 同域路径反代：'/aicis/'      -> normalizedBase='/aicis', isAbsoluteBase=false
        //   2) 跨域直连绝对 URL：'http://...' -> normalizedBase='http://...', isAbsoluteBase=true
        const rawBase = (cfg.baseUrl || '').replace(/\/+$/, '');
        const isAbsoluteBase = /^https?:\/\//i.test(rawBase);
        const normalizedBase = rawBase;
        const normalizedPath = cfg.iframePath.startsWith('/') ? cfg.iframePath : `/${cfg.iframePath}`;
        const iframeSrc = `${normalizedBase}${normalizedPath}?${cfg.iframeQuery}`;

        // postMessage 的 targetOrigin：同源走 location.origin，跨源走 baseUrl 的 origin。
        let postTargetOrigin = '*';
        if (isAbsoluteBase) {
            try {
                postTargetOrigin = new URL(normalizedBase).origin;
            } catch (_) {
                postTargetOrigin = '*';
            }
        } else {
            postTargetOrigin = window.location.origin;
        }

        const ssoLoginUrl = `${normalizedBase}/auth/sso_login`;

        const root = document.createElement('div');
        root.setAttribute('id', 'aicis-embed-root');
        root.style.position = 'fixed';
        root.style.right = `${cfg.right}px`;
        root.style.bottom = `${cfg.bottom}px`;
        root.style.zIndex = String(cfg.zIndex);
        root.style.fontFamily = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif';
        root.style.pointerEvents = 'none';
        root.style.display = 'flex';
        root.style.flexDirection = 'column';
        root.style.alignItems = 'flex-end';

        const panel = document.createElement('div');
        panel.style.width = `${cfg.width}px`;
        panel.style.height = `${cfg.height}px`;
        panel.style.maxWidth = 'calc(100vw - 20px)';
        panel.style.maxHeight = 'calc(100vh - 92px)';
        panel.style.marginBottom = '8px';
        panel.style.borderRadius = '16px';
        panel.style.overflow = 'hidden';
        panel.style.background = '#fff';
        panel.style.boxShadow = '0 18px 50px rgba(0, 0, 0, 0.24)';
        panel.style.display = cfg.startOpen ? 'block' : 'none';
        panel.style.pointerEvents = 'auto';

        const header = document.createElement('div');
        header.style.height = '40px';
        header.style.padding = '0 12px';
        header.style.background = '#ffffff';
        header.style.color = '#111111';
        header.style.borderBottom = '1px solid #e0e0e0';
        header.style.boxSizing = 'border-box';
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';
        header.style.fontSize = '13px';
        header.style.fontWeight = '600';
        header.style.gap = '8px';

        const titleEl = document.createElement('span');
        titleEl.textContent = cfg.title;
        titleEl.style.flex = '1';
        titleEl.style.minWidth = '0';
        titleEl.style.overflow = 'hidden';
        titleEl.style.textOverflow = 'ellipsis';
        titleEl.style.whiteSpace = 'nowrap';
        titleEl.style.color = '#111111';

        const exportBtn = document.createElement('button');
        exportBtn.type = 'button';
        exportBtn.textContent = '📄 导出 PDF';
        exportBtn.setAttribute('aria-label', '导出 PDF');
        exportBtn.title = '导出当前对话为 PDF';
        exportBtn.disabled = true;
        exportBtn.style.display = 'inline-flex';
        exportBtn.style.alignItems = 'center';
        exportBtn.style.gap = '4px';
        exportBtn.style.padding = '4px 10px';
        exportBtn.style.border = '1px solid #cccccc';
        exportBtn.style.borderRadius = '8px';
        exportBtn.style.background = '#f5f5f5';
        exportBtn.style.color = '#333333';
        exportBtn.style.fontSize = '12px';
        exportBtn.style.fontWeight = '600';
        exportBtn.style.cursor = 'pointer';
        exportBtn.style.pointerEvents = 'auto';
        exportBtn.style.flexShrink = '0';

        const closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.innerText = '✕';
        closeBtn.setAttribute('aria-label', '关闭聊天窗口');
        closeBtn.style.border = 'none';
        closeBtn.style.background = 'transparent';
        closeBtn.style.color = '#444444';
        closeBtn.style.cursor = 'pointer';
        closeBtn.style.fontSize = '14px';
        closeBtn.style.padding = '2px 6px';
        closeBtn.style.borderRadius = '6px';
        closeBtn.style.pointerEvents = 'auto';
        closeBtn.style.flexShrink = '0';

        const headerRight = document.createElement('div');
        headerRight.style.display = 'flex';
        headerRight.style.alignItems = 'center';
        headerRight.style.gap = '6px';
        headerRight.style.flexShrink = '0';
        headerRight.appendChild(exportBtn);
        headerRight.appendChild(closeBtn);

        header.appendChild(titleEl);
        header.appendChild(headerRight);

        const iframe = document.createElement('iframe');
        iframe.src = iframeSrc;
        iframe.title = cfg.title;
        iframe.style.width = '100%';
        iframe.style.height = 'calc(100% - 40px)';
        iframe.style.border = '0';
        iframe.setAttribute('allow', 'clipboard-write');
        iframe.setAttribute('referrerpolicy', 'no-referrer-when-downgrade');

        // ----------------- SSO 鉴权：拿 ticket → postMessage 发给 iframe -----------------
        // 缓存最近一次成功的 auth 结果，便于 iframe 重新 ready 时补发。
        let lastAuthPayload = null;
        let pendingAuthFetch = null;

        const sendAuthToIframe = () => {
            if (!lastAuthPayload || !iframe.contentWindow) return;
            try {
                iframe.contentWindow.postMessage(
                    { type: 'aicis-auth', ...lastAuthPayload },
                    postTargetOrigin,
                );
            } catch (_) {
                /* 忽略 */
            }
        };

        const fetchAuth = async () => {
            // 同一时间只跑一份，避免 iframe 多次索要时打多份请求。
            if (pendingAuthFetch) return pendingAuthFetch;
            const oassotoken = readCookie(cfg.ssoCookieName);
            if (!oassotoken) {
                console.warn('[AICIS] 未找到 cookie:', cfg.ssoCookieName, '— 嵌入式问答将无法绑定员工身份');
                return null;
            }
            pendingAuthFetch = (async () => {
                try {
                    const r = await fetch(ssoLoginUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ oassotoken }),
                        credentials: 'omit',
                    });
                    if (!r.ok) {
                        console.warn('[AICIS] sso_login 失败：HTTP', r.status, await r.text().catch(() => ''));
                        return null;
                    }
                    const data = await r.json();
                    lastAuthPayload = {
                        ticket: data.ticket,
                        employee_no: data.employee_no,
                        employee_name: data.employee_name || '',
                        expires_at: data.expires_at || '',
                    };
                    sendAuthToIframe();
                    return lastAuthPayload;
                } catch (e) {
                    console.warn('[AICIS] sso_login 异常：', e);
                    return null;
                } finally {
                    pendingAuthFetch = null;
                }
            })();
            return pendingAuthFetch;
        };

        // iframe 加载完成后：先看缓存有没有，没有就主动拉一次。
        iframe.addEventListener('load', () => {
            if (lastAuthPayload) {
                sendAuthToIframe();
            } else {
                fetchAuth();
            }
        });

        const launcher = document.createElement('button');
        launcher.type = 'button';
        launcher.setAttribute('aria-label', cfg.title);
        launcher.style.width = '58px';
        launcher.style.height = '58px';
        launcher.style.borderRadius = '50%';
        launcher.style.border = '4px solid #fff';
        launcher.style.cursor = 'pointer';
        launcher.style.background = '#c8e2fe';
        launcher.style.color = '#fff';
        launcher.style.fontSize = '24px';
        launcher.style.boxShadow = '0 10px 26px rgba(45, 106, 79, 0.4)';
        launcher.style.pointerEvents = 'auto';
        launcher.textContent = cfg.icon;

        const launcherWrap = document.createElement('div');
        launcherWrap.style.position = 'relative';
        launcherWrap.style.alignSelf = 'flex-end';
        launcherWrap.style.pointerEvents = 'none';

        const hoverTooltip = document.createElement('div');
        hoverTooltip.textContent = 'AICIS智能问答';
        hoverTooltip.style.position = 'absolute';
        hoverTooltip.style.right = '0';
        hoverTooltip.style.bottom = '100%';
        hoverTooltip.style.marginBottom = '6px';
        hoverTooltip.style.padding = '6px 10px';
        hoverTooltip.style.borderRadius = '8px';
        hoverTooltip.style.background = 'rgba(0, 0, 0, 0.75)';
        hoverTooltip.style.color = '#fff';
        hoverTooltip.style.fontSize = '12px';
        hoverTooltip.style.lineHeight = '1';
        hoverTooltip.style.whiteSpace = 'nowrap';
        hoverTooltip.style.pointerEvents = 'none';
        hoverTooltip.style.opacity = '0';
        hoverTooltip.style.visibility = 'hidden';
        hoverTooltip.style.transition = 'opacity 0.15s ease';
        hoverTooltip.style.zIndex = '1';

        const toggleTooltip = (show) => {
            hoverTooltip.style.opacity = show ? '1' : '0';
            hoverTooltip.style.visibility = show ? 'visible' : 'hidden';
        };

        const togglePanel = (open) => {
            const show = typeof open === 'boolean' ? open : panel.style.display === 'none';
            panel.style.display = show ? 'block' : 'none';
            launcher.setAttribute('aria-expanded', show ? 'true' : 'false');
        };

        launcher.addEventListener('click', () => {
            toggleTooltip(false);
            togglePanel();
        });
        launcher.addEventListener('mouseenter', () => toggleTooltip(true));
        launcher.addEventListener('mouseleave', () => toggleTooltip(false));
        launcher.addEventListener('focus', () => toggleTooltip(true));
        launcher.addEventListener('blur', () => toggleTooltip(false));
        closeBtn.addEventListener('click', () => togglePanel(false));
        exportBtn.addEventListener('click', () => {
            if (iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: 'aicis-open-export' }, postTargetOrigin);
            }
        });

        const syncExportBtnVisual = () => {
            exportBtn.style.opacity = exportBtn.disabled ? '0.45' : '1';
            exportBtn.style.cursor = exportBtn.disabled ? 'not-allowed' : 'pointer';
        };
        exportBtn.addEventListener('mouseenter', () => {
            if (!exportBtn.disabled) {
                exportBtn.style.background = '#ebebeb';
                exportBtn.style.borderColor = '#b0b0b0';
            }
        });
        exportBtn.addEventListener('mouseleave', () => {
            exportBtn.style.background = '#f5f5f5';
            exportBtn.style.borderColor = '#cccccc';
        });
        syncExportBtnVisual();

        window.addEventListener('message', (e) => {
            if (e.source !== iframe.contentWindow) return;
            const data = e.data;
            if (!data || typeof data !== 'object') return;
            if (data.type === 'aicis-export-state') {
                const n = Number(data.count) || 0;
                exportBtn.disabled = n < 1;
                syncExportBtnVisual();
            } else if (data.type === 'aicis-request-auth') {
                // iframe 启动时主动索要鉴权信息（如果之前已经成功拿过就直接补发）。
                if (lastAuthPayload) {
                    sendAuthToIframe();
                } else {
                    fetchAuth();
                }
            } else if (data.type === 'aicis-auth-invalid') {
                // iframe 收到 401，让父页清缓存重新换。
                lastAuthPayload = null;
                fetchAuth();
            }
        });

        panel.appendChild(header);
        panel.appendChild(iframe);
        root.appendChild(panel);
        launcherWrap.appendChild(hoverTooltip);
        launcherWrap.appendChild(launcher);
        root.appendChild(launcherWrap);
        document.body.appendChild(root);

        launcher.setAttribute('aria-expanded', cfg.startOpen ? 'true' : 'false');
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount, { once: true });
    } else {
        mount();
    }
})();
