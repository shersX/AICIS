(() => {
    if (window.__AICIS_CHATBOT_EMBEDDED__) return;
    window.__AICIS_CHATBOT_EMBEDDED__ = true;

    const defaults = {
        baseUrl: '',
        iframePath: '/',
        iframeQuery: 'embed=1',
        title: 'AICIS 智能助手',
        icon: '🧠',
        width: 380,
        height: 640,
        right: 20,
        bottom: 48,
        zIndex: 2147483000,
        startOpen: false,
    };

    const mount = () => {
        const userConfig = window.AICIS_CHATBOT_CONFIG || {};
        const cfg = { ...defaults, ...userConfig };

        const normalizedBase = cfg.baseUrl.replace(/\/+$/, '');
        const normalizedPath = cfg.iframePath.startsWith('/') ? cfg.iframePath : `/${cfg.iframePath}`;
        const iframeSrc = `${normalizedBase}${normalizedPath}?${cfg.iframeQuery}`;
        let postTargetOrigin = '*';
        try {
            postTargetOrigin = new URL(normalizedBase).origin;
        } catch (_) {
            /* keep * */
        }

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
        panel.style.marginBottom = '12px';
        panel.style.borderRadius = '16px';
        panel.style.overflow = 'hidden';
        panel.style.background = '#fff';
        panel.style.boxShadow = '0 18px 50px rgba(0, 0, 0, 0.24)';
        panel.style.display = cfg.startOpen ? 'block' : 'none';
        panel.style.pointerEvents = 'auto';

        const header = document.createElement('div');
        header.style.height = '40px';
        header.style.padding = '0 12px';
        header.style.background = '#00b140';
        header.style.color = '#fff';
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
        exportBtn.style.border = '1px solid rgba(255, 255, 255, 0.55)';
        exportBtn.style.borderRadius = '8px';
        exportBtn.style.background = 'rgba(255, 255, 255, 0.18)';
        exportBtn.style.color = '#fff';
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
        closeBtn.style.color = '#fff';
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

        const hoverTooltip = document.createElement('div');
        hoverTooltip.textContent = 'AICIS智能问答';
        hoverTooltip.style.padding = '6px 10px';
        hoverTooltip.style.marginBottom = '8px';
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
            if (!exportBtn.disabled) exportBtn.style.background = 'rgba(255, 255, 255, 0.3)';
        });
        exportBtn.addEventListener('mouseleave', () => {
            exportBtn.style.background = 'rgba(255, 255, 255, 0.18)';
        });
        syncExportBtnVisual();

        window.addEventListener('message', (e) => {
            if (e.source !== iframe.contentWindow) return;
            if (e.data && e.data.type === 'aicis-export-state') {
                const n = Number(e.data.count) || 0;
                exportBtn.disabled = n < 1;
                syncExportBtnVisual();
            }
        });

        panel.appendChild(header);
        panel.appendChild(iframe);
        root.appendChild(panel);
        root.appendChild(hoverTooltip);
        root.appendChild(launcher);
        document.body.appendChild(root);

        launcher.setAttribute('aria-expanded', cfg.startOpen ? 'true' : 'false');
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount, { once: true });
    } else {
        mount();
    }
})();
