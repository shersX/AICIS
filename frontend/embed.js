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
        bottom: 20,
        zIndex: 2147483000,
        startOpen: false,
    };

    const mount = () => {
        const userConfig = window.AICIS_CHATBOT_CONFIG || {};
        const cfg = { ...defaults, ...userConfig };

        const normalizedBase = cfg.baseUrl.replace(/\/+$/, '');
        const normalizedPath = cfg.iframePath.startsWith('/') ? cfg.iframePath : `/${cfg.iframePath}`;
        const iframeSrc = `${normalizedBase}${normalizedPath}?${cfg.iframeQuery}`;

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
        header.style.background = '#578873';
        header.style.color = '#fff';
        header.style.display = 'flex';
        header.style.alignItems = 'center';
        header.style.justifyContent = 'space-between';
        header.style.fontSize = '13px';
        header.style.fontWeight = '600';
        header.textContent = cfg.title;

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
        launcher.style.border = 'none';
        launcher.style.cursor = 'pointer';
        launcher.style.background = '#e9f5eb';
        launcher.style.color = '#fff';
        launcher.style.fontSize = '24px';
        launcher.style.boxShadow = '0 10px 26px rgba(45, 106, 79, 0.4)';
        launcher.style.pointerEvents = 'auto';
        launcher.textContent = cfg.icon;

        const togglePanel = (open) => {
            const show = typeof open === 'boolean' ? open : panel.style.display === 'none';
            panel.style.display = show ? 'block' : 'none';
            launcher.setAttribute('aria-expanded', show ? 'true' : 'false');
        };

        launcher.addEventListener('click', () => togglePanel());
        closeBtn.addEventListener('click', () => togglePanel(false));

        header.appendChild(closeBtn);
        panel.appendChild(header);
        panel.appendChild(iframe);
        root.appendChild(panel);
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
