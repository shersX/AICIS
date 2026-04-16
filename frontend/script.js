const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            API_URL: '/chat',
            abortController: null,
            userId: 'user_' + Math.random().toString(36).substring(2, 11),
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            isEmbedMode: false,
            exportModalVisible: false,
            exportModalLoading: false,
            exportRounds: [],
            exportSessionLabel: ''
        };
    },
    computed: {
        exportableRoundCount() {
            return this.buildConversationRounds(this.messages).length;
        },
        selectedExportRoundCount() {
            return this.exportRounds.filter((r) => r.selected).length;
        }
    },
    mounted() {
        const params = new URLSearchParams(window.location.search);
        this.isEmbedMode = params.get('embed') === '1';
        this.configureMarked();
        // 尝试从 localStorage 恢复用户ID
        const savedUserId = localStorage.getItem('userId');
        if (savedUserId) {
            this.userId = savedUserId;
        } else {
            localStorage.setItem('userId', this.userId);
        }
        if (this.isEmbedMode && window.parent !== window) {
            window.addEventListener('message', (e) => {
                if (e.source !== window.parent) return;
                if (e.data && e.data.type === 'aicis-open-export') {
                    this.openExportCurrent();
                }
            });
        }
        this.notifyParentExportState();
    },
    methods: {
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },
        
        parseMarkdown(text) {
            if (!text) return '';
            const html = marked.parse(text);
            // 对话内链接在新标签页打开，避免离开当前会话页
            return html.replace(/<a\s+/gi, '<a target="_blank" rel="noopener noreferrer" ');
        },
        
        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },
        
        handleCompositionStart() {
            this.isComposing = true;
        },
        
        handleCompositionEnd() {
            this.isComposing = false;
        },

        onRagStepsToggle(msg, event) {
            msg.ragStepsExpanded = event.target.open;
        },
        
        handleKeyDown(event) {
            // 如果是回车键且不是Shift+回车，且不在输入法组合中
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },
        
        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },
        
        async handleSend() {
            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;

            // Add user message
            this.messages.push({
                text: text,
                isUser: true
            });
            
            this.userInput = '';
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            // Show loading
            this.isLoading = true;

            // 立刻创建气泡，显示思考动画（二合一：思考 + 流式输出在同一个气泡）
            this.messages.push({ 
                text: '', 
                isUser: false, 
                isThinking: true, 
                ragTrace: null,
                ragSteps: [],
                ragStepsExpanded: true
            });
            const botMsgIdx = this.messages.length - 1;

            // 用于终止请求
            this.abortController = new AbortController();

            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: text,
                        user_id: this.userId,
                        session_id: this.sessionId
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    
                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);
                        
                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.type === 'content') {
                                    // 收到第一个 token 时关闭思考动画
                                    if (this.messages[botMsgIdx].isThinking) {
                                        this.messages[botMsgIdx].isThinking = false;
                                    }
                                    this.messages[botMsgIdx].text += data.content;
                                } else if (data.type === 'trace') {
                                    this.messages[botMsgIdx].ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    // 实时 RAG 检索步骤（与正文同气泡；每步后 nextTick 以便同缓冲区内多事件也能逐帧刷新）
                                    if (!this.messages[botMsgIdx].ragSteps) {
                                        this.messages[botMsgIdx].ragSteps = [];
                                    }
                                    this.messages[botMsgIdx].ragSteps.push(data.step);
                                    await this.$nextTick();
                                } else if (data.type === 'error') {
                                    this.messages[botMsgIdx].isThinking = false;
                                    this.messages[botMsgIdx].text += `\n[Error: ${data.content}]`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                if (error.name === 'AbortError') {
                    // 用户主动终止
                    this.messages[botMsgIdx].isThinking = false;
                    if (!this.messages[botMsgIdx].text) {
                        this.messages[botMsgIdx].text = '(已终止回答)';
                    } else {
                        this.messages[botMsgIdx].text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    console.error('Error:', error);
                    this.messages[botMsgIdx].isThinking = false;
                    this.messages[botMsgIdx].text = `出错了：${error.message}`;
                }
            } finally {
                const m = this.messages[botMsgIdx];
                if (m && m.ragSteps && m.ragSteps.length) {
                    m.ragStepsExpanded = false;
                }
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },
        
        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },
        
        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },
        
        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },
        
        handleNewChat() {
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },
        
        handleClearChat() {
            if (confirm('确定要清空当前对话吗？')) {
                this.messages = [];
            }
        },
        
        async handleHistory() {
            if (this.isEmbedMode) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await fetch(`/sessions/${this.userId}`);
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                console.error('Error loading sessions:', error);
                alert('加载历史记录失败：' + error.message);
            }
        },
        
        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';
            
            // 从后端加载历史消息
            try {
                const response = await fetch(`/sessions/${this.userId}/${sessionId}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                
                // 转换消息格式并显示
                this.messages = data.messages.map(msg => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null
                }));
                
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                console.error('Error loading session:', error);
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        /** 将连续的用户消息 + 其后若干条助手消息视为一轮（仅正文用于导出） */
        buildConversationRounds(messages) {
            const rounds = [];
            let i = 0;
            while (i < messages.length) {
                if (!messages[i].isUser) {
                    i += 1;
                    continue;
                }
                const userText = messages[i].text || '';
                let botText = '';
                let j = i + 1;
                while (j < messages.length && !messages[j].isUser) {
                    botText = messages[j].text || '';
                    j += 1;
                }
                rounds.push({
                    index: rounds.length,
                    userText,
                    botText
                });
                i = j;
            }
            return rounds;
        },

        apiMessagesToUiMessages(apiMessages) {
            return (apiMessages || []).map((msg) => ({
                text: msg.content,
                isUser: msg.type === 'human',
                ragTrace: null
            }));
        },

        /** 供打印/PDF 使用：保留 Markdown 结构与链接可点击（与主界面 parseMarkdown 一致地处理链接） */
        markdownToExportHtml(md) {
            if (!md || !String(md).trim()) {
                return '<p class="md-empty">（无内容）</p>';
            }
            let html = marked.parse(md);
            return html.replace(/<a\s+/gi, '<a target="_blank" rel="noopener noreferrer" ');
        },

        escapeXmlText(s) {
            return String(s == null ? '' : s)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        },

        truncateForPreview(text, maxLen = 48) {
            const t = (text || '').replace(/\s+/g, ' ').trim();
            if (t.length <= maxLen) return t || '（空）';
            return t.slice(0, maxLen) + '…';
        },

        openExportModalFromRounds(rounds, sessionLabel) {
            this.exportSessionLabel = sessionLabel || this.sessionId;
            this.exportRounds = rounds.map((r) => ({
                ...r,
                selected: true
            }));
            this.exportModalVisible = true;
        },

        openExportCurrent() {
            const rounds = this.buildConversationRounds(this.messages);
            if (!rounds.length) {
                alert('没有可导出的问答轮次。');
                return;
            }
            this.openExportModalFromRounds(rounds, this.sessionId);
        },

        notifyParentExportState() {
            if (!this.isEmbedMode || window.parent === window) return;
            try {
                window.parent.postMessage(
                    { type: 'aicis-export-state', count: this.exportableRoundCount },
                    '*'
                );
            } catch (_) {
                /* ignore */
            }
        },

        async openExportForSession(sid) {
            this.exportModalLoading = true;
            this.exportModalVisible = true;
            this.exportRounds = [];
            this.exportSessionLabel = sid;
            try {
                const response = await fetch(`/sessions/${this.userId}/${sid}`);
                if (!response.ok) throw new Error('无法加载会话');
                const data = await response.json();
                const ui = this.apiMessagesToUiMessages(data.messages);
                const rounds = this.buildConversationRounds(ui);
                this.openExportModalFromRounds(rounds, sid);
            } catch (e) {
                console.error(e);
                alert('加载会话失败：' + e.message);
                this.exportModalVisible = false;
            } finally {
                this.exportModalLoading = false;
            }
        },

        onExportOverlayClick() {
            if (this.exportModalLoading) return;
            this.closeExportModal();
        },

        closeExportModal() {
            this.exportModalVisible = false;
            this.exportModalLoading = false;
            this.exportRounds = [];
        },

        setAllExportRounds(val) {
            this.exportRounds.forEach((r) => {
                r.selected = val;
            });
        },

        escapeHtmlForDoc(s) {
            const div = document.createElement('div');
            div.textContent = s == null ? '' : String(s);
            return div.innerHTML;
        },

        confirmExportPdf() {
            const picked = this.exportRounds.filter((r) => r.selected);
            if (!picked.length) {
                alert('请至少选择一轮对话。');
                return;
            }
            const title = 'AICIS 对话导出';
            const sessionLabel = this.exportSessionLabel || this.sessionId;
            const docTitle = this.escapeXmlText(`${title} — ${sessionLabel}`);
            const sessionLine = this.escapeHtmlForDoc(sessionLabel);
            const timeLine = this.escapeHtmlForDoc(new Date().toLocaleString());
            const sections = picked
                .sort((a, b) => a.index - b.index)
                .map((r) => {
                    const u = this.escapeHtmlForDoc(r.userText);
                    const bHtml = this.markdownToExportHtml(r.botText);
                    return (
                        `<section class="round">` +
                        `<div class="round-num">第 ${r.index + 1} 轮</div>` +
                        `<div class="role">用户</div>` +
                        `<div class="block user">${u}</div>` +
                        `<div class="role">助手</div>` +
                        `<div class="block assistant markdown-export">${bHtml}</div>` +
                        `</section>`
                    );
                })
                .join('');

            const exportCss =
                'body{font-family:"Noto Sans SC","Microsoft YaHei",sans-serif;padding:24px;color:#222;line-height:1.55;}' +
                'h1{font-size:1.25rem;margin:0 0 8px;font-weight:600;}' +
                '.meta{color:#666;font-size:0.85rem;margin-bottom:20px;}' +
                '.round{margin-bottom:22px;page-break-inside:auto;break-inside:auto;}' +
                '.round + .round{page-break-before:always;break-before:page;}' +
                '.round-num{font-size:0.8rem;color:#888;margin-bottom:6px;}' +
                '.role{font-weight:600;color:#2c6e49;margin-top:8px;font-size:0.9rem;}' +
                '.block{margin:4px 0 12px;padding:10px 12px;border-radius:8px;word-break:break-word;}' +
                '.block.user{background:#f0f7f4;border:1px solid #cfe8dc;white-space:pre-wrap;}' +
                '.block.assistant{background:#fafafa;border:1px solid #e8e8e8;white-space:normal;}' +
                '.markdown-export p{margin:0.45em 0;}' +
                '.markdown-export p:first-child{margin-top:0;}' +
                '.markdown-export p:last-child{margin-bottom:0;}' +
                '.markdown-export .md-empty{color:#999;font-size:0.9rem;margin:0;}' +
                '.markdown-export strong,.markdown-export b{font-weight:600;}' +
                '.markdown-export em,.markdown-export i{font-style:italic;}' +
                '.markdown-export a{color:#0a58ca;text-decoration:underline;}' +
                '.markdown-export code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:0.88em;background:#ececec;padding:0.12em 0.35em;border-radius:4px;}' +
                '.markdown-export pre{background:#f0f0f0;padding:10px 12px;border-radius:8px;overflow-x:auto;margin:0.6em 0;white-space:pre-wrap;word-break:break-word;}' +
                '.markdown-export pre code{background:none;padding:0;font-size:0.85em;}' +
                '.markdown-export ul,.markdown-export ol{margin:0.45em 0;padding-left:1.35em;}' +
                '.markdown-export li{margin:0.2em 0;}' +
                '.markdown-export h1,.markdown-export h2,.markdown-export h3{margin:0.65em 0 0.35em;font-weight:600;line-height:1.3;}' +
                '.markdown-export h1{font-size:1.15rem;}' +
                '.markdown-export h2{font-size:1.05rem;}' +
                '.markdown-export h3{font-size:1rem;}' +
                '.markdown-export blockquote{margin:0.5em 0;padding:0.2em 0 0.2em 0.9em;border-left:3px solid #ccc;color:#444;}' +
                '.markdown-export table{border-collapse:collapse;margin:0.6em 0;font-size:0.92em;}' +
                '.markdown-export th,.markdown-export td{border:1px solid #ddd;padding:6px 10px;text-align:left;}' +
                '.markdown-export th{background:#eee;font-weight:600;}';

            const html =
                '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"/>' +
                `<title>${docTitle}</title>` +
                '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;600&display=swap"/>' +
                '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-light.min.css"/>' +
                `<style>${exportCss}</style></head><body>` +
                `<h1>${this.escapeHtmlForDoc(title)}</h1>` +
                `<div class="meta">会话：${sessionLine} · 导出时间：${timeLine}</div>` +
                sections +
                '</body></html>';

            // 必须在用户点击的同步阶段打开窗口，且不能使用 noopener（否则 Chrome 返回 null，无法 print）
            const w = window.open('about:blank', '_blank');
            if (!w) {
                alert('请允许本站点弹出窗口，以便打开打印预览。（浏览器地址栏旁可临时允许弹窗）');
                return;
            }

            w.document.open();
            w.document.write(html);
            w.document.close();
            w.document.title = `${title} — ${sessionLabel}`;

            let printScheduled = false;
            const schedulePrint = () => {
                if (printScheduled) return;
                printScheduled = true;
                setTimeout(() => {
                    try {
                        w.focus();
                        w.print();
                    } catch (err) {
                        console.error(err);
                    }
                }, 500);
            };

            if (w.document.readyState === 'complete') {
                schedulePrint();
            } else {
                w.addEventListener('load', schedulePrint, { once: true });
                setTimeout(schedulePrint, 800);
            }

            this.closeExportModal();
        },

        async deleteSession(sessionId) {
            if (!confirm(`确定要删除会话 "${sessionId}" 吗？`)) {
                return;
            }

            try {
                const response = await fetch(`/sessions/${this.userId}/${sessionId}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                console.error('Error deleting session:', error);
                alert('删除会话失败：' + error.message);
            }
        }
    },
    watch: {
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.notifyParentExportState();
                });
            },
            deep: true
        }
    }
}).mount('#app');
