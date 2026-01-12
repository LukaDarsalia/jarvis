/**
 * Voice Assistant Web Client
 *
 * Simplified streaming architecture:
 * - Server sends AV frames immediately (no server-side timing)
 * - Client buffers frames before playback
 * - Client handles all timing and synchronization
 */

class VoiceAssistant {
    constructor() {
        // Debug logging
        this.DEBUG = false;
        this.METRICS_LOG = true;

        // WebSocket
        this.ws = null;
        this.connectionId = null;
        this.isConnected = false;

        // Audio playback
        this.audioContext = null;
        this.audioSampleRate = 24000;
        this.nextAudioTime = 0;

        // Recording
        this.mediaStream = null;
        this.isRecording = false;
        this.recordingContext = null;
        this.audioProcessor = null;
        this.audioSource = null;

        // Video/Avatar
        this.videoEnabled = false;
        this.idleFrame = null;
        this.videoFps = 25;
        this.frameInterval = 1000 / 25;  // 40ms per frame at 25 FPS

        // Streaming state
        this.isGenerating = false;
        this.streamComplete = false;
        this.isPaused = false;
        this.generationTimeoutId = null;  // Watchdog timer for stuck generation
        this.lastActivityTime = 0;        // Last time we received data

        // Frame buffer and playback - LARGER BUFFER to handle MuseTalk being slower than realtime
        this.frameBuffer = new Map();    // frame_index -> frame payload
        this.replayFrames = new Map();   // last completed stream frames
        this.pendingReplayFrames = new Map();
        this.isReplaying = false;
        this.nextFrameIndex = null;
        this.highestFrameIndex = null;
        this.missingFrameSince = null;
        this.underrunLogged = false;
        this.isBuffering = true;         // Waiting for buffer to fill
        this.isPlaying = false;          // Currently playing frames
        this.playbackTimer = null;       // setTimeout handle
        this.lastVideoFrame = null;      // Fallback video frame
        this.framesPlayed = 0;
        this.framesReceived = 0;
        this.frameOrderStats = {
            outOfOrder: 0,
            duplicates: 0,
            gaps: 0,
            lateDrops: 0,
        };
        this.frameArrivalStats = {
            lastArrivalTime: null,
            gapSamples: [],          // Track last N gaps
            maxSamples: 50,
            underrunCount: 0,        // Times playback stalled
            burstCount: 0,           // Frames arriving < 5ms apart (burst)
        };
        this.audioDecodeStats = {
            count: 0,
            logEvery: 50,
            logFirst: 3,
        };

        // Buffer configuration - simple manual setting
        this.bufferConfig = {
            minFrames: 24,          // Initial buffer: ~1 second at 25fps (reduced from 48)
            maxFrames: 150,         // Maximum buffer size
            missingFrameToleranceMs: 400,
        };

        // Per-stream metrics (reset on each generation)
        this.streamMetrics = {
            id: 0,
            finalized: false,
            lastRtf: null,
            lastAudioDurationMs: null,
            lastGenerationTimeMs: null,
            framesWithMetrics: 0,
            firstFrameAt: null,
            lastFrameAt: null,
            completeAt: null,
            framesReceived: 0,
            firstFrameIndex: null,
            maxFrameIndex: null,
        };
        this.streamCounter = 0;
        this.runMetrics = this.createRunMetrics();

        // Playback stats for logging
        this.playbackStats = {
            startTime: 0,
            expectedPlayTime: 0,
            actualPlayTime: 0,
        };

        // Conversation state
        this.currentAssistantMessage = null;
        this.wordsSpoken = [];
        this.lastLoggedWord = null;

        // Config
        this.config = {
            vad: { speech_threshold_ms: 200, silence_threshold_ms: 1500 },
            llm: { temperature: 0.7, top_p: 0.9, max_new_tokens: 512, system_prompt: 'თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი' },
            tts: { backbone_temperature: 0.01, backbone_top_p: 0.999, depth_temperature: 0.01, depth_top_p: 0.999 },
            musetalk: { batch_size: 8, lookahead_chunks: 2 },
        };

        // DOM Elements
        this.elements = {};

        this.init();
    }

    log(...args) {
        if (this.DEBUG) {
            console.log(`[VA ${new Date().toISOString().substr(11, 12)}]`, ...args);
        }
    }

    metricLog(message) {
        if (!this.METRICS_LOG) return;
        console.log(`[VA METRICS ${new Date().toISOString().substr(11, 12)}] ${message}`);
    }

    formatMs(value) {
        if (!Number.isFinite(value)) return 'n/a';
        return `${value.toFixed(0)}ms`;
    }

    formatRate(value, unit = '') {
        if (!Number.isFinite(value) || value <= 0) return 'n/a';
        return `${value.toFixed(2)}${unit}`;
    }

    createRunMetrics() {
        return {
            vadSpeechStartAt: null,
            vadUtteranceAt: null,
            sttStartAt: null,
            sttEndAt: null,
            llmStartAt: null,
            llmFirstTokenAt: null,
            llmLastTokenAt: null,
            llmTokenCount: 0,
            ttsStartAt: null,
            ttsFirstAudioAt: null,
            ttsFirstFrameIndex: null,
            musetalkFirstFrameAt: null,
            musetalkLastFrameAt: null,
            musetalkFirstFrameIndex: null,
            musetalkLastFrameIndex: null,
            // Key latency metric: utterance end to first AV frame
            firstAVFrameAt: null,
        };
    }

    resetRunMetrics() {
        this.runMetrics = this.createRunMetrics();
    }

    async init() {
        this.cacheElements();
        this.bindEvents();
        this.initializeBufferControls();
        await this.initAudioContext();
        this.connect();
        this.loadConfig();
    }

    cacheElements() {
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            avatarContainer: document.getElementById('avatarContainer'),
            avatarImage: document.getElementById('avatarImage'),
            avatarLoading: document.getElementById('avatarLoading'),
            avatarStatus: document.getElementById('avatarStatus'),
            avatarMetrics: document.getElementById('avatarMetrics'),
            videoFps: document.getElementById('videoFps'),
            videoFrames: document.getElementById('videoFrames'),
            chatMessages: document.getElementById('chatMessages'),
            chatContainer: document.getElementById('chatContainer'),
            metricsPanel: document.getElementById('metricsPanel'),
            rtfValue: document.getElementById('rtfValue'),
            genTimeValue: document.getElementById('genTimeValue'),
            audioDurValue: document.getElementById('audioDurValue'),
            currentWordValue: document.getElementById('currentWordValue'),
            voiceInput: document.getElementById('voiceInput'),
            vadIndicator: document.getElementById('vadIndicator'),
            micBtn: document.getElementById('micBtn'),
            vadStatus: document.getElementById('vadStatus'),
            stopBtn: document.getElementById('stopBtn'),
            replayBtn: document.getElementById('replayBtn'),
            settingsBtn: document.getElementById('settingsBtn'),
            settingsPanel: document.getElementById('settingsPanel'),
            settingsOverlay: document.getElementById('settingsOverlay'),
            closeSettingsBtn: document.getElementById('closeSettingsBtn'),
            saveSettingsBtn: document.getElementById('saveSettingsBtn'),
            resetSettingsBtn: document.getElementById('resetSettingsBtn'),
            speechThreshold: document.getElementById('speechThreshold'),
            speechThresholdValue: document.getElementById('speechThresholdValue'),
            silenceThreshold: document.getElementById('silenceThreshold'),
            silenceThresholdValue: document.getElementById('silenceThresholdValue'),
            earlySilenceThreshold: document.getElementById('earlySilenceThreshold'),
            earlySilenceThresholdValue: document.getElementById('earlySilenceThresholdValue'),
            llmTemperature: document.getElementById('llmTemperature'),
            llmTemperatureValue: document.getElementById('llmTemperatureValue'),
            llmTopP: document.getElementById('llmTopP'),
            llmTopPValue: document.getElementById('llmTopPValue'),
            llmMaxTokens: document.getElementById('llmMaxTokens'),
            llmMaxTokensValue: document.getElementById('llmMaxTokensValue'),
            systemPrompt: document.getElementById('systemPrompt'),
            ttsBackboneTemp: document.getElementById('ttsBackboneTemp'),
            ttsBackboneTempValue: document.getElementById('ttsBackboneTempValue'),
            ttsBackboneTopP: document.getElementById('ttsBackboneTopP'),
            ttsBackboneTopPValue: document.getElementById('ttsBackboneTopPValue'),
            ttsDepthTemp: document.getElementById('ttsDepthTemp'),
            ttsDepthTempValue: document.getElementById('ttsDepthTempValue'),
            ttsDepthTopP: document.getElementById('ttsDepthTopP'),
            ttsDepthTopPValue: document.getElementById('ttsDepthTopPValue'),
            musetalkBatchSize: document.getElementById('musetalkBatchSize'),
            musetalkBatchSizeValue: document.getElementById('musetalkBatchSizeValue'),
            musetalkLookaheadChunks: document.getElementById('musetalkLookaheadChunks'),
            musetalkLookaheadChunksValue: document.getElementById('musetalkLookaheadChunksValue'),
            bufferFrames: document.getElementById('bufferFrames'),
            bufferCurrent: document.getElementById('bufferCurrent'),
        };
    }

    bindEvents() {
        this.elements.micBtn?.addEventListener('click', () => this.toggleRecording());
        this.elements.stopBtn?.addEventListener('click', () => this.stopGeneration());
        this.elements.replayBtn?.addEventListener('click', () => this.replayLastStream());
        this.elements.settingsBtn?.addEventListener('click', () => this.openSettings());
        this.elements.closeSettingsBtn?.addEventListener('click', () => this.closeSettings());
        this.elements.settingsOverlay?.addEventListener('click', () => this.closeSettings());
        this.elements.saveSettingsBtn?.addEventListener('click', () => this.saveSettings());
        this.elements.resetSettingsBtn?.addEventListener('click', () => this.resetSettings());

        // Sliders
        ['speechThreshold', 'silenceThreshold', 'earlySilenceThreshold', 'llmTemperature', 'llmTopP', 'llmMaxTokens',
         'ttsBackboneTemp', 'ttsBackboneTopP', 'ttsDepthTemp', 'ttsDepthTopP',
         'musetalkBatchSize', 'musetalkLookaheadChunks'].forEach(id => {
            const slider = this.elements[id];
            const valueEl = this.elements[id + 'Value'];
            if (slider && valueEl) {
                slider.addEventListener('input', () => valueEl.textContent = slider.value);
            }
        });

        // Buffer controls
        this.elements.bufferFrames?.addEventListener('input', () => this.updateBufferFrames());
    }

    // ============ Buffering ============

    initializeBufferControls() {
        if (this.elements.bufferFrames) {
            const frames = parseInt(this.elements.bufferFrames.value, 10);
            this.bufferConfig.minFrames = Number.isFinite(frames) && frames > 0 ? frames : 48;
        }
        this.updateBufferDisplay();
    }

    updateBufferFrames() {
        if (this.elements.bufferFrames) {
            const frames = parseInt(this.elements.bufferFrames.value, 10);
            this.bufferConfig.minFrames = Number.isFinite(frames) && frames > 0 ? frames : 48;
            this.updateBufferDisplay();
        }
    }

    updateBufferDisplay() {
        if (this.elements.bufferCurrent) {
            const frames = this.bufferConfig.minFrames;
            const ms = frames * this.frameInterval;
            this.elements.bufferCurrent.textContent = `ბუფერი: ${frames} ფრეიმი = ${ms}ms`;
        }
    }

    applyInitialBufferConfig() {
        // Just log the current buffer config
        this.log(
            `Initial buffer | ${this.bufferConfig.minFrames} frames | ${this.bufferConfig.minFrames * this.frameInterval}ms`
        );
    }

    startNewStreamMetrics() {
        this.streamCounter = (this.streamCounter || 0) + 1;
        this.streamMetrics = {
            id: this.streamCounter,
            finalized: false,
            lastRtf: null,
            lastAudioDurationMs: null,
            lastGenerationTimeMs: null,
            framesWithMetrics: 0,
            firstFrameAt: null,
            lastFrameAt: null,
            completeAt: null,
            framesReceived: 0,
            firstFrameIndex: null,
            maxFrameIndex: null,
            summary: null,
            summaryLogged: false,
        };
        this.frameOrderStats = {
            outOfOrder: 0,
            duplicates: 0,
            gaps: 0,
            lateDrops: 0,
        };
    }

    updateStreamMetrics(data) {
        if (data.rtf !== undefined) {
            const rtf = Number(data.rtf);
            if (Number.isFinite(rtf)) {
                this.streamMetrics.lastRtf = rtf;
                this.streamMetrics.framesWithMetrics += 1;
            }
        }
        if (data.audio_duration_ms !== undefined) {
            const duration = Number(data.audio_duration_ms);
            if (Number.isFinite(duration)) {
                this.streamMetrics.lastAudioDurationMs = duration;
            }
        }
        if (data.generation_time_ms !== undefined) {
            const genTime = Number(data.generation_time_ms);
            if (Number.isFinite(genTime)) {
                this.streamMetrics.lastGenerationTimeMs = genTime;
            }
        }

        const now = performance.now();
        if (this.streamMetrics.firstFrameAt === null) {
            this.streamMetrics.firstFrameAt = now;
        }
        this.streamMetrics.lastFrameAt = now;
        this.streamMetrics.framesReceived = this.framesReceived;

        const frameIndex = Number(data.frame_index);
        if (Number.isFinite(frameIndex)) {
            if (this.streamMetrics.firstFrameIndex === null) {
                this.streamMetrics.firstFrameIndex = frameIndex;
            }
            if (this.streamMetrics.maxFrameIndex === null || frameIndex > this.streamMetrics.maxFrameIndex) {
                this.streamMetrics.maxFrameIndex = frameIndex;
            }
        }
    }

    finalizeStreamMetrics(reason) {
        if (this.streamMetrics.finalized) return;
        this.streamMetrics.finalized = true;

        const finalRtf = this.streamMetrics.lastRtf;
        let durationMs = 0;
        if (this.streamMetrics.firstFrameIndex !== null && this.streamMetrics.maxFrameIndex !== null) {
            durationMs = (this.streamMetrics.maxFrameIndex - this.streamMetrics.firstFrameIndex + 1) * this.frameInterval;
        } else {
            durationMs = this.streamMetrics.framesReceived * this.frameInterval;
        }
        const fallbackDuration = this.streamMetrics.lastAudioDurationMs;
        if (!durationMs && fallbackDuration) {
            durationMs = fallbackDuration;
        }

        const firstFrameAt = this.streamMetrics.firstFrameAt;
        const lastFrameAt = this.streamMetrics.lastFrameAt;
        const completeAt = this.streamMetrics.completeAt;
        let generationMs = 0;
        if (firstFrameAt !== null) {
            const endAt = Math.max(
                completeAt ?? firstFrameAt,
                lastFrameAt ?? firstFrameAt,
            );
            generationMs = Math.max(0, endAt - firstFrameAt);
        }

        const observedRtf = durationMs > 0 && generationMs > 0 ? generationMs / durationMs : 0;

        this.streamMetrics.summary = {
            reason,
            observedRtf,
            ttsRtf: finalRtf,
            durationMs,
            generationMs,
        };

        if (!this.isPlaying && this.frameBuffer.size === 0) {
            this.emitFinalSummary();
        }
    }

    logStreamSummary({ reason, observedRtf, ttsRtf, durationMs, generationMs }) {
        const metrics = this.runMetrics;

        const vadLatencyMs = metrics.vadSpeechStartAt !== null && metrics.vadUtteranceAt !== null
            ? metrics.vadUtteranceAt - metrics.vadSpeechStartAt
            : null;

        const sttLatencyMs = metrics.sttStartAt !== null && metrics.sttEndAt !== null
            ? metrics.sttEndAt - metrics.sttStartAt
            : null;

        const llmFirstTokenMs = metrics.llmStartAt !== null && metrics.llmFirstTokenAt !== null
            ? metrics.llmFirstTokenAt - metrics.llmStartAt
            : null;

        const llmDurationMs = metrics.llmFirstTokenAt !== null && metrics.llmLastTokenAt !== null
            ? metrics.llmLastTokenAt - metrics.llmFirstTokenAt
            : null;

        const llmSpeed = llmDurationMs && metrics.llmTokenCount > 1
            ? (metrics.llmTokenCount - 1) / (llmDurationMs / 1000)
            : null;

        const ttsFirstAudioMs = metrics.ttsStartAt !== null && metrics.ttsFirstAudioAt !== null
            ? metrics.ttsFirstAudioAt - metrics.ttsStartAt
            : null;

        const musetalkFirstMs = metrics.ttsStartAt !== null && metrics.musetalkFirstFrameAt !== null
            ? metrics.musetalkFirstFrameAt - metrics.ttsStartAt
            : null;

        const hasMusetalkFrames = metrics.musetalkFirstFrameAt !== null && metrics.musetalkLastFrameAt !== null;
        const musetalkGenMs = hasMusetalkFrames ? metrics.musetalkLastFrameAt - metrics.musetalkFirstFrameAt : null;
        const musetalkFrames = hasMusetalkFrames && metrics.musetalkFirstFrameIndex !== null && metrics.musetalkLastFrameIndex !== null
            ? metrics.musetalkLastFrameIndex - metrics.musetalkFirstFrameIndex + 1
            : null;
        const musetalkDurationMs = musetalkFrames ? musetalkFrames * this.frameInterval : null;
        
        // Calculate effective FPS instead of RTF
        const musetalkEffFps = musetalkGenMs && musetalkFrames ? (musetalkFrames / musetalkGenMs * 1000) : null;
        
        // Calculate average frame arrival gap
        const arrivalGaps = this.frameArrivalStats.gapSamples;
        const avgArrivalGapMs = arrivalGaps.length > 0 
            ? arrivalGaps.reduce((a, b) => a + b, 0) / arrivalGaps.length 
            : null;
        const effectiveArrivalFps = avgArrivalGapMs ? (1000 / avgArrivalGapMs) : null;

        const ttsRtfText = this.formatRate(ttsRtf, '');
        
        // KEY METRIC: utterance end to first AV frame
        const utteranceToFirstAV = metrics.vadUtteranceAt !== null && metrics.firstAVFrameAt !== null
            ? metrics.firstAVFrameAt - metrics.vadUtteranceAt
            : null;

        this.metricLog(
            `Final (${reason}) | ⚡ First AV ${this.formatMs(utteranceToFirstAV)} | ` +
            `VAD ${this.formatMs(vadLatencyMs)} | STT ${this.formatMs(sttLatencyMs)} | ` +
            `LLM first ${this.formatMs(llmFirstTokenMs)} | LLM speed ${this.formatRate(llmSpeed, ' tok/s')} | ` +
            `TTS first ${this.formatMs(ttsFirstAudioMs)} | TTS RTF ${ttsRtfText} | ` +
            `MuseTalk first ${this.formatMs(musetalkFirstMs)} | MuseTalk eff_fps ${this.formatRate(musetalkEffFps, '')} | ` +
            `Arrival eff_fps ${this.formatRate(effectiveArrivalFps, '')} | ` +
            `Underruns ${this.frameArrivalStats.underrunCount} | Bursts ${this.frameArrivalStats.burstCount} | ` +
            `Audio ${this.formatMs(durationMs)} | Gen ${this.formatMs(generationMs)}`
        );
    }

    emitFinalSummary() {
        if (!this.streamMetrics.summary || this.streamMetrics.summaryLogged) return;
        this.logStreamSummary(this.streamMetrics.summary);
        this.streamMetrics.summaryLogged = true;
    }

    getContiguousFrameCount() {
        if (this.nextFrameIndex === null) return 0;
        let count = 0;
        let idx = this.nextFrameIndex;
        while (this.frameBuffer.has(idx)) {
            count += 1;
            idx += 1;
        }
        return count;
    }

    getLowestBufferedIndex() {
        let lowest = null;
        for (const key of this.frameBuffer.keys()) {
            if (lowest === null || key < lowest) {
                lowest = key;
            }
        }
        return lowest;
    }

    // ============ WebSocket ============

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.log('Connecting to:', wsUrl);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.log('WebSocket connected');
            this.updateConnectionStatus('connected');
        };

        this.ws.onclose = () => {
            this.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            setTimeout(() => this.connect(), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'connected':
                this.isConnected = true;
                this.connectionId = data.connection_id;
                this.log('Connected with ID:', this.connectionId);
                break;

            case 'musetalk_ready':
                this.videoEnabled = data.success;
                if (data.success) {
                    this.updateAvatarStatus('ready', 'მზადაა');
                    this.elements.avatarContainer?.classList.remove('disabled');
                    if (data.idle_frame) {
                        this.idleFrame = data.idle_frame;
                        this.displayFrame(data.idle_frame);
                    }
                    this.elements.avatarLoading?.classList.add('hidden');
                } else {
                    this.updateAvatarStatus('unavailable', 'მიუწვდომელია');
                    this.elements.avatarContainer?.classList.add('disabled');
                }
                break;

            case 'vad_status':
                this.updateVadStatus(data.status);
                break;

            case 'stt_start':
                // Preserve vadUtteranceAt since it was set just before this
                const savedVadUtteranceAt = this.runMetrics.vadUtteranceAt;
                this.resetRunMetrics();
                this.runMetrics.vadUtteranceAt = savedVadUtteranceAt;
                this.runMetrics.sttStartAt = performance.now();
                this.updateVadStatus('processing');
                this.setVadStatusText('ტრანსკრიფცია...');
                break;

            case 'stt_complete':
                this.runMetrics.sttEndAt = performance.now();
                const sttLatencyMs = this.runMetrics.sttStartAt !== null
                    ? this.runMetrics.sttEndAt - this.runMetrics.sttStartAt
                    : null;
                this.metricLog(`STT complete in ${this.formatMs(sttLatencyMs)}`);
                this.addMessage('user', data.text);
                this.setVadStatusText(this.isRecording ? 'მოსმენა...' : 'დააჭირეთ მიკროფონს საუბრის დასაწყებად');
                break;

            case 'llm_start':
                this.isGenerating = true;
                this.streamComplete = false;
                this.lastActivityTime = performance.now();
                this.startGenerationWatchdog();  // Start watchdog timer
                this.showStopButton();
                this.currentAssistantMessage = this.addMessage('assistant', '', true);
                this.wordsSpoken = [];
                this.lastLoggedWord = null;
                this.resetPlaybackState();
                this.prepareReplayCapture();
                this.startNewStreamMetrics();
                this.runMetrics.llmStartAt = performance.now();
                this.runMetrics.llmFirstTokenAt = null;
                this.runMetrics.llmLastTokenAt = null;
                this.runMetrics.llmTokenCount = 0;
                this.runMetrics.ttsStartAt = null;
                this.runMetrics.ttsFirstAudioAt = null;
                this.runMetrics.ttsFirstFrameIndex = null;
                this.runMetrics.musetalkFirstFrameAt = null;
                this.runMetrics.musetalkLastFrameAt = null;
                this.runMetrics.musetalkFirstFrameIndex = null;
                this.runMetrics.musetalkLastFrameIndex = null;
                this.applyInitialBufferConfig();
                this.updateAvatarStatus('loading', 'იტვირთება...');
                break;

            case 'llm_token':
                this.lastActivityTime = performance.now();  // Update activity
                if (this.runMetrics.llmFirstTokenAt === null) {
                    this.runMetrics.llmFirstTokenAt = performance.now();
                    const firstTokenMs = this.runMetrics.llmStartAt !== null
                        ? this.runMetrics.llmFirstTokenAt - this.runMetrics.llmStartAt
                        : null;
                    this.metricLog(`LLM first token in ${this.formatMs(firstTokenMs)}`);
                }
                this.runMetrics.llmTokenCount += 1;
                this.runMetrics.llmLastTokenAt = performance.now();
                if (this.currentAssistantMessage) {
                    this.updateAssistantMessage(data.full_text);
                }
                break;

            case 'llm_complete':
                if (this.runMetrics.llmLastTokenAt === null) {
                    this.runMetrics.llmLastTokenAt = performance.now();
                }
                const llmDurationMs = this.runMetrics.llmFirstTokenAt !== null && this.runMetrics.llmLastTokenAt !== null
                    ? this.runMetrics.llmLastTokenAt - this.runMetrics.llmFirstTokenAt
                    : null;
                const llmSpeed = llmDurationMs && this.runMetrics.llmTokenCount > 1
                    ? (this.runMetrics.llmTokenCount - 1) / (llmDurationMs / 1000)
                    : null;
                this.metricLog(`LLM done | tokens ${this.runMetrics.llmTokenCount} | speed ${this.formatRate(llmSpeed, ' tok/s')}`);
                if (this.currentAssistantMessage) {
                    this.updateAssistantMessage(data.text, true);
                }
                break;

            case 'tts_start':
                this.runMetrics.ttsStartAt = performance.now();
                this.updateAvatarStatus('speaking', 'საუბრობს');
                break;

            case 'synced_av_frame':
                this.lastActivityTime = performance.now();  // Update activity
                this.handleAVFrame(data);
                break;

            case 'tts_complete':
            case 'video_complete':
                this.stopGenerationWatchdog();  // Stop watchdog timer
                this.streamMetrics.completeAt = performance.now();
                this.finalizeStreamMetrics(data.type);
                this.streamComplete = true;
                this.isGenerating = false;
                this.hideStopButton();
                this.finalizeReplayCapture();

                // If still buffering or paused, force start/resume playback
                if ((this.isBuffering || this.isPaused) && this.frameBuffer.size > 0) {
                    this.log('Starting/resuming playback on stream complete');
                    this.isPaused = false;
                    this.startPlayback();
                }
                break;

            case 'error':
                this.log('Error:', data.message);
                this.stopGenerationWatchdog();  // Stop watchdog timer
                this.isGenerating = false;
                this.hideStopButton();
                this.stopPlayback();
                this.updateAvatarStatus('error', 'შეცდომა');
                break;
        }
    }

    sendMessage(type, payload = {}) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, ...payload }));
        }
    }

    // ============ Audio Context ============

    async initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.audioSampleRate
            });
            this.log('AudioContext initialized, sample rate:', this.audioContext.sampleRate);
        } catch (e) {
            console.error('Failed to initialize AudioContext:', e);
        }
    }

    // ============ Recording ============

    async startRecording() {
        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                this.setVadStatusText('ბრაუზერი ვერ ხსნის მიკროფონს');
                return;
            }

            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
            });

            this.recordingContext = new AudioContext({ sampleRate: 16000 });
            const source = this.recordingContext.createMediaStreamSource(this.mediaStream);
            const processor = this.recordingContext.createScriptProcessor(512, 1, 1);

            const BATCH_SIZE = 5;
            let audioBatch = [];

            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;

                audioBatch.push(new Float32Array(e.inputBuffer.getChannelData(0)));

                if (audioBatch.length >= BATCH_SIZE) {
                    if (this.ws?.readyState === WebSocket.OPEN && this.ws.bufferedAmount < 256 * 1024) {
                        const totalSamples = audioBatch.reduce((sum, arr) => sum + arr.length, 0);
                        const combined = new Float32Array(totalSamples);
                        let offset = 0;
                        for (const chunk of audioBatch) {
                            combined.set(chunk, offset);
                            offset += chunk.length;
                        }
                        this.ws.send(combined.buffer);
                    }
                    audioBatch = [];
                }
            };

            source.connect(processor);
            processor.connect(this.recordingContext.destination);

            this.audioSource = source;
            this.audioProcessor = processor;
            this.isRecording = true;

            this.sendMessage('recording_start');
        } catch (e) {
            console.error('Failed to start recording:', e);
            this.setVadStatusText('მიკროფონზე წვდომა უარყოფილია');
        }
    }

    stopRecording() {
        this.isRecording = false;
        this.sendMessage('recording_stop');

        this.audioProcessor?.disconnect();
        this.audioSource?.disconnect();
        this.mediaStream?.getTracks().forEach(track => track.stop());
        this.recordingContext?.close();

        this.audioProcessor = null;
        this.audioSource = null;
        this.mediaStream = null;
        this.recordingContext = null;

        // Force reset VAD visual state
        this.resetVadVisualState();

    }

    resetVadVisualState() {
        // Reset all VAD-related visual elements
        this.elements.vadIndicator?.classList.remove('speaking', 'processing');
        this.elements.vadStatus?.classList.remove('speaking', 'processing');
        this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
            this.elements.micBtn?.classList.remove('active');
        } else {
            if (this.audioContext?.state === 'suspended') {
                await this.audioContext.resume();
            }
            await this.startRecording();
            this.elements.micBtn?.classList.add('active');
            this.setVadStatusText('მოსმენა...');
        }
    }

    // ============ AV Frame Handling ============

    resetPlaybackState() {
        this.stopPlayback();
        this.frameBuffer = new Map();
        this.nextFrameIndex = null;
        this.highestFrameIndex = null;
        this.missingFrameSince = null;
        this.underrunLogged = false;
        this.isBuffering = true;
        this.isPlaying = false;
        this.isPaused = false;
        this.isReplaying = false;
        this.framesPlayed = 0;
        this.framesReceived = 0;
        this.lastVideoFrame = null;
        this.nextAudioTime = 0;
        
        // Reset frame arrival stats
        this.frameArrivalStats = {
            lastArrivalTime: null,
            gapSamples: [],
            maxSamples: 50,
            underrunCount: 0,
            burstCount: 0,
        };
    }

    prepareReplayCapture() {
        this.pendingReplayFrames = new Map();
        this.updateReplayButtonState();
    }

    finalizeReplayCapture() {
        if (this.pendingReplayFrames.size > 0) {
            this.replayFrames = this.pendingReplayFrames;
        }
        this.pendingReplayFrames = new Map();
        this.updateReplayButtonState();
    }

    updateReplayButtonState() {
        const replayBtn = this.elements.replayBtn;
        if (!replayBtn) return;

        const hasReplay = this.replayFrames && this.replayFrames.size > 0;
        if (!hasReplay) {
            replayBtn.classList.add('hidden');
            replayBtn.disabled = true;
            return;
        }

        replayBtn.classList.remove('hidden');
        const disabled = this.isGenerating || this.isPlaying || this.isReplaying;
        replayBtn.disabled = disabled;
    }

    replayLastStream() {
        if (!this.replayFrames || this.replayFrames.size === 0) {
            this.log('Replay requested but no stored stream');
            return;
        }
        if (this.isPlaying || this.isGenerating) {
            this.log('Replay blocked: currently playing or generating');
            return;
        }

        this.resetPlaybackState();
        this.isReplaying = true;
        this.streamComplete = true;
        this.isGenerating = false;
        this.wordsSpoken = [];

        this.frameBuffer = new Map(this.replayFrames);
        this.framesReceived = this.frameBuffer.size;
        this.nextFrameIndex = this.getLowestBufferedIndex();
        this.log(`Replaying last stream | Frames: ${this.framesReceived}`);
        this.updateReplayButtonState();
        this.startPlayback();
    }

    handleAVFrame(data) {
        const frameIndex = Number(data.frame_index);
        if (!Number.isFinite(frameIndex)) {
            this.log('Invalid frame index, skipping frame:', data.frame_index);
            return;
        }

        if (this.nextFrameIndex === null) {
            this.nextFrameIndex = frameIndex;
            this.log(`First frame index: ${frameIndex}`);
        } else if (!this.isPlaying && frameIndex < this.nextFrameIndex) {
            this.nextFrameIndex = frameIndex;
        }

        if (this.highestFrameIndex !== null) {
            if (frameIndex < this.highestFrameIndex) {
                this.frameOrderStats.outOfOrder += 1;
                if (this.frameOrderStats.outOfOrder <= 5 || this.frameOrderStats.outOfOrder % 25 === 0) {
                    this.log(
                        `Out-of-order frame: ${frameIndex} < ${this.highestFrameIndex} (count ${this.frameOrderStats.outOfOrder})`
                    );
                }
            } else if (frameIndex > this.highestFrameIndex + 1) {
                this.frameOrderStats.gaps += 1;
                this.log(
                    `Frame gap detected: expected ${this.highestFrameIndex + 1}, got ${frameIndex} (gap count ${this.frameOrderStats.gaps})`
                );
            }
        }

        if (this.highestFrameIndex === null || frameIndex > this.highestFrameIndex) {
            this.highestFrameIndex = frameIndex;
        }

        if (this.frameBuffer.has(frameIndex)) {
            this.frameOrderStats.duplicates += 1;
            if (this.frameOrderStats.duplicates <= 5 || this.frameOrderStats.duplicates % 25 === 0) {
                this.log(`Duplicate frame dropped: ${frameIndex} (count ${this.frameOrderStats.duplicates})`);
            }
            return;
        }

        if (this.isPlaying && this.nextFrameIndex !== null && frameIndex < this.nextFrameIndex) {
            this.frameOrderStats.lateDrops += 1;
            if (this.frameOrderStats.lateDrops <= 5 || this.frameOrderStats.lateDrops % 25 === 0) {
                this.log(`Late frame dropped: ${frameIndex} < ${this.nextFrameIndex} (count ${this.frameOrderStats.lateDrops})`);
            }
            return;
        }

        // Decode audio
        let audioFloat = null;
        if (data.audio) {
            audioFloat = this.decodeAudioBase64(data.audio);
            this.logDecodedAudioStats(audioFloat, frameIndex, data.audio.length);
        }

        // KEY METRIC: Time from utterance end to first AV frame
        if (this.runMetrics.firstAVFrameAt === null && (audioFloat || data.frame)) {
            this.runMetrics.firstAVFrameAt = performance.now();
            const utteranceToFirstAV = this.runMetrics.vadUtteranceAt !== null
                ? this.runMetrics.firstAVFrameAt - this.runMetrics.vadUtteranceAt
                : null;
            this.metricLog(`⚡ FIRST AV FRAME: ${this.formatMs(utteranceToFirstAV)} after utterance end`);
        }

        if (audioFloat && this.runMetrics.ttsFirstAudioAt === null) {
            this.runMetrics.ttsFirstAudioAt = performance.now();
            this.runMetrics.ttsFirstFrameIndex = frameIndex;
            const ttsFirstMs = this.runMetrics.ttsStartAt !== null
                ? this.runMetrics.ttsFirstAudioAt - this.runMetrics.ttsStartAt
                : null;
            this.metricLog(`TTS first audio in ${this.formatMs(ttsFirstMs)}`);
        }

        if (data.frame && this.runMetrics.musetalkFirstFrameAt === null) {
            this.runMetrics.musetalkFirstFrameAt = performance.now();
            this.runMetrics.musetalkFirstFrameIndex = frameIndex;
            const mtFirstMs = this.runMetrics.ttsStartAt !== null
                ? this.runMetrics.musetalkFirstFrameAt - this.runMetrics.ttsStartAt
                : null;
            this.metricLog(`MuseTalk first frame in ${this.formatMs(mtFirstMs)}`);
        }
        if (data.frame) {
            this.runMetrics.musetalkLastFrameAt = performance.now();
            this.runMetrics.musetalkLastFrameIndex = frameIndex;
        }

        const payload = {
            audio: audioFloat,
            frame: data.frame,
            frameIndex: frameIndex,
            timestampMs: data.timestamp_ms,
            word: data.word || '',
            rtf: data.rtf,
            generation_time_ms: data.generation_time_ms,
            audio_duration_ms: data.audio_duration_ms,
        };

        // Store frame
        this.frameBuffer.set(frameIndex, payload);
        this.pendingReplayFrames.set(frameIndex, payload);
        this.framesReceived++;
        
        // Track frame arrival timing
        const now = performance.now();
        if (this.frameArrivalStats.lastArrivalTime !== null) {
            const gapMs = now - this.frameArrivalStats.lastArrivalTime;
            this.frameArrivalStats.gapSamples.push(gapMs);
            if (this.frameArrivalStats.gapSamples.length > this.frameArrivalStats.maxSamples) {
                this.frameArrivalStats.gapSamples.shift();
            }
            // Track bursts (frames arriving very close together)
            if (gapMs < 5) {
                this.frameArrivalStats.burstCount++;
            }
        }
        this.frameArrivalStats.lastArrivalTime = now;
        
        this.updateStreamMetrics(data);

        // Update metrics display
        if (data.frame_index !== undefined) {
            this.updateVideoMetrics();
        }
        if (data.rtf !== undefined) {
            this.updateMetrics(data);
        }

        // Check if we should start playback
        if (this.isBuffering) {
            const contiguous = this.getContiguousFrameCount();
            if (contiguous >= this.bufferConfig.minFrames) {
                this.log(
                    `Buffer full (${contiguous}/${this.bufferConfig.minFrames} contiguous, ${this.frameBuffer.size} total), starting playback`
                );
                this.startPlayback();
            } else {
                if (contiguous === 0 && this.frameBuffer.size > 0 && this.nextFrameIndex !== null) {
                    if (!this.missingFrameSince) {
                        this.missingFrameSince = performance.now();
                    }
                    const waitMs = performance.now() - this.missingFrameSince;
                    const lowestIndex = this.getLowestBufferedIndex();
                    if (lowestIndex !== null && this.nextFrameIndex < lowestIndex && waitMs >= this.bufferConfig.missingFrameToleranceMs) {
                        this.log(`Buffering skip missing frame ${this.nextFrameIndex} -> ${lowestIndex} after ${waitMs.toFixed(0)}ms`);
                        this.nextFrameIndex = lowestIndex;
                        this.missingFrameSince = null;
                    }
                } else {
                    this.missingFrameSince = null;
                }

                if (this.framesReceived % 12 === 0) {
                    this.log(`Buffering: ${contiguous}/${this.bufferConfig.minFrames} contiguous (${this.frameBuffer.size} total)`);
                }
            }
        }
    }

    startPlayback() {
        if (this.isPlaying) return;

        this.isBuffering = false;
        this.isPlaying = true;
        if (this.nextFrameIndex === null) {
            this.nextFrameIndex = this.getLowestBufferedIndex();
        }
        this.nextAudioTime = this.audioContext?.currentTime || 0;

        // Initialize playback stats
        this.playbackStats.startTime = performance.now();
        this.playbackStats.framesAtStart = this.getContiguousFrameCount();

        const bufferedTotal = this.frameBuffer.size;
        const contiguous = this.getContiguousFrameCount();
        this.log(
            `Starting playback | Buffer: ${bufferedTotal} total, ${contiguous} contiguous | Min: ${this.bufferConfig.minFrames} | Next index: ${this.nextFrameIndex}`
        );
        this.updateAvatarStatus('speaking', 'საუბრობს');
        this.updateReplayButtonState();

        this.playNextFrame();
    }

    playNextFrame() {
        if (!this.isPlaying) return;

        if (this.nextFrameIndex === null && this.frameBuffer.size > 0) {
            this.nextFrameIndex = this.getLowestBufferedIndex();
        }

        if (this.nextFrameIndex !== null && this.frameBuffer.has(this.nextFrameIndex)) {
            const frame = this.frameBuffer.get(this.nextFrameIndex);
            this.frameBuffer.delete(this.nextFrameIndex);
            this.nextFrameIndex += 1;
            this.framesPlayed++;
            this.underrunLogged = false;
            this.missingFrameSince = null;

            // Play audio
            if (frame.audio) {
                this.playAudio(frame.audio);
            }

            // Display video frame
            if (frame.frame) {
                this.displayFrame(frame.frame);
                this.lastVideoFrame = frame.frame;
            } else if (this.lastVideoFrame) {
                this.displayFrame(this.lastVideoFrame);
            }

            // Update word highlighting
            if (frame.word && this.currentAssistantMessage) {
                if (this.lastLoggedWord !== frame.word) {
                    const elapsedMs = this.playbackStats.startTime
                        ? performance.now() - this.playbackStats.startTime
                        : null;
                    this.metricLog(`Word "${frame.word}" at ${this.formatMs(elapsedMs)}`);
                    this.lastLoggedWord = frame.word;
                }
                this.highlightWord(frame.word);
            }

            // Log every 25 frames (1 second of playback)
            if (this.framesPlayed % 25 === 0) {
                const elapsed = (performance.now() - this.playbackStats.startTime) / 1000;
                const expectedTime = this.framesPlayed * 0.04;  // 40ms per frame
                const drift = (elapsed - expectedTime) * 1000;
                this.log(`Played: ${this.framesPlayed} | Buffer: ${this.frameBuffer.size} | Drift: ${drift.toFixed(0)}ms | Received: ${this.framesReceived}`);
            }

            // Schedule next frame at 40ms intervals (25fps)
            this.playbackTimer = setTimeout(() => this.playNextFrame(), this.frameInterval);
            return;
        }

        // Buffer empty or missing expected frame
        if (this.streamComplete && this.frameBuffer.size === 0) {
            // All done
            const totalTime = (performance.now() - this.playbackStats.startTime) / 1000;
            this.log(`Playback complete | Played: ${this.framesPlayed} frames | Time: ${totalTime.toFixed(2)}s | Received: ${this.framesReceived}`);
            this.stopPlayback();
            this.updateAvatarStatus('ready', 'მზადაა');
            this.emitFinalSummary();

            // Show idle frame
            setTimeout(() => {
                if (this.idleFrame && !this.isPlaying) {
                    this.displayFrame(this.idleFrame);
                }
            }, 500);
            return;
        }

        if (this.frameBuffer.size > 0 && this.nextFrameIndex !== null) {
            if (!this.missingFrameSince) {
                this.missingFrameSince = performance.now();
            }
            const waitMs = performance.now() - this.missingFrameSince;
            const lowestIndex = this.getLowestBufferedIndex();
            if (lowestIndex !== null && this.nextFrameIndex < lowestIndex && waitMs >= this.bufferConfig.missingFrameToleranceMs) {
                this.log(`Skipping missing frame ${this.nextFrameIndex} -> ${lowestIndex} after ${waitMs.toFixed(0)}ms`);
                this.nextFrameIndex = lowestIndex;
                this.missingFrameSince = null;
            }
        } else {
            this.missingFrameSince = null;
        }

        // Buffer underrun - wait for more frames (log once)
        if (!this.underrunLogged) {
            this.log(`Buffer underrun at frame ${this.framesPlayed} | Waiting for more frames...`);
            this.underrunLogged = true;
            this.frameArrivalStats.underrunCount++;
        }

        // Check again soon
        this.playbackTimer = setTimeout(() => this.playNextFrame(), 20);
    }

    stopPlayback() {
        this.isPlaying = false;
        this.isBuffering = true;
        this.isPaused = false;
        this.missingFrameSince = null;
        this.underrunLogged = false;
        this.isReplaying = false;

        if (this.playbackTimer) {
            clearTimeout(this.playbackTimer);
            this.playbackTimer = null;
        }

        this.nextAudioTime = 0;
        this.updateReplayButtonState();
    }

    // ============ Generation Watchdog ============

    startGenerationWatchdog() {
        this.stopGenerationWatchdog();  // Clear any existing timer
        
        const WATCHDOG_CHECK_INTERVAL = 5000;  // Check every 5 seconds
        const INACTIVITY_TIMEOUT = 30000;       // Reset if no activity for 30 seconds
        
        this.generationTimeoutId = setInterval(() => {
            if (!this.isGenerating) {
                this.stopGenerationWatchdog();
                return;
            }
            
            const now = performance.now();
            const inactiveMs = now - this.lastActivityTime;
            
            if (inactiveMs > INACTIVITY_TIMEOUT) {
                this.log(`Generation watchdog: No activity for ${(inactiveMs / 1000).toFixed(1)}s, forcing reset`);
                this.handleGenerationTimeout();
            }
        }, WATCHDOG_CHECK_INTERVAL);
    }

    stopGenerationWatchdog() {
        if (this.generationTimeoutId) {
            clearInterval(this.generationTimeoutId);
            this.generationTimeoutId = null;
        }
    }

    handleGenerationTimeout() {
        this.stopGenerationWatchdog();
        
        // Reset generation state
        this.streamComplete = true;
        this.isGenerating = false;
        this.hideStopButton();
        
        // If we have frames, try to play them
        if (this.frameBuffer.size > 0) {
            this.log(`Timeout recovery: Playing ${this.frameBuffer.size} buffered frames`);
            this.isPaused = false;
            this.startPlayback();
        } else {
            this.stopPlayback();
            this.updateAvatarStatus('ready', 'მზადაა');
            
            // Show idle frame
            if (this.idleFrame) {
                this.displayFrame(this.idleFrame);
            }
        }
        
        // Notify user
        this.log('Generation timed out - state reset');
    }

    playAudio(audioFloat) {
        if (!this.audioContext || !audioFloat || audioFloat.length === 0) return;

        try {
            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume().catch(() => {});
            }

            const audioBuffer = this.audioContext.createBuffer(1, audioFloat.length, this.audioSampleRate);
            audioBuffer.getChannelData(0).set(audioFloat);

            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);

            const now = this.audioContext.currentTime;
            const startAt = this.nextAudioTime > now ? this.nextAudioTime : now;
            source.start(startAt);
            this.nextAudioTime = startAt + (audioFloat.length / this.audioSampleRate);
        } catch (e) {
            console.error('Failed to play audio:', e);
        }
    }

    decodeAudioBase64(audioBase64) {
        try {
            const binaryString = atob(audioBase64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            const rawBuffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);

            if (rawBuffer.byteLength % 4 === 0) {
                return new Float32Array(rawBuffer);
            }
            if (rawBuffer.byteLength % 2 === 0) {
                const int16 = new Int16Array(rawBuffer);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }
                return float32;
            }
            return null;
        } catch (e) {
            console.error('Failed to decode audio:', e);
            return null;
        }
    }

    logDecodedAudioStats(audioFloat, frameIndex, base64Len) {
        if (!audioFloat || audioFloat.length === 0) return;

        this.audioDecodeStats.count += 1;
        const count = this.audioDecodeStats.count;
        const expectedSamples = Math.round(this.audioSampleRate / this.videoFps);
        const shouldLog =
            count <= this.audioDecodeStats.logFirst ||
            count % this.audioDecodeStats.logEvery === 0 ||
            audioFloat.length !== expectedSamples;

        if (!shouldLog) return;

        let minVal = Infinity;
        let maxVal = -Infinity;
        let sumSq = 0;
        let nanCount = 0;

        for (let i = 0; i < audioFloat.length; i++) {
            const v = audioFloat[i];
            if (!Number.isFinite(v)) {
                nanCount += 1;
                continue;
            }
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
            sumSq += v * v;
        }

        if (minVal === Infinity) {
            minVal = 0;
            maxVal = 0;
        }

        const rmsVal = audioFloat.length > 0 ? Math.sqrt(sumSq / audioFloat.length) : 0;
        this.log(
            `Decoded audio ${count} frame=${frameIndex} samples=${audioFloat.length} ` +
            `expected=${expectedSamples} min=${minVal.toFixed(4)} max=${maxVal.toFixed(4)} ` +
            `rms=${rmsVal.toFixed(4)} nan=${nanCount} b64=${base64Len}`
        );
    }

    displayFrame(frameBase64) {
        if (this.elements.avatarImage) {
            this.elements.avatarImage.src = `data:image/jpeg;base64,${frameBase64}`;
        }
    }

    // ============ UI Updates ============

    updateConnectionStatus(status) {
        const statusEl = this.elements.connectionStatus;
        if (!statusEl) return;

        statusEl.className = 'connection-status ' + status;
        const textEl = statusEl.querySelector('.status-text');
        if (textEl) {
            textEl.textContent = status === 'connected' ? 'დაკავშირებულია' : 'გათიშულია';
        }
    }

    updateVadStatus(status) {
        const indicator = this.elements.vadIndicator;
        const statusEl = this.elements.vadStatus;
        const now = performance.now();

        // Always remove existing states first
        indicator?.classList.remove('speaking', 'processing');
        statusEl?.classList.remove('speaking', 'processing');

        const statusTexts = {
            'listening': 'მოსმენა...',
            'speech_start': 'საუბარი...',
            'speech_continue': 'საუბარი...',
            'speaking': 'საუბარი...',
            'utterance_complete': 'დამუშავება...',
            'processing': 'ტრანსკრიფცია...'
        };

        const isSpeaking = ['speaking', 'speech_start', 'speech_continue'].includes(status);
        if (isSpeaking && this.runMetrics.vadSpeechStartAt === null && !this.isGenerating) {
            this.resetRunMetrics();
        }

        if (isSpeaking) {
            if (this.runMetrics.vadSpeechStartAt === null) {
                this.runMetrics.vadSpeechStartAt = now;
            }
            indicator?.classList.add('speaking');
            statusEl?.classList.add('speaking');
        } else if (['processing', 'utterance_complete'].includes(status)) {
            statusEl?.classList.add('processing');
        }

        // Update text
        if (statusTexts[status]) {
            this.setVadStatusText(statusTexts[status]);
        } else if (status === 'listening' && !this.isRecording) {
            // If listening but not recording, show default text
            this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
        }

        if (status === 'utterance_complete') {
            this.runMetrics.vadUtteranceAt = now;
            const vadMs = this.runMetrics.vadSpeechStartAt !== null
                ? this.runMetrics.vadUtteranceAt - this.runMetrics.vadSpeechStartAt
                : null;
            this.metricLog(`VAD complete in ${this.formatMs(vadMs)}`);
        }
    }

    setVadStatusText(text) {
        if (this.elements.vadStatus) {
            this.elements.vadStatus.textContent = text;
        }
    }

    updateAvatarStatus(status, text) {
        const statusEl = this.elements.avatarStatus;
        if (!statusEl) return;

        const statusIndicator = statusEl.querySelector('.status-indicator');
        const statusText = statusEl.querySelector('.status-text');

        statusEl.classList.remove('ready', 'speaking', 'unavailable', 'error', 'loading');
        statusEl.classList.add(status);

        statusIndicator?.classList.remove('ready', 'speaking', 'unavailable', 'error', 'loading');
        statusIndicator?.classList.add(status);

        if (statusText) statusText.textContent = text;

        if (this.elements.avatarContainer) {
            if (status === 'speaking') {
                this.elements.avatarContainer.classList.add('speaking');
            } else {
                this.elements.avatarContainer.classList.remove('speaking');
            }
        }
    }

    updateVideoMetrics() {
        if (this.elements.videoFrames) {
            this.elements.videoFrames.textContent = `${this.framesReceived} კადრი`;
        }
        if (this.elements.videoFps && this.framesPlayed > 0) {
            this.elements.videoFps.textContent = `${this.videoFps} FPS`;
        }
    }

    updateMetrics(data) {
        if (data.rtf !== undefined && this.elements.rtfValue) {
            this.elements.rtfValue.textContent = data.rtf.toFixed(3);
            this.elements.rtfValue.style.color = data.rtf < 1 ? 'var(--success)' : data.rtf < 1.5 ? 'var(--warning)' : 'var(--error)';
        }
        if (data.generation_time_ms !== undefined && this.elements.genTimeValue) {
            this.elements.genTimeValue.textContent = `${data.generation_time_ms.toFixed(0)} ms`;
        }
        if (data.audio_duration_ms !== undefined && this.elements.audioDurValue) {
            this.elements.audioDurValue.textContent = `${data.audio_duration_ms.toFixed(0)} ms`;
        }
        if (this.elements.currentWordValue) {
            this.elements.currentWordValue.textContent = data.word || '-';
        }
    }

    addMessage(role, text, isStreaming = false) {
        const welcomeMsg = this.elements.chatMessages?.querySelector('.welcome-message');
        welcomeMsg?.remove();

        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;

        const avatarSvg = role === 'user'
            ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
            : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>';

        messageEl.innerHTML = `
            <div class="message-avatar">${avatarSvg}</div>
            <div class="message-content">
                <span class="message-text">${text || ''}</span>
                ${isStreaming ? '<span class="typing-indicator"><span></span><span></span><span></span></span>' : ''}
            </div>
        `;

        this.elements.chatMessages?.appendChild(messageEl);
        this.scrollToBottom();

        return messageEl;
    }

    updateAssistantMessage(text, complete = false) {
        if (!this.currentAssistantMessage) return;

        const textEl = this.currentAssistantMessage.querySelector('.message-text');
        const typingEl = this.currentAssistantMessage.querySelector('.typing-indicator');

        if (textEl) textEl.textContent = text;
        if (complete && typingEl) typingEl.remove();

        this.scrollToBottom();
    }

    highlightWord(word) {
        if (!this.currentAssistantMessage || !word) return;

        if (!this.wordsSpoken.includes(word)) {
            this.wordsSpoken.push(word);
        }

        const textEl = this.currentAssistantMessage.querySelector('.message-text');
        if (!textEl) return;

        const text = textEl.textContent;
        const words = text.split(' ');
        const highlightedWords = words.map(w => {
            const cleanWord = w.replace(/[,.!?;:]/g, '');
            if (this.wordsSpoken.includes(cleanWord) || this.wordsSpoken.includes(w)) {
                return `<span class="word-spoken">${w}</span>`;
            }
            if (cleanWord === word || w === word) {
                return `<span class="word-highlight">${w}</span>`;
            }
            return w;
        });

        textEl.innerHTML = highlightedWords.join(' ');
    }

    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    showStopButton() {
        this.elements.stopBtn?.classList.remove('hidden');
    }

    hideStopButton() {
        this.elements.stopBtn?.classList.add('hidden');
    }

    stopGeneration() {
        this.sendMessage('stop_generation', {});
        this.stopPlayback();
        this.hideStopButton();
        this.isGenerating = false;
        this.streamComplete = true;
        this.updateAvatarStatus('ready', 'მზადაა');
    }

    // ============ Settings ============

    openSettings() {
        this.elements.settingsPanel?.classList.remove('hidden');
        this.loadSettingsToUI();
    }

    closeSettings() {
        this.elements.settingsPanel?.classList.add('hidden');
    }

    loadSettingsToUI() {
        const setVal = (id, value) => {
            if (this.elements[id]) this.elements[id].value = value;
            if (this.elements[id + 'Value']) this.elements[id + 'Value'].textContent = value;
        };

        setVal('speechThreshold', this.config.vad.speech_threshold_ms);
        setVal('silenceThreshold', this.config.vad.silence_threshold_ms);
        setVal('earlySilenceThreshold', this.config.vad.early_silence_threshold_ms ?? 500);
        setVal('llmTemperature', this.config.llm.temperature);
        setVal('llmTopP', this.config.llm.top_p);
        setVal('llmMaxTokens', this.config.llm.max_new_tokens);
        if (this.elements.systemPrompt) this.elements.systemPrompt.value = this.config.llm.system_prompt;
        setVal('ttsBackboneTemp', this.config.tts.backbone_temperature);
        setVal('ttsBackboneTopP', this.config.tts.backbone_top_p);
        setVal('ttsDepthTemp', this.config.tts.depth_temperature);
        setVal('ttsDepthTopP', this.config.tts.depth_top_p);
        setVal('musetalkBatchSize', this.config.musetalk?.batch_size ?? 8);
        setVal('musetalkLookaheadChunks', this.config.musetalk?.lookahead_chunks ?? 2);
    }

    async saveSettings() {
        this.config.vad.speech_threshold_ms = parseInt(this.elements.speechThreshold?.value || 200);
        this.config.vad.silence_threshold_ms = parseInt(this.elements.silenceThreshold?.value || 1500);
        this.config.vad.early_silence_threshold_ms = parseInt(this.elements.earlySilenceThreshold?.value || 500);
        this.config.llm.temperature = parseFloat(this.elements.llmTemperature?.value || 0.7);
        this.config.llm.top_p = parseFloat(this.elements.llmTopP?.value || 0.9);
        this.config.llm.max_new_tokens = parseInt(this.elements.llmMaxTokens?.value || 512);
        this.config.llm.system_prompt = this.elements.systemPrompt?.value || '';
        this.config.tts.backbone_temperature = parseFloat(this.elements.ttsBackboneTemp?.value || 0.8);
        this.config.tts.backbone_top_p = parseFloat(this.elements.ttsBackboneTopP?.value || 0.9);
        this.config.tts.depth_temperature = parseFloat(this.elements.ttsDepthTemp?.value || 0.8);
        this.config.tts.depth_top_p = parseFloat(this.elements.ttsDepthTopP?.value || 0.9);
        this.config.musetalk.batch_size = parseInt(this.elements.musetalkBatchSize?.value || 8);
        this.config.musetalk.lookahead_chunks = parseInt(this.elements.musetalkLookaheadChunks?.value || 2);

        localStorage.setItem('voiceAssistantConfig', JSON.stringify(this.config));

        try {
            await fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.config)
            });
            this.log('Settings saved');
        } catch (e) {
            console.error('Failed to save settings:', e);
        }

        this.closeSettings();
    }

    resetSettings() {
        this.config = {
            vad: { speech_threshold_ms: 200, silence_threshold_ms: 1500 },
            llm: { temperature: 0.7, top_p: 0.9, max_new_tokens: 512, system_prompt: 'თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი' },
            tts: { backbone_temperature: 0.01, backbone_top_p: 0.999, depth_temperature: 0.01, depth_top_p: 0.999 },
            musetalk: { batch_size: 8, lookahead_chunks: 2 },
        };
        this.loadSettingsToUI();
    }

    loadConfig() {
        const saved = localStorage.getItem('voiceAssistantConfig');
        if (saved) {
            try {
                this.config = { ...this.config, ...JSON.parse(saved) };
            } catch (e) {
                console.error('Failed to load config:', e);
            }
        }
        this.loadSettingsToUI();

        fetch('/config')
            .then(res => res.json())
            .then(serverConfig => {
                if (serverConfig.vad) this.config.vad = { ...this.config.vad, ...serverConfig.vad };
                if (serverConfig.llm) this.config.llm = { ...this.config.llm, ...serverConfig.llm };
                if (serverConfig.tts) this.config.tts = { ...this.config.tts, ...serverConfig.tts };
                if (serverConfig.musetalk) this.config.musetalk = { ...this.config.musetalk, ...serverConfig.musetalk };
                this.loadSettingsToUI();
            })
            .catch(e => this.log('Could not load server config:', e));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new VoiceAssistant();
});
