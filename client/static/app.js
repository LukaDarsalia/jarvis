/**
 * Minimal Voice Assistant Web Client.
 * - Sends mic audio to server for VAD/STT/LLM/TTS
 * - Streams LLM text to chat
 * - Plays synced AV frames in strict index order (0,1,2,...)
 */

class SimpleVoiceAssistant {
    constructor() {
        this.DEBUG = true;

        // WebSocket
        this.ws = null;
        this.isConnected = false;
        this.connectionId = null;

        // Recording
        this.isRecording = false;
        this.mediaStream = null;
        this.recordingContext = null;
        this.audioProcessor = null;
        this.audioSource = null;

        // Audio playback
        this.audioContext = null;
        this.audioSampleRate = 24000;
        this.nextAudioTime = 0;

        // AV sync
        this.expectedIndex = 0;
        this.pendingFrames = new Map();
        this.streamComplete = false;
        this.lastVideoFrame = null;

        // WebCodecs
        this.useAudioDecoder = false;
        this.useVideoDecoder = false;
        this.audioDecoder = null;
        this.videoDecoder = null;
        this.audioDecodeQueue = [];
        this.videoDecodeQueue = [];
        this.videoCanvas = null;
        this.videoCtx = null;

        // UI
        this.currentAssistantMessage = null;
        this.elements = {};

        this.init();
    }

    log(...args) {
        if (this.DEBUG) {
            console.log(`[VA ${new Date().toISOString().substr(11, 12)}]`, ...args);
        }
    }

    async init() {
        this.cacheElements();
        this.bindEvents();
        await this.initAudioContext();
        await this.initDecoders();
        this.connect();
    }

    cacheElements() {
        const get = (id) => {
            const el = document.getElementById(id);
            if (!el) {
                throw new Error(`Missing required element: ${id}`);
            }
            return el;
        };
        const query = (root, selector, label) => {
            const el = root.querySelector(selector);
            if (!el) {
                throw new Error(`Missing required element: ${label}`);
            }
            return el;
        };

        const connectionStatus = get('connectionStatus');
        const avatarStatus = get('avatarStatus');
        const chatMessages = get('chatMessages');

        this.elements = {
            connectionStatus,
            connectionStatusText: query(connectionStatus, '.status-text', 'connectionStatusText'),
            avatarStatus,
            avatarStatusIndicator: query(avatarStatus, '.status-indicator', 'avatarStatusIndicator'),
            avatarStatusText: query(avatarStatus, '.status-text', 'avatarStatusText'),
            avatarImage: get('avatarImage'),
            avatarLoading: get('avatarLoading'),
            vadIndicator: get('vadIndicator'),
            vadStatus: get('vadStatus'),
            chatMessages,
            welcomeMessage: query(chatMessages, '.welcome-message', 'welcomeMessage'),
            micBtn: get('micBtn'),
            stopBtn: get('stopBtn'),
        };
    }

    bindEvents() {
        this.elements.micBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.stopBtn.addEventListener('click', () => this.stopGeneration());
    }

    async initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.audioSampleRate,
            });
            this.log('AudioContext initialized, sample rate:', this.audioContext.sampleRate);
        } catch (e) {
            console.error('Failed to initialize AudioContext:', e);
        }
    }

    async initDecoders() {
        const audioConfig = {
            codec: 'pcm-f32le',
            sampleRate: this.audioSampleRate,
            numberOfChannels: 1,
        };
        const videoConfig = { codec: 'jpeg' };

        if (window.AudioDecoder) {
            const support = await AudioDecoder.isConfigSupported(audioConfig).catch(() => null);
            if (support && support.supported) {
                try {
                    this.audioDecoder = new AudioDecoder({
                        output: (audioData) => this.onAudioDecoded(audioData),
                        error: (err) => {
                            console.error('AudioDecoder error:', err);
                            this.useAudioDecoder = false;
                            this.audioDecoder = null;
                            this.audioDecodeQueue = [];
                        },
                    });
                    this.audioDecoder.configure(audioConfig);
                    this.useAudioDecoder = true;
                } catch (e) {
                    console.warn('AudioDecoder unavailable:', e);
                    this.audioDecoder = null;
                    this.useAudioDecoder = false;
                }
            } else {
                this.useAudioDecoder = false;
                this.audioDecoder = null;
                this.log('AudioDecoder not supported for pcm-f32le, using PCM fallback.');
            }
        }

        if (window.VideoDecoder) {
            const support = await VideoDecoder.isConfigSupported(videoConfig).catch(() => null);
            if (support && support.supported) {
                try {
                    this.videoDecoder = new VideoDecoder({
                        output: (videoFrame) => this.onVideoDecoded(videoFrame),
                        error: (err) => {
                            console.error('VideoDecoder error:', err);
                            this.useVideoDecoder = false;
                            this.videoDecoder = null;
                            if (this.videoCanvas) {
                                this.videoCanvas.remove();
                                this.videoCanvas = null;
                                this.videoCtx = null;
                            }
                            this.elements.avatarImage.classList.remove('hidden');
                        },
                    });
                    this.videoDecoder.configure(videoConfig);
                    this.useVideoDecoder = true;
                    this.setupVideoCanvas();
                } catch (e) {
                    console.warn('VideoDecoder unavailable:', e);
                    this.videoDecoder = null;
                    this.useVideoDecoder = false;
                }
            } else {
                this.useVideoDecoder = false;
                this.videoDecoder = null;
                this.log('VideoDecoder not supported for jpeg, using image fallback.');
            }
        }
    }

    setupVideoCanvas() {
        const canvas = document.createElement('canvas');
        canvas.className = this.elements.avatarImage.className;
        canvas.width = this.elements.avatarImage.width || 512;
        canvas.height = this.elements.avatarImage.height || 512;
        const parent = this.elements.avatarImage.parentElement;
        if (parent) {
            parent.insertBefore(canvas, this.elements.avatarImage);
            this.elements.avatarImage.classList.add('hidden');
        }
        this.videoCanvas = canvas;
        this.videoCtx = canvas.getContext('2d');
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.log('Connecting to:', wsUrl);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateConnectionStatus('connected');
            this.log('WebSocket connected');
        };

        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            this.log('WebSocket disconnected');
            setTimeout(() => this.connect(), 2000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
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
                this.connectionId = data.connection_id || null;
                this.log('Connected with ID:', this.connectionId);
                break;

            case 'musetalk_ready':
                this.elements.avatarLoading.classList.add('hidden');
                if (data.success) {
                    this.updateAvatarStatus('ready', 'მზადაა');
                    if (data.idle_frame) {
                        this.displayJpegFrame(data.idle_frame);
                    }
                } else {
                    this.updateAvatarStatus('unavailable', 'მიუწვდომელია');
                }
                break;

            case 'vad_status':
                this.updateVadStatus(data.status);
                break;

            case 'stt_start':
                this.setVadStatusText('ტრანსკრიფცია...');
                break;

            case 'stt_complete':
                this.addMessage('user', data.text || '');
                this.setVadStatusText('მოსმენა...');
                break;

            case 'llm_start':
                this.isGenerating = true;
                this.showStopButton();
                this.currentAssistantMessage = this.addMessage('assistant', '', true);
                this.updateAvatarStatus('loading', 'იტვირთება...');
                break;

            case 'llm_token':
                this.updateAssistantMessage(data.full_text || '');
                break;

            case 'llm_complete':
                this.updateAssistantMessage(data.text || '', true);
                break;

            case 'tts_start':
                this.resetStreamState();
                this.updateAvatarStatus('speaking', 'საუბრობს');
                break;

            case 'synced_av_frame':
                this.handleAVFrame(data);
                break;

            case 'tts_complete':
            case 'video_complete':
                this.streamComplete = true;
                this.maybeFinishStream();
                break;

            case 'error':
                this.log('Error:', data.message);
                this.isGenerating = false;
                this.hideStopButton();
                this.updateAvatarStatus('error', 'შეცდომა');
                break;
        }
    }

    // ============ Recording ============

    async startRecording() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                this.setVadStatusText('ბრაუზერი ვერ ხსნის მიკროფონს');
                return;
            }

            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
            });

            this.recordingContext = new AudioContext({ sampleRate: 16000 });
            const source = this.recordingContext.createMediaStreamSource(this.mediaStream);
            const processor = this.recordingContext.createScriptProcessor(512, 1, 1);

            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
                if (this.ws.bufferedAmount > 256 * 1024) return;

                const chunk = new Float32Array(e.inputBuffer.getChannelData(0));
                this.ws.send(chunk.buffer);
            };

            source.connect(processor);
            processor.connect(this.recordingContext.destination);

            this.audioSource = source;
            this.audioProcessor = processor;
            this.isRecording = true;

            this.sendMessage('recording_start');
            this.setVadStatusText('მოსმენა...');
            this.elements.micBtn.classList.add('active');
        } catch (e) {
            console.error('Failed to start recording:', e);
            this.setVadStatusText('მიკროფონზე წვდომა უარყოფილია');
        }
    }

    stopRecording() {
        this.isRecording = false;
        this.sendMessage('recording_stop');

        if (this.audioProcessor) this.audioProcessor.disconnect();
        if (this.audioSource) this.audioSource.disconnect();
        if (this.mediaStream) this.mediaStream.getTracks().forEach((track) => track.stop());
        if (this.recordingContext) this.recordingContext.close();

        this.audioProcessor = null;
        this.audioSource = null;
        this.mediaStream = null;
        this.recordingContext = null;

        this.elements.micBtn.classList.remove('active');
        this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
            return;
        }

        if (this.audioContext && this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        await this.startRecording();
    }

    stopGeneration() {
        this.sendMessage('stop_generation');
        this.hideStopButton();
        this.isGenerating = false;
        this.resetStreamState();
        this.updateAvatarStatus('ready', 'მზადაა');
    }

    // ============ AV Playback ============

    resetStreamState() {
        this.expectedIndex = 0;
        this.pendingFrames.clear();
        this.streamComplete = false;
        this.nextAudioTime = this.audioContext ? this.audioContext.currentTime : 0;
    }

    handleAVFrame(data) {
        const index = Number(data.frame_index);
        if (!Number.isFinite(index)) {
            return;
        }

        this.pendingFrames.set(index, {
            index,
            audio: data.audio || '',
            video: data.frame || '',
            word: data.word || '',
        });

        this.drainFrames();
    }

    drainFrames() {
        while (this.pendingFrames.has(this.expectedIndex)) {
            const frame = this.pendingFrames.get(this.expectedIndex);
            this.pendingFrames.delete(this.expectedIndex);
            this.playFrame(frame);
            this.expectedIndex += 1;
        }

        this.maybeFinishStream();
    }

    maybeFinishStream() {
        if (!this.streamComplete || this.pendingFrames.size > 0) {
            return;
        }
        this.isGenerating = false;
        this.hideStopButton();
        this.updateAvatarStatus('ready', 'მზადაა');
    }

    playFrame(frame) {
        if (frame.audio) {
            this.queueAudio(frame.audio, frame.index);
        }

        if (frame.video) {
            this.queueVideo(frame.video, frame.index);
        } else if (this.lastVideoFrame) {
            this.displayJpegFrame(this.lastVideoFrame);
        }
    }

    queueAudio(base64, index) {
        const bytes = this.base64ToBytes(base64);
        if (!bytes || bytes.length === 0) return;

        if (this.useAudioDecoder && this.audioDecoder) {
            this.audioDecodeQueue.push(index);
            const chunk = new EncodedAudioChunk({
                type: 'key',
                timestamp: index * 40000,
                data: bytes,
            });
            this.audioDecoder.decode(chunk);
            return;
        }

        const pcm = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        this.schedulePcm(pcm);
    }

    onAudioDecoded(audioData) {
        const samples = new Float32Array(audioData.numberOfFrames * audioData.numberOfChannels);
        audioData.copyTo(samples, { planeIndex: 0 });
        audioData.close();
        this.audioDecodeQueue.shift();
        this.schedulePcm(samples);
    }

    schedulePcm(pcm) {
        if (!this.audioContext || !pcm || pcm.length === 0) return;

        const audioBuffer = this.audioContext.createBuffer(1, pcm.length, this.audioSampleRate);
        audioBuffer.getChannelData(0).set(pcm);

        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        const now = this.audioContext.currentTime;
        const startAt = this.nextAudioTime > now ? this.nextAudioTime : now;
        source.start(startAt);
        this.nextAudioTime = startAt + pcm.length / this.audioSampleRate;
    }

    queueVideo(base64, index) {
        if (!this.useVideoDecoder || !this.videoDecoder) {
            this.displayJpegFrame(base64);
            return;
        }

        const bytes = this.base64ToBytes(base64);
        if (!bytes || bytes.length === 0) return;

        this.videoDecodeQueue.push(index);
        const chunk = new EncodedVideoChunk({
            type: 'key',
            timestamp: index * 40000,
            data: bytes,
        });
        this.videoDecoder.decode(chunk);
    }

    onVideoDecoded(videoFrame) {
        if (this.videoCtx && this.videoCanvas) {
            if (this.videoCanvas.width !== videoFrame.displayWidth || this.videoCanvas.height !== videoFrame.displayHeight) {
                this.videoCanvas.width = videoFrame.displayWidth;
                this.videoCanvas.height = videoFrame.displayHeight;
            }
            this.videoCtx.drawImage(videoFrame, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
        }
        videoFrame.close();
        this.videoDecodeQueue.shift();
    }

    displayJpegFrame(base64) {
        this.lastVideoFrame = base64;
        if (this.useVideoDecoder && this.videoCanvas) {
            const img = new Image();
            img.onload = () => {
                this.videoCanvas.width = img.width;
                this.videoCanvas.height = img.height;
                this.videoCtx.drawImage(img, 0, 0, img.width, img.height);
            };
            img.src = `data:image/jpeg;base64,${base64}`;
            return;
        }

        this.elements.avatarImage.src = `data:image/jpeg;base64,${base64}`;
    }

    base64ToBytes(base64) {
        try {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            return bytes;
        } catch (e) {
            console.error('Failed to decode base64:', e);
            return null;
        }
    }

    // ============ UI Helpers ============

    updateConnectionStatus(status) {
        const el = this.elements.connectionStatus;
        el.className = `connection-status ${status}`;
        this.elements.connectionStatusText.textContent = status === 'connected' ? 'დაკავშირებულია' : 'გათიშულია';
    }

    updateAvatarStatus(status, text) {
        const statusEl = this.elements.avatarStatus;
        const indicator = this.elements.avatarStatusIndicator;
        const textEl = this.elements.avatarStatusText;

        statusEl.classList.remove('ready', 'speaking', 'unavailable', 'error', 'loading');
        statusEl.classList.add(status);

        indicator.classList.remove('ready', 'speaking', 'unavailable', 'error', 'loading');
        indicator.classList.add(status);

        textEl.textContent = text;
    }

    updateVadStatus(status) {
        const indicator = this.elements.vadIndicator;
        const statusEl = this.elements.vadStatus;

        indicator.classList.remove('speaking', 'processing');
        statusEl.classList.remove('speaking', 'processing');

        const isSpeaking = ['speaking', 'speech_start', 'speech_continue'].includes(status);
        if (isSpeaking) {
            indicator.classList.add('speaking');
            statusEl.classList.add('speaking');
        } else if (['processing', 'utterance_complete'].includes(status)) {
            statusEl.classList.add('processing');
        }
    }

    setVadStatusText(text) {
        this.elements.vadStatus.textContent = text;
    }

    addMessage(role, text, isStreaming = false) {
        if (this.elements.welcomeMessage) {
            this.elements.welcomeMessage.remove();
            this.elements.welcomeMessage = null;
        }

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

        this.elements.chatMessages.appendChild(messageEl);
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

    scrollToBottom() {
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    showStopButton() {
        this.elements.stopBtn.classList.remove('hidden');
    }

    hideStopButton() {
        this.elements.stopBtn.classList.add('hidden');
    }

    sendMessage(type, payload = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, ...payload }));
        }
    }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new SimpleVoiceAssistant();
});
