/**
 * Voice Assistant Web Client
 * Handles WebSocket communication, audio recording/playback, and UI state
 */

class VoiceAssistant {
    constructor() {
        // WebSocket
        this.ws = null;
        this.connectionId = null;
        this.isConnected = false;
        
        // Audio
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorklet = null;
        this.isRecording = false;
        this.audioQueue = [];
        this.isPlaying = false;
        this.currentPlaybackNode = null;
        
        // State
        this.currentMode = 'voice_to_voice';
        this.isGenerating = false;
        this.currentAssistantMessage = null;
        this.wordsSpoken = [];
        
        // TTS-only mode state
        this.currentTTSText = '';
        this.currentTTSAudioChunks = [];
        this.currentTTSMessage = null;
        
        // Config
        this.config = {
            vad: {
                speech_threshold_ms: 200,
                silence_threshold_ms: 1500,
            },
            llm: {
                temperature: 0.7,
                top_p: 0.9,
                max_new_tokens: 512,
                system_prompt: 'თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი, რომლის მოვალეობაცაა დაეხმაროს მომხმარებლებს საბანკო თემებში'
            },
            tts: {
                backbone_temperature: 0.8,
                backbone_top_p: 0.9,
                depth_temperature: 0.8,
                depth_top_p: 0.9
            }
        };
        
        // DOM Elements
        this.elements = {};
        
        // Initialize
        this.init();
    }
    
    async init() {
        this.cacheElements();
        this.bindEvents();
        await this.initAudioContext();
        this.connect();
        this.loadConfig();
    }
    
    cacheElements() {
        this.elements = {
            // Connection
            connectionStatus: document.getElementById('connectionStatus'),
            
            // Mode
            modeBtns: document.querySelectorAll('.mode-btn'),
            
            // Chat
            chatMessages: document.getElementById('chatMessages'),
            chatContainer: document.getElementById('chatContainer'),
            
            // Metrics
            metricsPanel: document.getElementById('metricsPanel'),
            rtfValue: document.getElementById('rtfValue'),
            genTimeValue: document.getElementById('genTimeValue'),
            audioDurValue: document.getElementById('audioDurValue'),
            currentWordValue: document.getElementById('currentWordValue'),
            
            // Voice Input
            voiceInput: document.getElementById('voiceInput'),
            vadIndicator: document.getElementById('vadIndicator'),
            micBtn: document.getElementById('micBtn'),
            vadStatus: document.getElementById('vadStatus'),
            
            // Text Input
            textInput: document.getElementById('textInput'),
            textArea: document.getElementById('textArea'),
            sendBtn: document.getElementById('sendBtn'),
            
            // Stop
            stopBtn: document.getElementById('stopBtn'),
            
            // Settings
            settingsBtn: document.getElementById('settingsBtn'),
            settingsPanel: document.getElementById('settingsPanel'),
            settingsOverlay: document.getElementById('settingsOverlay'),
            closeSettingsBtn: document.getElementById('closeSettingsBtn'),
            saveSettingsBtn: document.getElementById('saveSettingsBtn'),
            resetSettingsBtn: document.getElementById('resetSettingsBtn'),
            
            // Setting inputs
            speechThreshold: document.getElementById('speechThreshold'),
            speechThresholdValue: document.getElementById('speechThresholdValue'),
            silenceThreshold: document.getElementById('silenceThreshold'),
            silenceThresholdValue: document.getElementById('silenceThresholdValue'),
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
        };
    }
    
    bindEvents() {
        // Mode switching
        this.elements.modeBtns.forEach(btn => {
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode));
        });
        
        // Microphone
        this.elements.micBtn.addEventListener('click', () => this.toggleRecording());
        
        // Text input
        this.elements.sendBtn.addEventListener('click', () => this.sendTextInput());
        this.elements.textArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendTextInput();
            }
        });
        this.elements.textArea.addEventListener('input', () => this.autoResizeTextarea());
        
        // Stop button
        this.elements.stopBtn.addEventListener('click', () => this.stopGeneration());
        
        // Settings
        this.elements.settingsBtn.addEventListener('click', () => this.openSettings());
        this.elements.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.elements.settingsOverlay.addEventListener('click', () => this.closeSettings());
        this.elements.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.elements.resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        
        // Settings sliders
        this.bindSlider('speechThreshold', 'speechThresholdValue');
        this.bindSlider('silenceThreshold', 'silenceThresholdValue');
        this.bindSlider('llmTemperature', 'llmTemperatureValue');
        this.bindSlider('llmTopP', 'llmTopPValue');
        this.bindSlider('llmMaxTokens', 'llmMaxTokensValue');
        this.bindSlider('ttsBackboneTemp', 'ttsBackboneTempValue');
        this.bindSlider('ttsBackboneTopP', 'ttsBackboneTopPValue');
        this.bindSlider('ttsDepthTemp', 'ttsDepthTempValue');
        this.bindSlider('ttsDepthTopP', 'ttsDepthTopPValue');
    }
    
    bindSlider(sliderId, valueId) {
        const slider = this.elements[sliderId];
        const valueSpan = this.elements[valueId];
        if (slider && valueSpan) {
            slider.addEventListener('input', () => {
                valueSpan.textContent = slider.value;
            });
        }
    }
    
    // ============ WebSocket ============
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        console.log('Connecting to:', wsUrl);
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus('connected');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            // Reconnect after 3 seconds
            setTimeout(() => this.connect(), 3000);
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
        const handlers = {
            'connected': () => {
                this.isConnected = true;
                this.connectionId = data.connection_id;
                console.log('Connected with ID:', this.connectionId);
                this.sendMessage('set_mode', { mode: this.currentMode });
            },
            
            'tts_cache_ready': () => {
                console.log('TTS cache ready');
            },
            
            'mode_changed': () => {
                console.log('Mode changed to:', data.mode);
            },
            
            'vad_status': () => {
                this.updateVadStatus(data.status);
            },
            
            'stt_start': () => {
                this.updateVadStatus('processing');
                this.setVadStatusText('ტრანსკრიფცია...');
            },
            
            'stt_complete': () => {
                this.addMessage('user', data.text);
                this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
            },
            
            'user_message': () => {
                this.addMessage('user', data.text);
            },
            
            'llm_start': () => {
                this.isGenerating = true;
                this.showStopButton();
                this.currentAssistantMessage = this.addMessage('assistant', '', true);
                this.wordsSpoken = [];
                this.currentTTSAudioChunks = [];
            },
            
            'llm_token': () => {
                if (this.currentAssistantMessage) {
                    this.updateAssistantMessage(data.full_text);
                }
            },
            
            'llm_complete': () => {
                if (this.currentAssistantMessage) {
                    this.updateAssistantMessage(data.text, true);
                }
            },
            
            'tts_start': () => {
                console.log('TTS starting for:', data.text);
                this.currentTTSText = data.text;
                this.currentTTSAudioChunks = [];
                
                // In TTS-only mode, show user message and prepare audio message
                if (this.currentMode === 'tts_only') {
                    this.isGenerating = true;
                    this.showStopButton();
                    // Add user message (the text to synthesize)
                    this.addMessage('user', data.text);
                    // Create audio message placeholder
                    this.currentTTSMessage = this.addAudioMessage();
                }
            },
            
            'tts_audio': () => {
                this.handleTTSAudio(data);
            },
            
            'tts_complete': () => {
                console.log('TTS complete');
                this.hideStopButton();
                this.isGenerating = false;
                
                // Finalize audio message in TTS-only mode
                if (this.currentMode === 'tts_only' && this.currentTTSMessage) {
                    this.finalizeTTSMessage();
                }
            },
            
            'error': () => {
                console.error('Server error:', data.message);
                this.hideStopButton();
                this.isGenerating = false;
            },
            
            'conversation_cleared': () => {
                this.clearChat();
            }
        };
        
        const handler = handlers[data.type];
        if (handler) {
            handler();
        } else {
            console.log('Unknown message type:', data.type, data);
        }
    }
    
    sendMessage(type, payload = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, ...payload }));
        }
    }
    
    // ============ Audio ============
    
    async initAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000
            });
            console.log('AudioContext initialized with sample rate:', this.audioContext.sampleRate);
        } catch (e) {
            console.error('Failed to initialize AudioContext:', e);
        }
    }
    
    async startRecording() {
        try {
            // Guard for browsers/contexts where mediaDevices is missing (non-HTTPS or legacy browsers)
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
                if (legacyGetUserMedia) {
                    navigator.mediaDevices = navigator.mediaDevices || {};
                    navigator.mediaDevices.getUserMedia = (constraints) => new Promise((resolve, reject) => {
                        legacyGetUserMedia.call(navigator, constraints, resolve, reject);
                    });
                }
            }
            
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.error('getUserMedia not available. Use HTTPS/localhost and a supported browser.');
                this.setVadStatusText('ბრაუზერი ვერ ხსნის მიკროფონს (HTTPS/უფლებები)');
                return;
            }

            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            const recordingContext = new AudioContext({ sampleRate: 16000 });
            const source = recordingContext.createMediaStreamSource(this.mediaStream);
            
            const bufferSize = 512;
            const processor = recordingContext.createScriptProcessor(bufferSize, 1, 1);
            
            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                const audioData = new Float32Array(inputData);
                
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(audioData.buffer);
                }
            };
            
            source.connect(processor);
            processor.connect(recordingContext.destination);
            
            this.recordingContext = recordingContext;
            this.audioProcessor = processor;
            this.audioSource = source;
            this.isRecording = true;
            
            console.log('Recording started');
        } catch (e) {
            console.error('Failed to start recording:', e);
            this.setVadStatusText('მიკროფონზე წვდომა უარყოფილია');
        }
    }
    
    stopRecording() {
        this.isRecording = false;
        
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        if (this.audioSource) {
            this.audioSource.disconnect();
            this.audioSource = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.recordingContext) {
            this.recordingContext.close();
            this.recordingContext = null;
        }
        
        console.log('Recording stopped');
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
            this.elements.micBtn.classList.remove('active');
            this.elements.vadIndicator.classList.remove('speaking');
            this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
        } else {
            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            await this.startRecording();
            this.elements.micBtn.classList.add('active');
            this.setVadStatusText('მოსმენა...');
        }
    }
    
    handleTTSAudio(data) {
        try {
            const binaryString = atob(data.audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            const alignedBuffer = new ArrayBuffer(bytes.length);
            new Uint8Array(alignedBuffer).set(bytes);
            const audioData = new Float32Array(alignedBuffer);
            
            if (audioData.length === 0) {
                return;
            }
            
            // Update metrics
            this.updateMetrics(data);
            
            // Store audio chunk for TTS message
            this.currentTTSAudioChunks.push(audioData.slice());
            
            // Highlight current word in message (for non-TTS-only modes)
            if (data.word && this.currentAssistantMessage && this.currentMode !== 'tts_only') {
                this.highlightWord(data.word);
            }
            
            // Update TTS message progress
            if (this.currentMode === 'tts_only' && this.currentTTSMessage) {
                this.updateTTSMessageProgress(data.word);
            }
            
            // Queue audio for playback
            this.queueAudio(audioData);
            
        } catch (e) {
            console.error('Failed to decode audio:', e);
        }
    }
    
    queueAudio(audioData) {
        this.audioQueue.push(audioData);
        
        if (!this.isPlaying) {
            this.playNextAudio();
        }
    }
    
    async playNextAudio() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }
        
        this.isPlaying = true;
        
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        const audioData = this.audioQueue.shift();
        
        const audioBuffer = this.audioContext.createBuffer(1, audioData.length, 24000);
        audioBuffer.getChannelData(0).set(audioData);
        
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        
        source.onended = () => {
            this.playNextAudio();
        };
        
        source.start();
        this.currentPlaybackNode = source;
    }
    
    stopAudioPlayback() {
        this.audioQueue = [];
        if (this.currentPlaybackNode) {
            try {
                this.currentPlaybackNode.stop();
            } catch (e) {}
            this.currentPlaybackNode = null;
        }
        this.isPlaying = false;
    }
    
    // Play stored audio (for clickable audio messages)
    async playStoredAudio(audioChunks) {
        if (!audioChunks || audioChunks.length === 0) return;
        
        this.stopAudioPlayback();
        
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        // Concatenate all chunks
        const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const fullAudio = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of audioChunks) {
            fullAudio.set(chunk, offset);
            offset += chunk.length;
        }
        
        // Create and play buffer
        const audioBuffer = this.audioContext.createBuffer(1, fullAudio.length, 24000);
        audioBuffer.getChannelData(0).set(fullAudio);
        
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
        
        this.currentPlaybackNode = source;
    }
    
    // ============ UI Updates ============
    
    updateConnectionStatus(status) {
        const statusEl = this.elements.connectionStatus;
        statusEl.className = 'connection-status ' + status;
        
        const textEl = statusEl.querySelector('.status-text');
        const texts = {
            'connected': 'დაკავშირებულია',
            'disconnected': 'გათიშულია',
        };
        textEl.textContent = texts[status] || 'კავშირის დამყარება...';
    }
    
    setMode(mode) {
        this.currentMode = mode;
        
        this.elements.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        if (mode === 'voice_to_voice') {
            this.elements.voiceInput.classList.remove('hidden');
            this.elements.textInput.classList.add('hidden');
        } else {
            this.elements.voiceInput.classList.add('hidden');
            this.elements.textInput.classList.remove('hidden');
        }
        
        if (mode !== 'voice_to_voice' && this.isRecording) {
            this.stopRecording();
            this.elements.micBtn.classList.remove('active');
        }
        
        this.sendMessage('set_mode', { mode });
    }
    
    updateVadStatus(status) {
        const indicator = this.elements.vadIndicator;
        const statusEl = this.elements.vadStatus;
        
        indicator.classList.remove('speaking', 'processing');
        statusEl.classList.remove('speaking', 'processing');
        
        const statusTexts = {
            'listening': 'მოსმენა...',
            'speaking': 'საუბარი...',
            'utterance_complete': 'დამუშავება...',
            'processing': 'ტრანსკრიფცია...'
        };
        
        if (status === 'speaking') {
            indicator.classList.add('speaking');
            statusEl.classList.add('speaking');
        } else if (status === 'processing' || status === 'utterance_complete') {
            statusEl.classList.add('processing');
        }
        
        if (statusTexts[status]) {
            this.setVadStatusText(statusTexts[status]);
        }
    }
    
    setVadStatusText(text) {
        this.elements.vadStatus.textContent = text;
    }
    
    updateMetrics(data) {
        this.elements.rtfValue.textContent = data.rtf.toFixed(3);
        this.elements.genTimeValue.textContent = `${data.generation_time_ms.toFixed(0)} ms`;
        this.elements.audioDurValue.textContent = `${data.audio_duration_ms.toFixed(0)} ms`;
        this.elements.currentWordValue.textContent = data.word || '-';
        
        const rtfEl = this.elements.rtfValue;
        if (data.rtf < 1) {
            rtfEl.style.color = 'var(--success)';
        } else if (data.rtf < 1.5) {
            rtfEl.style.color = 'var(--warning)';
        } else {
            rtfEl.style.color = 'var(--error)';
        }
    }
    
    addMessage(role, text, isStreaming = false) {
        const welcomeMsg = this.elements.chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
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
    
    // Add audio message (for TTS-only mode)
    addAudioMessage() {
        const welcomeMsg = this.elements.chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }
        
        const messageEl = document.createElement('div');
        messageEl.className = 'message assistant audio-message';
        
        messageEl.innerHTML = `
            <div class="message-avatar">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                    <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                </svg>
            </div>
            <div class="message-content audio-content">
                <div class="audio-player">
                    <button class="play-btn" disabled>
                        <svg class="play-icon" viewBox="0 0 24 24" fill="currentColor">
                            <polygon points="5 3 19 12 5 21 5 3"/>
                        </svg>
                        <svg class="pause-icon hidden" viewBox="0 0 24 24" fill="currentColor">
                            <rect x="6" y="4" width="4" height="16"/>
                            <rect x="14" y="4" width="4" height="16"/>
                        </svg>
                    </button>
                    <div class="audio-info">
                        <span class="audio-status">გენერაცია...</span>
                        <span class="audio-word"></span>
                    </div>
                </div>
            </div>
        `;
        
        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
        
        return messageEl;
    }
    
    updateTTSMessageProgress(word) {
        if (!this.currentTTSMessage) return;
        
        const wordEl = this.currentTTSMessage.querySelector('.audio-word');
        if (wordEl && word) {
            wordEl.textContent = word;
        }
    }
    
    finalizeTTSMessage() {
        if (!this.currentTTSMessage) return;
        
        const playBtn = this.currentTTSMessage.querySelector('.play-btn');
        const statusEl = this.currentTTSMessage.querySelector('.audio-status');
        const wordEl = this.currentTTSMessage.querySelector('.audio-word');
        
        // Store audio chunks in the button's dataset
        const audioChunks = [...this.currentTTSAudioChunks];
        
        if (audioChunks.length > 0) {
            playBtn.disabled = false;
            
            // Calculate duration
            const totalSamples = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
            const durationSec = totalSamples / 24000;
            statusEl.textContent = `${durationSec.toFixed(1)}წმ`;
            wordEl.textContent = '';
            
            // Add click handler
            let isPlaying = false;
            playBtn.addEventListener('click', async () => {
                const playIcon = playBtn.querySelector('.play-icon');
                const pauseIcon = playBtn.querySelector('.pause-icon');
                
                if (isPlaying) {
                    this.stopAudioPlayback();
                    isPlaying = false;
                    playIcon.classList.remove('hidden');
                    pauseIcon.classList.add('hidden');
                } else {
                    await this.playStoredAudio(audioChunks);
                    isPlaying = true;
                    playIcon.classList.add('hidden');
                    pauseIcon.classList.remove('hidden');
                    
                    // Reset when done (approximate)
                    setTimeout(() => {
                        isPlaying = false;
                        playIcon.classList.remove('hidden');
                        pauseIcon.classList.add('hidden');
                    }, durationSec * 1000 + 100);
                }
            });
        } else {
            statusEl.textContent = 'შეცდომა';
        }
        
        this.currentTTSMessage = null;
        this.currentTTSAudioChunks = [];
    }
    
    updateAssistantMessage(text, complete = false) {
        if (!this.currentAssistantMessage) return;
        
        const contentEl = this.currentAssistantMessage.querySelector('.message-content');
        const textEl = contentEl.querySelector('.message-text');
        const typingEl = contentEl.querySelector('.typing-indicator');
        
        textEl.textContent = text;
        
        if (complete && typingEl) {
            typingEl.remove();
        }
        
        this.scrollToBottom();
    }
    
    highlightWord(word) {
        if (!this.currentAssistantMessage || !word) return;
        
        if (!this.wordsSpoken.includes(word)) {
            this.wordsSpoken.push(word);
        }
        
        const textEl = this.currentAssistantMessage.querySelector('.message-text');
        const text = textEl.textContent;
        
        const words = text.split(' ');
        const highlightedWords = words.map((w, i) => {
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
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }
    
    clearChat() {
        this.elements.chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M8 14s1.5 2 4 2 4-2 4-2"/>
                        <line x1="9" y1="9" x2="9.01" y2="9"/>
                        <line x1="15" y1="9" x2="15.01" y2="9"/>
                    </svg>
                </div>
                <h2>გამარჯობა!</h2>
                <p>მე ვარ თქვენი ციფრული ასისტენტი. როგორ შემიძლია დაგეხმაროთ?</p>
            </div>
        `;
    }
    
    showStopButton() {
        this.elements.stopBtn.classList.remove('hidden');
    }
    
    hideStopButton() {
        this.elements.stopBtn.classList.add('hidden');
    }
    
    stopGeneration() {
        this.sendMessage('stop_generation', {});
        this.stopAudioPlayback();
        this.hideStopButton();
        this.isGenerating = false;
    }
    
    sendTextInput() {
        const text = this.elements.textArea.value.trim();
        if (!text || this.isGenerating) return;
        
        this.sendMessage('text_input', { text });
        this.elements.textArea.value = '';
        this.autoResizeTextarea();
    }
    
    autoResizeTextarea() {
        const textarea = this.elements.textArea;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
    }
    
    // ============ Settings ============
    
    openSettings() {
        this.elements.settingsPanel.classList.remove('hidden');
        this.loadSettingsToUI();
    }
    
    closeSettings() {
        this.elements.settingsPanel.classList.add('hidden');
    }
    
    loadSettingsToUI() {
        this.elements.speechThreshold.value = this.config.vad.speech_threshold_ms;
        this.elements.speechThresholdValue.textContent = this.config.vad.speech_threshold_ms;
        this.elements.silenceThreshold.value = this.config.vad.silence_threshold_ms;
        this.elements.silenceThresholdValue.textContent = this.config.vad.silence_threshold_ms;
        
        this.elements.llmTemperature.value = this.config.llm.temperature;
        this.elements.llmTemperatureValue.textContent = this.config.llm.temperature;
        this.elements.llmTopP.value = this.config.llm.top_p;
        this.elements.llmTopPValue.textContent = this.config.llm.top_p;
        this.elements.llmMaxTokens.value = this.config.llm.max_new_tokens;
        this.elements.llmMaxTokensValue.textContent = this.config.llm.max_new_tokens;
        this.elements.systemPrompt.value = this.config.llm.system_prompt;
        
        this.elements.ttsBackboneTemp.value = this.config.tts.backbone_temperature;
        this.elements.ttsBackboneTempValue.textContent = this.config.tts.backbone_temperature;
        this.elements.ttsBackboneTopP.value = this.config.tts.backbone_top_p;
        this.elements.ttsBackboneTopPValue.textContent = this.config.tts.backbone_top_p;
        this.elements.ttsDepthTemp.value = this.config.tts.depth_temperature;
        this.elements.ttsDepthTempValue.textContent = this.config.tts.depth_temperature;
        this.elements.ttsDepthTopP.value = this.config.tts.depth_top_p;
        this.elements.ttsDepthTopPValue.textContent = this.config.tts.depth_top_p;
    }
    
    async saveSettings() {
        this.config.vad.speech_threshold_ms = parseInt(this.elements.speechThreshold.value);
        this.config.vad.silence_threshold_ms = parseInt(this.elements.silenceThreshold.value);
        
        this.config.llm.temperature = parseFloat(this.elements.llmTemperature.value);
        this.config.llm.top_p = parseFloat(this.elements.llmTopP.value);
        this.config.llm.max_new_tokens = parseInt(this.elements.llmMaxTokens.value);
        this.config.llm.system_prompt = this.elements.systemPrompt.value;
        
        this.config.tts.backbone_temperature = parseFloat(this.elements.ttsBackboneTemp.value);
        this.config.tts.backbone_top_p = parseFloat(this.elements.ttsBackboneTopP.value);
        this.config.tts.depth_temperature = parseFloat(this.elements.ttsDepthTemp.value);
        this.config.tts.depth_top_p = parseFloat(this.elements.ttsDepthTopP.value);
        
        localStorage.setItem('voiceAssistantConfig', JSON.stringify(this.config));
        
        try {
            const response = await fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.config)
            });
            
            if (response.ok) {
                console.log('Settings saved');
            }
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
        
        this.closeSettings();
    }
    
    resetSettings() {
        this.config = {
            vad: {
                speech_threshold_ms: 200,
                silence_threshold_ms: 1500,
            },
            llm: {
                temperature: 0.7,
                top_p: 0.9,
                max_new_tokens: 512,
                system_prompt: 'თქვენ ხართ თიბისი ბანკის ციფრული ასისტენტი, რომლის მოვალეობაცაა დაეხმაროს მომხმარებლებს საბანკო თემებში'
            },
            tts: {
                backbone_temperature: 0.8,
                backbone_top_p: 0.9,
                depth_temperature: 0.8,
                depth_top_p: 0.9
            }
        };
        
        this.loadSettingsToUI();
    }
    
    loadConfig() {
        const saved = localStorage.getItem('voiceAssistantConfig');
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                this.config = { ...this.config, ...parsed };
            } catch (e) {
                console.error('Failed to load config:', e);
            }
        }
        
        fetch('/config')
            .then(res => res.json())
            .then(serverConfig => {
                if (serverConfig.vad) this.config.vad = { ...this.config.vad, ...serverConfig.vad };
                if (serverConfig.llm) this.config.llm = { ...this.config.llm, ...serverConfig.llm };
                if (serverConfig.tts) this.config.tts = { ...this.config.tts, ...serverConfig.tts };
            })
            .catch(e => console.log('Could not load server config:', e));
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new VoiceAssistant();
});
