/**
 * Voice Assistant Web Client
 * Handles WebSocket communication, audio recording/playback, and UI state
 */

class VoiceAssistant {
    constructor() {
        // Debug logging toggle
        this.DEBUG = true;
        
        // WebSocket
        this.ws = null;
        this.connectionId = null;
        this.isConnected = false;
        
        // Audio
        this.audioContext = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.currentPlaybackNode = null;
        
        // Video/Avatar
        this.videoEnabled = false;
        this.isDisplayingVideo = false;
        this.videoStartTime = null;
        this.idleFrame = null;
        this.lastFrameIndex = -1;
        this.totalFramesReceived = 0;
        this.videoFps = 25;  // Target FPS
        this.videoFrameInterval = 1000 / 25;  // 40ms per frame
        this.videoDisplayTimer = null;
        this.videoComplete = false;   // Set when server signals video_complete
        
        // Synced A/V playback
        this.syncedQueue = [];  // Queue of {audio, frame} pairs
        this.isSyncedPlayback = false;
        this.recordedSyncedFrames = []; // Stored synced A/V for replay
        this.loopFrames = [];          // Stored loopable frames to play between generations
        this.frameReplayTimer = null;
        this.isReplayingAV = false;
        this.isLoopingFrames = false;
        
        // Adaptive buffering
        this.bufferConfig = {
            buffer_ms: 160,           // Target buffer size in ms (default: 4 frames)
            frame_buffer: 4,          // Number of frames to buffer before playback
            tts_rtf_mean: 0,
            tts_rtf_std: 0,
            is_calibrated: false,
            buffer_source: 'adaptive',
            manual_buffer_ms: null,
        };
        this.isBuffering = false;      // Whether we're waiting for buffer to fill
        this.bufferStartTime = null;   // When we started buffering
        this.playbackStutters = 0;     // Count of buffer underruns
        this.lastBufferUpdate = 0;     // Timestamp of last buffer config update
        
        // State
        this.isGenerating = false;
        this.currentAssistantMessage = null;
        this.wordsSpoken = [];
        
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
            },
            musetalk: {
                start_after_chunks: 3,
                lookahead_chunks: 2
            },
            buffer: {
                manual_buffer_ms: null
            }
        };
        
        // DOM Elements
        this.elements = {};
        
        // Initialize
        this.init();
    }
    
    // Debug logging helper
    log(...args) {
        if (this.DEBUG) {
            console.log(`[VA ${new Date().toISOString().substr(11, 12)}]`, ...args);
        }
    }
    
    logState(context) {
        if (this.DEBUG) {
            console.log(`[VA STATE @ ${context}]`, {
                isBuffering: this.isBuffering,
                isSyncedPlayback: this.isSyncedPlayback,
                isDisplayingVideo: this.isDisplayingVideo,
                isGenerating: this.isGenerating,
                videoComplete: this.videoComplete,
                syncedQueueLen: this.syncedQueue.length,
                recordedFramesLen: this.recordedSyncedFrames.length,
                bufferConfig: this.bufferConfig,
            });
        }
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
            
            // Avatar/Video
            avatarContainer: document.getElementById('avatarContainer'),
            avatarImage: document.getElementById('avatarImage'),
            avatarLoading: document.getElementById('avatarLoading'),
            avatarStatus: document.getElementById('avatarStatus'),
            avatarMetrics: document.getElementById('avatarMetrics'),
            videoFps: document.getElementById('videoFps'),
            videoFrames: document.getElementById('videoFrames'),
            
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
            musetalkStartChunks: document.getElementById('musetalkStartChunks'),
            musetalkStartChunksValue: document.getElementById('musetalkStartChunksValue'),
            musetalkLookaheadChunks: document.getElementById('musetalkLookaheadChunks'),
            musetalkLookaheadChunksValue: document.getElementById('musetalkLookaheadChunksValue'),
            bufferSize: document.getElementById('bufferSize'),
            bufferAutoToggle: document.getElementById('bufferAutoToggle'),
            bufferCurrent: document.getElementById('bufferCurrent'),
        };
    }
    
    bindEvents() {
        // Microphone
        this.elements.micBtn.addEventListener('click', () => this.toggleRecording());
        
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
        this.bindSlider('musetalkStartChunks', 'musetalkStartChunksValue');
        this.bindSlider('musetalkLookaheadChunks', 'musetalkLookaheadChunksValue');
        if (this.elements.bufferAutoToggle) {
            this.elements.bufferAutoToggle.addEventListener('change', () => this.toggleBufferMode());
        }
        if (this.elements.bufferSize) {
            this.elements.bufferSize.addEventListener('input', () => {
                const val = parseFloat(this.elements.bufferSize.value);
                this.config.buffer.manual_buffer_ms = isNaN(val) ? null : val;
            });
        }
        
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
            setTimeout(() => this.connect(), 10000);
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
        this.log(`[WS MSG] type=${data.type}`, data.type === 'synced_av_frame' ? `frame=${data.frame_index}` : data);
        
        const handlers = {
            'connected': () => {
                this.isConnected = true;
                this.connectionId = data.connection_id;
                this.log('[WS] Connected with ID:', this.connectionId);
            },
            
            'tts_cache_ready': () => {
                this.log('[TTS] Cache ready, session:', data.session_id, 'success:', data.success);
            },
            
            'musetalk_ready': () => {
                this.log('[MUSETALK] Ready:', data.success, 'session:', data.session_id);
                this.logState('musetalk_ready');
                if (data.success) {
                    this.videoEnabled = true;
                    this.updateAvatarStatus('ready', 'მზადაა');
                    if (this.elements.avatarContainer) {
                        this.elements.avatarContainer.classList.remove('disabled');
                    }

                    // Show idle frame if available
                    if (data.idle_frame) {
                        this.idleFrame = data.idle_frame;
                        this.displayFrame(data.idle_frame);
                    }
                    
                    // Update buffer config from server
                    if (data.buffer_config) {
                        this.updateBufferConfig(data.buffer_config);
                    }

                    // Hide loading indicator
                    if (this.elements.avatarLoading) {
                        this.elements.avatarLoading.classList.add('hidden');
                    }
                } else {
                    this.videoEnabled = false;
                    this.updateAvatarStatus('unavailable', 'მიუწვდომელია');
                    if (this.elements.avatarContainer) {
                        this.elements.avatarContainer.classList.add('disabled');
                    }
                    console.log('MuseTalk not available:', data.reason);
                }
            },
            
            'buffer_config': () => {
                // Handle buffer config updates from server
                this.updateBufferConfig(data);
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
                // Only reset status text if not still recording
                if (this.isRecording) {
                    this.setVadStatusText('მოსმენა...');
                } else {
                    this.setVadStatusText('დააჭირეთ მიკროფონს საუბრის დასაწყებად');
                }
            },
            
            'llm_start': () => {
                this.isGenerating = true;
                this.showStopButton();
                this.currentAssistantMessage = this.addMessage('assistant', '', true);
                this.wordsSpoken = [];
                
                // Reset video state
                this.totalFramesReceived = 0;
                this.lastFrameIndex = -1;
                this.videoStartTime = null;
                
                // Show loading status during TTS/MuseTalk initialization
                this.updateAvatarStatus('loading', 'იტვირთება...');
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
                this.log('[TTS] Starting, text:', data.text ? data.text.substring(0, 50) : '(empty)', 'video_enabled:', data.video_enabled);
                this.logState('tts_start_begin');

                this.videoComplete = false;
                this.recordedSyncedFrames = [];
                // Stop any idle/loop playback so new frames can take over
                this.stopFrameReplay();
                
                // Update buffer config from server if provided
                if (data.buffer_config) {
                    this.log('[BUFFER] Config from tts_start:', data.buffer_config);
                    this.updateBufferConfig(data.buffer_config);
                }
                
                // Reset playback state
                this.playbackStutters = 0;

                // Check if video is enabled for this session
                if (data.video_enabled) {
                    this.updateAvatarStatus('speaking', 'საუბრობს');
                    // Don't start playback yet - wait for buffer to fill
                    this.isBuffering = true;
                    this.bufferStartTime = performance.now();
                    this.log(`[BUFFER] Started buffering, waiting for ${this.bufferConfig.frame_buffer} frames (${this.bufferConfig.buffer_ms}ms)`);
                } else {
                    this.log('[BUFFER] Video NOT enabled, isBuffering stays:', this.isBuffering);
                }
                this.logState('tts_start_end');
            },
            
            'tts_complete': () => {
                this.log('[TTS] Complete');
                this.logState('tts_complete');
                this.hideStopButton();
                this.isGenerating = false;
            },
            
            'synced_av_frame': () => {
                // Handle synchronized audio+video frame
                this.handleSyncedAVFrame(data);
            },
            
            'video_complete': () => {
                this.log('[VIDEO] Complete, total frames received:', this.totalFramesReceived);
                this.logState('video_complete_begin');
                this.videoComplete = true;

                if (this.recordedSyncedFrames && this.recordedSyncedFrames.length > 0) {
                    const lastFrame = this.recordedSyncedFrames[this.recordedSyncedFrames.length - 1];
                    if (lastFrame && lastFrame.frame) {
                        this.idleFrame = lastFrame.frame;
                    }
                }

                const haveFrames = this.syncedQueue.length > 0 || (this.recordedSyncedFrames && this.recordedSyncedFrames.length > 0);
                this.log('[VIDEO] haveFrames:', haveFrames, 'isBuffering:', this.isBuffering, 'isSyncedPlayback:', this.isSyncedPlayback);
                
                if (this.isBuffering && haveFrames) {
                    this.log(`[BUFFER] Generation finished while buffering with ${this.syncedQueue.length} queued frames; starting fallback playback`);
                    this.isBuffering = false;
                    // Prefer smooth replay using recorded frames/audio when available
                    if (this.recordedSyncedFrames && this.recordedSyncedFrames.length > 0) {
                        this.playRecordedAV();
                    } else {
                        this.startSyncedPlayback();
                    }
                    return;
                }

                // If already playing or draining, let it finish
                if (this.isSyncedPlayback || this.syncedQueue.length > 0) {
                    return;
                }

                // Nothing to play; stop immediately
                this.stopVideoPlayback();
                this.updateAvatarStatus('ready', 'მზადაა');
                
                // Show idle frame after a short delay
                setTimeout(() => {
                    if (this.idleFrame && !this.isDisplayingVideo) {
                        this.displayFrame(this.idleFrame);
                    }
                }, 500);

                // Save last frames for looping between generations
                const maxLoopFrames = 60; // about 2.4s at 25fps
                if (this.recordedSyncedFrames && this.recordedSyncedFrames.length > 0) {
                    this.loopFrames = this.recordedSyncedFrames.slice(-maxLoopFrames);
                }
                // Start looping previous clip if available
                if (!this.isGenerating && this.loopFrames.length > 0) {
                    this.startFrameReplay(this.loopFrames, true);
                }
            },
            
            'error': () => {
                this.log('[ERROR] Server error:', data.message);
                this.logState('error');
                this.hideStopButton();
                this.isGenerating = false;
                this.stopVideoPlayback();
                this.updateAvatarStatus('error', 'შეცდომა');
            },
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
            this.log('[WS SEND]', type, payload);
            this.ws.send(JSON.stringify({ type, ...payload }));
        } else {
            this.log('[WS SEND FAILED] WebSocket not open, type:', type);
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
            
            // Audio batching: accumulate frames before sending
            // 512 samples @ 16kHz = 32ms per frame
            // Batch 5 frames = 160ms = ~6 packets/sec instead of ~31/sec
            const BATCH_SIZE = 5;
            const MAX_BUFFERED_AMOUNT = 256 * 1024;  // 256KB backpressure limit
            let audioBatch = [];
            
            processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                
                const inputData = e.inputBuffer.getChannelData(0);
                const audioData = new Float32Array(inputData);
                
                // Accumulate audio frames
                audioBatch.push(audioData);
                
                // Send when batch is full
                if (audioBatch.length >= BATCH_SIZE) {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        // Backpressure check - don't send if socket is backed up
                        if (this.ws.bufferedAmount > MAX_BUFFERED_AMOUNT) {
                            // Drop this batch to prevent backlog spiral
                            this.log('[AUDIO] Dropping batch - socket backed up:', this.ws.bufferedAmount);
                            audioBatch = [];
                            return;
                        }
                        
                        // Concatenate batch into single buffer
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
            processor.connect(recordingContext.destination);
            
            this.recordingContext = recordingContext;
            this.audioProcessor = processor;
            this.audioSource = source;
            this.isRecording = true;
            
            // Notify server that recording started (for TTS pre-initialization)
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'recording_start' }));
                this.log('[WS SEND] recording_start');
            }
            
            console.log('Recording started');
        } catch (e) {
            console.error('Failed to start recording:', e);
            this.setVadStatusText('მიკროფონზე წვდომა უარყოფილია');
        }
    }
    
    stopRecording() {
        this.isRecording = false;
        
        // Notify server that recording stopped
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'recording_stop' }));
            this.log('[WS SEND] recording_stop');
        }
        
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
    
    stopAudioPlayback() {
        if (this.currentPlaybackNode) {
            try {
                this.currentPlaybackNode.stop();
            } catch (e) {}
            this.currentPlaybackNode = null;
        }
    }
    
    // ============ Video/Avatar ============
    
    handleSyncedAVFrame(data) {
        /**
         * Handle synchronized audio+video frame.
         * Each frame contains 40ms of audio paired with its video frame.
         * We play them together to ensure lip sync.
         * 
         * Adaptive buffering: Wait for buffer to fill before starting playback
         * to ensure smooth playback without stutters.
         */
        if (!data.frame && !data.audio) {
            this.log('[SYNC FRAME] Skipped - no frame or audio');
            return;
        }
        
        this.totalFramesReceived++;
        
        // Log every 10th frame to avoid spam
        if (this.totalFramesReceived % 10 === 1 || this.totalFramesReceived <= 5) {
            this.log(`[SYNC FRAME] #${data.frame_index} received, total: ${this.totalFramesReceived}, queue: ${this.syncedQueue.length}, buffering: ${this.isBuffering}, playing: ${this.isSyncedPlayback}`);
        }
        let decodedAudio = null;
        if (data.audio) {
            decodedAudio = this.decodeAudioBase64(data.audio);
        }
        // Record frames/audio for potential replay/fallback even if not TTS-only
        if (data.frame || data.audio) {
            this.recordedSyncedFrames.push({
                audio: data.audio,
                audioFloat: decodedAudio,
                frame: data.frame,
                frameIndex: data.frame_index,
                timestampMs: data.timestamp_ms,
                word: data.word || '',
            });
        }
        
        // Queue the synced pair
        this.syncedQueue.push({
            audio: data.audio,
            frame: data.frame,
            frameIndex: data.frame_index,
            timestampMs: data.timestamp_ms,
            word: data.word || '',
            // Include metrics for display when this frame is played
            rtf: data.rtf,
            generation_time_ms: data.generation_time_ms,
            audio_duration_ms: data.audio_duration_ms,
            receivedAt: performance.now(),
        });
        
        // Update video metrics (FPS, frame count)
        if (data.frame_index !== undefined) {
            this.updateVideoMetrics(data);
        }
        
        // Update TTS metrics if present
        if (data.rtf !== undefined) {
            this.updateMetrics(data);
        }
        
        // Update avatar status
        if (!this.isDisplayingVideo && this.videoEnabled) {
            this.updateAvatarStatus('speaking', 'საუბრობს');
        }
        
        // Adaptive buffering: Wait for buffer to fill before starting playback
        if (this.isBuffering) {
            const targetFrames = Number.isFinite(this.bufferConfig.frame_buffer) ? this.bufferConfig.frame_buffer : 4;
            
            if (this.syncedQueue.length >= targetFrames) {
                // Buffer is full, start playback
                const bufferTime = performance.now() - this.bufferStartTime;
                this.log(`[BUFFER] Full! ${this.syncedQueue.length}/${targetFrames} frames in ${bufferTime.toFixed(0)}ms, starting playback`);
                this.isBuffering = false;
                this.startSyncedPlayback();
            } else {
                // Still buffering
                if (this.syncedQueue.length % 2 === 0) {
                    this.log(`[BUFFER] Filling: ${this.syncedQueue.length}/${targetFrames} frames`);
                }
            }
        } else if (!this.isSyncedPlayback) {
            // Not buffering and not playing - start immediately (shouldn't normally happen)
            this.log(`[BUFFER] WARNING: Not buffering and not playing! Starting immediate playback. queue=${this.syncedQueue.length}`);
            this.logState('immediate_playback_fallback');
            this.startSyncedPlayback();
        }
    }
    
    startSyncedPlayback() {
        if (this.isSyncedPlayback) {
            this.log('[PLAYBACK] Already playing, skipping startSyncedPlayback');
            return;
        }
        
        this.log('[PLAYBACK] Starting synced playback, queue:', this.syncedQueue.length);
        this.logState('startSyncedPlayback');
        
        this.isSyncedPlayback = true;
        this.isDisplayingVideo = true;
        this.videoStartTime = performance.now();
        this.isBuffering = false;
        
        let framesPlayed = 0;
        
        const playNextSyncedFrame = () => {
            if (!this.isSyncedPlayback) {
                this.log('[PLAYBACK] Stopped, exiting playNextSyncedFrame');
                return;
            }
            
            if (this.syncedQueue.length > 0) {
                // Stop any idle/loop replay as soon as real frames start
                if (this.isReplayingAV) {
                    this.log('[PLAYBACK] Stopping frame replay for real frames');
                    this.stopFrameReplay();
                }

                const syncedData = this.syncedQueue.shift();
                framesPlayed++;
                
                // Play audio immediately
                if (syncedData.audio) {
                    this.playSyncedAudio(syncedData.audio);
                }
                
                // Display video frame immediately
                if (syncedData.frame) {
                    this.displayFrame(syncedData.frame);
                    this.lastFrameIndex = syncedData.frameIndex;
                }
                
                // Update TTS metrics if available
                if (syncedData.rtf !== undefined) {
                    this.updateMetrics({
                        rtf: syncedData.rtf,
                        generation_time_ms: syncedData.generation_time_ms,
                        audio_duration_ms: syncedData.audio_duration_ms,
                        word: syncedData.word,
                    });
                }
                
                // Handle word highlighting
                if (syncedData.word) {
                    this.wordsSpoken.push(syncedData.word);
                    if (this.currentAssistantMessage) {
                        this.highlightWord(syncedData.word);
                    }
                }
                // Log every 25th frame (about once per second)
                if (framesPlayed % 25 === 0) {
                    this.log(`[PLAYBACK] Played ${framesPlayed} frames, queue: ${this.syncedQueue.length}, videoComplete: ${this.videoComplete}`);
                }
            } else {
                // Buffer underrun - no frames available
                if (this.videoComplete) {
                    // Nothing more coming; stop cleanly
                    this.log(`[PLAYBACK] Complete! Played ${framesPlayed} frames total, stutters: ${this.playbackStutters}`);
                    this.stopVideoPlayback();
                    this.updateAvatarStatus('ready', 'მზადაა');
                    if (this.idleFrame && !this.isDisplayingVideo) {
                        this.displayFrame(this.idleFrame);
                    }
                    return;
                } else {
                    this.playbackStutters++;
                    if (this.playbackStutters % 5 === 1) {
                        this.log(`[PLAYBACK] Buffer underrun #${this.playbackStutters}, queue empty, waiting for more frames`);
                    }
                }
            }
            
            // Schedule next frame at 25 FPS (40ms intervals)
            this.videoDisplayTimer = setTimeout(() => {
                requestAnimationFrame(playNextSyncedFrame);
            }, this.videoFrameInterval);
        };
        
        requestAnimationFrame(playNextSyncedFrame);
    }
    
    updateBufferConfig(config) {
        /**
         * Update adaptive buffer configuration from server.
         * The server calculates optimal buffer size based on:
         * - TTS RTF (real-time factor)
         * - Network latency statistics
         * - Generation speed measurements
         */
        if (!config) {
            this.log('[BUFFER CONFIG] Received null config, ignoring');
            return;
        }
        
        this.log('[BUFFER CONFIG] Updating:', config);
        const oldConfig = { ...this.bufferConfig };
        
        if (config.buffer_ms !== undefined) {
            this.bufferConfig.buffer_ms = Number(config.buffer_ms);
        }
        if (config.frame_buffer !== undefined) {
            this.bufferConfig.frame_buffer = Number(config.frame_buffer);
        }
        if (config.tts_rtf_mean !== undefined) {
            this.bufferConfig.tts_rtf_mean = config.tts_rtf_mean;
        }
        if (config.tts_rtf_std !== undefined) {
            this.bufferConfig.tts_rtf_std = config.tts_rtf_std;
        }
        if (config.buffer_source !== undefined) {
            this.bufferConfig.buffer_source = config.buffer_source;
        }
        if (config.manual_buffer_ms !== undefined) {
            this.bufferConfig.manual_buffer_ms = config.manual_buffer_ms === null ? null : Number(config.manual_buffer_ms);
            if (this.config.buffer) {
                this.config.buffer.manual_buffer_ms = this.bufferConfig.manual_buffer_ms;
            }
        }
        if (config.is_calibrated !== undefined) {
            this.bufferConfig.is_calibrated = config.is_calibrated;
        }
        
        // Log if config changed significantly
        if (Math.abs(oldConfig.buffer_ms - this.bufferConfig.buffer_ms) > 10 ||
            oldConfig.frame_buffer !== this.bufferConfig.frame_buffer) {
            const source = this.bufferConfig.buffer_source || 'adaptive';
            const bufferVal = Number.isFinite(this.bufferConfig.buffer_ms) ? this.bufferConfig.buffer_ms : 0;
            const frameVal = Number.isFinite(this.bufferConfig.frame_buffer) ? this.bufferConfig.frame_buffer : 0;
            console.log(`Buffer config updated (${source}): ${bufferVal.toFixed(0)}ms, ${frameVal} frames`);
            console.log(`  TTS RTF: ${this.bufferConfig.tts_rtf_mean.toFixed(3)} ± ${this.bufferConfig.tts_rtf_std.toFixed(3)}`);
        }
        
        this.lastBufferUpdate = performance.now();
        this.renderBufferHint();
    }
    
    playSyncedAudio(audioBase64) {
        /**
         * Play a single 40ms audio chunk immediately.
         * This is synchronized with video frame display.
         */
        if (!this.audioContext || !audioBase64) return;
        
        try {
            const float32 = this.decodeAudioBase64(audioBase64);
            if (!float32) return;
            
            // Create audio buffer (40ms at 24kHz = 960 samples)
            const audioBuffer = this.audioContext.createBuffer(1, float32.length, 24000);
            audioBuffer.getChannelData(0).set(float32);
            
            // Play immediately
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            
            // Store for potential cleanup
            this.currentPlaybackNode = source;
        } catch (e) {
            console.error('Failed to play synced audio:', e);
        }
    }

    decodeAudioBase64(audioBase64) {
        try {
            const binaryString = atob(audioBase64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            return new Float32Array(bytes.buffer);
        } catch (e) {
            console.error('Failed to decode audio base64:', e);
            return null;
        }
    }
    
    stopSyncedPlayback() {
        this.log('[PLAYBACK] Stopping synced playback, queue remaining:', this.syncedQueue.length);
        this.logState('stopSyncedPlayback');
        
        this.isSyncedPlayback = false;
        this.isDisplayingVideo = false;
        this.isBuffering = false;
        this.isReplayingAV = false;
        this.isLoopingFrames = false;
        
        if (this.videoDisplayTimer) {
            clearTimeout(this.videoDisplayTimer);
            this.videoDisplayTimer = null;
        }
        if (this.frameReplayTimer) {
            clearTimeout(this.frameReplayTimer);
            this.frameReplayTimer = null;
        }
        
        // Log playback stats
        if (this.playbackStutters > 0) {
            this.log(`[PLAYBACK] Ended with ${this.playbackStutters} stutters`);
        }
        
        // Clear remaining synced data
        const remainingFrames = this.syncedQueue.length;
        this.syncedQueue = [];
        if (remainingFrames > 0) {
            this.log(`[PLAYBACK] Cleared ${remainingFrames} remaining frames from queue`);
        }
    }
    
    displayFrame(frameBase64) {
        if (!this.elements.avatarImage) return;
        
        try {
            this.elements.avatarImage.src = `data:image/jpeg;base64,${frameBase64}`;
        } catch (e) {
            console.error('Failed to display frame:', e);
        }
    }
    
    stopVideoPlayback() {
        this.log('[VIDEO] Stopping video playback');
        this.logState('stopVideoPlayback');
        
        this.isDisplayingVideo = false;
        this.isSyncedPlayback = false;
        this.isBuffering = false;
        
        if (this.videoDisplayTimer) {
            clearTimeout(this.videoDisplayTimer);
            this.videoDisplayTimer = null;
        }
        
        // Log playback stats
        if (this.playbackStutters > 0) {
            this.log(`[VIDEO] Playback ended with ${this.playbackStutters} buffer underruns`);
        }
        
        // Clear remaining frames
        const remainingSynced = this.syncedQueue.length;
        this.syncedQueue = [];
        
        if (remainingSynced > 0) {
            this.log(`[VIDEO] Cleared ${remainingSynced} synced frames`);
        }
    }
    
    updateAvatarStatus(status, text) {
        if (!this.elements.avatarStatus) return;
        
        const statusIndicator = this.elements.avatarStatus.querySelector('.status-indicator');
        const statusText = this.elements.avatarStatus.querySelector('.status-text');
        
        // Remove all status classes
        this.elements.avatarStatus.classList.remove('ready', 'speaking', 'unavailable', 'error');
        this.elements.avatarStatus.classList.add(status);
        
        if (statusIndicator) {
            statusIndicator.classList.remove('ready', 'speaking', 'unavailable', 'error');
            statusIndicator.classList.add(status);
        }
        
        if (statusText) {
            statusText.textContent = text;
        }
        
        // Add speaking animation class to avatar container
        if (this.elements.avatarContainer) {
            if (status === 'speaking') {
                this.elements.avatarContainer.classList.add('speaking');
            } else {
                this.elements.avatarContainer.classList.remove('speaking');
            }
        }
    }
    
    updateVideoMetrics(data) {
        // Calculate effective FPS
        const elapsed = (performance.now() - (this.videoStartTime || performance.now())) / 1000;
        const effectiveFps = elapsed > 0 ? (this.totalFramesReceived / elapsed).toFixed(1) : 0;
        
        if (this.elements.videoFps) {
            this.elements.videoFps.textContent = `${effectiveFps} FPS`;
        }
        
        if (this.elements.videoFrames) {
            this.elements.videoFrames.textContent = `${data.frames_generated || this.totalFramesReceived} კადრი`;
        }
    }
    
    // Play stored audio (for clickable audio messages)
    async playStoredAudio(audioChunks) {
        if (!audioChunks || audioChunks.length === 0) return;
        
        this.stopAudioPlayback();
        this.stopSyncedPlayback();
        
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

    playRecordedAV(framesOverride = null) {
        // Replay stored synced audio+video frames (used in TTS-only mode and as fallback)
        const frames = framesOverride || this.recordedSyncedFrames;
        if (!frames || frames.length === 0) {
            this.log('[REPLAY] No frames to replay');
            return;
        }
        
        this.log('[REPLAY] Starting recorded AV playback, frames:', frames.length);
        
        // Reset playback state
        this.stopSyncedPlayback();
        this.stopAudioPlayback();
        this.syncedQueue = frames.map(frame => ({ ...frame }));
        this.isBuffering = false;
        this.videoComplete = true; // no more frames will arrive
        this.isSyncedPlayback = false;
        this.isDisplayingVideo = false;
        this.playbackStutters = 0;
        this.totalFramesReceived = frames.length;
        this.lastFrameIndex = -1;
        this.videoStartTime = performance.now();

        // Extract audio floats for smooth continuous playback
        const audioChunks = frames
            .map(f => f.audioFloat)
            .filter(Boolean);
        if (audioChunks.length > 0) {
            this.playStoredAudio(audioChunks);
        }

        // Replay frames based on their timestamps to stay in sync with audio
        this.startFrameReplay(frames, false);
    }

    startFrameReplay(frames, loop = false) {
        if (!frames || frames.length === 0) {
            this.log('[FRAME REPLAY] No frames to replay');
            return;
        }

        this.log('[FRAME REPLAY] Starting, frames:', frames.length, 'loop:', loop);
        this.isReplayingAV = true;
        this.isLoopingFrames = loop;
        let index = 0;
        let start = performance.now();

        const step = () => {
            if (!this.isReplayingAV) {
                return;
            }

            const frame = frames[index];
            if (frame && frame.frame) {
                this.displayFrame(frame.frame);
                this.lastFrameIndex = frame.frameIndex;
            }

            index += 1;

            if (index >= frames.length) {
                if (loop) {
                    index = 0;
                    start = performance.now();
                } else {
                    this.frameReplayTimer = null;
                    return;
                }
            }

            const nextTs = frames[index]?.timestampMs ?? (index * 40);
            const elapsed = performance.now() - start;
            const delay = Math.max(0, nextTs - elapsed);
            this.frameReplayTimer = setTimeout(step, delay);
        };

        step();
    }

    stopFrameReplay() {
        if (this.isReplayingAV || this.frameReplayTimer) {
            this.log('[FRAME REPLAY] Stopping');
        }
        this.isReplayingAV = false;
        this.isLoopingFrames = false;
        if (this.frameReplayTimer) {
            clearTimeout(this.frameReplayTimer);
            this.frameReplayTimer = null;
        }
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
    
    updateVadStatus(status) {
        const indicator = this.elements.vadIndicator;
        const statusEl = this.elements.vadStatus;
        
        indicator.classList.remove('speaking', 'processing');
        statusEl.classList.remove('speaking', 'processing');
        
        // Map server VAD statuses to display text
        const statusTexts = {
            'listening': 'მოსმენა...',
            'speech_start': 'საუბარი...',
            'speech_continue': 'საუბარი...',
            'speaking': 'საუბარი...',
            'utterance_complete': 'დამუშავება...',
            'processing': 'ტრანსკრიფცია...'
        };
        
        // Treat speech_start and speech_continue as speaking
        const isSpeaking = status === 'speaking' || status === 'speech_start' || status === 'speech_continue';
        
        if (isSpeaking) {
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
        // Only update metrics if they are present in the data
        if (data.rtf !== undefined && this.elements.rtfValue) {
            this.elements.rtfValue.textContent = data.rtf.toFixed(3);
            
            const rtfEl = this.elements.rtfValue;
            if (data.rtf < 1) {
                rtfEl.style.color = 'var(--success)';
            } else if (data.rtf < 1.5) {
                rtfEl.style.color = 'var(--warning)';
            } else {
                rtfEl.style.color = 'var(--error)';
            }
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
        this.stopVideoPlayback();
        this.hideStopButton();
        this.isGenerating = false;
        this.updateAvatarStatus('ready', 'მზადაა');
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

        this.elements.musetalkStartChunks.value = this.config.musetalk.start_after_chunks;
        this.elements.musetalkStartChunksValue.textContent = this.config.musetalk.start_after_chunks;
        this.elements.musetalkLookaheadChunks.value = this.config.musetalk.lookahead_chunks;
        this.elements.musetalkLookaheadChunksValue.textContent = this.config.musetalk.lookahead_chunks;
        
        this.updateBufferSettingUI();
    }
    
    updateBufferSettingUI() {
        const manualValue = this.config.buffer?.manual_buffer_ms;
        const auto = manualValue === null || manualValue === undefined;
        
        if (this.elements.bufferAutoToggle) {
            this.elements.bufferAutoToggle.checked = auto;
        }
        if (this.elements.bufferSize) {
            this.elements.bufferSize.disabled = auto;
            const fallback = Number.isFinite(this.bufferConfig.buffer_ms) ? Math.round(this.bufferConfig.buffer_ms) : 0;
            this.elements.bufferSize.value = auto ? fallback : manualValue ?? fallback;
        }
        this.renderBufferHint();
    }
    
    renderBufferHint() {
        if (!this.elements.bufferCurrent) return;
        const bufferValue = Number.isFinite(this.bufferConfig.buffer_ms) ? this.bufferConfig.buffer_ms : 0;
        const source = this.bufferConfig.buffer_source || 'adaptive';
        this.elements.bufferCurrent.textContent = `ამჟამინდელი ბუფერი: ${bufferValue.toFixed(0)} ms (${source})`;
    }
    
    toggleBufferMode() {
        const auto = this.elements.bufferAutoToggle?.checked ?? true;
        if (this.elements.bufferSize) {
            this.elements.bufferSize.disabled = auto;
        }
        if (auto) {
            this.config.buffer.manual_buffer_ms = null;
        } else if (this.elements.bufferSize) {
            const val = parseFloat(this.elements.bufferSize.value);
            this.config.buffer.manual_buffer_ms = isNaN(val) ? null : val;
        }
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

        this.config.musetalk.start_after_chunks = parseInt(this.elements.musetalkStartChunks.value);
        this.config.musetalk.lookahead_chunks = parseInt(this.elements.musetalkLookaheadChunks.value);
        
        const autoBuffer = this.elements.bufferAutoToggle?.checked ?? true;
        if (autoBuffer) {
            this.config.buffer.manual_buffer_ms = null;
        } else {
            const manualVal = parseFloat(this.elements.bufferSize?.value ?? '');
            this.config.buffer.manual_buffer_ms = isNaN(manualVal) ? null : manualVal;
        }
        
        localStorage.setItem('voiceAssistantConfig', JSON.stringify(this.config));
        
        try {
            const response = await fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.config)
            });
            
            if (response.ok) {
                const updated = await response.json();
                console.log('Settings saved');
                if (updated?.buffer) {
                    this.config.buffer.manual_buffer_ms = updated.buffer.manual_buffer_ms ?? this.config.buffer.manual_buffer_ms;
                    this.bufferConfig.buffer_source = updated.buffer.buffer_source || this.bufferConfig.buffer_source;
                    this.updateBufferConfig({
                        buffer_ms: updated.buffer.current_buffer_ms,
                        frame_buffer: updated.buffer.frame_buffer,
                        buffer_source: updated.buffer.buffer_source,
                        manual_buffer_ms: updated.buffer.manual_buffer_ms,
                    });
                    this.updateBufferSettingUI();
                }
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
            },
            musetalk: {
                start_after_chunks: 3,
                lookahead_chunks: 2
            },
            buffer: {
                manual_buffer_ms: null
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
        
        this.loadSettingsToUI();
        
        fetch('/config')
            .then(res => res.json())
            .then(serverConfig => {
                if (serverConfig.vad) this.config.vad = { ...this.config.vad, ...serverConfig.vad };
                if (serverConfig.llm) this.config.llm = { ...this.config.llm, ...serverConfig.llm };
                if (serverConfig.tts) this.config.tts = { ...this.config.tts, ...serverConfig.tts };
                if (serverConfig.musetalk) this.config.musetalk = { ...this.config.musetalk, ...serverConfig.musetalk };
                if (serverConfig.buffer) {
                    this.config.buffer = { ...this.config.buffer, manual_buffer_ms: serverConfig.buffer.manual_buffer_ms ?? this.config.buffer.manual_buffer_ms };
                    this.updateBufferConfig({
                        buffer_ms: serverConfig.buffer.current_buffer_ms,
                        frame_buffer: serverConfig.buffer.frame_buffer,
                        buffer_source: serverConfig.buffer.buffer_source,
                        manual_buffer_ms: serverConfig.buffer.manual_buffer_ms,
                    });
                }
                this.loadSettingsToUI();
            })
            .catch(e => console.log('Could not load server config:', e));
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.voiceAssistant = new VoiceAssistant();
});
