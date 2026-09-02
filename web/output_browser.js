import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const DRAG_TYPE = "application/x-comfyui-iat-output";
const FAVORITES_KEY = "comfyui-iat-output-favorites";
const panelInstances = new Set();
let activeAudioController = null;
let sharedAudioContext = null;
const englishCollator = new Intl.Collator("en", {
    numeric: true,
    sensitivity: "base",
    ignorePunctuation: false,
});
const pinyinCollator = new Intl.Collator("zh-Hans-CN-u-co-pinyin", {
    usage: "sort",
    numeric: true,
    sensitivity: "base",
});
const kindOrder = { folder: 0, image: 1, video: 2, audio: 3, other: 4 };
const hanCharacterPattern = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
const latinCharacterPattern = /[A-Za-z]/;

const translations = {
    en: {
        play: "Play",
        pause: "Pause",
        parentFolder: "Parent folder",
        refresh: "Refresh",
        searchAllOutput: "Search file or folder",
        outputFolderPath: "Output folder path",
        favoriteOutputFolders: "Favorite output folders",
        displayMode: "Display mode",
        thumbnailView: "Thumbnail view",
        listView: "List view",
        sortBy: "Sort by",
        updated: "Updated",
        name: "Name",
        download: "Download",
        loadWorkflow: "Load workflow",
        open: "Open",
        addToFavorites: "Add to favorites",
        removeFromFavorites: "Remove from favorites",
        noFilesFound: "No files found",
        item: "item",
        items: "items",
        unableRead: "Unable to read",
        unableOutputFolder: "Unable to read output folder",
        unableSearch: "Unable to search output folder",
        webAudioUnsupported: "Web Audio is not supported",
        output: "output",
        outputBrowser: "Output Browser",
    },
    zh: {
        play: "播放",
        pause: "暂停",
        parentFolder: "上层文件夹",
        refresh: "刷新",
        searchAllOutput: "搜索文件或文件夹",
        outputFolderPath: "输出文件夹路径",
        favoriteOutputFolders: "收藏的输出文件夹",
        displayMode: "显示模式",
        thumbnailView: "图片模式",
        listView: "列表模式",
        sortBy: "排序方式",
        updated: "更新时间",
        name: "名称",
        download: "下载",
        loadWorkflow: "加载工作流",
        open: "打开",
        addToFavorites: "添加到收藏",
        removeFromFavorites: "取消收藏",
        noFilesFound: "没有找到文件",
        item: "项",
        items: "项",
        unableRead: "无法读取",
        unableOutputFolder: "无法读取 output 文件夹",
        unableSearch: "无法搜索 output 文件夹",
        output: "output",
        outputBrowser: "输出浏览器",
    },
};

function getComfyLocale() {
    try {
        const configured = app.uiSettings?.getSettingValue?.("Comfy.Locale")
            || app.uiSettings?.getSettingValue?.("locale")
            || app.ui?.settings?.getSettingValue?.("Comfy.Locale")
            || app.ui?.settings?.getSettingValue?.("locale")
            || localStorage.getItem("Comfy.Settings.Locale")
            || localStorage.getItem("Comfy.Locale");
        return String(configured || navigator.language || "en").toLowerCase();
    } catch {
        return String(navigator.language || "en").toLowerCase();
    }
}

const locale = getComfyLocale();
const language = locale.startsWith("zh") ? "zh" : "en";
const t = (key) => translations[language][key] || translations.en[key] || key;

function getNameSortInfo(name) {
    const normalized = name.normalize("NFKC");
    const firstCharacter = Array.from(normalized)[0] || "";
    if (latinCharacterPattern.test(firstCharacter)) return { rank: 1, value: normalized };
    if (hanCharacterPattern.test(firstCharacter)) return { rank: 2, value: normalized };
    return { rank: 0, value: normalized };
}

function compareNames(leftName, rightName) {
    const left = getNameSortInfo(leftName);
    const right = getNameSortInfo(rightName);
    if (left.rank !== right.rank) return left.rank - right.rank;
    const collator = left.rank === 2 ? pinyinCollator : englishCollator;
    return collator.compare(left.value, right.value)
        || englishCollator.compare(leftName, rightName);
}

function itemTime(item) {
    return item.kind === "folder" ? (item.modified || 0) : (item.created || 0);
}

function protectUserText(element) {
    element.classList.add("notranslate", "lite-search-item-type");
    element.setAttribute("translate", "no");
    return element;
}

function readFavorites() {
    try {
        const value = JSON.parse(localStorage.getItem(FAVORITES_KEY) || "[]");
        return Array.isArray(value)
            ? [...new Set(value.filter((path) => typeof path === "string" && path))]
            : [];
    } catch {
        return [];
    }
}

function writeFavorites(favorites) {
    try {
        localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
    } catch {
        // The panel still works for this session if browser storage is unavailable.
    }
}

function installStyles() {
    if (document.querySelector('link[data-iat-output-browser="true"]')) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./output_browser.css", import.meta.url).href;
    link.dataset.iatOutputBrowser = "true";
    document.head.append(link);
}

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "";
    const totalSeconds = Math.round(seconds);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const remainder = totalSeconds % 60;
    return hours
        ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
        : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function formatDateTime(timestamp) {
    if (!Number.isFinite(timestamp) || timestamp <= 0) return "";
    return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en-US", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    }).format(new Date(timestamp));
}

function updateMediaMeta(element, item, meta) {
    const parts = [formatSize(item.size)];
    if (item.kind === "video" || item.kind === "audio") {
        const duration = formatDuration(element.duration);
        if (duration) parts.push(duration);
    }
    if (item.kind === "video" && element.videoWidth && element.videoHeight) {
        parts.push(`${element.videoWidth}x${element.videoHeight}`);
    }
    if (item.kind === "image" && element.naturalWidth && element.naturalHeight) {
        parts.push(`${element.naturalWidth}x${element.naturalHeight}`);
    }
    meta.textContent = parts.join(" · ");
    meta.title = meta.textContent;
}

function createAudioSpectrum(audio, label, onError) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "iat-output-spectrum";
    button.setAttribute("aria-label", `${t("play")} ${label}`);
    button.setAttribute("aria-pressed", "false");
    button.title = `${t("play")} ${label}`;

    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-hidden", "true");
    button.append(canvas);

    let waveform = null;
    let animationFrame = 0;
    let disposed = false;
    const waveformRequest = new AbortController();

    const buildWaveform = (audioBuffer, binCount = 2048) => {
        const peaks = new Float32Array(binCount);
        const channelData = Array.from(
            { length: audioBuffer.numberOfChannels },
            (_, channel) => audioBuffer.getChannelData(channel)
        );
        const samplesPerBin = audioBuffer.length / binCount;
        let maximum = 0;
        for (let bin = 0; bin < binCount; bin += 1) {
            const start = Math.floor(bin * samplesPerBin);
            const end = Math.min(audioBuffer.length, Math.ceil((bin + 1) * samplesPerBin));
            const step = Math.max(1, Math.floor((end - start) / 128));
            let peak = 0;
            for (const channel of channelData) {
                for (let sample = start; sample < end; sample += step) {
                    peak = Math.max(peak, Math.abs(channel[sample]));
                }
            }
            peaks[bin] = peak;
            maximum = Math.max(maximum, peak);
        }
        if (maximum > 0) {
            for (let index = 0; index < peaks.length; index += 1) {
                peaks[index] /= maximum;
            }
        }
        return peaks;
    };

    const loadWaveform = async () => {
        try {
            const response = await fetch(audio.src, { signal: waveformRequest.signal });
            if (!response.ok) throw new Error(`${t("unableRead")} ${label}`);
            const encodedAudio = await response.arrayBuffer();
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) throw new Error(t("webAudioUnsupported"));
            sharedAudioContext ||= new AudioContextClass();
            const audioBuffer = await sharedAudioContext.decodeAudioData(encodedAudio);
            if (disposed) return;
            waveform = buildWaveform(audioBuffer);
            draw();
        } catch (error) {
            if (error.name !== "AbortError" && !disposed) {
                button.classList.add("iat-output-spectrum-error");
                onError(error.message || `${t("unableRead")} ${label}`);
                draw();
            }
        }
    };

    const draw = () => {
        if (disposed) return;
        const scale = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.round(canvas.clientWidth * scale));
        const height = Math.max(1, Math.round(canvas.clientHeight * scale));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height);
        const gap = Math.max(1, Math.round(1.5 * scale));
        const barCount = Math.max(12, Math.floor(width / Math.max(3, 3.5 * scale)));
        const barWidth = Math.max(1, (width - gap * (barCount - 1)) / barCount);
        const playing = !audio.paused && !audio.ended;
        const progress = Number.isFinite(audio.duration) && audio.duration > 0
            ? Math.min(1, Math.max(0, audio.currentTime / audio.duration))
            : 0;
        const drawBars = (color) => {
            context.fillStyle = color;
            for (let index = 0; index < barCount; index += 1) {
                const start = waveform
                    ? Math.floor((index / barCount) * waveform.length)
                    : 0;
                const end = waveform
                    ? Math.max(start + 1, Math.ceil(((index + 1) / barCount) * waveform.length))
                    : 0;
                let level = 0;
                for (let sample = start; sample < end; sample += 1) {
                    level = Math.max(level, waveform[sample] || 0);
                }
                const barHeight = Math.max(1.5 * scale, level * height * .88);
                const x = index * (barWidth + gap);
                context.fillRect(x, (height - barHeight) / 2, barWidth, barHeight);
            }
        };
        if (waveform) {
            drawBars("rgba(255, 255, 255, .34)");
            if (progress > 0) {
                context.save();
                context.beginPath();
                context.rect(0, 0, width * progress, height);
                context.clip();
                drawBars("#67d58b");
                context.restore();
            }
        }
        animationFrame = playing ? requestAnimationFrame(draw) : 0;
    };

    const setPlayingState = (playing) => {
        button.classList.toggle("iat-output-spectrum-playing", playing);
        button.setAttribute("aria-pressed", String(playing));
        button.setAttribute("aria-label", `${playing ? t("pause") : t("play")} ${label}`);
        button.title = `${playing ? t("pause") : t("play")} ${label}`;
        if (animationFrame) cancelAnimationFrame(animationFrame);
        animationFrame = requestAnimationFrame(draw);
    };

    const controller = {
        pause() {
            audio.pause();
        },
        dispose() {
            disposed = true;
            audio.pause();
            if (activeAudioController === controller) activeAudioController = null;
            if (animationFrame) cancelAnimationFrame(animationFrame);
            waveformRequest.abort();
            resizeObserver.disconnect();
        },
    };

    button.addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!audio.paused) {
            audio.pause();
            return;
        }
        activeAudioController?.pause();
        try {
            await audio.play();
        } catch (error) {
            onError(error.message || `${t("unableRead")} ${label}`);
        }
    });
    button.addEventListener("dblclick", (event) => event.stopPropagation());
    audio.addEventListener("play", () => {
        if (activeAudioController && activeAudioController !== controller) {
            activeAudioController.pause();
        }
        activeAudioController = controller;
        setPlayingState(true);
    });
    audio.addEventListener("pause", () => {
        if (activeAudioController === controller) activeAudioController = null;
        setPlayingState(false);
    });
    audio.addEventListener("ended", () => {
        if (activeAudioController === controller) activeAudioController = null;
        setPlayingState(false);
    });

    const resizeObserver = new ResizeObserver(() => {
        if (audio.paused) draw();
    });
    resizeObserver.observe(canvas);
    animationFrame = requestAnimationFrame(draw);
    loadWaveform();
    return { button, controller };
}

function iconButton(icon, label, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "iat-output-icon-button";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.innerHTML = `<i class="pi ${icon}" aria-hidden="true"></i>`;
    button.addEventListener("click", onClick);
    return button;
}

async function loadWorkflow(item) {
    const response = await fetch(api.apiURL(item.url));
    if (!response.ok) throw new Error(`${t("unableRead")} ${item.name}`);
    const blob = await response.blob();
    const file = new File([blob], item.name, { type: blob.type });
    await app.handleFile(file);
}

function downloadItem(item) {
    const link = document.createElement("a");
    link.href = api.apiURL(item.url);
    link.download = item.name;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
}

function isCanvasDrop(event) {
    const target = event.target;
    return target instanceof Element && Boolean(
        target.closest("canvas, .graph-canvas-container, .litegraph")
    );
}

function readDraggedItem(event) {
    const value = event.dataTransfer?.getData(DRAG_TYPE);
    if (!value) return null;
    try {
        return JSON.parse(value);
    } catch {
        return null;
    }
}

function hasDraggedItem(event) {
    return Array.from(event.dataTransfer?.types || []).includes(DRAG_TYPE);
}

function installCanvasDropHandler() {
    document.addEventListener("dragover", (event) => {
        if (isCanvasDrop(event) && hasDraggedItem(event)) {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
        }
    }, true);

    document.addEventListener("drop", async (event) => {
        const item = isCanvasDrop(event) ? readDraggedItem(event) : null;
        if (!item) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        try {
            await loadWorkflow(item);
        } catch (error) {
            for (const panel of panelInstances) panel.showError(error.message);
        }
    }, true);
}

class OutputBrowserPanel {
    constructor(container) {
        this.container = container;
        this.currentPath = "";
        this.parentPath = "";
        this.query = "";
        this.displayMode = "grid";
        this.sortMode = "time";
        this.items = [];
        this.favorites = readFavorites();
        this.audioControllers = new Set();
        this.searchTimer = 0;
        this.searchRequest = null;
        this.renderShell();
        this.load("");
    }

    renderShell() {
        this.container.classList.add("iat-output-panel");

        const pathBar = document.createElement("div");
        pathBar.className = "iat-output-pathbar";

        this.upButton = iconButton("pi-arrow-up", t("parentFolder"), () => this.load(this.parentPath));
        this.upButton.disabled = true;

        const searchWrap = document.createElement("label");
        searchWrap.className = "iat-output-search";
        searchWrap.innerHTML = '<i class="pi pi-search" aria-hidden="true"></i>';
        const search = document.createElement("input");
        search.type = "search";
        search.placeholder = t("searchAllOutput");
        search.setAttribute("aria-label", t("searchAllOutput"));
        search.addEventListener("input", () => {
            this.query = search.value.trim();
            clearTimeout(this.searchTimer);
            this.searchRequest?.abort();
            if (!this.query) {
                this.load(this.currentPath);
                return;
            }
            const query = this.query;
            this.searchTimer = window.setTimeout(() => this.search(query), 250);
        });
        this.searchInput = search;
        searchWrap.append(search);

        this.breadcrumbs = document.createElement("nav");
        this.breadcrumbs.className = "iat-output-breadcrumbs";
        this.breadcrumbs.setAttribute("aria-label", t("outputFolderPath"));
        pathBar.append(
            this.upButton,
            this.breadcrumbs,
            iconButton("pi-refresh", t("refresh"), () => this.refresh()),
        );

        this.favoritesBar = document.createElement("nav");
        this.favoritesBar.className = "iat-output-favorites";
        this.favoritesBar.setAttribute("aria-label", t("favoriteOutputFolders"));

        const controls = document.createElement("div");
        controls.className = "iat-output-controls";

        const viewOptions = document.createElement("div");
        viewOptions.className = "iat-output-view-options";

        const viewSwitch = document.createElement("div");
        viewSwitch.className = "iat-output-segmented";
        viewSwitch.setAttribute("role", "group");
        viewSwitch.setAttribute("aria-label", t("displayMode"));
        const addViewButton = (mode, icon, label) => {
            const button = iconButton(icon, label, () => this.setDisplayMode(mode));
            button.dataset.mode = mode;
            viewSwitch.append(button);
            return button;
        };
        this.gridButton = addViewButton("grid", "pi-images", t("thumbnailView"));
        this.listButton = addViewButton("list", "pi-list", t("listView"));

        const sortWrap = document.createElement("label");
        sortWrap.className = "iat-output-sort";
        sortWrap.innerHTML = '<i class="pi pi-sort-alt" aria-hidden="true"></i>';
        const sort = document.createElement("select");
        sort.setAttribute("aria-label", t("sortBy"));
        sort.append(new Option(t("updated"), "time"), new Option(t("name"), "name"));
        sort.value = this.sortMode;
        sort.addEventListener("change", () => {
            this.sortMode = sort.value;
            this.renderItems();
        });
        sortWrap.append(sort);
        viewOptions.append(viewSwitch, sortWrap);
        controls.append(searchWrap, viewOptions);

        this.status = document.createElement("div");
        this.status.className = "iat-output-status";
        this.status.hidden = true;

        this.content = document.createElement("div");
        this.content.className = "iat-output-content";
        this.content.dataset.view = this.displayMode;
        this.updateDisplayButtons();
        this.container.replaceChildren(
            pathBar,
            this.favoritesBar,
            controls,
            this.status,
            this.content,
        );
        this.renderFavorites();
    }

    async load(path) {
        clearTimeout(this.searchTimer);
        this.searchRequest?.abort();
        this.searchRequest = null;
        this.query = "";
        if (this.searchInput) this.searchInput.value = "";
        this.setLoading(true);
        this.hideError();
        try {
            const response = await api.fetchApi(`/iat/output-browser?path=${encodeURIComponent(path)}`);
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || t("unableOutputFolder"));
            this.currentPath = data.path;
            this.parentPath = data.parent;
            this.items = [...data.directories.map((item) => ({ ...item, kind: "folder" })), ...data.files];
            this.upButton.disabled = !data.path;
            this.renderBreadcrumbs();
            this.renderItems();
        } catch (error) {
            this.showError(error.message);
            this.content.replaceChildren();
        } finally {
            this.setLoading(false);
        }
    }

    async search(query) {
        this.searchRequest?.abort();
        const request = new AbortController();
        this.searchRequest = request;
        this.setLoading(true);
        this.hideError();
        try {
            const response = await api.fetchApi(
                `/iat/output-browser?search=${encodeURIComponent(query)}`,
                { signal: request.signal }
            );
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || t("unableSearch"));
            if (this.query !== query || request.signal.aborted) return;
            this.items = [...data.directories.map((item) => ({ ...item, kind: "folder" })), ...data.files];
            this.renderItems();
        } catch (error) {
            if (error.name !== "AbortError") {
                this.showError(error.message);
                this.content.replaceChildren();
            }
        } finally {
            if (this.searchRequest === request) {
                this.searchRequest = null;
                this.setLoading(false);
            }
        }
    }

    refresh() {
        if (this.query) this.search(this.query);
        else this.load(this.currentPath);
    }

    setLoading(loading) {
        this.container.classList.toggle("iat-output-loading", loading);
    }

    showError(message) {
        this.status.textContent = message;
        this.status.hidden = false;
    }

    hideError() {
        this.status.hidden = true;
        this.status.textContent = "";
    }

    renderBreadcrumbs() {
        const parts = this.currentPath ? this.currentPath.split("/") : [];
        const nodes = [];
        const addButton = (label, path) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "iat-output-crumb";
            protectUserText(button);
            button.textContent = label;
            button.title = path ? `output/${path}` : "output";
            button.disabled = path === this.currentPath;
            button.addEventListener("click", () => this.load(path));
            nodes.push(button);
        };

        addButton(t("output"), "");
        parts.forEach((part, index) => {
            const separator = document.createElement("i");
            separator.className = "pi pi-chevron-right iat-output-crumb-separator";
            separator.setAttribute("aria-hidden", "true");
            nodes.push(separator);
            addButton(part, parts.slice(0, index + 1).join("/"));
        });
        this.breadcrumbs.replaceChildren(...nodes);
        this.breadcrumbs.scrollLeft = this.breadcrumbs.scrollWidth;
    }

    isFavorite(path) {
        return this.favorites.includes(path);
    }

    toggleFavorite(path) {
        this.favorites = this.isFavorite(path)
            ? this.favorites.filter((favorite) => favorite !== path)
            : [...this.favorites, path];
        writeFavorites(this.favorites);
        this.renderFavorites();
        this.syncFavoriteButtons();
    }

    syncFavoriteButtons() {
        for (const button of this.content.querySelectorAll(".iat-output-folder-favorite")) {
            this.updateFavoriteButton(button, button.dataset.path, button.dataset.name);
        }
    }

    updateFavoriteButton(button, path, name) {
        const active = this.isFavorite(path);
        const icon = button.querySelector(".pi");
        button.classList.toggle("iat-output-favorite-active", active);
        button.setAttribute("aria-pressed", String(active));
        button.setAttribute("aria-label", `${active ? t("removeFromFavorites") : t("addToFavorites")} ${name}`);
        button.title = active ? t("removeFromFavorites") : t("addToFavorites");
        icon?.classList.toggle("pi-star", !active);
        icon?.classList.toggle("pi-star-fill", active);
    }

    renderFavorites() {
        const nodes = this.favorites.map((path) => {
            const entry = document.createElement("span");
            entry.className = "iat-output-favorite-entry";

            const open = document.createElement("button");
            open.type = "button";
            open.className = "iat-output-favorite-link";
            open.title = `output/${path}`;
            const icon = document.createElement("i");
            icon.className = "pi pi-folder";
            icon.setAttribute("aria-hidden", "true");
            const name = document.createElement("span");
            protectUserText(name);
            name.textContent = path.split("/").pop() || path;
            open.append(icon, name);
            open.addEventListener("click", () => this.load(path));

            const remove = iconButton("pi-star-fill", `${t("removeFromFavorites")} ${name.textContent}`, () => {
                this.toggleFavorite(path);
            });
            remove.classList.add("iat-output-favorite-remove");
            entry.append(open, remove);
            return entry;
        });
        this.favoritesBar.replaceChildren(...nodes);
        this.favoritesBar.hidden = nodes.length === 0;
    }

    setDisplayMode(mode) {
        if (this.displayMode === mode) return;
        this.displayMode = mode;
        this.content.dataset.view = mode;
        this.updateDisplayButtons();
    }

    updateDisplayButtons() {
        for (const [button, mode] of [[this.gridButton, "grid"], [this.listButton, "list"]]) {
            const active = this.displayMode !== mode;
            button.classList.toggle("iat-output-active", active);
            button.setAttribute("aria-pressed", String(active));
        }
    }

    renderItems() {
        this.clearAudioControllers();
        const visible = this.items
            .sort((left, right) => {
                const typeOrder = (kindOrder[left.kind] ?? 5) - (kindOrder[right.kind] ?? 5);
                if (typeOrder) return typeOrder;
                if (this.sortMode === "time") {
                    const timeOrder = itemTime(right) - itemTime(left);
                    if (timeOrder) return timeOrder;
                    return englishCollator.compare(left.name, right.name);
                }
                return compareNames(left.name, right.name);
            });
        if (!visible.length) {
            const empty = document.createElement("div");
            empty.className = "iat-output-empty";
            const emptyText = document.createElement("span");
            emptyText.textContent = t("noFilesFound");
            empty.innerHTML = '<i class="pi pi-images" aria-hidden="true"></i>';
            empty.append(emptyText);
            this.content.replaceChildren(empty);
            return;
        }
        const folders = visible.filter((item) => item.kind === "folder");
        const media = visible.filter((item) => item.kind !== "folder");
        const sections = [];
        if (folders.length) {
            const folderGrid = document.createElement("div");
            folderGrid.className = "iat-output-folder-grid";
            folderGrid.append(...folders.map((item) => this.createFolder(item)));
            sections.push(folderGrid);
        }
        if (media.length) {
            const mediaGrid = document.createElement("div");
            mediaGrid.className = "iat-output-media-grid";
            mediaGrid.append(...media.map((item) => this.createMedia(item)));
            sections.push(mediaGrid);
        }
        this.content.replaceChildren(...sections);
    }

    createFolder(item) {
        const folder = document.createElement("div");
        folder.className = "iat-output-folder";
        folder.title = item.path;
        const open = document.createElement("button");
        open.type = "button";
        open.className = "iat-output-folder-open";
        open.setAttribute("aria-label", `${t("open")} ${item.name}`);
        const preview = document.createElement("span");
        preview.className = "iat-output-folder-preview";
        preview.innerHTML = '<i class="pi pi-folder" aria-hidden="true"></i>';
        const details = document.createElement("span");
        details.className = "iat-output-folder-details";
        const name = document.createElement("span");
        name.className = "iat-output-name";
        protectUserText(name);
        name.textContent = item.name;
        const date = document.createElement("span");
        date.className = "iat-output-meta";
        date.textContent = formatDateTime(item.modified);
        date.title = date.textContent;
        const count = document.createElement("span");
        count.className = "iat-output-meta";
        const itemCount = item.item_count || 0;
        count.textContent = `${itemCount} ${itemCount === 1 ? t("item") : t("items")}`;
        count.title = count.textContent;
        details.append(name, count, date);
        const favorite = iconButton("pi-star", `${t("addToFavorites")} ${item.name}`, (event) => {
            event.stopPropagation();
            this.toggleFavorite(item.path);
        });
        favorite.classList.add("iat-output-folder-favorite");
        favorite.dataset.path = item.path;
        favorite.dataset.name = item.name;
        open.append(preview, details);
        open.addEventListener("click", () => this.load(item.path));
        folder.append(open, favorite);
        this.updateFavoriteButton(favorite, item.path, item.name);
        return folder;
    }

    createMedia(item) {
        const card = document.createElement("article");
        card.className = "iat-output-card";
        card.draggable = true;
        card.title = item.name;
        card.addEventListener("dragstart", (event) => {
            event.dataTransfer.effectAllowed = "copy";
            event.dataTransfer.setData(DRAG_TYPE, JSON.stringify(item));
            event.dataTransfer.setData("text/plain", item.path);
        });
        card.addEventListener("dblclick", () => this.openItem(item));

        const meta = document.createElement("span");
        meta.className = "iat-output-meta";
        meta.textContent = formatSize(item.size);
        const date = document.createElement("span");
        date.className = "iat-output-meta iat-output-date";
        date.textContent = formatDateTime(item.created);
        date.title = date.textContent;

        let preview;
        if (item.kind === "audio") {
            preview = document.createElement("div");
            preview.className = "iat-output-preview iat-output-audio-preview";
            const audio = document.createElement("audio");
            audio.preload = "metadata";
            audio.src = api.apiURL(item.url);
            audio.setAttribute("draggable", "false");
            audio.addEventListener("loadedmetadata", () => updateMediaMeta(audio, item, meta));
            audio.addEventListener("dblclick", (event) => event.stopPropagation());
            const spectrum = createAudioSpectrum(audio, item.name, (message) => this.showError(message));
            this.audioControllers.add(spectrum.controller);
            preview.append(spectrum.button, audio);
        } else if (item.kind === "other") {
            preview = document.createElement("div");
            preview.className = "iat-output-preview iat-output-file-preview";
            preview.innerHTML = '<i class="pi pi-file" aria-hidden="true"></i>';
        } else {
            preview = document.createElement(item.kind === "video" ? "video" : "img");
            preview.className = "iat-output-preview";
            preview.src = api.apiURL(item.url);
            preview.setAttribute("draggable", "false");
        }
        if (item.kind === "video") {
            preview.muted = true;
            preview.loop = true;
            preview.preload = "metadata";
            preview.addEventListener("loadedmetadata", () => updateMediaMeta(preview, item, meta));
            card.addEventListener("mouseenter", () => preview.play().catch(() => {}));
            card.addEventListener("mouseleave", () => {
                preview.pause();
                preview.currentTime = 0;
            });
        } else if (item.kind === "image") {
            preview.loading = "lazy";
            preview.alt = "";
            preview.addEventListener("load", () => updateMediaMeta(preview, item, meta));
        }

        const details = document.createElement("div");
        details.className = "iat-output-details";
        const name = document.createElement("span");
        name.className = "iat-output-name";
        protectUserText(name);
        name.textContent = item.name;
        const actions = document.createElement("div");
        actions.className = "iat-output-actions";
        const download = iconButton("pi-download", t("download"), () => downloadItem(item));
        const load = iconButton("pi-file-import", t("loadWorkflow"), () => this.openItem(item));
        actions.append(download, load);
        details.append(name, meta, date, actions);
        card.append(preview, details);
        return card;
    }

    async openItem(item) {
        this.hideError();
        try {
            await loadWorkflow(item);
        } catch (error) {
            this.showError(error.message);
        }
    }

    clearAudioControllers() {
        for (const controller of this.audioControllers) controller.dispose();
        this.audioControllers.clear();
    }

    destroy() {
        this.clearAudioControllers();
        panelInstances.delete(this);
        this.container.replaceChildren();
    }
}

app.registerExtension({
    name: "ComfyUI.IAT.OutputBrowser",
    setup() {
        installStyles();
        installCanvasDropHandler();
        app.extensionManager.registerSidebarTab({
            id: "iat-output-browser",
            icon: "pi pi-images",
            title: t("outputBrowser"),
            tooltip: t("outputBrowser"),
            type: "custom",
            render: (container) => {
                const panel = new OutputBrowserPanel(container);
                panelInstances.add(panel);
                return () => panel.destroy();
            },
        });
    },
});
