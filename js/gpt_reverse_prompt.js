import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "GPTReversePrompt by IAT";
const PROVIDER_DEFAULTS = {
    "OpenAI-Compatible": {
        base_url: "https://api.openai.com/v1",
        model: "gpt-4.1-mini",
    },
    "Gemini": {
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        model: "gemini-2.5-flash",
    },
    "Qwen OpenAI-Compatible": {
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: "qwen-vl-plus",
    },
};

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function uniqueStrings(values) {
    return [...new Set(values.filter((value) => typeof value === "string" && value.trim()))];
}

function getFetchApi() {
    if (api?.fetchApi) {
        return api.fetchApi.bind(api);
    }
    if (app?.api?.fetchApi) {
        return app.api.fetchApi.bind(app.api);
    }
    return null;
}

function showToast(severity, summary, detail) {
    app.extensionManager?.toast?.add?.({
        severity,
        summary,
        detail,
        life: severity === "error" ? 8000 : 3500,
    });
}

app.registerExtension({
    name: "comfyui_iat.gpt_reverse_prompt",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);

            if (this.__iatGptWidgetsAttached) {
                return;
            }
            this.__iatGptWidgetsAttached = true;

            const modelWidget = getWidget(this, "model");
            const providerWidget = getWidget(this, "provider");
            const baseUrlWidget = getWidget(this, "base_url");
            if (!modelWidget) {
                return;
            }

            let statusWidget = null;
            let modelSelectWidget = null;

            const setStatus = (message) => {
                if (statusWidget) {
                    statusWidget.value = message;
                }
                this.setDirtyCanvas(true, true);
            };

            const syncSelectOptions = (models = []) => {
                const values = uniqueStrings([`${modelWidget.value ?? ""}`, ...models]);
                const options = values.length ? values : ["[manual input]"];
                modelSelectWidget.options = modelSelectWidget.options || {};
                modelSelectWidget.options.values = options;

                if (typeof modelWidget.value === "string" && modelWidget.value.trim() && options.includes(modelWidget.value)) {
                    modelSelectWidget.value = modelWidget.value;
                } else {
                    modelSelectWidget.value = options[0];
                }
            };

            const applyProviderDefaults = (providerName, force = false) => {
                const defaults = PROVIDER_DEFAULTS[providerName];
                if (!defaults) {
                    return;
                }

                const knownBaseUrls = Object.values(PROVIDER_DEFAULTS).map((item) => item.base_url);
                const knownModels = Object.values(PROVIDER_DEFAULTS).map((item) => item.model);

                if (baseUrlWidget && (force || !`${baseUrlWidget.value ?? ""}`.trim() || knownBaseUrls.includes(baseUrlWidget.value))) {
                    baseUrlWidget.value = defaults.base_url;
                }
                if (force || !`${modelWidget.value ?? ""}`.trim() || knownModels.includes(modelWidget.value)) {
                    modelWidget.value = defaults.model;
                }
                syncSelectOptions([]);
            };

            modelSelectWidget = this.addWidget(
                "combo",
                "available_models",
                modelWidget.value || "[manual input]",
                (value) => {
                    if (typeof value === "string" && value && !value.startsWith("[")) {
                        modelWidget.value = value;
                        setStatus(`Selected model: ${value}`);
                    }
                },
                { values: [modelWidget.value || "[manual input]"] },
            );
            modelSelectWidget.serializeValue = () => undefined;

            const refreshButton = this.addWidget("button", "refresh_models", null, async () => {
                const fetchApi = getFetchApi();
                if (!fetchApi) {
                    const message = "ComfyUI frontend API is unavailable; cannot refresh model list.";
                    setStatus(message);
                    showToast("error", "IAT Model Refresh", message);
                    return;
                }

                const payload = {
                    provider: `${getWidget(this, "provider")?.value ?? "OpenAI-Compatible"}`,
                    api_key: `${getWidget(this, "api_key")?.value ?? ""}`,
                    base_url: `${baseUrlWidget?.value ?? ""}`,
                    timeout_seconds: Number(getWidget(this, "timeout_seconds")?.value ?? 60),
                };

                setStatus("Refreshing models...");
                try {
                    const response = await fetchApi("/iat/gpt/models", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(payload),
                    });

                    let data = null;
                    try {
                        data = await response.json();
                    } catch {
                        data = null;
                    }

                    if (!response.ok || !data?.ok) {
                        throw new Error(data?.error || `Model refresh failed (HTTP ${response.status}).`);
                    }

                    const models = Array.isArray(data.models) ? data.models : [];
                    syncSelectOptions(models);
                    const summary = models.length ? `Loaded ${models.length} models.` : "No models returned.";
                    setStatus(summary);
                    showToast("success", "IAT Model Refresh", summary);
                } catch (err) {
                    const detail = err?.message || "Model refresh failed.";
                    setStatus(detail);
                    showToast("error", "IAT Model Refresh", detail);
                    console.error("[IAT] refresh_models failed", err);
                }
            });
            refreshButton.serializeValue = () => undefined;

            statusWidget = this.addWidget(
                "text",
                "model_status",
                "Edit api_key/base_url, then click refresh_models to query /models.",
                () => {},
                { multiline: false },
            );
            statusWidget.serializeValue = () => undefined;

            if (providerWidget) {
                const originalProviderCallback = providerWidget.callback;
                providerWidget.callback = (value, ...args) => {
                    originalProviderCallback?.call(providerWidget, value, ...args);
                    applyProviderDefaults(value, true);
                    setStatus(`Provider switched to ${value}.`);
                };
                applyProviderDefaults(providerWidget.value);
            }

            syncSelectOptions([]);
        };
    },
});
